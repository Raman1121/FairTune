import os

os.environ["TORCH_HOME"] = os.path.dirname(os.getcwd())

import datetime
import random
import sys
import re
import time
import warnings
import timm
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import presets
import torch
import torch.utils.data
import torchvision
import transforms
from utils import *
from training_utils import *
from parse_args import *
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
import yaml
from pprint import pprint

import optuna
from optuna.trial import TrialState

############################################################################

def create_opt_mask(trial, args, num_blocks):

    mask_length = None
    #if(args.tuning_method == 'tune_full_block' or args.tuning_method == 'tune_attention_blocks_random'):
    if(args.tuning_method in ['tune_full_block', 'tune_attention_blocks_random', 'tune_layernorm_blocks_random', 'tune_attention_layernorm', 'tune_attention_mlp', 'tune_layernorm_mlp', 'tune_attention_layernorm_mlp']):
        mask_length = num_blocks
    elif(args.tuning_method == 'auto_peft1'):
        mask_length = num_blocks * 3
    elif(args.tuning_method == 'tune_attention_params_random' or args.tuning_method == 'auto_peft2'):
        mask_length = num_blocks * 4
    elif(args.tuning_method == 'fullft'):
        return None
    else:
        raise NotImplementedError

    mask = np.zeros(mask_length, dtype=np.int8)

    for i in range(mask_length):
        mask[i] = trial.suggest_int("Mask Idx {}".format(i), 0, 1)

    return mask

def create_model(args):

    print("Creating model")
    print("TUNING METHOD: ", args.tuning_method)
    model = utils.get_timm_model(args.model, num_classes=args.num_classes)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in [
            "class_token",
            "position_embedding",
            "relative_position_bias_table",
        ]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))

    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay
        if len(custom_keys_weight_decay) > 0
        else None,
    )

    return model, parameters

def create_results_df(args):

    test_results_df = None
    
    if(args.sens_attribute == 'gender'):
        if(args.use_metric == 'acc'):
            test_results_df = pd.DataFrame(
                    columns=[
                        "Tuning Method",
                        "Train Percent",
                        "LR",
                        "Test Acc Overall",
                        "Test Acc Male",
                        "Test Acc Female",
                        "Test Acc Difference",
                    ]
                )  
        elif(args.use_metric == 'auc'):
            test_results_df = pd.DataFrame(
                    columns=[
                        "Tuning Method",
                        "Train Percent",
                        "LR",
                        "Test AUC Overall",
                        "Test AUC Male",
                        "Test AUC Female",
                        "Test AUC Difference",
                    ]
                )  
    elif(args.sens_attribute == 'skin_type' or args.sens_attribute == 'age' or args.sens_attribute == 'race' or args.sens_attribute == 'age_sex'):
        if(args.use_metric == 'acc'):
            test_results_df = pd.DataFrame(
                    columns=[
                        "Tuning Method",
                        "Train Percent",
                        "LR",
                        "Test Acc Overall",
                        "Test Acc (Best)",
                        "Test Acc (Worst)",
                        "Test Acc Difference",
                    ]
                )  
        elif(args.use_metric == 'auc'):
            test_results_df = pd.DataFrame(
                    columns=[
                        "Tuning Method",
                        "Train Percent",
                        "LR",
                        "Test AUC Overall",
                        "Test AUC (Best)",
                        "Test AUC (Worst)",
                        "Test AUC Difference",
                    ]
                )  
    else:
        raise NotImplementedError

    return test_results_df

def define_dataloaders(args):
    with open("config.yaml") as file:
        yaml_data = yaml.safe_load(file)

    print("Creating dataset")
    (
            dataset,
            dataset_val,
            dataset_test,
            train_sampler,
            val_sampler,
            test_sampler,
    ) = get_fairness_data(args, yaml_data)

    args.num_classes = len(dataset.classes)
    print("DATASET: ", args.dataset)
    print("Size of training dataset: ", len(dataset))
    print("Size of validation dataset: ", len(dataset_val))
    print("Size of test dataset: ", len(dataset_test))
    print("Number of classes: ", args.num_classes)
    pprint(dataset.class_to_idx)

    collate_fn = None
    mixup_transforms = get_mixup_transforms(args)

    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    print("Creating data loaders")

    if(args.dataset == 'papila'):
        drop_last = False
    else:
        drop_last = True

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=drop_last
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        #drop_last=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )

    return data_loader, data_loader_val, data_loader_test

def objective(trial):

    args = get_args_parser().parse_args()
    device = torch.device(args.device)

    if(args.dev_mode):
        args.disable_plotting = True
        args.disable_checkpointing = True

    # try:
    #     _temp_trainable_params_df = pd.read_csv('_temp_trainable_params_df.csv')
    # except:
    #     _temp_trainable_params_df = pd.DataFrame(columns=['Trainable Params'])

    # Saving results to a dataframe
    results_df_savedir = os.path.join(args.model, args.dataset, "Optuna Results")
    if not os.path.exists(results_df_savedir):
        os.makedirs(results_df_savedir, exist_ok=True)
    results_df_name = "Fairness_Optuna_" + args.sens_attribute + "_" + args.tuning_method + "_" + args.model + "_" + args.objective_metric + ".csv"

    if("auc" in args.objective_metric):
        args.use_metric = 'auc'

    try:
        results_df = pd.read_csv(os.path.join(results_df_savedir, results_df_name))
    except:
        results_df = create_results_df(args)

    args.output_dir = os.path.join(os.getcwd(), args.model, args.dataset)

    args.distributed = False
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # Create datasets and dataloaders here
    data_loader, data_loader_val, data_loader_test = define_dataloaders(args)

    # Create the model here
    model, parameters = create_model(args)

    mask = create_opt_mask(trial, args, len(model.blocks))
    print("Mask: ", mask)
    print("\n")

    masking_vector = utils.get_masked_model(model, args.tuning_method, mask=list(mask))

    trainable_params, all_param = utils.check_tunable_params(model, True)
    trainable_percentage = 100 * trainable_params / all_param

    # _row = [round(trainable_percentage, 3)]
    # _temp_trainable_params_df.loc[len(_temp_trainable_params_df)] = _row
    # _temp_trainable_params_df.to_csv('_temp_trainable_params_df.csv', index=False)

    model.to(device)

    # Create the optimizer, criterion, lr_scheduler here
    criterion = nn.CrossEntropyLoss(
                label_smoothing=args.label_smoothing, reduction="none"
            )
    ece_criterion = utils.ECELoss()
    args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer = get_optimizer(args, parameters)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None    

    lr_scheduler = get_lr_scheduler(args, optimizer)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()

    model_ema = None

    for epoch in range(args.start_epoch, args.epochs):
        
        train_one_epoch_fairness(
            model,
            criterion,
            ece_criterion,
            optimizer,
            data_loader,
            device,
            epoch,
            args,
            model_ema,
            scaler,
        )
        lr_scheduler.step()

        if args.sens_attribute == "gender":
            (
                val_acc,
                val_male_acc,
                val_female_acc,
                val_auc,
                val_male_auc,
                val_female_auc,
                val_loss,
                val_max_loss,
            ) = evaluate_fairness_gender(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )
            print(
                "Val Acc: {:.2f}, Val Male Acc {:.2f}, Val Female Acc {:.2f}, Val Loss: {:.2f}, Val MAX LOSS: {:.2f}".format(
                    val_acc,
                    val_male_acc,
                    val_female_acc,
                    torch.mean(val_loss),
                    val_max_loss,
                )
            )
            print(
                    "Val AUC: {:.2f}, Val Male AUC {:.2f}, Val Female AUC {:.2f}".format(
                        val_auc,
                        val_male_auc,
                        val_female_auc,
                    )
                )

        elif args.sens_attribute == "skin_type":
            if(args.skin_type == 'multi'):
                (
                    val_acc,
                    val_acc_type0,
                    val_acc_type1,
                    val_acc_type2,
                    val_acc_type3,
                    val_acc_type4,
                    val_acc_type5,
                    val_auc,
                    val_auc_type0,
                    val_auc_type1,
                    val_auc_type2,
                    val_auc_type3,
                    val_auc_type4,
                    val_auc_type5,
                    val_loss,
                    val_max_loss,
                ) = evaluate_fairness_skin_type(
                    model,
                    criterion,
                    ece_criterion,
                    data_loader_val,
                    args=args,
                    device=device,
                )
                print(
                    "Val Acc: {:.2f}, Val Type 0 Acc: {:.2f}, Val Type 1 Acc: {:.2f}, Val Type 2 Acc: {:.2f}, Val Type 3 Acc: {:.2f}, Val Type 4 Acc: {:.2f}, Val Type 5 Acc: {:.2f}, Val Loss: {:.2f}, Val MAX LOSS: {:.2f}".format(
                        val_acc,
                        val_acc_type0,
                        val_acc_type1,
                        val_acc_type2,
                        val_acc_type3,
                        val_acc_type4,
                        val_acc_type5,
                        torch.mean(val_loss),
                        val_max_loss,
                    )
                )
                print(
                    "Val AUC: {:.2f}, Val Type 0 AUC: {:.2f}, Val Type 1 AUC: {:.2f}, Val Type 2 AUC: {:.2f}, Val Type 3 AUC: {:.2f}, Val Type 4 AUC: {:.2f}, Val Type 5 AUC: {:.2f}".format(
                        val_auc,
                        val_auc_type0,
                        val_auc_type1,
                        val_auc_type2,
                        val_auc_type3,
                        val_auc_type4,
                        val_auc_type5,
                    )
                )
                print("\n")
            elif(args.skin_type == 'binary'):
                (
                    val_acc,
                    val_acc_type0,
                    val_acc_type1,
                    val_auc,
                    val_auc_type0,
                    val_auc_type1,
                    val_loss,
                    val_max_loss,
                ) = evaluate_fairness_skin_type_binary(
                    model,
                    criterion,
                    ece_criterion,
                    data_loader_val,
                    args=args,
                    device=device,
                )
                print(
                    "Val Acc: {:.2f}, Val Type 0 Acc: {:.2f}, Val Type 1 Acc: {:.2f}, Val Loss: {:.2f}, Val MAX LOSS: {:.2f}".format(
                        val_acc,
                        val_acc_type0,
                        val_acc_type1,
                        torch.mean(val_loss),
                        val_max_loss,
                    )
                )
                print(
                    "Val AUC: {:.2f}, Val Type 0 AUC: {:.2f}, Val Type 1 AUC: {:.2f}".format(
                        val_auc,
                        val_auc_type0,
                        val_auc_type1,
                    )
                )
                print("\n")


        elif args.sens_attribute == "age":
            if(args.age_type == 'multi'):
                (
                    val_acc,
                    acc_age0_avg,
                    acc_age1_avg,
                    acc_age2_avg,
                    acc_age3_avg,
                    acc_age4_avg,
                    val_auc,
                    auc_age0_avg,
                    auc_age1_avg,
                    auc_age2_avg,
                    auc_age3_avg,
                    auc_age4_avg,
                    val_loss,
                    val_max_loss,
                ) = evaluate_fairness_age(
                    model,
                    criterion,
                    ece_criterion,
                    data_loader_val,
                    args=args,
                    device=device,
                )
                print(
                    "Val Acc: {:.2f}, Val Age Group0 Acc: {:.2f}, Val Age Group1 Acc: {:.2f}, Val Age Group2 Acc: {:.2f}, Val Age Group3 Acc: {:.2f}, Val Age Group4 Acc: {:.2f}, Val Loss: {:.2f}, Val MAX LOSS: {:.2f}".format(
                        val_acc,
                        acc_age0_avg,
                        acc_age1_avg,
                        acc_age2_avg,
                        acc_age3_avg,
                        acc_age4_avg,
                        torch.mean(val_loss),
                        val_max_loss,
                    )
                )
                print(
                    "Val AUC: {:.2f}, Val Age Group0 AUC: {:.2f}, Val Age Group1 AUC: {:.2f}, Val Age Group2 AUC: {:.2f}, Val Age Group3 AUC: {:.2f}, Val Age Group4 AUC: {:.2f}".format(
                        val_auc,
                        auc_age0_avg,
                        auc_age1_avg,
                        auc_age2_avg,
                        auc_age3_avg,
                        auc_age4_avg,
                    )
                )
            elif(args.age_type == 'binary'):
                (
                    val_acc,
                    acc_age0_avg,
                    acc_age1_avg,
                    val_auc,
                    auc_age0_avg,
                    auc_age1_avg,
                    val_loss,
                    val_max_loss,
                ) = evaluate_fairness_age_binary(
                    model,
                    criterion,
                    ece_criterion,
                    data_loader_val,
                    args=args,
                    device=device,
                )
                print(
                    "Val Acc: {:.2f}, Val Age Group0 Acc: {:.2f}, Val Age Group1 Acc: {:.2f}, Val Loss: {:.2f}, Val MAX LOSS: {:.2f}".format(
                        val_acc,
                        acc_age0_avg,
                        acc_age1_avg,
                        torch.mean(val_loss),
                        val_max_loss,
                    )
                )
                print(
                    "Val AUC: {:.2f}, Val Age Group0 AUC: {:.2f}, Val Age Group1 AUC: {:.2f}".format(
                        val_auc,
                        auc_age0_avg,
                        auc_age1_avg,
                    )
                )
                print("\n")
            else:
                raise NotImplementedError("Age type not supported. Choose from 'multi' or 'binary'")
        
        elif(args.sens_attribute == 'race'):
            if(args.cal_equiodds):
                val_acc, acc_race0_avg, acc_race1_avg, val_auc, auc_race0_avg, auc_race1_avg, val_loss, val_max_loss, equiodds_diff, equiodds_ratio, dpd, dpr = evaluate_fairness_race_binary(model, criterion, ece_criterion, data_loader_val, args=args, device=device)
            else:
                val_acc, acc_race0_avg, acc_race1_avg, val_auc, auc_race0_avg, auc_race1_avg, val_loss, val_max_loss = evaluate_fairness_race_binary(model, criterion, ece_criterion, data_loader_val, args=args, device=device)
        
        elif(args.sens_attribute == 'age_sex'):
            assert args.dataset == 'chexpert'
            if(args.cal_equiodds):
                (
                    val_acc, val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3, val_auc, val_auc_type0, val_auc_type1, val_auc_type2, val_auc_type3, val_loss, val_max_loss, equiodds_diff,  equiodds_ratio,  dpd,  dpr
                ) = evaluate_fairness_age_sex(
                    model,
                    criterion,
                    ece_criterion,
                    data_loader_val,
                    args=args,
                    device=device,
                )
            else:
                (val_acc, val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3, val_auc, val_auc_type0, val_auc_type1, val_auc_type2, val_auc_type3, val_loss, val_max_loss
                ) = evaluate_fairness_age_sex(
                    model,
                    criterion,
                    ece_criterion,
                    data_loader_val,
                    args=args,
                    device=device,
                )

        else:
            raise NotImplementedError("Sensitive attribute not implemented")

        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
        
            if model_ema:
                    checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            # utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoints', f"model_{epoch}.pth"))7
            ckpt_path = os.path.join(
                args.output_dir,
                "checkpoints",
                "checkpoint_" + args.tuning_method + ".pth",
            )
            if not args.disable_checkpointing:
                utils.save_on_master(checkpoint, ckpt_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    # Obtaining the performance on val set
    print("Training finished | Evaluating on the val set")

    if args.sens_attribute == "gender":
        (
            val_acc,
            val_male_acc,
            val_female_acc,
            val_auc,
            val_male_auc,
            val_female_auc,
            val_loss,
            val_max_loss
        ) = evaluate_fairness_gender(
            model,
            criterion,
            ece_criterion,
            data_loader_val,
            args=args,
            device=device,
        )

        max_acc = max(val_male_acc, val_female_acc)
        min_acc = min(val_male_acc, val_female_acc)
        acc_diff = abs(max_acc - min_acc)

        max_auc = max(val_male_auc, val_female_auc)
        min_auc = min(val_male_auc, val_female_auc)
        auc_diff = abs(max_auc - min_auc)

        print("\n")
        print("Val Male Accuracy: ", val_male_acc)
        print("Val Female Accuracy: ", val_female_acc)
        print("Difference in sub-group performance: ", acc_diff)
        print("\n")
        print("Val Male AUC: ", val_male_auc)
        print("Val Female AUC: ", val_female_auc)
        print("Difference in sub-group performance (AUC): ", auc_diff)

    elif args.sens_attribute == "skin_type":
        if(args.skin_type == 'multi'):
            (
                val_acc,
                val_acc_type0,
                val_acc_type1,
                val_acc_type2,
                val_acc_type3,
                val_acc_type4,
                val_acc_type5,
                val_auc,
                val_auc_type0,
                val_auc_type1,
                val_auc_type2,
                val_auc_type3,
                val_auc_type4,
                val_auc_type5,
                val_loss,
                val_max_loss
            ) = evaluate_fairness_skin_type(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )

            max_acc = max(val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3, val_acc_type4, val_acc_type5)
            min_acc = min(val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3, val_acc_type4, val_acc_type5)
            acc_diff = abs(max_acc - min_acc)

            max_auc = max(val_auc_type0, val_auc_type1, val_auc_type2, val_auc_type3, val_auc_type4, val_auc_type5)
            min_auc = min(val_auc_type0, val_auc_type1, val_auc_type2, val_auc_type3, val_auc_type4, val_auc_type5)
            auc_diff = abs(max_auc - min_auc)

            print("\n")
            print("Val Type 0 Accuracy: ", val_acc_type0)
            print("Val Type 1 Accuracy: ", val_acc_type1)
            print("Val Type 2 Accuracy: ", val_acc_type2)
            print("Val Type 3 Accuracy: ", val_acc_type3)
            print("Val Type 4 Accuracy: ", val_acc_type4)
            print("Val Type 5 Accuracy: ", val_acc_type5)
            print("Difference in sub-group performance (Accuracy): ", acc_diff)

            print("\n")
            print("Val Type 0 AUC: ", val_auc_type0)
            print("Val Type 1 AUC: ", val_auc_type1)
            print("Val Type 2 AUC: ", val_auc_type2)
            print("Val Type 3 AUC: ", val_auc_type3)
            print("Val Type 4 AUC: ", val_auc_type4)
            print("Val Type 5 AUC: ", val_auc_type5)
            print("Difference in sub-group performance (AUC): ", auc_diff)

        elif(args.skin_type == 'binary'):
            (
                val_acc,
                val_acc_type0,
                val_acc_type1,
                val_auc,
                val_auc_type0,
                val_auc_type1,
                val_loss,
                val_max_loss,
            ) = evaluate_fairness_skin_type_binary(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )

            max_acc = max(val_acc_type0, val_acc_type1)
            min_acc = min(val_acc_type0, val_acc_type1)
            acc_diff = abs(max_acc - min_acc)

            max_auc = max(val_auc_type0, val_auc_type1)
            min_auc = min(val_auc_type0, val_auc_type1)
            auc_diff = abs(max_auc - min_auc)

            print("\n")
            print("Val Type 0 Accuracy: ", val_acc_type0)
            print("Val Type 1 Accuracy: ", val_acc_type1)
            print("Difference in sub-group performance (Accuracy): ", acc_diff)

            print("\n")
            print("Overall Val AUC: ", val_auc)
            print("Val Type 0 AUC: ", val_auc_type0)
            print("Val Type 1 AUC: ", val_auc_type1)
            print("Difference in sub-group performance (AUC): ", auc_diff)


    elif(args.sens_attribute == 'age'):
        if(args.age_type == 'multi'):
            (
                val_acc,
                acc_age0_avg,
                acc_age1_avg,
                acc_age2_avg,
                acc_age3_avg,
                acc_age4_avg,
                val_auc,
                auc_age0_avg,
                auc_age1_avg,
                auc_age2_avg,
                auc_age3_avg,
                auc_age4_avg,
                val_loss,
                val_max_loss
            ) = evaluate_fairness_age(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )

            max_acc = max(acc_age0_avg, acc_age1_avg, acc_age2_avg, acc_age3_avg, acc_age4_avg)
            min_acc = min(acc_age0_avg, acc_age1_avg, acc_age2_avg, acc_age3_avg, acc_age4_avg)
            acc_diff = abs(max_acc - min_acc)

            max_auc = max(auc_age0_avg, auc_age1_avg, auc_age2_avg, auc_age3_avg, auc_age4_avg)
            min_auc = min(auc_age0_avg, auc_age1_avg, auc_age2_avg, auc_age3_avg, auc_age4_avg)
            auc_diff = abs(max_auc - min_auc)

            print("\n")
            print("val Age Group 0 Accuracy: ", acc_age0_avg)
            print("val Age Group 1 Accuracy: ", acc_age1_avg)
            print("val Age Group 2 Accuracy: ", acc_age2_avg)
            print("val Age Group 3 Accuracy: ", acc_age3_avg)
            print("val Age Group 4 Accuracy: ", acc_age4_avg)
            print("Difference in sub-group performance (Accuracy): ", acc_diff)

            print("\n")
            print("val Age Group 0 AUC: ", auc_age0_avg)
            print("val Age Group 1 AUC: ", auc_age1_avg)
            print("val Age Group 2 AUC: ", auc_age2_avg)
            print("val Age Group 3 AUC: ", auc_age3_avg)
            print("val Age Group 4 AUC: ", auc_age4_avg)
            print("Difference in sub-group performance (AUC): ", auc_diff)
        
        elif(args.age_type == 'binary'):
            (
                val_acc,
                acc_age0_avg,
                acc_age1_avg,
                val_auc,
                auc_age0_avg,
                auc_age1_avg,
                val_loss,
                val_max_loss
            ) = evaluate_fairness_age_binary(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )

            max_acc = max(acc_age0_avg, acc_age1_avg)
            min_acc = min(acc_age0_avg, acc_age1_avg)
            acc_diff = abs(max_acc - min_acc)

            max_auc = max(auc_age0_avg, auc_age1_avg)
            min_auc = min(auc_age0_avg, auc_age1_avg)
            auc_diff = abs(max_auc - min_auc)

            print("\n")
            print("val Age Group 0 Accuracy: ", acc_age0_avg)
            print("val Age Group 1 Accuracy: ", acc_age1_avg)
            print("Difference in sub-group performance (Accuracy): ", acc_diff)

            print("\n")
            print("val Age Group 0 AUC: ", auc_age0_avg)
            print("val Age Group 1 AUC: ", auc_age1_avg)
            print("Difference in sub-group performance (AUC): ", auc_diff)

        else:
            raise NotImplementedError("Age type not supported. Choose from 'multi' or 'binary'")
        
    elif(args.sens_attribute == 'race'):
        if(args.cal_equiodds):
            val_acc, acc_race0_avg, acc_race1_avg, val_auc, auc_race0_avg, auc_race1_avg, val_loss, val_max_loss, equiodds_diff, equiodds_ratio, dpd, dpr = evaluate_fairness_race_binary(model, criterion, ece_criterion, data_loader_val, args=args, device=device)
        else:
            val_acc, acc_race0_avg, acc_race1_avg, val_auc, auc_race0_avg, auc_race1_avg, val_loss, val_max_loss = evaluate_fairness_race_binary(model, criterion, ece_criterion, data_loader_val, args=args, device=device)

        max_acc = max(acc_race0_avg, acc_race1_avg)
        min_acc = min(acc_race0_avg, acc_race1_avg)
        acc_diff = abs(max_acc - min_acc)

        max_auc = max(auc_race0_avg, auc_race1_avg)
        min_auc = min(auc_race0_avg, auc_race1_avg)
        auc_diff = abs(max_auc - min_auc)

        print("\n")
        print("val Race Group 0 Accuracy: ", acc_race0_avg)
        print("val Race Group 1 Accuracy: ", acc_race1_avg)
        print("Difference in sub-group performance (Accuracy): ", acc_diff)

        print("\n")
        print("val Race Group 0 AUC: ", auc_race0_avg)
        print("val Race Group 1 AUC: ", auc_race0_avg)
        print("Difference in sub-group performance (AUC): ", auc_diff)

    elif(args.sens_attribute == 'age_sex'):
        assert args.dataset == 'chexpert'
        if(args.cal_equiodds):
            (
                val_acc, val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3, val_auc, val_auc_type0, val_auc_type1, val_auc_type2, val_auc_type3, val_loss, val_max_loss, equiodds_diff,  equiodds_ratio,  dpd,  dpr
            ) = evaluate_fairness_age_sex(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )
        else:
            (val_acc, val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3, val_auc, val_auc_type0, val_auc_type1, val_auc_type2, val_auc_type3, val_loss, val_max_loss
            ) = evaluate_fairness_age_sex(
                model,
                criterion,
                ece_criterion,
                data_loader_val,
                args=args,
                device=device,
            )

        max_acc = max(val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3)
        min_acc = min(val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3)
        acc_diff = abs(max_acc - min_acc)

        max_auc = max(val_auc_type0, val_auc_type1, val_auc_type2, val_auc_type3)
        min_auc = min(val_auc_type0, val_auc_type1, val_auc_type2, val_auc_type3)
        auc_diff = abs(max_auc - min_auc)

        print("\n")
        print("val AgeSex Group 0 Accuracy: ", val_acc_type0)
        print("val AgeSex Group 1 Accuracy: ", val_acc_type1)
        print("val AgeSex Group 2 Accuracy: ", val_acc_type2)
        print("val AgeSex Group 3 Accuracy: ", val_acc_type3)
        print("Difference in sub-group performance (Accuracy): ", acc_diff)

        print("\n")
        print("val AgeSex Group 0 AUC: ", val_auc_type0)
        print("val AgeSex Group 1 AUC: ", val_auc_type1)
        print("val AgeSex Group 2 AUC: ", val_auc_type2)
        print("val AgeSex Group 3 AUC: ", val_auc_type3)

    else:
        raise NotImplementedError("Sensitive attribute not implemented")

    print("Val overall accuracy: ", val_acc)
    print("Val Max Accuracy: ", round(max_acc, 3))
    print("Val Min Accuracy: ", round(min_acc, 3))
    print("Val Accuracy Difference: ", round(acc_diff, 3))
    print("Val loss: ", round(torch.mean(val_loss).item(), 3))
    print("Val max loss: ", round(val_max_loss.item(), 3))

    print("Val overall AUC: ", val_auc)
    print("Val Max AUC: ", round(max_auc, 3))
    print("Val Min AUC: ", round(min_auc, 3))
    print("Val AUC Difference: ", round(auc_diff, 3))
    

    # Adding results to the dataframe
    if(args.sens_attribute == 'gender'):
        if(args.use_metric == 'acc'):
            _row = [args.tuning_method, trainable_percentage, args.lr, round(val_acc, 3), round(val_male_acc, 3), round(val_female_acc, 3), round(acc_diff, 3)]
        if(args.use_metric == 'auc'):
            _row = [args.tuning_method, trainable_percentage, args.lr, round(val_auc, 3), round(val_male_auc, 3), round(val_female_auc, 3), round(auc_diff, 3)]

    elif(args.sens_attribute == 'age' or args.sens_attribute == 'skin_type' or args.sens_attribute == 'race' or args.sens_attribute == 'age_sex'):
        if(args.use_metric == 'acc'):
            _row = [args.tuning_method, round(trainable_percentage, 3), args.lr, round(val_acc, 3), round(max_acc, 3), round(min_acc, 3), round(acc_diff, 3)]
        if(args.use_metric == 'auc'):
            _row = [args.tuning_method, round(trainable_percentage, 3), args.lr, round(val_auc, 3), round(max_auc, 3), round(min_auc, 3), round(auc_diff, 3)]

    results_df.loc[len(results_df)] = _row
    print("!!! Saving the results dataframe at {}".format(os.path.join(results_df_savedir, results_df_name)))
    results_df.to_csv(os.path.join(results_df_savedir, results_df_name), index=False)

    # Pruning
    if(args.objective_metric == 'min_acc'):
        trial.report(min_acc, epoch)
    elif(args.objective_metric == 'min_auc'):
        trial.report(min_auc, epoch)

    elif(args.objective_metric == 'acc_diff'):
        trial.report(acc_diff, epoch)
    elif(args.objective_metric == 'auc_diff'):
        trial.report(auc_diff, epoch)

    elif(args.objective_metric == 'max_loss'):
        trial.report(val_max_loss, epoch)

    elif(args.objective_metric == 'overall_acc'):
        trial.report(val_acc, epoch)
    elif(args.objective_metric == 'overall_auc'):
        trial.report(val_auc, epoch)
    else:
        raise NotImplementedError("Objective metric not implemented")

    if(trial.should_prune()):
        raise optuna.exceptions.TrialPruned()    
    
    if(args.objective_metric == 'acc_diff'):
        try:
            return acc_diff.item()
        except:
            return acc_diff

    elif(args.objective_metric == 'auc_diff'):
        try:
            return auc_diff.item()
        except:
            return auc_diff

    elif(args.objective_metric == 'min_acc'):
        try:
            return min_acc.item()
        except:
            return min_acc
    elif(args.objective_metric == 'min_auc'):
        try:
            return min_auc.item()
        except:
            return min_auc


    elif(args.objective_metric == 'max_loss'):
        try:
            return val_max_loss.item()
        except:
            return val_max_loss
    elif(args.objective_metric == 'overall_acc'):
        try:
            return val_acc.item()
        except:
            return val_acc
    elif(args.objective_metric == 'overall_auc'):
        try:
            return val_auc.item()
        except:
            return val_auc
    
    else:
        raise NotImplementedError("Objective metric not implemented")

if __name__ == "__main__":
    
    args = get_args_parser().parse_args()
    args.plots_save_dir = os.path.join(os.getcwd(), "plots", "optuna_plots", args.model, args.dataset, args.tuning_method, args.sens_attribute)

    if(args.dev_mode):
        args.disable_plotting = True
        args.disable_checkpointing = True

    if not os.path.exists(args.plots_save_dir):
        os.makedirs(args.plots_save_dir, exist_ok = True)

    if(args.objective_metric == 'acc_diff' or args.objective_metric == 'auc_diff' or args.objective_metric == 'max_loss'):
        direction = 'minimize'
    elif(args.objective_metric == 'min_acc' or args.objective_metric == 'overall_acc' or args.objective_metric == 'overall_auc' or args.objective_metric == 'min_auc'):
        direction = 'maximize'
    else:
        raise NotImplementedError

    # Pruners
    if(args.pruner == 'SuccessiveHalving'):
        pruner = optuna.pruners.SuccessiveHalvingPruner()
    elif(args.pruner == 'MedianPruner'):
        pruner = optuna.pruners.MedianPruner()
    elif(args.pruner == 'Hyperband'):
        pruner = optuna.pruners.HyperbandPruner()
    else:
        raise NotImplementedError


    if(not args.disable_storage):
        study_name = args.dataset + "_" + args.tuning_method + "_" + args.sens_attribute + "_" + args.objective_metric
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        storage_dir = os.path.join(os.getcwd(), "Optuna_StorageDB")
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        #storage_name = os.path.join(storage_dir, "sqlite:///{}.db".format(study_name))
        storage_name = "sqlite:///{}.db".format(study_name)
        print("!!! Creating the study DB at {}".format(storage_name))
        study = optuna.create_study(direction=direction, pruner=pruner, storage=storage_name)
    else:
        study = optuna.create_study(direction=direction, pruner=pruner)
        
    study.optimize(objective, n_trials=args.num_trials, show_progress_bar=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    #print("Pruned trials: ", pruned_trials)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    best_mask = []
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        if(key == 'lr'):
            continue
        best_mask.append(value)

    # Save the best mask
    best_mask = np.array(best_mask).astype(np.int8)
    mask_savedir = os.path.join(args.model, args.dataset, "Optuna_Masks", args.sens_attribute)
    if not os.path.exists(mask_savedir):
        os.makedirs(mask_savedir)

    print("!!! Saving the best mask at {}".format(os.path.join(mask_savedir, args.tuning_method + "_best_mask_" + args.objective_metric + "_" + str(trial.value) + ".npy")))
    if(not args.dev_mode):
        np.save(os.path.join(mask_savedir, args.tuning_method + "_best_mask_" + args.objective_metric + "_" + str(trial.value) + ".npy"), best_mask)
    

    # Save these results to a dataframe
    stats_df_savedir = os.path.join(args.model, args.dataset, "Optuna Run Stats")
    if not os.path.exists(stats_df_savedir):
        os.makedirs(stats_df_savedir)
    stats_df_name = "Run_Stats_" + args.sens_attribute + "_" + args.tuning_method + "_" + args.model + "_" + args.objective_metric + ".csv"
    best_params_df_name = "Best_Params" + args.sens_attribute + "_" + args.tuning_method + "_" + args.model + "_" + args.objective_metric + ".csv"

    df = study.trials_dataframe()
    try:
        df = df.drop(['datetime_start', 'datetime_complete', 'duration', 'system_attrs_completed_rung_0'], axis=1)     # Drop unnecessary columns
    except:
        pass
    df = df.rename(columns={'value': args.objective_metric})
    df.to_csv(os.path.join(stats_df_savedir, stats_df_name), index=False)

    # Save the best params
    cols = ["Col {}".format(i) for i in range(len(trial.params))]
    best_params_df = pd.DataFrame(columns=cols)
    best_params_df.loc[len(best_params_df)] = list(trial.params.values())
    best_params_df.to_csv(os.path.join(stats_df_savedir, best_params_df_name), index=False)

    #################### Plotting ####################

    # 1. Parameter importance plots

    # a) Bar Plot

    if(not args.disable_plotting):
        try:
            param_imp_plot = optuna.visualization.matplotlib.plot_param_importances(study)
            param_imp_plot.figure.tight_layout()
            param_imp_plot.figure.savefig(os.path.join(args.plots_save_dir, "param_importance_{}.jpg".format(args.objective_metric)), format="jpg")
        except:
            print("Error in plotting parameter importance plot")

        # b) Contour Plot
        try:
            contour_fig = plt.figure()
            contour_plot = optuna.visualization.matplotlib.plot_contour(study)
        except:
            print("Error in plotting contour plot")
        #contour_fig.savefig(os.path.join(args.plots_save_dir, "contour_plot.jpg"), format="jpg")
        
        # print(contour_plot)
        #contour_plot.figure.savefig(os.path.join(args.plots_save_dir, "contour_plot_{}.jpg".format(args.objective_metric)), format="jpg")

        
        # 2. Slice plot
        #fig2 = plt.figure()
        # slice_plot = optuna.visualization.matplotlib.plot_slice(study)
        # print(slice_plot)
        # slice_plot.figure.savefig(os.path.join(args.plots_save_dir, "slice_plot_{}.jpg".format(args.objective_metric)), format="jpg")
        # fig2.add_axes(axes)
        # plt.savefig(os.path.join(args.plots_save_dir, "slice_plot.jpg"), format="jpg")
        # plt.close(fig2)
        
        # 3. Optimization history plot
        try:
            history_plot = optuna.visualization.matplotlib.plot_optimization_history(study)
            history_plot.figure.tight_layout()
            history_plot.figure.savefig(os.path.join(args.plots_save_dir, "optimization_history_{}.jpg".format(args.objective_metric)), format="jpg")
        except:
            print("Error in plotting optimization history plot")

        # 4. High-dimensional parameter relationships plot
        try:
            parallel_plot = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            parallel_plot.figure.tight_layout()
            parallel_plot.figure.savefig(os.path.join(args.plots_save_dir, "parallel_coordinate_{}.jpg".format(args.objective_metric)), format="jpg")
        except:
            print("Error in plotting parallel coordinate plot")

        # 5. Pareto front plot
        try:
            pareto_plot = optuna.visualization.matplotlib.plot_pareto_front(study)
            pareto_plot.figure.tight_layout()
            pareto_plot.figure.savefig(os.path.join(args.plots_save_dir, "pareto_plot_{}.jpg".format(args.objective_metric)), format="jpg")
        except:
            print("Error in plotting Pareto front plot")

        # 6. Parameter Rank plot
        try:
            #param_rank_plot = optuna.visualization.matplotlib.plot_param_importances(study, target=lambda t: t.params[args.objective_metric])
            param_rank_plot = optuna.visualization.matplotlib.plot_rank(study)
            param_rank_plot.figure.tight_layout()
            param_rank_plot.figure.savefig(os.path.join(args.plots_save_dir, "rank_plot_{}.jpg".format(args.objective_metric)), format="jpg")
        except:
            print("Error in plotting parameter rank plot")

        



    
    

