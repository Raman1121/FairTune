import os

os.environ["TORCH_HOME"] = os.path.dirname(os.getcwd())

import datetime
import random
import re
import time
import json
import warnings
import timm
import pandas as pd
import numpy as np
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

"""
    This script would test only the baselines (Full FT and Linear Readout) and the best performing masks after the search.
"""


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
                        "Mask Path"
                    ]
                )  
        elif(args.use_metric == 'auc'):
            if(args.cal_equiodds):
                test_results_df = pd.DataFrame(
                        columns=[
                            "Tuning Method",
                            "Train Percent",
                            "LR",
                            "Test AUC Overall",
                            "Test AUC Male",
                            "Test AUC Female",
                            "Test AUC Difference",
                            "Mask Path",
                            "EquiOdd_diff", 
                            "EquiOdd_ratio", 
                            "DPD", 
                            "DPR", 
                        ]
                    )
            else:
                test_results_df = pd.DataFrame(
                        columns=[
                            "Tuning Method",
                            "Train Percent",
                            "LR",
                            "Test AUC Overall",
                            "Test AUC Male",
                            "Test AUC Female",
                            "Test AUC Difference",
                            "Mask Path"
                        ]
                    ) 
    elif(args.sens_attribute == 'skin_type' or args.sens_attribute == 'age' or args.sens_attribute == 'race' or args.sens_attribute == 'age_sex'):
        if(args.use_metric == 'acc'):
            test_results_df = pd.DataFrame(
                    columns=[
                        "Tuning Method","Train Percent", "LR", "Test Acc Overall", "Test Acc (Best)", "Test Acc (Worst)", "Test Acc Difference", "Mask Path"
                    ]
                )  
        elif(args.use_metric == 'auc'):

            if(args.cal_equiodds):
                cols = ["Tuning Method", "Train Percent", "LR", "Test AUC Overall", "Test AUC (Best)", "Test AUC (Worst)", "Test AUC Difference", "EquiOdd_diff", "EquiOdd_ratio", "DPD", "DPR", "Mask Path"]
            else:
                cols = ["Tuning Method", "Train Percent", "LR", "Test AUC Overall", "Test AUC (Best)", "Test AUC (Worst)", "Test AUC Difference", "Mask Path"]
            test_results_df = pd.DataFrame(
                    columns=cols
                )  
    else:
        raise NotImplementedError

    return test_results_df



def plot_auc(args, AUC_list, best_AUC_list, worst_AUC_list, is_train):

    assert is_train is not None

    print("Saving AUC List")

    ALL_AUC = []
    ALL_BEST_AUC = []
    ALL_WORST_AUC = []

    print(AUC_list)
    print(best_AUC_list)
    print(worst_AUC_list)

    # Handling nan values, replacing them with previous values
    for i in range(len(AUC_list)):
        if(np.isnan(AUC_list[i])):
            ALL_AUC.append(AUC_list[i-1])
        else:
            ALL_AUC.append(AUC_list[i])
        
    for i in range(len(best_AUC_list)):
        if(np.isnan(best_AUC_list[i])):
            ALL_BEST_AUC.append(best_AUC_list[i-1])
        else:
            ALL_BEST_AUC.append(best_AUC_list[i])
        
    for i in range(len(worst_AUC_list)):
        if(np.isnan(worst_AUC_list[i])):
            ALL_WORST_AUC.append(worst_AUC_list[i-1])
        else:
            ALL_WORST_AUC.append(worst_AUC_list[i])

    # List for AUC Difference
    ALL_AUC_DIFF = []
    for i in range(len(ALL_AUC)):
        ALL_AUC_DIFF.append(ALL_BEST_AUC[i] - ALL_WORST_AUC[i])


    # Save the validation ACC, best and worst ACC as json_files
    if(is_train):
        json_filename = 'AUC_' + args.dataset + '_' + args.tuning_method + '_' + args.sens_attribute + '_' + 'training.json'
    else:
        json_filename = 'AUC_' + args.dataset + '_' + args.tuning_method + '_' + args.sens_attribute + '_' + 'validation.json'

    json_dict = {
        'AUC': ALL_AUC,
        'Best AUC': ALL_BEST_AUC,
        'Worst AUC': ALL_WORST_AUC,
        'AUC Difference': ALL_AUC_DIFF
    }

    # Save the file to disk
    print("Saving json file at: ", os.path.join(args.fig_savepath, json_filename))
    with open(os.path.join(args.fig_savepath, json_filename), 'w') as fp:
        json.dump(json_dict, fp)


def plot_acc(args, ACC_list, best_ACC_list, worst_ACC_list, is_train):

    assert is_train is not None

    print("Saving ACC List")

    ALL_ACC = []
    ALL_BEST_ACC = []
    ALL_WORST_ACC = []

    print(ACC_list)
    print(best_ACC_list)
    print(worst_ACC_list)

    # Handling nan values, replacing them with previous values
    for i in range(len(ACC_list)):
        if(np.isnan(ACC_list[i])):
            ALL_ACC.append(ACC_list[i-1])
        else:
            ALL_ACC.append(ACC_list[i])
        
    for i in range(len(best_ACC_list)):
        if(np.isnan(best_ACC_list[i])):
            ALL_BEST_ACC.append(best_ACC_list[i-1])
        else:
            ALL_BEST_ACC.append(best_ACC_list[i])
        
    for i in range(len(worst_ACC_list)):
        if(np.isnan(worst_ACC_list[i])):
            ALL_WORST_ACC.append(worst_ACC_list[i-1])
        else:
            ALL_WORST_ACC.append(worst_ACC_list[i])

    # List for ACC Difference
    ALL_ACC_DIFF = []
    for i in range(len(ALL_ACC)):
        ALL_ACC_DIFF.append(ALL_BEST_ACC[i] - ALL_WORST_ACC[i])

    # if(is_train):
    #     figname = args.dataset + '_' + args.tuning_method + '_' + args.sens_attribute + '_' + 'training.png'
    #     label1 = 'Train ACC'
    #     label2 = 'Train Best ACC'
    #     label3 = 'Train Worst ACC'
    #     title = 'Overall, Best and Worst Sub-group Training ACC'
    # else:
    #     figname = args.dataset + '_' + args.tuning_method + '_' + args.sens_attribute + '_' + 'validation.png'
    #     label1 = 'Val ACC'
    #     label2 = 'Val Best ACC'
    #     label3 = 'Val Worst ACC'
    #     title = 'Overall, Best and Worst Sub-group Validation ACC'

    # Plot the validation ACC, best and worst ACC in the same figure as a line plot
    # plt.figure(figsize=(12, 16))
    # plt.plot(ALL_ACC, label=label1)
    # plt.plot(ALL_BEST_ACC, label=label2)
    # plt.plot(ALL_WORST_ACC, label=label3)
    # plt.plot(ALL_ACC_DIFF, label='ACC Difference')
    # plt.xlabel('Epochs')
    # plt.ylabel('ACC')
    # plt.title(title)
    # plt.xticks(list(range(len(ALL_WORST_ACC))))
    # plt.legend()
    # plt.savefig(os.path.join(args.fig_savepath, figname))

    # if(is_train):
    #     print("Training ACC Plot saved at: ", os.path.join(args.fig_savepath, figname))
    # else:
    #     print("Validation ACC Plot saved at: ", os.path.join(args.fig_savepath, figname))

    # plt.close()

    # Save the validation ACC, best and worst ACC as json_files
    if(is_train):
        json_filename = 'ACC_' + args.dataset + '_' + args.tuning_method + '_' + args.sens_attribute + '_' + 'training.json'
    else:
        json_filename = 'ACC_' + args.dataset + '_' + args.tuning_method + '_' + args.sens_attribute + '_' + 'validation.json'

    json_dict = {
        'ACC': ALL_ACC,
        'Best ACC': ALL_BEST_ACC,
        'Worst ACC': ALL_WORST_ACC,
        'ACC Difference': ALL_ACC_DIFF
    }

    # Save the file to disk
    print("Saving json file at: ", os.path.join(args.fig_savepath, json_filename))
    with open(os.path.join(args.fig_savepath, json_filename), 'w') as fp:
        json.dump(json_dict, fp)



def main(args):
    assert args.sens_attribute is not None, "Sensitive attribute not provided"
    
    os.makedirs(args.fig_savepath, exist_ok=True)

    # Making directory for saving checkpoints
    if args.output_dir:
        utils.mkdir(args.output_dir)
        utils.mkdir(os.path.join(args.output_dir, "checkpoints"))

    try:
        # results_df = pd.read_csv(os.path.join(args.output_dir, args.results_df))
        test_results_df = pd.read_csv(
            os.path.join(args.output_dir, args.test_results_df)
        )
        print("Reading existing results dataframe")
    except:
        print("Creating new results dataframe")
        test_results_df = create_results_df(args)

    print("!!!!!!!! COLUMNS !!!!!!")
    print(test_results_df.columns)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    with open("config.yaml") as file:
        yaml_data = yaml.safe_load(file)

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
    #pprint(dataset.class_to_idx)

    collate_fn = None
    mixup_transforms = get_mixup_transforms(args)

    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

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

    print("TUNING METHOD: ", args.tuning_method)
    print("Creating model")

    if(args.tuning_method == 'train_from_scratch'):
        model = utils.get_timm_model(args.model, num_classes=args.num_classes, pretrained=False)
    else:
        model = utils.get_timm_model(args.model, num_classes=args.num_classes)

    # Calculate the sum of model parameters
    total_params = sum([p.sum() for p in model.parameters()])
    print("Sum of parameters: ", total_params)

    base_model = model

    if(args.tuning_method == 'fullft' or args.tuning_method == 'train_from_scratch'):
        pass
    elif(args.tuning_method == 'linear_readout'):
        utils.disable_module(model)
        utils.enable_module(model.head)
    else:
        assert args.mask_path is not None
        mask = np.load(args.mask_path)
        mask = get_masked_model(model, args.tuning_method, mask=mask)
        print("LOADED MASK: ", mask)

        if(np.all(np.array(mask) == 1)):  
            # If the mask contains all ones
            args.tuning_method = 'Vanilla_'+args.tuning_method
            print("Mask contains all ones. Changing tuning method to: ", args.tuning_method)
            

    # Check Tunable Params
    trainable_params, all_param = utils.check_tunable_params(model, True)
    trainable_percentage = 100 * trainable_params / all_param

    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing, reduction="none"
        )

    ece_criterion = utils.ECELoss()

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

    # Optimizer
    optimizer = get_optimizer(args, parameters)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # LR Scheduler
    lr_scheduler = get_lr_scheduler(args, optimizer)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # From training_utils.py
    model_ema = get_model_ema(model_without_ddp, args)
    print("model ema", model_ema)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.disable_training:
        print("Training Process Skipped")
    else:


        ALL_TRAIN_AUC = []
        ALL_TRAIN_BEST_AUC = []
        ALL_TRAIN_WORST_AUC = []
        ALL_TRAIN_ACC = []
        ALL_TRAIN_BEST_ACC = []
        ALL_TRAIN_WORST_ACC = []

        ALL_VAL_AUC = []
        ALL_VAL_BEST_AUC = []
        ALL_VAL_WORST_AUC = []
        ALL_VAL_ACC = []
        ALL_VAL_BEST_ACC = []
        ALL_VAL_WORST_ACC = []
        

        print("Start training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            (
                train_acc,
                train_best_acc,
                train_worst_acc,
                train_auc,
                train_best_auc,
                train_worst_auc,
            ) = train_one_epoch_fairness(
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

            ALL_TRAIN_AUC.append(train_auc)
            ALL_TRAIN_BEST_AUC.append(train_best_auc)
            ALL_TRAIN_WORST_AUC.append(train_worst_auc)

            ALL_TRAIN_ACC.append(train_acc)
            ALL_TRAIN_BEST_ACC.append(train_best_acc)
            ALL_TRAIN_WORST_ACC.append(train_worst_acc)

            if args.sens_attribute == "gender":
                if(args.cal_equiodds):
                    (
                        val_acc,
                        val_male_acc,
                        val_female_acc,
                        val_auc,
                        val_male_auc,
                        val_female_auc,
                        val_loss,
                        val_max_loss,
                        equiodds_diff, 
                        equiodds_ratio, 
                        dpd, 
                        dpr

                    ) = evaluate_fairness_gender(
                        model,
                        criterion,
                        ece_criterion,
                        data_loader_val,
                        args=args,
                        device=device,
                    )
                else:
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

                best_val_acc = max(val_male_acc, val_female_acc)
                worst_val_acc = min(val_male_acc, val_female_acc)

                best_val_auc = max(val_male_auc, val_female_auc)
                worst_val_auc = min(val_male_auc, val_female_auc)

                ALL_VAL_AUC.append(val_auc)
                ALL_VAL_BEST_AUC.append(best_val_auc)
                ALL_VAL_WORST_AUC.append(worst_val_auc)

                ALL_VAL_ACC.append(val_acc)
                ALL_VAL_BEST_ACC.append(best_val_acc)
                ALL_VAL_WORST_ACC.append(worst_val_acc)

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
                    if(args.cal_equiodds):
                        (
                            val_acc, val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3, val_acc_type4, val_acc_type5, val_auc, val_auc_type0, val_auc_type1, val_auc_type2, val_auc_type3, val_auc_type4, val_auc_type5, val_loss, val_max_loss, equiodds_diff,  equiodds_ratio,  dpd,  dpr
                        ) = evaluate_fairness_skin_type(
                            model,
                            criterion,
                            ece_criterion,
                            data_loader_val,
                            args=args,
                            device=device,
                        )
                    else:
                        (val_acc, val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3, val_acc_type4, val_acc_type5, val_auc, val_auc_type0, val_auc_type1, val_auc_type2, val_auc_type3, val_auc_type4, val_auc_type5, val_loss, val_max_loss
                        ) = evaluate_fairness_skin_type(
                            model,
                            criterion,
                            ece_criterion,
                            data_loader_val,
                            args=args,
                            device=device,
                        )

                    best_val_acc = max(val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3, val_acc_type4, val_acc_type5)
                    worst_val_acc = min(val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3, val_acc_type4, val_acc_type5)

                    best_val_auc = max(val_auc_type0, val_auc_type1, val_auc_type2, val_auc_type3, val_auc_type4, val_auc_type5)
                    worst_val_auc = min(val_auc_type0, val_auc_type1, val_auc_type2, val_auc_type3, val_auc_type4, val_auc_type5)

                    ALL_VAL_AUC.append(val_auc)
                    ALL_VAL_BEST_AUC.append(best_val_auc)
                    ALL_VAL_WORST_AUC.append(worst_val_auc)

                    ALL_VAL_ACC.append(val_acc)
                    ALL_VAL_BEST_ACC.append(best_val_acc)
                    ALL_VAL_WORST_ACC.append(worst_val_acc)

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
                    if(args.cal_equiodds):
                        (
                            val_acc,
                            val_acc_type0,
                            val_acc_type1,
                            val_auc,
                            val_auc_type0,
                            val_auc_type1,
                            val_loss,
                            val_max_loss,
                            equiodds_diff, 
                            equiodds_ratio, 
                            dpd, 
                            dpr
                        ) = evaluate_fairness_skin_type_binary(
                            model,
                            criterion,
                            ece_criterion,
                            data_loader_val,
                            args=args,
                            device=device,
                        )
                    else:
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

                    best_val_acc = max(val_acc_type0, val_acc_type1)
                    worst_val_acc = min(val_acc_type0, val_acc_type1)

                    best_val_auc = max(val_auc_type0, val_auc_type1)
                    worst_val_auc = min(val_auc_type0, val_auc_type1)

                    ALL_VAL_AUC.append(val_auc)
                    ALL_VAL_BEST_AUC.append(best_val_auc)
                    ALL_VAL_WORST_AUC.append(worst_val_auc)

                    ALL_VAL_ACC.append(val_acc)
                    ALL_VAL_BEST_ACC.append(best_val_acc)
                    ALL_VAL_WORST_ACC.append(worst_val_acc)
                    
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
                    if(args.cal_equiodds):
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
                            equiodds_diff,
                            equiodds_ratio,
                            dpd,
                            dpr
                        ) = evaluate_fairness_age(
                            model,
                            criterion,
                            ece_criterion,
                            data_loader_val,
                            args=args,
                            device=device,
                        )
                    else:
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

                    best_val_acc = max(acc_age0_avg, acc_age1_avg, acc_age2_avg, acc_age3_avg, acc_age4_avg)
                    worst_val_acc = min(acc_age0_avg, acc_age1_avg, acc_age2_avg, acc_age3_avg, acc_age4_avg)

                    best_val_auc = max(auc_age0_avg, auc_age1_avg, auc_age2_avg, auc_age3_avg, auc_age4_avg)
                    worst_val_auc = min(auc_age0_avg, auc_age1_avg, auc_age2_avg, auc_age3_avg, auc_age4_avg)

                    ALL_VAL_AUC.append(val_auc)
                    ALL_VAL_BEST_AUC.append(best_val_auc)
                    ALL_VAL_WORST_AUC.append(worst_val_auc)

                    ALL_VAL_ACC.append(val_acc)
                    ALL_VAL_BEST_ACC.append(best_val_acc)
                    ALL_VAL_WORST_ACC.append(worst_val_acc)

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
                    if(args.cal_equiodds):
                        (
                            val_acc,
                            acc_age0_avg,
                            acc_age1_avg,
                            val_auc,
                            auc_age0_avg,
                            auc_age1_avg,
                            val_loss,
                            val_max_loss,
                            equiodds_diff, 
                            equiodds_ratio, 
                            dpd, 
                            dpr
                        ) = evaluate_fairness_age_binary(
                            model,
                            criterion,
                            ece_criterion,
                            data_loader_val,
                            args=args,
                            device=device,
                        )
                    else:
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
                    
                    best_val_acc = max(acc_age0_avg, acc_age1_avg)
                    worst_val_acc = min(acc_age0_avg, acc_age1_avg)

                    best_val_auc = max(auc_age0_avg, auc_age1_avg)
                    worst_val_auc = min(auc_age0_avg, auc_age1_avg)

                    ALL_VAL_AUC.append(val_auc)
                    ALL_VAL_BEST_AUC.append(best_val_auc)
                    ALL_VAL_WORST_AUC.append(worst_val_auc)

                    ALL_VAL_ACC.append(val_acc)
                    ALL_VAL_BEST_ACC.append(best_val_acc)
                    ALL_VAL_WORST_ACC.append(worst_val_acc)

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

                best_val_acc = max(val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3)
                worst_val_acc = min(val_acc_type0, val_acc_type1, val_acc_type2, val_acc_type3)

                best_val_auc = max(val_auc_type0, val_auc_type1, val_auc_type2, val_auc_type3)
                worst_val_auc = min(val_auc_type0, val_auc_type1, val_auc_type2, val_auc_type3)

                ALL_VAL_AUC.append(val_auc)
                ALL_VAL_BEST_AUC.append(best_val_auc)
                ALL_VAL_WORST_AUC.append(worst_val_auc)

                ALL_VAL_ACC.append(val_acc)
                ALL_VAL_BEST_ACC.append(best_val_acc)
                ALL_VAL_WORST_ACC.append(worst_val_acc)

                print(
                        "Val Acc: {:.2f}, Val Young Male Acc: {:.2f}, Val Old Male Acc: {:.2f}, Val Young Female Acc: {:.2f}, Val Old Female Acc: {:.2f}, Val Loss: {:.2f}, Val MAX LOSS: {:.2f}".format(
                            val_acc,
                            val_acc_type0,
                            val_acc_type1,
                            val_acc_type2,
                            val_acc_type3,
                            torch.mean(val_loss),
                            val_max_loss,
                        )
                    )
                print(
                    "Val AUC: {:.2f}, Val Young Male AUC: {:.2f}, Val Old Male AUC: {:.2f}, Val Young Female AUC: {:.2f}, Val Old Female AUC: {:.2f}".format(
                        val_auc,
                        val_auc_type0,
                        val_auc_type1,
                        val_auc_type2,
                        val_auc_type3,
                    )
                )
                print("\n")

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
        

        # Obtaining the performance on test set
        print("Obtaining the performance on test set")
        if args.sens_attribute == "gender":
            if(args.cal_equiodds):
                (
                    test_acc,
                    test_male_acc,
                    test_female_acc,
                    test_auc,
                    test_male_auc,
                    test_female_auc,
                    test_loss,
                    test_max_loss,
                    equiodds_diff, 
                    equiodds_ratio, 
                    dpd, 
                    dpr
                ) = evaluate_fairness_gender(
                    model,
                    criterion,
                    ece_criterion,
                    data_loader_test,
                    args=args,
                    device=device,
                )
            else:
                (
                    test_acc,
                    test_male_acc,
                    test_female_acc,
                    test_auc,
                    test_male_auc,
                    test_female_auc,
                    test_loss,
                    test_max_loss,
                ) = evaluate_fairness_gender(
                    model,
                    criterion,
                    ece_criterion,
                    data_loader_test,
                    args=args,
                    device=device,
                )
            print("\n")
            print("Overall Test Accuracy: ", test_acc)
            print("Test Male Accuracy: ", test_male_acc)
            print("Test Female Accuracy: ", test_female_acc)
            print("\n")
            print("Overall Test AUC: ", test_auc)
            print("Test Male AUC: ", test_male_auc)
            print("Test Female AUC: ", test_female_auc)
            if(args.cal_equiodds):
                print("\n")
                print("EquiOdds Difference: ", equiodds_diff)
                print("EquiOdds Ratio: ", equiodds_ratio)
                print("DPD: ", dpd)
                print("DPR: ", dpr)

        elif args.sens_attribute == "skin_type":
            if(args.skin_type == 'multi'):
                if(args.cal_equiodds):
                    (
                        test_acc,
                        test_acc_type0,
                        test_acc_type1,
                        test_acc_type2,
                        test_acc_type3,
                        test_acc_type4,
                        test_acc_type5,
                        test_auc,
                        test_auc_type0,
                        test_auc_type1,
                        test_auc_type2,
                        test_auc_type3,
                        test_auc_type4,
                        test_auc_type5,
                        test_loss,
                        test_max_loss,
                        equiodds_diff, 
                        equiodds_ratio, 
                        dpd, 
                        dpr
                    ) = evaluate_fairness_skin_type(
                        model,
                        criterion,
                        ece_criterion,
                        data_loader_test,
                        args=args,
                        device=device,
                    )
                else:
                    (
                        test_acc,
                        test_acc_type0,
                        test_acc_type1,
                        test_acc_type2,
                        test_acc_type3,
                        test_acc_type4,
                        test_acc_type5,
                        test_auc,
                        test_auc_type0,
                        test_auc_type1,
                        test_auc_type2,
                        test_auc_type3,
                        test_auc_type4,
                        test_auc_type5,
                        test_loss,
                        test_max_loss,
                    ) = evaluate_fairness_skin_type(
                        model,
                        criterion,
                        ece_criterion,
                        data_loader_test,
                        args=args,
                        device=device,
                    )
                print("\n")
                print("Overall Test accuracy: ", test_acc)
                print("Test Type 0 Accuracy: ", test_acc_type0)
                print("Test Type 1 Accuracy: ", test_acc_type1)
                print("Test Type 2 Accuracy: ", test_acc_type2)
                print("Test Type 3 Accuracy: ", test_acc_type3)
                print("Test Type 4 Accuracy: ", test_acc_type4)
                print("Test Type 5 Accuracy: ", test_acc_type5)
                print("\n")
                print("Overall Test AUC: ", test_auc)
                print("Test Type 0 AUC: ", test_auc_type0)
                print("Test Type 1 AUC: ", test_auc_type1)
                print("Test Type 2 AUC: ", test_auc_type2)
                print("Test Type 3 AUC: ", test_auc_type3)
                print("Test Type 4 AUC: ", test_auc_type4)
                print("Test Type 5 AUC: ", test_auc_type5)
            
            elif(args.skin_type == 'binary'):
                if(args.cal_equiodds):
                    (
                        test_acc,
                        test_acc_type0,
                        test_acc_type1,
                        test_auc,
                        test_auc_type0,
                        test_auc_type1,
                        test_loss,
                        test_max_loss,
                        equiodds_diff, 
                        equiodds_ratio, 
                        dpd, 
                        dpr
                    ) = evaluate_fairness_skin_type_binary(
                        model,
                        criterion,
                        ece_criterion,
                        data_loader_test,
                        args=args,
                        device=device,
                    )
                else:
                    (
                        test_acc,
                        test_acc_type0,
                        test_acc_type1,
                        test_auc,
                        test_auc_type0,
                        test_auc_type1,
                        test_loss,
                        test_max_loss,
                    ) = evaluate_fairness_skin_type_binary(
                        model,
                        criterion,
                        ece_criterion,
                        data_loader_test,
                        args=args,
                        device=device,
                    )

                print("\n")
                print("Overall Test accuracy: ", test_acc)
                print("Test Type 0 Accuracy: ", test_acc_type0)
                print("Test Type 1 Accuracy: ", test_acc_type1)
                print("\n")
                print("Overall Test AUC: ", test_auc)
                print("Test Type 0 AUC: ", test_auc_type0)
                print("Test Type 1 AUC: ", test_auc_type1)
                if(args.cal_equiodds):
                    print("\n")
                    print("EquiOdds Difference: ", equiodds_diff)
                    print("EquiOdds Ratio: ", equiodds_ratio)
                    print("DPD: ", dpd)
                    print("DPR: ", dpr)

        elif(args.sens_attribute == 'age'):
            if(args.age_type == 'multi'):
                if(args.cal_equiodds):
                        (test_acc,test_acc_type0,test_acc_type1,test_acc_type2,test_acc_type3,test_acc_type4,test_auc,test_auc_type0,test_auc_type1,test_auc_type2,test_auc_type3,test_auc_type4,test_loss,test_max_loss,equiodds_diff,equiodds_ratio,dpd,dpr
                    ) = evaluate_fairness_age(
                        model,
                        criterion,
                        ece_criterion,
                        data_loader_test,
                        args=args,
                        device=device,
                    )
                else:
                    (test_acc, test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3, test_acc_type4, test_auc, test_auc_type0, test_auc_type1, test_auc_type2, test_auc_type3, test_auc_type4, test_loss, test_max_loss,
                    ) = evaluate_fairness_age(
                        model,
                        criterion,
                        ece_criterion,
                        data_loader_test,
                        args=args,
                        device=device,
                    )
                print("\n")
                print("Overall Test  accuracy: ", test_acc)
                print("Test Age Group 0 Accuracy: ", test_acc_type0)
                print("Test Age Group 1 Accuracy: ", test_acc_type1)
                print("Test Age Group 2 Accuracy: ", test_acc_type2)
                print("Test Age Group 3 Accuracy: ", test_acc_type3)
                print("Test Age Group 4 Accuracy: ", test_acc_type4)
                print("\n")
                print("Overall Test AUC: ", test_auc)
                print("Test Age Group 0 AUC: ", test_auc_type0)
                print("Test Age Group 1 AUC: ", test_auc_type1)
                print("Test Age Group 2 AUC: ", test_auc_type2)
                print("Test Age Group 3 AUC: ", test_auc_type3)
                print("Test Age Group 4 AUC: ", test_auc_type4)

                if(args.cal_equiodds):
                    print("\n")
                    print("EquiOdds Difference: ", equiodds_diff)
                    print("EquiOdds Ratio: ", equiodds_ratio)
                    print("DPD: ", dpd)
                    print("DPR: ", dpr)

            elif(args.age_type == 'binary'):
                if(args.cal_equiodds):
                    (
                        test_acc,
                        test_acc_type0,
                        test_acc_type1,
                        test_auc,
                        test_auc_type0,
                        test_auc_type1,
                        test_loss,
                        test_max_loss,
                        equiodds_diff, 
                        equiodds_ratio, 
                        dpd, 
                        dpr
                    ) = evaluate_fairness_age_binary(
                        model,
                        criterion,
                        ece_criterion,
                        data_loader_test,
                        args=args,
                        device=device,
                    )
                else: 
                    (
                        test_acc,
                        test_acc_type0,
                        test_acc_type1,
                        test_auc,
                        test_auc_type0,
                        test_auc_type1,
                        test_loss,
                        test_max_loss,
                    ) = evaluate_fairness_age_binary(
                        model,
                        criterion,
                        ece_criterion,
                        data_loader_test,
                        args=args,
                        device=device,
                    )
                print("\n")
                print("Overall Test accuracy: ", test_acc)
                print("Test Age Group 0 Accuracy: ", test_acc_type0)
                print("Test Age Group 1 Accuracy: ", test_acc_type1)
                print("\n")
                print("Overall Test AUC: ", test_auc)
                print("Test Age Group 0 AUC: ", test_auc_type0)
                print("Test Age Group 1 AUC: ", test_auc_type1)
                if(args.cal_equiodds):
                    print("\n")
                    print("EquiOdds Difference: ", equiodds_diff)
                    print("EquiOdds Ratio: ", equiodds_ratio)
                    print("DPD: ", dpd)
                    print("DPR: ", dpr)
            else:
                raise NotImplementedError("Age type not supported. Choose from 'multi' or 'binary'")
        
        elif(args.sens_attribute == 'race'):
            if(args.cal_equiodds):
                test_acc, test_acc_type0, test_acc_type1, test_auc, test_auc_type0, test_auc_type1, test_loss, test_max_loss, equiodds_diff, equiodds_ratio, dpd, dpr = evaluate_fairness_race_binary(model, criterion, ece_criterion, data_loader_test, args=args, device=device)
            else:
                est_acc, test_acc_type0, test_acc_type1, test_auc, test_auc_type0, test_auc_type1, test_loss, test_max_loss = evaluate_fairness_race_binary(model, criterion, ece_criterion, data_loader_test, args=args, device=device)

        elif(args.sens_attribute == 'age_sex'):
            if(args.cal_equiodds):
                (test_acc,test_acc_type0,test_acc_type1,test_acc_type2,test_acc_type3,test_auc,test_auc_type0,test_auc_type1,test_auc_type2,test_auc_type3,test_loss,test_max_loss,equiodds_diff,equiodds_ratio,dpd,dpr
            ) = evaluate_fairness_age_sex(
                model,
                criterion,
                ece_criterion,
                data_loader_test,
                args=args,
                device=device,
            )
            else:
                (test_acc, test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3, test_auc, test_auc_type0, test_auc_type1, test_auc_type2, test_auc_type3, test_loss, test_max_loss,
                ) = evaluate_fairness_age_sex(
                    model,
                    criterion,
                    ece_criterion,
                    data_loader_test,
                    args=args,
                    device=device,
                )

            print("\n")
            print("Overall Test  accuracy: ", test_acc)
            print("Test Young Male Accuracy: ", test_acc_type0)
            print("Test Old Male Accuracy: ", test_acc_type1)
            print("Test Young Female Accuracy: ", test_acc_type2)
            print("Test Old Female Accuracy: ", test_acc_type3)
            print("\n")
            print("Overall Test AUC: ", test_auc)
            print("Test Young Male AUC: ", test_auc_type0)
            print("Test Old Male AUC: ", test_auc_type1)
            print("Test Young Female AUC: ", test_auc_type2)
            print("Test Old Female AUC: ", test_auc_type3)

            if(args.cal_equiodds):
                print("\n")
                print("EquiOdds Difference: ", equiodds_diff)
                print("EquiOdds Ratio: ", equiodds_ratio)
                print("DPD: ", dpd)
                print("DPR: ", dpr)
        else:
            raise NotImplementedError("Sensitive Attribute not supported")

        print("Test loss: ", round(torch.mean(test_loss).item(), 3))
        print("Test max loss: ", round(test_max_loss.item(), 3))

        # Add these results to CSV
        # Here we are adding results on the test set

        if(args.mask_path is not None):
            mask_path = args.mask_path.split('/')[-1]
        else:
            mask_path = 'None'

        if(args.sens_attribute == 'gender'):

            if(args.use_metric == 'acc'):
                new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_acc, test_male_acc, test_female_acc, round(abs(test_male_acc - test_female_acc), 3), mask_path]
            if(args.use_metric == 'auc'):
                if(args.cal_equiodds):
                    new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_auc, test_male_auc, test_female_auc, round(abs(test_male_auc - test_female_auc), 3), round(equiodds_diff, 3), round(equiodds_ratio, 3), round(dpd, 3), round(dpr, 3), mask_path]
                else:
                    new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_auc, test_male_auc, test_female_auc, round(abs(test_male_auc - test_female_auc), 3), mask_path]

        elif(args.sens_attribute == 'skin_type'):
            
            if(args.skin_type == 'multi'):
                best_acc = max(test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3, test_acc_type4, test_acc_type5)
                worst_acc = min(test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3, test_acc_type4, test_acc_type5)

                best_auc = max(test_auc_type0, test_auc_type1, test_auc_type2, test_auc_type3, test_auc_type4, test_auc_type5)
                worst_auc = min(test_auc_type0, test_auc_type1, test_auc_type2, test_auc_type3, test_auc_type4, test_auc_type5)
            
            elif(args.skin_type == 'binary'):
                best_acc = max(test_acc_type0, test_acc_type1)
                worst_acc = min(test_acc_type0, test_acc_type1)

                best_auc = max(test_auc_type0, test_auc_type1)
                worst_auc = min(test_auc_type0, test_auc_type1)

            if(args.use_metric == 'acc'):
                new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_acc, best_acc, worst_acc, round(abs(best_acc - worst_acc), 3), mask_path]
            elif(args.use_metric == 'auc'):
                if(args.cal_equiodds):
                    print("Saving with equiodds")
                    new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_auc, best_auc, worst_auc, round(abs(best_auc - worst_auc), 3), round(equiodds_diff, 3), round(equiodds_ratio, 3), round(dpd, 3), round(dpr, 3), mask_path]
                else:
                    new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_auc, best_auc, worst_auc, round(abs(best_auc - worst_auc), 3), mask_path]

        elif(args.sens_attribute == 'age'):
            if(args.age_type == 'multi'):
                best_acc = max(test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3, test_acc_type4)
                worst_acc = min(test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3, test_acc_type4)
                best_auc = max(test_auc_type0, test_auc_type1, test_auc_type2, test_auc_type3, test_auc_type4)
                worst_auc = min(test_auc_type0, test_auc_type1, test_auc_type2, test_auc_type3, test_auc_type4)

                if(args.use_metric == 'acc'):
                    if(args.cal_equiodds):
                        new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_acc, best_acc, worst_acc, round(abs(best_acc - worst_acc), 3), round(equiodds_diff, 3), round(equiodds_ratio, 3), round(dpd, 3), round(dpr, 3), mask_path]
                    else:
                        new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_acc, best_acc, worst_acc, round(abs(best_acc - worst_acc), 3), mask_path]
                if(args.use_metric == 'auc'):
                    if(args.cal_equiodds):
                        new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_auc, best_auc, worst_auc, round(abs(best_auc - worst_auc), 3), round(equiodds_diff, 3), round(equiodds_ratio, 3), round(dpd, 3), round(dpr, 3), mask_path]
                    else:
                        new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_auc, best_auc, worst_auc, round(abs(best_auc - worst_auc), 3), mask_path]
                        
            elif(args.age_type == 'binary'):
                best_acc = max(test_acc_type0, test_acc_type1)
                worst_acc = min(test_acc_type0, test_acc_type1)

                best_auc = max(test_auc_type0, test_auc_type1)
                worst_auc = min(test_auc_type0, test_auc_type1)

                if(args.use_metric == 'acc'):
                    new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_acc, best_acc, worst_acc, round(abs(best_acc - worst_acc), 3), mask_path]
                elif(args.use_metric == 'auc'):
                    if(args.cal_equiodds):
                        print("Saving with equiodds")
                        new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_auc, best_auc, worst_auc, round(abs(best_auc - worst_auc), 3), round(equiodds_diff, 3), round(equiodds_ratio, 3), round(dpd, 3), round(dpr, 3), mask_path]
                    else:
                        new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_auc, best_auc, worst_auc, round(abs(best_auc - worst_auc), 3), mask_path]
            else:
                raise NotImplementedError("Age type not supported. Choose from 'multi' or 'binary'")
        
        elif(args.sens_attribute == 'race'):
            best_acc = max(test_acc_type0, test_acc_type1)
            worst_acc = min(test_acc_type0, test_acc_type1)

            best_auc = max(test_auc_type0, test_auc_type1)
            worst_auc = min(test_auc_type0, test_auc_type1)

            if(args.use_metric == 'acc'):
                new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_acc, best_acc, worst_acc, round(abs(best_acc - worst_acc), 3), mask_path]
            elif(args.use_metric == 'auc'):
                if(args.cal_equiodds):
                    new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_auc, best_auc, worst_auc, round(abs(best_auc - worst_auc), 3), round(equiodds_diff, 3), round(equiodds_ratio, 3), round(dpd, 3), round(dpr, 3), mask_path]
                else:
                    new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_auc, best_auc, worst_auc, round(abs(best_auc - worst_auc), 3), mask_path]
        
        elif(args.sens_attribute == 'age_sex'):
            best_acc = max(test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3)
            worst_acc = min(test_acc_type0, test_acc_type1, test_acc_type2, test_acc_type3)
            best_auc = max(test_auc_type0, test_auc_type1, test_auc_type2, test_auc_type3)
            worst_auc = min(test_auc_type0, test_auc_type1, test_auc_type2, test_auc_type3)

            if(args.use_metric == 'acc'):
                if(args.cal_equiodds):
                    new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_acc, best_acc, worst_acc, round(abs(best_acc - worst_acc), 3), round(equiodds_diff, 3), round(equiodds_ratio, 3), round(dpd, 3), round(dpr, 3), mask_path]
                else:
                    new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_acc, best_acc, worst_acc, round(abs(best_acc - worst_acc), 3), mask_path]
            if(args.use_metric == 'auc'):
                if(args.cal_equiodds):
                    new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_auc, best_auc, worst_auc, round(abs(best_auc - worst_auc), 3), round(equiodds_diff, 3), round(equiodds_ratio, 3), round(dpd, 3), round(dpr, 3), mask_path]
                else:
                    new_row2 = [args.tuning_method, round(trainable_percentage, 3), args.lr, test_auc, best_auc, worst_auc, round(abs(best_auc - worst_auc), 3), mask_path]
        
        else:
            raise NotImplementedError("Sensitive attribute not implemented")
            
        test_results_df.loc[len(test_results_df)] = new_row2

        print(
            "Saving test results df at: {}".format(
                os.path.join(args.output_dir, args.test_results_df)
            )
        )

        test_results_df.to_csv(
            os.path.join(args.output_dir, args.test_results_df), index=False
        )
        
        if(args.use_metric == 'auc'):
            # Potting the training AUC, Best AUC and Worst AUC
            plot_auc(args, ALL_TRAIN_AUC, ALL_TRAIN_BEST_AUC, ALL_TRAIN_WORST_AUC, is_train=True)

            # Potting the validation AUC, Best AUC and Worst AUC
            plot_auc(args, ALL_VAL_AUC, ALL_VAL_BEST_AUC, ALL_VAL_WORST_AUC, is_train=False)
        else:
            # Potting the training AUC, Best AUC and Worst AUC
            plot_acc(args, ALL_TRAIN_ACC, ALL_TRAIN_BEST_ACC, ALL_TRAIN_WORST_ACC, is_train=True)

            # Potting the validation ACC, Best ACC and Worst ACC
            plot_acc(args, ALL_VAL_ACC, ALL_VAL_BEST_ACC, ALL_VAL_WORST_ACC, is_train=False)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.output_dir = os.path.join(os.getcwd(), args.model, args.dataset)

    if("auc" in args.objective_metric):
        args.use_metric = 'auc'
    
    if(args.use_metric == 'acc'):
        args.test_results_df = "NEW_TEST_SET_RESULTS_" + args.sens_attribute + "_" + args.objective_metric + ".csv"
    elif(args.use_metric == 'auc'):
        if(args.cal_equiodds):
            args.test_results_df = "Equiodds_AUC_NEW_TEST_SET_RESULTS_" + args.sens_attribute + "_" + args.objective_metric + ".csv"
        else:
            args.test_results_df = "AUC_NEW_TEST_SET_RESULTS_" + args.sens_attribute + "_" + args.objective_metric + ".csv"

    current_wd = os.getcwd()
    args.fig_savepath = os.path.join(args.output_dir, "plots/")

    main(args)
