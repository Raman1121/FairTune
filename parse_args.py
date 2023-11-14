import argparse


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="PyTorch Classification Training", add_help=add_help
    )

    # parser.add_argument("--data-path", default="/disk/scratch2/raman/ALL_DATASETS/HAM10000_dataset/", type=str, help="dataset path")
    parser.add_argument(
        "--dataset",
        default=None,
        required=True,
        type=str,
        help="Dataset for finetuning.",
    )
    parser.add_argument(
        "--train_subset_ratio",
        default=None,
        type=float,
        help="Subset of the training dataset to be used",
    )
    parser.add_argument(
        "--val_subset_ratio",
        default=None,
        type=float,
        help="Subset of the validation dataset to be used",
    )
    parser.add_argument(
        "--dataset_basepath",
        default="/home/co-dutt1/rds/hpc-work/ALL_DATASETS/",
        required=False,
        type=str,
        help="Base path for all the datasets.",
    )
    # parser.add_argument("--fig_savepath", required=True, type=str, help="Base path for saving figures.")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device (Use cuda or cpu Default: cuda)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=32,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=16,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 16)",
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    # parser.add_argument("--inner_lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument(
        "--lr_scaler",
        default=1,
        type=float,
        help="Multiplier for the LR used in the inner loop weight update",
    )

    parser.add_argument(
        "--outer_lr", default=0.1, type=float, help="outer loop learning rate"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing",
        default=0.0,
        type=float,
        help="label smoothing (default: 0.0)",
        dest="label_smoothing",
    )
    parser.add_argument(
        "--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)"
    )
    parser.add_argument(
        "--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)"
    )
    parser.add_argument(
        "--lr-scheduler",
        default="steplr",
        type=str,
        help="the lr scheduler (default: steplr)",
    )
    parser.add_argument(
        "--lr-scheduler-outer",
        default="constant",
        type=str,
        help="the lr scheduler (default: constant)",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        default=0,
        type=int,
        help="the number of epochs to warmup (default: 0)",
    )
    parser.add_argument(
        "--lr-warmup-method",
        default="constant",
        type=str,
        help="the warmup method (default: constant)",
    )
    parser.add_argument(
        "--lr-warmup-decay", default=0.01, type=float, help="the decay for lr"
    )
    parser.add_argument(
        "--lr-step-size",
        default=30,
        type=int,
        help="decrease lr every step-size epochs",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    parser.add_argument(
        "--lr-min",
        default=0.0,
        type=float,
        help="minimum lr of lr schedule (default: 0.0)",
    )
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument(
        "--output-dir", default=".", type=str, help="path to save outputs"
    )
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--auto-augment",
        default=None,
        type=str,
        help="auto augment policy (default: None)",
    )
    parser.add_argument(
        "--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy"
    )
    parser.add_argument(
        "--augmix-severity", default=3, type=int, help="severity of augmix policy"
    )
    parser.add_argument(
        "--random-erase",
        default=0.0,
        type=float,
        help="random erasing probability (default: 0.0)",
    )

    # Mixed precision training parameters
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp for mixed precision training",
    )

    # distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--model-ema",
        action="store_true",
        help="enable tracking Exponential Moving Average of model parameters",
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only.",
    )
    parser.add_argument(
        "--interpolation",
        default="bilinear",
        type=str,
        help="the interpolation method (default: bilinear)",
    )
    parser.add_argument(
        "--val-resize-size",
        default=256,
        type=int,
        help="the resize size used for validation (default: 256)",
    )
    parser.add_argument(
        "--val-crop-size",
        default=224,
        type=int,
        help="the central crop size used for validation (default: 224)",
    )
    parser.add_argument(
        "--train-crop-size",
        default=224,
        type=int,
        help="the random crop size used for training (default: 224)",
    )
    parser.add_argument(
        "--clip-grad-norm",
        default=None,
        type=float,
        help="the maximum gradient norm (default None)",
    )
    parser.add_argument(
        "--ra-sampler",
        action="store_true",
        help="whether to use Repeated Augmentation in training",
    )
    parser.add_argument(
        "--ra-reps",
        default=3,
        type=int,
        help="number of repetitions for Repeated Augmentation (default: 3)",
    )
    parser.add_argument(
        "--weights",
        default="IMAGENET1K_V1",
        type=str,
        help="the weights enum name to load",
    )

    # RANDOM SEARCH AND TUNING METHOD PARAMETERS
    parser.add_argument(
        "--tuning_method",
        default="fullft",
        type=str,
        help="Type of fine-tuning method to use",
    )
    parser.add_argument("--masking_vector_idx", type=int, default=None)

    parser.add_argument('--masking_vector', metavar='N', type=float, nargs='+',
                        help='Elements of the masking vector')
    
    parser.add_argument("--subnetwork_mask_name", type=str, default=None)

    parser.add_argument("--mask_path", type=str, default=None)
    
    parser.add_argument(
        "--exp_vector_path",
        type=str,
        default="/home/co-dutt1/rds/hpc-work/Layer-Masking/Experiment_Vectors/",
    )

    # MASK GENERATION PARAMETERS
    parser.add_argument(
        "--mask_gen_method",
        default="random",
        type=str,
    )
    parser.add_argument(
        "--sigma",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--use_adaptive_threshold",
        action="store_true",
        help="Use adaptive thresholding for masking",
    )
    parser.add_argument(
        "--thr_ema_decay",
        default=0.99,
        type=float,
    )
    parser.add_argument(
        "--use_gumbel_sigmoid",
        action="store_true",
        help="Use sigmoid function to the weight mask",
    )

    ## LOGGING AND MISC PARAMETERS
    parser.add_argument(
        "--wandb_logging",
        action="store_true",
        help="To enable/ disable wandb logging.",
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default=None, help="Path to the checkpoint directory"
    )
    parser.add_argument(
        "--disable_checkpointing",
        action="store_true",
        help="Disable saving of model checkopints",
    )
    parser.add_argument(
        "--disable_training",
        action="store_true",
        help="To disable/ skip the training process.",
    )
    parser.add_argument(
        "--disable_plotting",
        action="store_true",
        help="To disable/ skip the plotting.",
    )
    parser.add_argument(
        "--dev_mode",
        action="store_true",
        help="Dev mode disables plotting, checkpointing, etc",
    )

    # FAIRNESS Arguements
    parser.add_argument('--sens_attribute',
                        type=str,
                        default=None,
                        help='Sensitive attribute to be used for fairness')
    parser.add_argument('--age_type', type=str, default='multi', choices=['binary', 'multi'])
    parser.add_argument('--skin_type', type=str, default='multi', choices=['binary', 'multi'])
    parser.add_argument('--use_metric', type=str, default='auc', choices=['acc', 'auc'])

    # HPARAM OPT (HPO) Arguements
    parser.add_argument("--objective_metric", type=str, default="min_acc", choices=["min_acc", "min_auc", "acc_diff", "auc_diff", "max_loss", "overall_acc", "overall_auc"])
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--pruner", type=str, default='SuccessiveHalving', choices=['SuccessiveHalving', 'MedianPruner', 'Hyperband'])
    parser.add_argument(
        "--disable_storage",
        action="store_true",
        help="Disable creating a storage DB for the experiment",
    )
    parser.add_argument(
        "--use_multi_objective",
        help="Use multi-objective optimization for HPO",
        action="store_true",
    )

    # FAIRPRUNE ARGUEMENTS
    parser.add_argument(
        "--pruning_ratio",
        default=0.35,
        type=float,
        help="Pruning ratio in FairPrune",
    )
    parser.add_argument(
        "--b_param",
        default=0.33,
        type=float,
        help="Beta Parameter in FairPrune",
    )

    parser.add_argument(
        "--cal_equiodds",
        action='store_true',
        help="Calculate Equalized odds and DPD",
    )

    # FSCL ARGUEMENTS
    parser.add_argument(
        "--fscl",
        action='store_true',
        help="To perform FSCL",
    )
    parser.add_argument(
        "--train_encoder_lr",
        default=0.1,
        type=float,
        help="Learning rate for training the encoder in FSCL",
    )
    parser.add_argument(
        "--train_classifier_lr",
        default=0.1,
        type=float,
        help="Learning rate for training the classifier in FSCL",
    )
    parser.add_argument(
        "--fscl_eval_only",
        action='store_true',
        help="To perform FSCL eval using the trained model",
    )
    parser.add_argument(
        "--temperature",
        default=0.1,
        type=float,
        help="Temperature Parameter in FSCL",
    )
    parser.add_argument(
        "--contrast_mode",
        default='all',
        type=str,
        help="Contrast Mode in FSCL",
    )
    parser.add_argument(
        "--base_temperature",
        default=0.1,
        type=float,
        help="Temperature Parameter in FSCL",
    )
    parser.add_argument('--group_norm', type=int, default=0, help='group normalization')

    parser.add_argument('--method', type=str, default='FSCL',
                        choices=['FSCL','FSCL*','SupCon', 'SimCLR'], help='choose method')

    
    # HAM10000 LABELS
    parser.add_argument(
        "--label_type",
        default='binary',
        type=str,
        help="Binary/ Multi labels to be used",
    )

    return parser
