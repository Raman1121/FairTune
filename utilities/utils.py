import os

os.environ["TORCH_HOME"] = os.path.dirname(os.getcwd())

import copy
import loralib as lora
import datetime
import re
import errno
import hashlib
import numpy as np
from scipy.interpolate import interp1d

import time
from collections import defaultdict, deque, OrderedDict
from typing import List, Optional, Tuple
import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sklm

from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio, demographic_parity_difference, demographic_parity_ratio

###################### HELPER FUNCTIONS #######################


def disable_module(module):
    for p in module.parameters():
        p.requires_grad = False


def enable_module(module):
    for p in module.parameters():
        p.requires_grad = True

def read_numpy_from_cmd(args):
    mask_str = args.masking_vector
    mask = np.array(args.masking_vector)
    return mask

def convert_numpy_array_to_string(mask):
    mask_string = np.array2string(mask, separator=', ', formatter={'all': lambda x: f'{int(x)}'})
    return mask_string


def check_tunable_params(model, verbose=True):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0

    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            if verbose:
                print(name)
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.5f}"
    )

    return trainable_params, all_param


def enable_from_vector(vector, model):
    print("Vector: ", vector)

    disable_module(model)

    for idx, block in enumerate(model.blocks):
        if vector[idx] == 1:
            print("Enabling attention in Block {}".format(idx))
            enable_module(block.attn)
        else:
            # print("Disabling attention in Block {}".format(idx))
            disable_module(block.attn)


def tune_attention_params_random(model, mask, model_type="timm"):
    assert mask is not None

    attn_params = [
        p
        for name_p, p in model.named_parameters()
        if ".attn." in name_p or "attention" in name_p
    ]
    vector = []

    print(mask)

    for idx, p in enumerate(attn_params):
        if mask[idx] == 1:
            p.requires_grad = True
            vector.append(1)
        else:
            p.requires_grad = False
            vector.append(0)

    try:
        # Timm Model
        model.head.weight.requires_grad = True
        model.head.bias.requires_grad = True
    except:
        # HF Model
        model.classifier.weight.requires_grad = True
        model.classifier.bias.requires_grad = True

    # POSITION EMBEDDING
    if model_type == "timm":
        try:
            model.pos_embed.requires_grad = True
        except:
            print("no pos embedding")

    # PATCH EMBEDDING
    if model_type == "timm":
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print("no patch embed")

    # print("MASKING VECTOR: ", vector)

    assert vector == mask

    return vector


def tune_blocks_random(model, mask, segment):
    vector = []

    for idx, block in enumerate(model.blocks):
        if mask is None:
            bit = int(np.random.random(1)[0] > 0.5)
        else:
            bit = mask[idx]

        if bit == 1:
            #print("Enabling {} in Block {}".format(segment, idx))
            if segment == "attention":
                enable_module(block.attn)
            elif segment == "layernorm":
                enable_module(block.norm1)
                enable_module(block.norm2)
            elif segment == "mlp":
              enable_module(block.mlp)
            elif segment == "full":
                enable_module(block)

            vector.append(1)
        else:
            #print("Disabling {} in Block {}".format(segment, idx))
            if segment == "attention":
                disable_module(block.attn)
            elif segment == "layernorm":
                disable_module(block.norm1)
                disable_module(block.norm2)
            elif segment == "mlp":
              disable_module(block.mlp)
            elif segment == "full":
                disable_module(block)

            vector.append(0)

    # print(mask)
    # print(vector)
    # if mask is not None:
    #     assert mask == vector

    return vector


def tune_attention_layers(model):
    vector = []

    for name_p, p in model.named_parameters():
        if ".attn." in name_p or "attention" in name_p:
            vector.append(1)
            p.requires_grad = True
        else:
            p.requires_grad = False

        # Timm Model
        model.head.weight.requires_grad = True
        model.head.bias.requires_grad = True

        # POSITION EMBEDDING
        try:
            model.pos_embed.requires_grad = True
        except:
            print("no pos embedding")

        # PATCH EMBEDDING

        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print("no patch embed")

    return vector


def tune_layernorm_layers(model):
    disable_module(model)

    vector = []

    for n, p in model.named_parameters():
        if "norm" in n or "head" in n:
            vector.append(1)
            p.requires_grad = True

    return vector


def tune_layernorm_random(model):
    disable_module(model)

    vector = []

    for n, p in model.named_parameters():
        if "norm" in n or "head" in n:
            if np.random.random(1)[0] >= 0.5:
                vector.append(1)
                p.requires_grad = True
            else:
                vector.append(0)

    return vector


def get_model_for_bitfit(model, model_type):
    trainable_components = ["bias"]

    # Disale all the gradients
    for param in model.parameters():
        param.requires_grad = False

    # Add classification head to trainable components
    if trainable_components:
        trainable_components = trainable_components + ["pooler.dense.bias"]

    if model_type == "timm":
        trainable_components = trainable_components + ["head"]
    elif model_type == "hf":
        trainable_components = trainable_components + ["classifier"]

    vector = []

    for name, param in model.named_parameters():
        for component in trainable_components:
            if component in name:
                vector.append(1)
                param.requires_grad = True
                break

    return vector


def get_model_bitfit_random(model):
    trainable_components = ["bias"]

    # Disale all the gradients
    for param in model.parameters():
        param.requires_grad = False

    # Add classification head to trainable components
    if trainable_components:
        trainable_components = trainable_components + ["pooler.dense.bias"]

    trainable_components = trainable_components + ["head"]

    vector = []

    for name, param in model.named_parameters():
        for component in trainable_components:
            if component in name:
                if np.random.random(1)[0] >= 0.5:
                    vector.append(1)
                    param.requires_grad = True
                else:
                    vector.append(0)

    return vector

def auto_peft1(model, mask):

    vector = []

    for i in range(len(mask)):
        block_idx = i//3
        bit_idx = i%3

        bit = mask[i]
        vector.append(bit)

        # print("Bit: {} | Block {} | Bit Idx {}".format(bit, block_idx, bit_idx))

        '''
            bit_idx = 0 for norm layers
            bit_idx = 1 for attn layers
            bit_idx = 2 for mlp layers
        '''
        if(bit == 1):
            if(bit_idx == 0):
                enable_module(model.blocks[block_idx].norm1)
                enable_module(model.blocks[block_idx].norm2)
            if(bit_idx == 1):
                enable_module(model.blocks[block_idx].attn)
            if(bit_idx == 2):
                enable_module(model.blocks[block_idx].mlp)

        if(bit == 0):
            if(bit_idx == 0):
                disable_module(model.blocks[block_idx].norm1)
                disable_module(model.blocks[block_idx].norm2)
            if(bit_idx == 1):
                disable_module(model.blocks[block_idx].attn)
            if(bit_idx == 2):
                disable_module(model.blocks[block_idx].mlp)    

    return vector


def auto_peft2(model, mask):

    vector = []

    for i in range(len(mask)):
        block_idx = i//4
        bit_idx = i%4

        bit = mask[i]

        vector.append(bit)

        # print("Bit: {} | Block {} | Bit Idx {}".format(bit, block_idx, bit_idx))

        '''
            bit_idx = 0 for norm1 layer
            bit_idx = 1 for attn layers
            bit_idx = 2 for norm2 layer
            bit_idx = 3 for mlp layers
        '''

        if(bit == 1):
            if(bit_idx == 0):
                enable_module(model.blocks[block_idx].norm1)
            if(bit_idx == 1):
                enable_module(model.blocks[block_idx].attn)
            if(bit_idx == 2):
                enable_module(model.blocks[block_idx].norm2)
            if(bit_idx == 3):
                enable_module(model.blocks[block_idx].mlp)

        if(bit == 0):
            if(bit_idx == 0):
                disable_module(model.blocks[block_idx].norm1)
            if(bit_idx == 1):
                disable_module(model.blocks[block_idx].attn)
            if(bit_idx == 2):
                disable_module(model.blocks[block_idx].norm2)
            if(bit_idx == 3):
                disable_module(model.blocks[block_idx].mlp)

    return vector



def create_lora_model(
    model,
    lora_r: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.0,
    tune_k=False,
    block_mask=None,
):
    lora_model = copy.deepcopy(model)

    tune_list = [True, True, True] if tune_k else [True, False, True]

    block_mask = (
        [1] * len(model.blocks) if block_mask is None else block_mask
    )  # Apply LoRA to all attention layers if mask is not given.

    for idx, block in enumerate(lora_model.blocks):
        if block_mask[idx] == 1:
            in_d = block.attn.qkv.in_features
            out_d = block.attn.qkv.out_features
            block.attn.qkv = lora.MergedLinear(
                in_d,
                out_d,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                enable_lora=tune_list,
            )

    lora_model.load_state_dict(model.state_dict(), strict=False)
    lora.mark_only_lora_as_trainable(lora_model)

    return lora_model




def get_timm_model(encoder, num_classes, **kwargs):
    """
    Returns a timm model for a given encoder.
    """

    assert num_classes is not None, "Number of classes cannot be None"

    pretrained = kwargs["pretrained"] if "pretrained" in kwargs else True
    print("Pretrained: ", pretrained)

    if encoder == "resnet50":
        model = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    if encoder == "vit_base":
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif encoder == "vit_base_moco":
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=num_classes,
        )

        # Load the MOCO checkpoint
        CKPT_PATH = "checkpoints/vit_base_moco_0299.pth.tar"
        checkpoint = torch.load(CKPT_PATH)
        new_state_dict = utils.convert_ssl_to_timm(checkpoint)
        mssg = model.load_state_dict(new_state_dict, strict=False)

    elif encoder == "vit_base_mae":
        model = timm.create_model(
            "vit_base_patch16_224.mae",
            pretrained=True,
            num_classes=num_classes,
        )

    elif encoder == "vit_base_clip":
        model = timm.create_model(
            "timm/vit_base_patch16_clip_224.openai",
            pretrained=True,
            num_classes=num_classes,
        )

    elif encoder == "vit_base_dino":
        model = timm.create_model(
            "vit_base_patch16_224.dino",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif encoder == "vit_large":
        model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif encoder == "vit_huge":
        model = timm.create_model(
            "vit_huge_patch14_224", pretrained=pretrained, num_classes=num_classes
        )

    return model


def get_masked_model(model, method, **kwargs):
    if method == "fullft":
        pass
    elif method == "train_from_scratch":
        pass
    elif method == "tune_attention":
        disable_module(model)
        vector = tune_attention_layers(model)
    elif method == "tune_attention_params_random":
        disable_module(model)
        vector = tune_attention_params_random(model, kwargs["mask"])
    elif method == "tune_attention_blocks_random":
        disable_module(model)
        vector = tune_blocks_random(model, kwargs["mask"], segment="attention")
    elif method == "bitfit":
        vector = get_model_for_bitfit(model, "timm")
    elif method == "tune_bitfit_random":
        vector = get_model_bitfit_random(model)
    elif method == "tune_layernorm":
        disable_module(model)
        vector = tune_layernorm_layers(model)
    elif method == "tune_layernorm_random":
        disable_module(model)
        vector = tune_layernorm_random(model)
    elif method == "tune_layernorm_blocks_random":
        disable_module(model)
        vector = tune_blocks_random(model, kwargs["mask"], segment="layernorm")
    elif method == "tune_mlp_blocks_random":
        disable_module(model)
        vector = tune_blocks_random(model, kwargs["mask"], segment="mlp")

    # New Methods
    elif method == "tune_attention_layernorm":
        disable_module(model)
        vector = tune_blocks_random(model, kwargs["mask"], segment="attention")
        vector = tune_blocks_random(model, kwargs["mask"], segment="layernorm")
    elif method == "tune_attention_mlp":
        disable_module(model)
        vector = tune_blocks_random(model, kwargs["mask"], segment="attention")
        vector = tune_blocks_random(model, kwargs["mask"], segment="mlp")
    elif method == "tune_layernorm_mlp":
        disable_module(model)
        vector = tune_blocks_random(model, kwargs["mask"], segment="layernorm")
        vector = tune_blocks_random(model, kwargs["mask"], segment="mlp")
    elif method == "tune_attention_layernorm_mlp":
        disable_module(model)
        vector = tune_blocks_random(model, kwargs["mask"], segment="attention")
        vector = tune_blocks_random(model, kwargs["mask"], segment="layernorm")
        vector = tune_blocks_random(model, kwargs["mask"], segment="mlp")
    elif method == "tune_full_block":
        disable_module(model)
        vector = tune_blocks_random(model, kwargs["mask"], segment="full")
    elif method == 'auto_peft1':
        disable_module(model)
        vector = auto_peft1(model, kwargs["mask"])
    elif method == 'auto_peft2':
        disable_module(model)
        vector = auto_peft2(model, kwargs["mask"])
    else:
        raise NotImplementedError

    enable_module(model.head)

    return vector


def get_model_from_vector(model, method, vector):
    if method == "tune_attention_blocks_random":
        disable_module(model)
        enable_from_vector(vector, model)


def create_random_mask(mask_length, generation_method, device, **kwargs):
    # return np.random.randint(low=0, high=2, size=mask_length)
    # return nn.Parameter(torch.ones(mask_length).to(device))
    # return nn.Parameter(torch.randn(low=0, high=2, size=(mask_length,), dtype=torch.float32, requires_grad=True).to(device))

    # Generate a random mask with values close to 1
    if generation_method == "random":
        sigma = kwargs["sigma"]
        epsilon = 0.05
        mask = nn.Parameter(
            1
            + sigma
            * torch.randn(mask_length, dtype=torch.float32, requires_grad=True).to(
                device
            )
        )

    elif generation_method == "random_gumbel":
        sigma = kwargs["sigma"]
        mask = nn.Parameter(
            0
            + sigma
            * torch.randn(mask_length, dtype=torch.float32, requires_grad=True).to(
                device
            )
        )

    elif generation_method == "constant":
        sigma = kwargs["sigma"]
        mask = nn.Parameter(
            sigma
            + torch.ones(mask_length, dtype=torch.float32, requires_grad=True).to(
                device
            )
        )

    elif generation_method == "searched":
        """
        Generate a random mask with first and last three values centered around 1 and the rest centered around 0.5
        """
        sigma = kwargs["sigma"]
        tensor = torch.zeros(mask_length, dtype=torch.float32)
        tensor[:3] = 1 + sigma * torch.randn(3, dtype=torch.float32)
        tensor[-3:] = 1 + sigma * torch.randn(3, dtype=torch.float32)
        tensor[3:-3] = 0.5 + 0.1 * sigma * torch.randn(
            mask_length - 6, dtype=torch.float32
        )  # Initialize these values with a smaller deviation (sigma)
        tensor = tensor.requires_grad_(True).to(device)
        mask = nn.Parameter(tensor)
        # mask = nn.Parameter(torch.zeros(mask_length, dtype=torch.float32, requires_grad=True).to(device))

        # # Initialize first and last three values centered around 1
        # mask[:3] = 1 + sigma * torch.randn(3, dtype=torch.float32).to(device)
        # mask[-3:] = 1 + sigma * torch.randn(3, dtype=torch.float32).to(device)

        # # Initialize the rest of the values centered around 0.5
        # mask[3:-3] = 0.5 + sigma * torch.randn(mask_length-6, dtype=torch.float32).to(device)

    return mask


def gumbel_sigmoid(
    logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10
) -> torch.Tensor:
    uniform = logits.new_empty([2] + list(logits.shape)).uniform_(0, 1)

    noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
    res = torch.sigmoid((logits + noise) / tau)

    if hard:
        res = ((res > 0.5).type_as(res) - res).detach() + res

    return res


def plot_changes(fine_tuned_ckpt, base_model, args):
    """
    Plots the changes in different layers of a model
    """

    if args.model == "vit_base":
        num_layers = 12
    if args.model == "vit_large":
        num_layers = 24
    if args.model == "vit_huge":
        num_layers = 32

    fine_tuned_model = get_timm_model(args.model, args.num_classes)
    ckpt = torch.load(fine_tuned_ckpt)
    fine_tuned_model.load_state_dict(ckpt["model"], strict=True)
    fine_tuned_model = fine_tuned_model.cpu()

    def _calc_mean_diff(ft_p, base_p):
        return np.mean(np.abs(np.array(ft_p.data - base_p.data)))

    def _get_component_name(name):
        return re.split(r".[0-9]+.", name)[1]

    def _get_component_layer(name):
        return int(name.split(".")[1])

    base_model = base_model.cpu()
    # fine_tuned_model = base_model.cpu()
    print(fine_tuned_ckpt)

    changes = []
    for ft_name, ft_param in fine_tuned_model.named_parameters():
        if ft_param.requires_grad and (".attn." in ft_name or "attention" in ft_name):
            for base_name, base_param in base_model.named_parameters():
                if ft_name == base_name:
                    changes.append(
                        {
                            "name": ft_name,
                            "value": _calc_mean_diff(ft_param, base_param),
                        }
                    )

    keys = list(set(_get_component_name(c["name"]) for c in changes))
    keys_mapper = {k: i for i, k in enumerate(keys)}

    total_weights = np.zeros(len(keys))
    for change in changes:
        total_weights[keys_mapper[_get_component_name(change["name"])]] += change[
            "value"
        ]

    keys = [keys[i] for i in np.argsort(-total_weights)]
    keys_mapper = {k: i for i, k in enumerate(keys)}

    avg_column = np.zeros(len(keys))
    values_map = np.zeros((len(keys), num_layers + 1))
    for change in changes:
        avg_column[keys_mapper[_get_component_name(change["name"])]] += change["value"]
        values_map[
            keys_mapper[_get_component_name(change["name"])],
            _get_component_layer(change["name"]),
        ] = change["value"]
    avg_column /= num_layers
    values_map[:, -1] = avg_column

    print(values_map)

    fig, ax = plt.subplots(figsize=(num_layers, len(keys)))
    xticklabels = [f"Layer {i}" for i in range(num_layers)]
    xticklabels.append("Avg.")
    yticklabels = keys
    sns.heatmap(
        values_map,
        cmap="Blues",
        ax=ax,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
    )

    filename = args.dataset + "_" + args.tuning_method + "_" + str(vector_idx)
    plt.savefig(os.path.join(args.fig_savepath, filename + ".png"))


# Calibration error scores in the form of loss metrics
class ECELoss(nn.Module):
    """
    Compute ECE (Expected Calibration Error)
    """

    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return 0.0
        else:
            return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def accuracy_by_gender(output, target, sens_attribute, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
       for the whole dataset, male population, and female population separately.
    """
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        # Calculate accuracy for the whole dataset
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))

        # Calculate accuracy for the male population
        male_indices = [i for i, gender in enumerate(sens_attribute) if gender == 'M']
        male_correct = correct[:, male_indices]
        # correct[male_indices]

        res_male = []
        for k in topk:
            correct_k = male_correct[:k].flatten().sum(dtype=torch.float32)
            res_male.append(correct_k * (100.0 / len(male_indices)))

        # Calculate accuracy for the female population
        female_indices = [i for i, gender in enumerate(sens_attribute) if gender == 'F']
        female_correct = correct[:, female_indices]
        # correct[female_indices]
        
        res_female = []
        for k in topk:
            correct_k = female_correct[:k].flatten().sum(dtype=torch.float32)
            res_female.append(correct_k * (100.0 / len(female_indices)))

        return res, res_male, res_female

def auc_by_gender(output, target, sens_attribute, topk=(1,)):
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        maxk = 1
        pos_label = 1

        output_with_softmax = torch.softmax(output, dim=1).cpu().detach().data.numpy()
        target = target.cpu().detach().data

        if(output_with_softmax.shape[1] == 2):
            output_with_softmax = output_with_softmax[:, 1]

        try:
            score = sklm.roc_auc_score(target, output_with_softmax, multi_class='ovr')
        except:
            #import pdb; pdb.set_trace()
            score = np.nan

        try:
            type0_indices = [i for i, gender in enumerate(sens_attribute) if gender == 'M']
            type0_output = output_with_softmax[type0_indices]
            type0_target = target[type0_indices]
            type0_score = sklm.roc_auc_score(type0_target, type0_output, multi_class='ovr')
        except:
            #import pdb; pdb.set_trace()
            type0_score = np.nan

        try:
            type1_indices = [i for i, gender in enumerate(sens_attribute) if gender == 'F']
            type1_output = output_with_softmax[type1_indices]
            type1_target = target[type1_indices]
            type1_score = sklm.roc_auc_score(type1_target, type1_output, multi_class='ovr')
        except:
            #import pdb; pdb.set_trace()
            type1_score = np.nan

        return score, type0_score, type1_score
        

def accuracy_by_skin_type(output, target, sens_attribute, topk=(1,), num_skin_types=6):
    """Computes the accuracy over the k top predictions for the specified values of k
       for the whole dataset and different skin types separately.
    """
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        # Calculate accuracy for the whole dataset
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))

        # Calculate accuracy for the each skin type
        type0_indices = [i for i, _skin_type in enumerate(sens_attribute) if _skin_type == 0]
        type0_correct = correct[:, type0_indices]
        res_type0 = []
        for k in topk:
            correct_k = type0_correct[:k].flatten().sum(dtype=torch.float32)
            try:
                res_type0.append(correct_k * (100.0 / len(type0_indices)))
            except:
                res_type0.append(torch.tensor(0.0))

        type1_indices = [i for i, _skin_type in enumerate(sens_attribute) if _skin_type == 1]
        type1_correct = correct[:, type1_indices]
        res_type1 = []
        for k in topk:
            correct_k = type1_correct[:k].flatten().sum(dtype=torch.float32)
            try:    
                res_type1.append(correct_k * (100.0 / len(type1_indices)))
            except:
                res_type1.append(torch.tensor(0.0))

        type2_indices = [i for i, _skin_type in enumerate(sens_attribute) if _skin_type == 2]
        type2_correct = correct[:, type2_indices]
        res_type2 = []
        for k in topk:
            correct_k = type2_correct[:k].flatten().sum(dtype=torch.float32)
            try:
                res_type2.append(correct_k * (100.0 / len(type2_indices)))
            except:
                res_type2.append(torch.tensor(0.0))

        type3_indices = [i for i, _skin_type in enumerate(sens_attribute) if _skin_type == 3]
        type3_correct = correct[:, type3_indices]
        res_type3 = []
        for k in topk:
            correct_k = type3_correct[:k].flatten().sum(dtype=torch.float32)
            try:
                res_type3.append(correct_k * (100.0 / len(type3_indices)))
            except:
                res_type3.append(torch.tensor(0.0))

        type4_indices = [i for i, _skin_type in enumerate(sens_attribute) if _skin_type == 4]
        type4_correct = correct[:, type4_indices]
        res_type4 = []
        for k in topk:
            correct_k = type4_correct[:k].flatten().sum(dtype=torch.float32)
            try:
                res_type4.append(correct_k * (100.0 / len(type4_indices)))
            except:
                res_type4.append(torch.tensor(0.0))

        type5_indices = [i for i, _skin_type in enumerate(sens_attribute) if _skin_type == 5]
        type5_correct = correct[:, type5_indices]
        res_type5 = []
        for k in topk:
            correct_k = type5_correct[:k].flatten().sum(dtype=torch.float32)
            
            try:
                res_type5.append(correct_k * (100.0 / len(type5_indices)))
            except:
                res_type5.append(torch.tensor(0.0))
                
        return res, res_type0, res_type1, res_type2, res_type3, res_type4, res_type5

def accuracy_by_skin_type_binary(output, target, sens_attribute, topk=(1,), num_skin_types=2):
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        # Calculate accuracy for the whole dataset
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))

        # Calculate accuracy for the each skin type
        type0_indices = [i for i, _skin_type in enumerate(sens_attribute) if _skin_type == 0]
        type0_correct = correct[:, type0_indices]
        res_type0 = []
        for k in topk:
            correct_k = type0_correct[:k].flatten().sum(dtype=torch.float32)
            try:
                res_type0.append(correct_k * (100.0 / len(type0_indices)))
            except:
                res_type0.append(torch.tensor(0.0))

        type1_indices = [i for i, _skin_type in enumerate(sens_attribute) if _skin_type == 1]
        type1_correct = correct[:, type1_indices]
        res_type1 = []
        for k in topk:
            correct_k = type1_correct[:k].flatten().sum(dtype=torch.float32)
            try:    
                res_type1.append(correct_k * (100.0 / len(type1_indices)))
            except:
                res_type1.append(torch.tensor(0.0))

        return res, res_type0, res_type1
    


def auc_by_skin_type(output, target, sens_attribute, topk=(1,), num_skin_types=6):
    """Computes the AUC over the k top predictions for the specified values of k
       for the whole dataset and different skin types separately.
    """
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        maxk = 1
        pos_label = 1

        output_with_softmax = torch.softmax(output, dim=1).cpu().detach().data.numpy()
        target = target.cpu().detach().data

        #print("TARGET: ", target)

        # Calculate auc for the whole dataset
        try:
            score = sklm.roc_auc_score(target, output_with_softmax, multi_class='ovr')
        except:
            score = np.nan

        #print("Output with softmax: ", output_with_softmax)
        #print("Sens attribute: ", sens_attribute)

        # Calculate AUC for the each skin type
        try:
            type0_indices = [i for i, _skin_type in enumerate(sens_attribute) if _skin_type == 0]
            type0_output = output_with_softmax[type0_indices]
            type0_target = target[type0_indices]
            type0_score = sklm.roc_auc_score(type0_target, type0_output, multi_class='ovr')
        except:
            type0_score = np.nan

        try:
            type1_indices = [i for i, _skin_type in enumerate(sens_attribute) if _skin_type == 1]
            #type1_output = output_with_softmax[:, type1_indices]
            type1_output = output_with_softmax[type1_indices]
            type1_target = target[type1_indices]
            type1_score = sklm.roc_auc_score(type1_target, type1_output, multi_class='ovr')
        except:
            type1_score = np.nan

        try:
            type2_indices = [i for i, _skin_type in enumerate(sens_attribute) if _skin_type == 2]
            #type2_output = output_with_softmax[:, type2_indices]
            type2_output = output_with_softmax[type2_indices]
            type2_target = target[type2_indices]
            type2_score = sklm.roc_auc_score(type2_target, type2_output, multi_class='ovr')
        except:
            type2_score = np.nan

        try:
            type3_indices = [i for i, _skin_type in enumerate(sens_attribute) if _skin_type == 3]
            #type3_output = output_with_softmax[:, type3_indices]
            type3_output = output_with_softmax[type3_indices]
            type3_target = target[type3_indices]
            type3_score = sklm.roc_auc_score(type3_target, type3_output, multi_class='ovr')
        except:
            type3_score = np.nan

        try:
            type4_indices = [i for i, _skin_type in enumerate(sens_attribute) if _skin_type == 4]
            type4_output = output_with_softmax[type4_indices]
            type4_target = target[type4_indices]
            type4_score = sklm.roc_auc_score(type4_target, type4_output, multi_class='ovr')
        except:
            type4_score = np.nan

        try:
            type5_indices = [i for i, _skin_type in enumerate(sens_attribute) if _skin_type == 5]
            #type5_output = output_with_softmax[:, type5_indices]
            type5_output = output_with_softmax[type5_indices]
            type5_target = target[type5_indices]
            type5_score = sklm.roc_auc_score(type5_target, type5_output, multi_class='ovr')
        except:
            type5_score = np.nan

        return score, type0_score, type1_score, type2_score, type3_score, type4_score, type5_score


def auc_by_skin_type_binary(output, target, sens_attribute, topk=(1,), num_skin_types=2):
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        maxk = 1
        pos_label = 1

        output_with_softmax = torch.softmax(output, dim=1).cpu().detach().data.numpy()
        target = target.cpu().detach().data

        if(output_with_softmax.shape[1] == 2):
            output_with_softmax = output_with_softmax[:, 1]

        # Calculate auc for the whole dataset
        try:
            score = sklm.roc_auc_score(target, output_with_softmax, multi_class='ovr')
        except:
            #import pdb; pdb.set_trace()
            score = np.nan
                
        # Calculate AUC for the each skin type
        try:
            type0_indices = [i for i, _skin_type in enumerate(sens_attribute) if _skin_type == 0]
            type0_output = output_with_softmax[type0_indices]
            type0_target = target[type0_indices]
            type0_score = sklm.roc_auc_score(type0_target, type0_output, multi_class='ovr')
        except:
            type0_score = np.nan

        try:
            type1_indices = [i for i, _skin_type in enumerate(sens_attribute) if _skin_type == 1]
            #type1_output = output_with_softmax[:, type1_indices]
            type1_output = output_with_softmax[type1_indices]
            type1_target = target[type1_indices]
            type1_score = sklm.roc_auc_score(type1_target, type1_output, multi_class='ovr')
        except:
            type1_score = np.nan

        return score, type0_score, type1_score

def accuracy_by_age(output, target, sens_attribute, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
       for the whole dataset and different age groups separately.
    """
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        # Calculate accuracy for the whole dataset
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))

        # Calculate accuracy for the each age group
        type0_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 0]
        type0_correct = correct[:, type0_indices]
        res_type0 = []
        for k in topk:
            correct_k = type0_correct[:k].flatten().sum(dtype=torch.float32)
            try:
                res_type0.append(correct_k * (100.0 / len(type0_indices)))
            except:
                res_type0.append(torch.tensor(0.0))

        type1_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 1]
        type1_correct = correct[:, type1_indices]
        res_type1 = []
        for k in topk:
            correct_k = type1_correct[:k].flatten().sum(dtype=torch.float32)
            try:    
                res_type1.append(correct_k * (100.0 / len(type1_indices)))
            except:
                res_type1.append(torch.tensor(0.0))

        type2_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 2]
        type2_correct = correct[:, type2_indices]
        res_type2 = []
        for k in topk:
            correct_k = type2_correct[:k].flatten().sum(dtype=torch.float32)
            try:
                res_type2.append(correct_k * (100.0 / len(type2_indices)))
            except:
                res_type2.append(torch.tensor(0.0))

        type3_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 3]
        type3_correct = correct[:, type3_indices]
        res_type3 = []
        for k in topk:
            correct_k = type3_correct[:k].flatten().sum(dtype=torch.float32)
            try:
                res_type3.append(correct_k * (100.0 / len(type3_indices)))
            except:
                res_type3.append(torch.tensor(0.0))

        type4_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 4]
        type4_correct = correct[:, type4_indices]
        res_type4 = []
        for k in topk:
            correct_k = type4_correct[:k].flatten().sum(dtype=torch.float32)
            try:
                res_type4.append(correct_k * (100.0 / len(type4_indices)))
            except:
                res_type4.append(torch.tensor(0.0))

                
        return res, res_type0, res_type1, res_type2, res_type3, res_type4


def auc_by_age(output, target, sens_attribute, topk=(1,)):

    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        maxk = 1
        pos_label = 1

        output_with_softmax = torch.softmax(output, dim=1).cpu().detach().data.numpy()
        target = target.cpu().detach().data

        # If it is a binary classification, we need to convert the output to a shape (n_samples,). More info: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        if(output_with_softmax.shape[1] == 2):
            output_with_softmax = output_with_softmax[:, 1]

        # Calculate auc for the whole dataset
        try:
            score = sklm.roc_auc_score(target, output_with_softmax, multi_class='ovr')
        except:
            #import pdb; pdb.set_trace()
            score = np.nan

        # Calculate AUC for the each age group
        try:
            type0_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 0]
            type0_output = output_with_softmax[type0_indices]
            type0_target = target[type0_indices]
            type0_score = sklm.roc_auc_score(type0_target, type0_output, multi_class='ovr')
        except:
            #import pdb; pdb.set_trace()
            type0_score = np.nan

        try:
            type1_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 1]
            type1_output = output_with_softmax[type1_indices]
            type1_target = target[type1_indices]
            type1_score = sklm.roc_auc_score(type1_target, type1_output, multi_class='ovr')
        except:
            type1_score = np.nan
        
        try:
            type2_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 2]
            type2_output = output_with_softmax[type2_indices]
            type2_target = target[type2_indices]
            type2_score = sklm.roc_auc_score(type2_target, type2_output, multi_class='ovr')
        except:
            type2_score = np.nan

        try:
            type3_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 3]
            type3_output = output_with_softmax[type3_indices]
            type3_target = target[type3_indices]
            type3_score = sklm.roc_auc_score(type3_target, type3_output, multi_class='ovr')
        except:
            type3_score = np.nan

        try:
            type4_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 4]
            type4_output = output_with_softmax[type4_indices]
            type4_target = target[type4_indices]
            type4_score = sklm.roc_auc_score(type4_target, type4_output, multi_class='ovr')
        except:
            type4_score = np.nan

        return score, type0_score, type1_score, type2_score, type3_score, type4_score


def accuracy_by_age_binary(output, target, sens_attribute, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
       for the whole dataset and different age groups separately.
    """
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        # Calculate accuracy for the whole dataset
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))

        # Calculate accuracy for the each age group
        type0_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 0]
        type0_correct = correct[:, type0_indices]
        res_type0 = []
        for k in topk:
            correct_k = type0_correct[:k].flatten().sum(dtype=torch.float32)
            try:
                res_type0.append(correct_k * (100.0 / len(type0_indices)))
            except:
                res_type0.append(torch.tensor(0.0))

        type1_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 1]
        type1_correct = correct[:, type1_indices]
        res_type1 = []
        for k in topk:
            correct_k = type1_correct[:k].flatten().sum(dtype=torch.float32)
            try:    
                res_type1.append(correct_k * (100.0 / len(type1_indices)))
            except:
                res_type1.append(torch.tensor(0.0))
                
        return res, res_type0, res_type1

def auc_by_age_binary(output, target, sens_attribute, topk=(1,)):
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        maxk = 1
        pos_label = 1

        output_with_softmax = torch.softmax(output, dim=1).cpu().detach().data.numpy()
        target = target.cpu().detach().data

        # If it is a binary classification, we need to convert the output to a shape (n_samples,). More info: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        if(output_with_softmax.shape[1] == 2):
            output_with_softmax = output_with_softmax[:, 1]

        # Calculate auc for the whole dataset
        try:
            score = sklm.roc_auc_score(target, output_with_softmax, multi_class='ovr')
        except:
            #import pdb; pdb.set_trace()
            score = np.nan
    
        # Calculate AUC for the each age group
        try:
            type0_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 0]
            type0_output = output_with_softmax[type0_indices]
            type0_target = target[type0_indices]
            type0_score = sklm.roc_auc_score(type0_target, type0_output, multi_class='ovr')
        except:
            #import pdb; pdb.set_trace()
            type0_score = np.nan

        try:
            type1_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 1]
            type1_output = output_with_softmax[type1_indices]
            type1_target = target[type1_indices]
            type1_score = sklm.roc_auc_score(type1_target, type1_output, multi_class='ovr')
        except:
            #import pdb; pdb.set_trace()
            type1_score = np.nan

        return score, type0_score, type1_score


def accuracy_by_age_sex(output, target, sens_attribute, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
       for the whole dataset and different age groups separately.
    """
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        # Calculate accuracy for the whole dataset
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))

        # Calculate accuracy for the each age group
        type0_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 0]
        type0_correct = correct[:, type0_indices]
        res_type0 = []
        for k in topk:
            correct_k = type0_correct[:k].flatten().sum(dtype=torch.float32)
            try:
                res_type0.append(correct_k * (100.0 / len(type0_indices)))
            except:
                res_type0.append(torch.tensor(0.0))

        type1_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 1]
        type1_correct = correct[:, type1_indices]
        res_type1 = []
        for k in topk:
            correct_k = type1_correct[:k].flatten().sum(dtype=torch.float32)
            try:    
                res_type1.append(correct_k * (100.0 / len(type1_indices)))
            except:
                res_type1.append(torch.tensor(0.0))

        type2_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 2]
        type2_correct = correct[:, type2_indices]
        res_type2 = []
        for k in topk:
            correct_k = type2_correct[:k].flatten().sum(dtype=torch.float32)
            try:
                res_type2.append(correct_k * (100.0 / len(type2_indices)))
            except:
                res_type2.append(torch.tensor(0.0))

        type3_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 3]
        type3_correct = correct[:, type3_indices]
        res_type3 = []
        for k in topk:
            correct_k = type3_correct[:k].flatten().sum(dtype=torch.float32)
            try:
                res_type3.append(correct_k * (100.0 / len(type3_indices)))
            except:
                res_type3.append(torch.tensor(0.0))


                
        return res, res_type0, res_type1, res_type2, res_type3

def auc_by_age_sex(output, target, sens_attribute, topk=(1,)):

    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        maxk = 1
        pos_label = 1

        output_with_softmax = torch.softmax(output, dim=1).cpu().detach().data.numpy()
        target = target.cpu().detach().data

        # If it is a binary classification, we need to convert the output to a shape (n_samples,). More info: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        if(output_with_softmax.shape[1] == 2):
            output_with_softmax = output_with_softmax[:, 1]

        # Calculate auc for the whole dataset
        try:
            score = sklm.roc_auc_score(target, output_with_softmax, multi_class='ovr')
        except:
            #import pdb; pdb.set_trace()
            score = np.nan

        # Calculate AUC for the each age group
        try:
            type0_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 0]
            type0_output = output_with_softmax[type0_indices]
            type0_target = target[type0_indices]
            type0_score = sklm.roc_auc_score(type0_target, type0_output, multi_class='ovr')
        except:
            #import pdb; pdb.set_trace()
            type0_score = np.nan

        try:
            type1_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 1]
            type1_output = output_with_softmax[type1_indices]
            type1_target = target[type1_indices]
            type1_score = sklm.roc_auc_score(type1_target, type1_output, multi_class='ovr')
        except:
            type1_score = np.nan
        
        try:
            type2_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 2]
            type2_output = output_with_softmax[type2_indices]
            type2_target = target[type2_indices]
            type2_score = sklm.roc_auc_score(type2_target, type2_output, multi_class='ovr')
        except:
            type2_score = np.nan

        try:
            type3_indices = [i for i, _age_group in enumerate(sens_attribute) if _age_group == 3]
            type3_output = output_with_softmax[type3_indices]
            type3_target = target[type3_indices]
            type3_score = sklm.roc_auc_score(type3_target, type3_output, multi_class='ovr')
        except:
            type3_score = np.nan


        return score, type0_score, type1_score, type2_score, type3_score

def accuracy_by_race_binary(output, target, sens_attribute, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
       for the whole dataset and different age groups separately.
    """
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        # Calculate accuracy for the whole dataset
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))

        # Calculate accuracy for the each race group
        type0_indices = [i for i, _race_group in enumerate(sens_attribute) if _race_group == 0] # Black Population
        type0_correct = correct[:, type0_indices]
        res_type0 = []
        for k in topk:
            correct_k = type0_correct[:k].flatten().sum(dtype=torch.float32)
            try:
                res_type0.append(correct_k * (100.0 / len(type0_indices)))
            except:
                res_type0.append(torch.tensor(0.0))

        type1_indices = [i for i, _race_group in enumerate(sens_attribute) if _race_group == 1] # Asian/White Population
        type1_correct = correct[:, type1_indices]
        res_type1 = []
        for k in topk:
            correct_k = type1_correct[:k].flatten().sum(dtype=torch.float32)
            try:    
                res_type1.append(correct_k * (100.0 / len(type1_indices)))
            except:
                res_type1.append(torch.tensor(0.0))
                
        return res, res_type0, res_type1


def auc_by_race_binary(output, target, sens_attribute, topk=(1,)):
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        maxk = 1
        pos_label = 1

        output_with_softmax = torch.softmax(output, dim=1).cpu().detach().data.numpy()
        target = target.cpu().detach().data

        # If it is a binary classification, we need to convert the output to a shape (n_samples,). More info: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        if(output_with_softmax.shape[1] == 2):
            output_with_softmax = output_with_softmax[:, 1]

        # Calculate auc for the whole dataset
        try:
            score = sklm.roc_auc_score(target, output_with_softmax, multi_class='ovr')
        except:
            #import pdb; pdb.set_trace()
            score = np.nan
    
        # Calculate AUC for the each age group
        try:
            type0_indices = [i for i, _race_group in enumerate(sens_attribute) if _race_group == 0] # Black Population
            type0_output = output_with_softmax[type0_indices]
            type0_target = target[type0_indices]
            type0_score = sklm.roc_auc_score(type0_target, type0_output, multi_class='ovr')
        except:
            #import pdb; pdb.set_trace()
            type0_score = np.nan

        try:
            type1_indices = [i for i, _race_group in enumerate(sens_attribute) if _race_group == 1] # Asian/White Population
            type1_output = output_with_softmax[type1_indices]
            type1_target = target[type1_indices]
            type1_score = sklm.roc_auc_score(type1_target, type1_output, multi_class='ovr')
        except:
            #import pdb; pdb.set_trace()
            type1_score = np.nan

        return score, type0_score, type1_score


# Equalized odds Difference
def equiodds_difference(preds, labels, attrs):
    print("Preds: ", np.unique(preds))
    print("Labels: ", np.unique(labels.cpu()))
    print("Attrs: ", np.unique(attrs, return_counts=True))
    print("\n")
    return round(equalized_odds_difference(
                                labels.cpu(),
                                preds,
                                sensitive_features=attrs), 3)

# Equalized odds Ratio
def equiodds_ratio(preds, labels, attrs):
    return round(equalized_odds_ratio(
                        labels.cpu(),
                        preds,
                        sensitive_features=attrs), 3)

# Demographic Parity Difference
def dpd(preds, labels, attrs):
    return round(demographic_parity_difference(
                        labels.cpu(),
                        preds,
                        sensitive_features=attrs), 3)

# Demographic Parity Ratio
def dpr(preds, labels, attrs):
    return round(demographic_parity_ratio(
                        labels.cpu(),
                        preds,
                        sensitive_features=attrs), 3)

# def conditional_errors_binary(preds, labels, attrs):
#     """
#     Compute the conditional errors of A = 0/1. All the arguments need to be one-dimensional vectors.
#     :param preds: The predicted label given by a model.
#     :param labels: The groundtruth label.
#     :param attrs: The label of sensitive attribute.
#     :return: Overall classification error, error | A = 0, error | A = 1.
#     """
#     with torch.inference_mode():
#         batch_size = labels.size(0)
#         if labels.ndim == 2:
#             labels = labels.max(dim=1)[1]

#         preds = np.array(np.argmax(preds, axis=1))
#         labels = labels.cpu().detach().data.numpy()
#         attrs = np.array(attrs)

#         # import pdb; pdb.set_trace()
    
#         assert preds.shape == labels.shape and labels.shape == attrs.shape
#         cls_error = 1 - np.mean((preds == labels).astype('float'))
#         idx = attrs == 0
#         error_0 = 1 - np.mean((preds[idx] == labels[idx]).astype('float'))
#         error_1 = 1 - np.mean((preds[~idx] == labels[~idx]).astype('float'))
#         return cls_error, error_0, error_1

# def eqodd_at_specificity(output, labels, attrs, specificity):

#     with torch.inference_mode():
        
#         # if labels.ndim == 2:
#         #     labels = labels.max(dim=1)[1]

#         pred_probs = torch.softmax(output, dim=1).cpu().detach().data.numpy()
#         preds = np.argmax(pred_probs, axis=1)
        
#         labels = labels.cpu().detach().data.numpy()
#         attrs = np.array(attrs)
    
#         assert preds.shape == labels.shape and labels.shape == attrs.shape
#         fprs, tprs, thress = sklm.roc_curve(labels, preds)
#         thresh = interp1d(1 - fprs, thress)(specificity)
    
#         return cal_eqodd(preds, labels, attrs, threshold = thresh)

# def eqodd_at_sensitivity(output, labels, attrs, sensitivity):

#     with torch.inference_mode():
#         batch_size = labels.size(0)
#         if labels.ndim == 2:
#             labels = labels.max(dim=1)[1]

#         pred_probs = torch.softmax(output, dim=1).cpu().detach().data.numpy()
#         preds = np.argmax(pred_probs, axis=1)
#         attrs = np.array(attrs)
        
#         labels = labels.cpu().detach().data.numpy()
    
#         assert preds.shape == labels.shape and labels.shape == attrs.shape
#         fprs, tprs, thress = sklm.roc_curve(labels, preds)
#         thresh = interp1d(tprs, thress)(sensitivity)
        
#         return cal_eqodd(preds, labels, attrs, threshold = thresh)

# def cal_eqodd(pred_probs, labels, attrs, threshold=0.5):

#     with torch.inference_mode():
#         # import pdb; pdb.set_trace()
        
#         # if(not isinstance(labels, torch.Tensor)):
#         #     labels = torch.tensor(labels)
        
#         # if labels.ndim == 2:
#         #     labels = labels.max(dim=1)[1]

#         # pred_probs = torch.softmax(output, dim=1).cpu().detach().data.numpy()
#         labels = labels.cpu().detach().data

#         tol_predicted = (pred_probs > threshold).astype('float')
#         sens_idx = attrs == 0
#         target_idx = labels == 0
#         cls_error, error_0, error_1 = conditional_errors_binary(tol_predicted, labels, attrs)
#         cond_00 = np.mean((tol_predicted[np.logical_and(sens_idx, target_idx)]))
#         cond_10 = np.mean((tol_predicted[np.logical_and(~sens_idx, target_idx)]))
#         cond_01 = np.mean((tol_predicted[np.logical_and(sens_idx, ~target_idx)]))
#         cond_11 = np.mean((tol_predicted[np.logical_and(~sens_idx, ~target_idx)]))
#         return (1 - 0.5 * (np.abs(cond_00 - cond_10) + np.abs(cond_01 - cond_11)))

# def cal_eqopp(pred_probs, labels, attrs, threshold=0.5):
#     with torch.inference_mode():
#         labels = labels.cpu().detach().data
#         #tol_predicted = (pred_probs > threshold).astype('float')
#         tol_predicted = np.argmax(pred_probs, axis=1)
#         sens_idx = attrs == 0
#         target_idx = labels == 0

#         cond_00 = np.mean((tol_predicted[np.logical_and(sens_idx, target_idx)]))
#         cond_10 = np.mean((tol_predicted[np.logical_and(~sens_idx, target_idx)]))
#         cond_01 = np.mean((tol_predicted[np.logical_and(sens_idx, ~target_idx)]))
#         cond_11 = np.mean((tol_predicted[np.logical_and(~sens_idx, ~target_idx)]))

#         eqOpp1 = 1 - np.abs(cond_00 - cond_10)
#         eqOpp0 = 1 - np.abs(cond_01 - cond_11)

#         # import pdb; pdb.set_trace()

#         return eqOpp0, eqOpp1
            
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def auc(output, target, **kwargs):
    """Computes the top-1 AUC (Area Under the Curve)"""
    with torch.inference_mode():
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        # Get the predicted probabilities for the positive class (top-1)
        # pred_probs = torch.softmax(output, dim=1)[:, 0]
        maxk = 1
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t().flatten().cpu().numpy()
        #print("OUTPUT: ", torch.softmax(output, dim=1).cpu().data.numpy())
        #TODO: Check if the pred contains probabilities: elements should be between 0 and 1
        # pred_probs = torch.sigmoid(output).cpu().data.numpy()

        # Convert the tensors to NumPy arrays
        target = target.cpu().data.numpy()
        #print("TARGETS: ", target)
        # pred_probs = pred_probs.detach().cpu().numpy()
        # print(pred.shape, target.shape)
        # Calculate the AUC using sklearn's roc_auc_score function
        # fpr, tpr, thresholds = sklm.roc_curve(target, pred_probs, pos_label=1)
        pos_label = 1 if kwargs["pos_label"] is None else kwargs["pos_label"]
        fpr, tpr, thresholds = sklm.roc_curve(target, pred, pos_label=pos_label)
        auc = sklm.auc(fpr, tpr)

        return auc

def roc_auc_score_multiclass(output, target):
    with torch.inference_mode():
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

    maxk = 1
    output_with_softmax = torch.softmax(output, dim=1).cpu().detach().data.numpy()
    # best_prob, best_prob_idx = output_with_softmax.topk(maxk, 1, True, True)
    # best_prob = best_prob.t().flatten().cpu().detach().data.numpy()
    target = target.cpu().detach().data.numpy()

    #print("output_with_softmax: ", output_with_softmax)
    #print("TARGET: ", target)
    pos_label = 1
    score = sklm.roc_auc_score(target, output_with_softmax, multi_class='ovr')

    return score


# def roc_auc_score_multiclass(pred_class, actual_class, average="macro"):
#     with torch.inference_mode():
#         batch_size = actual_class.size(0)
#         if actual_class.ndim == 2:
#             actual_class = actual_class.max(dim=1)[1]

#     maxk = 1
#     _, pred_class = pred_class.topk(maxk, 1, True, True)
#     pred_class = pred_class.t().flatten().cpu().numpy()

#     actual_class = actual_class.cpu().data.numpy()

#     roc_auc = sklm.roc_auc_score(actual_class, pred_class, average=average, multi_class="ovr")

    # creating a set of all the unique classes using the actual class list
    # unique_class = set(actual_class)
    # roc_auc_dict = {}
    # for per_class in unique_class:
    #     # creating a list of all the classes except the current class
    #     other_class = [x for x in unique_class if x != per_class]

    #     # marking the current class as 1 and all other classes as 0
    #     new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    #     new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #     # using the sklearn metrics method to calculate the roc_auc_score
    #     roc_auc = sklm.roc_auc_score(new_actual_class, new_pred_class, average=average, multi_class="ovr")
    #     roc_auc_dict[per_class] = roc_auc

    # return roc_auc_dict


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    #     args.rank = int(os.environ["RANK"])
    #     args.world_size = int(os.environ["WORLD_SIZE"])
    #     args.gpu = int(os.environ["LOCAL_RANK"])
    # elif "SLURM_PROCID" in os.environ:
    #     args.rank = int(os.environ["SLURM_PROCID"])
    #     args.gpu = args.rank % torch.cuda.device_count()
    # elif hasattr(args, "rank"):
    #     pass
    # else:
    #     print("Not using distributed mode")
    #     args.distributed = False
    #     return

    # args.distributed = True

    # torch.cuda.set_device(args.gpu)
    # args.dist_backend = "nccl"
    # print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    # torch.distributed.init_process_group(
    #     backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    # )
    # torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)

    args.distributed = False
    return


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights. Original implementation taken from:
    https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/scripts/average_checkpoints.py#L16

    Args:
      inputs (List[str]): An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        model_params = state["model"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                f"For checkpoint {f}, expected list of params: {params_keys}, but found: {model_params_keys}"
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def store_model_weights(model, checkpoint_path, checkpoint_key="model", strict=True):
    """
    This method can be used to prepare weights files for new models. It receives as
    input a model architecture and a checkpoint from the training script and produces
    a file with the weights ready for release.

    Examples:
        from torchvision import models as M

        # Classification
        model = M.mobilenet_v3_large(weights=None)
        print(store_model_weights(model, './class.pth'))

        # Quantized Classification
        model = M.quantization.mobilenet_v3_large(weights=None, quantize=False)
        model.fuse_model(is_qat=True)
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
        _ = torch.ao.quantization.prepare_qat(model, inplace=True)
        print(store_model_weights(model, './qat.pth'))

        # Object Detection
        model = M.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
        print(store_model_weights(model, './obj.pth'))

        # Segmentation
        model = M.segmentation.deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=True)
        print(store_model_weights(model, './segm.pth', strict=False))

    Args:
        model (pytorch.nn.Module): The model on which the weights will be loaded for validation purposes.
        checkpoint_path (str): The path of the checkpoint we will load.
        checkpoint_key (str, optional): The key of the checkpoint where the model weights are stored.
            Default: "model".
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

    Returns:
        output_path (str): The location where the weights are saved.
    """
    # Store the new model next to the checkpoint_path
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = os.path.dirname(checkpoint_path)

    # Deep copy to avoid side effects on the model object.
    model = copy.deepcopy(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load the weights to the model to validate that everything works
    # and remove unnecessary weights (such as auxiliaries, etc.)
    if checkpoint_key == "model_ema":
        del checkpoint[checkpoint_key]["n_averaged"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            checkpoint[checkpoint_key], "module."
        )
    model.load_state_dict(checkpoint[checkpoint_key], strict=strict)

    tmp_path = os.path.join(output_dir, str(model.__hash__()))
    torch.save(model.state_dict(), tmp_path)

    sha256_hash = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hh = sha256_hash.hexdigest()

    output_path = os.path.join(output_dir, "weights-" + str(hh[:8]) + ".pth")
    os.replace(tmp_path, output_path)

    return output_path


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = (
                    f"{prefix}.{name}" if prefix != "" and "." in key else name
                )
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append(
                {"params": params[key], "weight_decay": params_weight_decay[key]}
            )
    return param_groups
