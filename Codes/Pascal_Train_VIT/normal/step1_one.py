import argparse
import datetime
import json
import os
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Iterable
import sys

sys.path.append('/nfs/turbo/coe-hunseok/sshoouri/Codes/')
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml

import utils
import utils.data_constants as data_constants
from multimae import multimae
from multimae.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from multimae.output_adapters import (ConvNeXtAdapter, DPTOutputAdapter,
                                      SegmenterMaskTransformerAdapter)
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import create_model
from utils.data_constants import COCO_SEMSEG_NUM_CLASSES
from utils.datasets_semseg import build_semseg_dataset, simple_transform 
from utils.dist import collect_results_cpu
from utils.log_images import log_semseg_wandb
from utils.optim_factory import LayerDecayValueAssigner, create_optimizer
from utils.pos_embed import interpolate_pos_embed_multimae
from utils.semseg_metrics import mean_iou

from utils.data_constants import NYU_MEAN, NYU_STD

from utils.dataset_regression import build_regression_dataset, build_regression_dataset_normal


import warnings
import cv2
import os.path
import numpy as np
import glob
import math
import torch
import json


def normalize_tensor(input_tensor, dim):
    norm = torch.norm(input_tensor, p='fro', dim=dim, keepdim=True)
    zero_mask = (norm == 0)
    norm[zero_mask] = 1
    out = input_tensor.div(norm)
    out[zero_mask.expand_as(out)] = 0
    return out

class NormalsMeter(object):
    def __init__(self, ignore_index=255):
        self.sum_deg_diff = 0
        self.total = 0
        self.ignore_index = ignore_index

    @torch.no_grad()
    def update(self, pred, gt):
        pred = pred.permute(0, 3, 1, 2) # [B, C, H, W]
        pred = 2 * pred / 255 - 1 # reverse post-processing
        valid_mask = (gt != self.ignore_index).all(dim=1)

        pred = normalize_tensor(pred, dim=1)
        gt = normalize_tensor(gt, dim=1)
        deg_diff = torch.rad2deg(2 * torch.atan2(torch.norm(pred - gt, dim=1), torch.norm(pred + gt, dim=1)))
        deg_diff = torch.masked_select(deg_diff, valid_mask)

        self.sum_deg_diff += torch.sum(deg_diff).cpu().item()
        self.total += deg_diff.numel()

    def get_score(self, verbose=False):
        eval_result = dict()
        eval_result['mean'] = self.sum_deg_diff / self.total

        return eval_result
    
@torch.no_grad()
def normal_metrics(pred, gt, mask_valid=None):
    # map to the original scale 
    ignore_index=255
    sum_deg_diff = 0
    total = 0
    #pred = pred.permute(0, 3, 1, 2) # [B, C, H, W]
    #pred = 2 * pred / 255 - 1 # reverse post-processing
    pred=(pred+1)*255/2
    gt=(gt+1)*255/2
    valid_mask = (gt != ignore_index).all(dim=1)

    pred = normalize_tensor(pred, dim=1)
    gt = normalize_tensor(gt, dim=1)
    deg_diff = torch.rad2deg(2 * torch.atan2(torch.norm(pred - gt, dim=1), torch.norm(pred + gt, dim=1)))
    deg_diff = torch.masked_select(deg_diff, valid_mask)

    sum_deg_diff += torch.sum(deg_diff).cpu().item()
    total += deg_diff.numel()


    metrics = {
        'mean': sum_deg_diff / total
    }
    return metrics



DOMAIN_CONF = {
    'rgb': {
        'channels': 3,
        'stride_level': 1,
        'aug_type': 'image',
        'input_adapter': partial(PatchedInputAdapter, num_channels=3),
    },
    'normal': {
        'channels': 3,
        'stride_level': 1,
        'aug_type': 'mask',
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
    },
    'semseg': {
        'stride_level': 4,
        'aug_type': 'mask',
        'input_adapter': partial(SemSegInputAdapter, num_classes=COCO_SEMSEG_NUM_CLASSES,
                                 dim_class_emb=64, interpolate_class_emb=False,
                                 emb_padding_idx=COCO_SEMSEG_NUM_CLASSES),
    },
    'pseudo_semseg': {
        'aug_type': 'mask'
    },
    'mask_valid': {
        'stride_level': 1,
        'aug_type': 'mask',
    },
}

def get_args():
    
   
   

    parser = argparse.ArgumentParser('MultiMAE depth fine-tuning script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=900, type=int)
    parser.add_argument('--save_ckpt_freq', default=200, type=int)
    parser.add_argument('--tmp', default=False, action='store_true')

    # Task parameters
    parser.add_argument('--in_domains', default='rgb', type=str,
                        help='Input domain names, separated by hyphen')
    parser.add_argument('--decoder_main_tasks', type=str, default='rgb',
                        help='for convnext & DPT adapters, separate tasks with a hyphen')
    parser.add_argument('--standardize_depth', action='store_true')
    parser.add_argument('--no_standardize_depth', action='store_false', dest='standardize_depth')
    parser.set_defaults(standardize_depth=False)

    # Model parameters
    parser.add_argument('--model', default='multivit_base', type=str, metavar='MODEL',
                        help='Name of MultiViT model to train')
    parser.add_argument('--num_global_tokens', default=1, type=int,
                        help='number of global tokens to add to encoder')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='base patch size for image-like modalities')
    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size for backbone')
    parser.add_argument('--drop_path_encoder', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')

    parser.add_argument('--output_adapter', type=str, default='dpt',
                        choices=['dpt', 'convnext'],
                        help='One of [dpt, convnext] (default: dpt)')
    parser.add_argument('--decoder_dim', default=1536, type=int,
                        help='Token dimension inside the decoder layers (for convnext adapter)')
    parser.add_argument('--decoder_depth', default=2, type=int,
                        help='Depth of decoder (for convnext adapter)')
    parser.add_argument('--drop_path_decoder', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--decoder_preds_per_patch', type=int, default=16,
                        help='Predictions per patch for convnext adapter')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")
    parser.add_argument('--decoder_decay', type=float, default=None,
                        help='decoder weight decay')

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate  (default: 1e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=0.0, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (0.0)')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA')
    parser.add_argument('--scale_input_lr', action='store_true')
    parser.add_argument('--no_scale_input_lr', action='store_false', dest='scale_input_lr')
    parser.set_defaults(scale_input_lr=True)
    parser.add_argument('--freeze_transformer', action='store_true')
    parser.add_argument('--no_freeze_transformer', action='store_false', dest='freeze_transformer')
    parser.set_defaults(freeze_transformer=False)

    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Finetuning parameters
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--loss', default='berhu',
                        help='Loss to use. One of [l1, l2, berhu] (default: berhu)')

    # Dataset parameters
    parser.add_argument('--data_path', default=data_constants.NYU_TRAIN_PATH, type=str, help='dataset path')
    parser.add_argument('--eval_data_path', default=data_constants.NYU_TEST_PATH, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--aug_name', default='nyu-augs', type=str)
    parser.add_argument('--color_augs', default=False, action='store_true')
    parser.add_argument('--no_color_augs', dest='color_augs', default=False, action='store_false')
    parser.add_argument('--eval_freq', default=1, type=int, help="frequency of evaluation")
    parser.add_argument('--max_train_images', default=1000, type=int, help='number of train images')
    parser.add_argument('--max_val_images', default=100, type=int, help='number of validation images')
    parser.add_argument('--max_test_images', default=100, type=int, help='number of test images')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--test', action='store_true',
                        help='Perform testing only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                    help='Enabling distributed evaluation')
    parser.add_argument('--no_dist_eval', action='store_false', dest='dist_eval',
                    help='Disabling distributed evaluation')
    parser.set_defaults(dist_eval=False)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--no_find_unused_params', action='store_false', dest='find_unused_params')
    parser.set_defaults(find_unused_params=True)

    # Wandb logging
    parser.add_argument('--log_wandb', default=False, action='store_true',
                        help='log training and validation metrics to wandb')
    parser.add_argument('--no_log_wandb', dest='log_wandb', default=False, action='store_false',
                        help='log training and validation metrics to wandb')
    parser.add_argument('--wandb_project', default=None, type=str,
                        help='log training and validation metrics to wandb')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='user or team name of wandb')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='run name on wandb')
    parser.add_argument('--wandb_group', default='', type=str)
    parser.add_argument('--log_images_wandb', action='store_true')
    parser.add_argument('--log_images_freq', default=5, type=int,
                        help="Frequency of image logging (in epochs)")
    parser.add_argument('--show_user_warnings', default=False, action='store_true')


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    
    return parser

sys.argv = ['--config-file','my_config']
args = get_args().parse_args("")

args.finetune='/nfs/turbo/coe-hunseok/sshoouri/pascal_finetune/main_seg/step1_sara_m/checkpoint-best.pth'
args.input_size=512
args.data_path='/nfs/turbo/coe-hunseok/sshoouri/Video_nyu_one_time_interval/train'
args.eval_data_path='/nfs/turbo/coe-hunseok/sshoouri/Video_nyu_one_time_interval/test'
args.num_classes=21
args.dataset_name='nyu'
args.dist_eval=True
args.seg_reduce_zero_label=True
args.eval_freq=5
args.find_unused_params=False
args.batch_size=25

args.lr=0.0001
args.weight_decay=1e-6


args.warmup_epochs=1
args.wandb_project='multimae-finetune-semseg'
args.output_dir='/nfs/turbo/coe-hunseok/sshoouri/pascal_finetune/normal/step1_sara_m_one'
if os.path.exists(args.output_dir)==False:
     os.mkdir(args.output_dir)
    
args.dist_on_itp=True

utils.init_distributed_mode(args)
device = torch.device(args.device)
import os
import pickle
import shutil
import tempfile

import torch
import torch.distributed as dist

seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)

cudnn.benchmark = True
#args.in_domains = args.in_domains.split('-')
#args.in_domains = ['rgb' ,'rgb_one_after' ,'rgb_one_before']
if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

num_tasks = utils.get_world_size()
global_rank = utils.get_rank()

if global_rank == 0 and args.log_wandb:
        log_writer = utils.WandbLogger(args)
else:
        log_writer = None

args.in_domains =['rgb']
args.out_domains = ['normal']
args.all_domains = list(set(args.in_domains) | set(args.out_domains))
args.use_mask_valid=False
if args.use_mask_valid:
        args.all_domains.append('mask_valid')
if 'rgb' not in args.all_domains:
        args.all_domains.append('rgb')

args.decoder_main_tasks = args.decoder_main_tasks.split('-')
for task in args.decoder_main_tasks:
        assert task in args.in_domains, f'Readout task {task} must be in in_domains.'

additional_targets = {domain: DOMAIN_CONF[domain]['aug_type'] for domain in args.all_domains}
num_tasks = utils.get_world_size()
global_rank = utils.get_rank()

from utils.pascal_context import PASCALContext
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil
import pdb
import torchvision
from pascal_utils import transforms
train_transforms = torchvision.transforms.Compose([ # from ATRC
            transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            transforms.RandomCrop(size=(512, 512), cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size= (512, 512)),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),])
# Testing 
valid_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=(512, 512)),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),])

from utils.mypath import db_paths, PROJECT_ROOT_DIR
train_dataset = PASCALContext(db_paths['PASCALContext'], download=False, split=['train'], transform=train_transforms, retname=True,
                                          do_semseg=False,
                                          do_edge=False,
                                          do_normals=True,
                                          do_sal=False,
                                          do_human_parts=False,
                                          overfit=False)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank,shuffle=True,drop_last=True)
trainloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,
                             num_workers=1, collate_fn=collate_mil, pin_memory=True, sampler=train_sampler)
test_dataset=PASCALContext(db_paths['PASCALContext'], download=False, split=['val'], transform=valid_transforms, retname=True,
                                          do_semseg=False,
                                          do_edge=False,
                                          do_normals=True,
                                          do_sal=False,
                                          do_human_parts=False,
                                          overfit=False)
if len(test_dataset) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
else:
                sampler_val = torch.utils.data.SequentialSampler(test_dataset)

testloader=DataLoader(test_dataset, sampler=sampler_val, batch_size=args.batch_size, drop_last=False,
                            num_workers=1, pin_memory=True, collate_fn=collate_mil)


args.learnable_pos_emb=False
args.output_adapter='convnext'
args.decoder_interpolate_mode='bilinear'
args.decoder_main_task='rgb'
input_adapters = {
        domain: DOMAIN_CONF[domain]['input_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=args.patch_size,
            image_size=args.input_size,
            learnable_pos_emb=args.learnable_pos_emb,
        )
        for domain in args.in_domains
}

if args.model != 'multivit_base' and args.output_adapter == 'dpt':
        raise NotImplementedError('Unsupported backbone: DPT head is fixed for ViT-B.')

adapters_dict = {
        'segmenter': partial(SegmenterMaskTransformerAdapter, depth=args.decoder_depth, drop_path_rate=args.drop_path_decoder),
        'convnext': partial(ConvNeXtAdapter,embed_dim=args.decoder_dim, preds_per_patch=args.decoder_preds_per_patch, depth=args.decoder_depth,
                            interpolate_mode=args.decoder_interpolate_mode, main_tasks=['rgb']),
        'dpt': partial(DPTOutputAdapter, stride_level=1, main_tasks=['rgb'], head_type='regression'),
}
output_adapters = {
        domain: adapters_dict[args.output_adapter](
            num_classes=DOMAIN_CONF[domain]['channels'],
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size=args.patch_size,
            main_tasks=args.decoder_main_tasks
        )
        for domain in args.out_domains
}
model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        drop_path_rate=args.drop_path_encoder,
)


if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu')
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        
        for k in list(checkpoint_model.keys()):
            if "output_adapters" in k:
                del checkpoint_model[k]

        
        # Interpolate position embedding
        interpolate_pos_embed_multimae(model, checkpoint_model)

        # Load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
model.to(device)


model.zero_grad()
import math
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iterations, gamma=0.9, min_lr=0., last_epoch=-1):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # slight abuse: last_epoch refers to last iteration
        factor = (1 - self.last_epoch /
                  float(self.max_iterations)) ** self.gamma
        return [(base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]



def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    #return LambdaLR(optimizer, lr_lambda, last_epoch)
    return PolynomialLR(optimizer, 40000, gamma=0.9, min_lr=0)
def concrete_stretched(alpha, l=0., r = 1.):
    u = torch.zeros_like(alpha).uniform_().clamp_(0.0001, 0.9999)
    s = (torch.sigmoid(u.log() - (1-u).log() + alpha)).detach()
    u = s*(r-l) + l
    t = u.clamp(0, 1000)
    z = t.clamp(-1000, 1)
    dz_dt = (t < 1).float().to(alpha.device).detach()
    dt_du = (u > 0).float().to(alpha.device).detach()
    du_ds = r - l
    ds_dalpha = (s*(1-s)).detach()
    dz_dalpha = dz_dt*dt_du*du_ds*ds_dalpha
    return z.detach(), dz_dalpha.detach()

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np

class L1Loss(nn.Module):
    """
    from ATRC
    L1 loss with ignore regions.
    normalize: normalization for surface normals
    """
    def __init__(self, normalize=False, ignore_index=255):
        super().__init__()
        self.normalize = normalize
        self.ignore_index = ignore_index

    def forward(self, out, label, reduction='mean'):

        if self.normalize:
            out = nn.functional.normalize(out, p=2, dim=1)

        mask = (label != self.ignore_index).all(dim=1, keepdim=True)
        n_valid = torch.sum(mask).item()
        masked_out = torch.masked_select(out, mask)
        masked_label = torch.masked_select(label, mask)
        if reduction == 'mean':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='sum') / max(n_valid, 1)
        elif reduction == 'sum':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='sum')
        elif reduction == 'none':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='none')
        
        
        
import json
from torch import optim as optim

args.alpha_init=5

depth_params = {}
finetune_params = []
alpha_params = []

parameter_group_names = {}
parameter_group_vars = {}
skip_weight_decay_list = model.no_weight_decay()

total_batch_size = args.batch_size * utils.get_world_size()
num_training_steps_per_epoch = len(trainloader) // total_batch_size


model_without_ddp = model
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model = %s" % str(model_without_ddp))
print('number of params: {} M'.format(n_parameters / 1e6))

num_layers = model_without_ddp.get_num_layers()

no_lr_scale_list = []

assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))

def get_num_layer_for_vit(var_name, num_max_layer=14):
    #print(var_name[7:len(var_name)])
    var_name=var_name[7:len(var_name)]
    
    #print(var_name)
    if var_name in ("cls_token", "mask_token", "pos_embed", "global_tokens"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("input_adapters"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks") or var_name.startswith("encoder"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1

if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))


print("Skip weight decay list: ", skip_weight_decay_list)
args.distributed=True
if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

get_num_layer=assigner.get_layer_id if assigner is not None else None
get_layer_scale=assigner.get_scale if assigner is not None else None
for name,param in model.named_parameters():
       # print(p)
       
        p0 = torch.zeros_like(param.data).copy_(param) #original BERT
        p1 = torch.zeros_like(param.data) #params to be fine-tuned
        p1.requires_grad = True

        p1.grad = torch.zeros_like(param.data)
        alpha = torch.zeros_like(param.data).to(device) + args.alpha_init
        alpha.requires_grad = True
        alpha.grad = torch.zeros_like(param.data)
        name1 = name
        depth_params[name1]= [p0, p1, alpha]
        finetune_params.append(depth_params[name1][1])
        alpha_params.append(depth_params[name1][2])
        
        
        
        
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_weight_decay_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        elif args.decoder_decay is not None and (name.startswith("decoder.") or name in decoder_list):
            group_name = "decoder_decay"
            this_weight_decay = args.decoder_decay
        else:
            group_name = "decay"
            this_weight_decay = args.weight_decay

        # Assign layer ID for LR scaling
        skip_scale = False
        if get_num_layer is not None:
            layer_id = get_num_layer_for_vit(name)
            print(name)
            print(layer_id)
            group_name = "layer_%d_%s" % (layer_id, group_name)
            if name in no_lr_scale_list:
                skip_scale = True
                group_name = f'{group_name}_no_lr_scale'
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None and not skip_scale:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(depth_params[name1][1])
        parameter_group_names[group_name]["params"].append(name)

args.per_params_alpha=1        
if args.per_params_alpha == 1:
        per_params_alpha = {}
        for n, p in model.named_parameters(): 
            alpha = torch.zeros((1)).to(device) + args.alpha_init
            alpha.requires_grad=True
            alpha.grad = torch.zeros_like(alpha)
            
            name = n
            per_params_alpha[name] = alpha
            alpha_params.append(alpha)

            
print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
#model_device = list(model.named_parameters())[0][1].device
args.adam_epsilon=1e-8

no_decay=["global_tokens", "input_adapters.rgb.pos_emb","bias" ,"LayerNorm.weight"]

optimizer_grouped_parameters = [
        {
            "params": [p[1] for n, p in depth_params.items() if not any(nd in n for nd in no_decay) and p[1].requires_grad is True],
            "weight_decay": args.weight_decay,
        },
        {"params": [p[1] for n, p in depth_params.items() if any(nd in n for nd in no_decay) and p[1].requires_grad is True], "weight_decay": 0.0},
]

optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.opt_eps)

t_total=args.epochs*len(trainloader) 

args.warmup_epoch=13

scheduler = PolynomialLR(optimizer, 120000, gamma=0.9, min_lr=0)

print("Use step level LR & WD scheduler!")


args.adam_epsilon=1e-8
alpha_optimizer = optim.AdamW(alpha_params, lr = 0.1, eps=args.adam_epsilon)

args.warmup_steps=args.warmup_epoch*len(trainloader) 
scheduler2 = PolynomialLR(alpha_optimizer , 120000, gamma=0.9, min_lr=0)


@torch.no_grad()
def evaluate(model, tasks_loss_fn, data_loader, device, epoch, in_domains,
             log_images=False, mode='val', return_all_layers=False, standardize_depth=True):
    # Switch to evaluation mode
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if mode == 'val':
        header = '(Eval) Epoch: [{}]'.format(epoch)
    elif mode == 'test':
        header = '(Test) Epoch: [{}]'.format(epoch)
    else:
        raise ValueError(f'Invalid eval mode {mode}')
    print_freq = 20

    pred_images = None
    gt_images = None
    performance_meter=NormalsMeter()
    for x in metric_logger.log_every(data_loader, print_freq, header):
       

        
        
        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items() 
             if task not in 'meta' 
        }
        
        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in 'image'
        }

        
        

        # Mask invalid input values
        for task in input_dict:
            if task in ['rgb']:
                continue
            channels = input_dict[task].shape[1]
            #input_dict[task][~tasks_dict['mask_valid'].repeat_interleave(repeats=channels, dim=1)] = 0.0

        # Forward + backward
        with torch.cuda.amp.autocast(enabled=False):
            preds = model(input_dict['image'], return_all_layers=return_all_layers)
            task_losses = {
                task: tasks_loss_fn[task](preds[task].float(), tasks_dict['normals'])
                for task in preds
            }
            loss = sum(task_losses.values())
                
        

        loss_value = loss.item()
        task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}
        #metrics = normal_metrics(preds['normal'], tasks_dict['normals'], mask_valid=None)

        #metric_logger.update(**metrics)
        
        output = preds['normal'].permute(0, 2, 3, 1)
        output = (F.normalize(output, p = 2, dim = 3) + 1.0) * 255 / 2.0
        targets = tasks_dict['normals'] 
        performance_meter.update(output ,targets)

        if log_images and pred_images is None and utils.is_main_process():
            # Just log images of first batch
            pred_images = {task: v.detach().cpu().float() for task, v in preds.items()}
            gt_images = {task: v.detach().cpu().float() for task, v in input_dict.items()}
            gt_images.update({task: v.detach().cpu().float() for task, v in tasks_dict.items() if task not in gt_images})

        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)

    # Do before metrics so that void is not replaced
    if log_images and utils.is_main_process():
        prefix = 'val/img' if mode == 'val' else 'test/img'
        log_taskonomy_wandb(pred_images, gt_images, prefix=prefix, image_count=8)

    # gather the stats from all processes
    res=performance_meter.get_score(verbose = True)
    metric_logger.update(mean=res['mean'])
    metric_logger.synchronize_between_processes()

    print(f'* Loss {metric_logger.loss.global_avg:.3f}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



tasks_loss_fn = {
            'normal': L1Loss(normalize=False, ignore_index=255)
}


print(f"Start training for {args.epochs} epochs")
start_time = time.time()
min_val_loss = np.inf

get_num_layer=assigner.get_layer_id 

args.act_diff_l0_pen_per_layer="2e-10,"*4+"2e-9,"*7+"1e-8"
import copy
args.save_ckpt_freq=50000
args.eval_freq=5
standardize_depth =True
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
return_all_layers = args.output_adapter in ['dpt']
total_layers=13
args.sparsity_pen=5e-08
args.sparsity_penalty_per_layer=None
if args.sparsity_penalty_per_layer is None:
        sparsity_pen = [args.sparsity_pen] * total_layers  # NB(anon)
else:
        sparsity_pen = args.sparsity_penalty_per_layer
        assert len(sparsity_pen) == total_layers,  "invalid sparsity penalty per layer: # of layers mismatch"
    
start_time = time.time()
min_val_loss = np.inf

args.eval_freq=4
args.distributed=False

args.concrete_lower=-1.5
args.concrete_upper=1.5


args.per_layer_alpha=0
args.max_grad_norm=10
max_norm=args.clip_grad

import numpy as np
from tqdm import tqdm, trange

args.gradient_accumulation_steps =1
args.save_ckp=2


for epoch in range(args.start_epoch, args.epochs):
    if args.distributed:
            trainloader.sampler.set_epoch(epoch)
    log_images = args.log_wandb and args.log_images_wandb and (epoch % args.log_images_freq == 0)
    if args.distributed:
            trainloader.sampler.set_epoch(epoch)
    if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    start_steps=epoch * num_training_steps_per_epoch
    loss1=0
    loss2=0
    for step, x in enumerate(metric_logger.log_every(trainloader, print_freq, header)):
        it = start_steps + step  # global training iteration
        

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items() 
             if task not in 'meta' 
        }
        
        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in 'image'
        }
        
        for task in input_dict:
            if task in ['rgb']:
                continue
            channels = input_dict[task].shape[1]

        nonzero_params = 0
        grad_params = {}
        log_ratio = np.log(-args.concrete_lower / args.concrete_upper)
        l0_pen = [0] * total_layers
        l0_pen_sum = 0
        if args.per_params_alpha:
            per_params_z = {}
            per_params_z_grad = {}
        for n, p in model.named_parameters():
                
                if n not in depth_params:
                    print(" n not in depth_params")
                    embed()
                assert(n in depth_params)
                if "output_adapters" in n:
                    #print("decoder")
                    nonzero_params += p.numel()
                    
                    p.data.copy_(depth_params[n][0].data + depth_params[n][1].data)
                else:

                    if args.per_params_alpha == 1:
                        params_z, params_z_grad = concrete_stretched(per_params_alpha[n], args.concrete_lower,
                                args.concrete_upper)
                        per_params_z[n] = params_z
                        per_params_z_grad[n] = params_z_grad

                    z, z_grad = concrete_stretched(depth_params[n][2], args.concrete_lower,
                                                   args.concrete_upper)
         
                    ind = get_num_layer_for_vit(n)
                    l0_pen[ind] += torch.sigmoid(depth_params[n][2] - log_ratio).sum()
                    l0_pen_sum += torch.sigmoid(depth_params[n][2] - log_ratio).sum()
                    
                    if args.per_layer_alpha == 1:
                        z2 = per_layer_z[ind]
                    elif args.per_params_alpha == 1:
                        z2 =  per_params_z[n]
                    else:
                        z2 = 1

                    grad_params[n] = [depth_params[n][1] * z2, z * z2, z_grad, depth_params[n][1] * z]

                    if args.per_params_alpha == 1:
                        l0_pen[ind] += torch.sigmoid(per_params_alpha[n] - log_ratio).sum()
                
                    p.data.copy_(depth_params[n][0].data + (z2*z).data*depth_params[n][1].data)
                    nonzero_params += ((z2*z)>0).float().detach().sum().item()
        
        model.train()
       
        # Forward + backward
        with torch.cuda.amp.autocast(enabled=False):
            # print('=====> input_dict:', {k: v.shape for k, v in input_dict.items()})

            preds = model(input_dict['image'], return_all_layers=return_all_layers)
            task_losses = {
                task: tasks_loss_fn[task](preds[task].float(), tasks_dict['normals'])
                for task in preds
            }
            loss = sum(task_losses.values())
            
        loss_value = loss.item()
        
        
        task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}
        metrics = normal_metrics(preds['normal'], tasks_dict['normals'], mask_valid=None)
 
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
       
        loss.backward()
                
        if (step + 1) % args.gradient_accumulation_steps == 0:
            for n, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    if "output_adapters" in n:
                        depth_params[n][1].grad.copy_(p.grad.data)
                    else:
                        try:
                            depth_params[n][1].grad.copy_(p.grad.data * grad_params[n][1].data)
                        except:
                            embed()
                        depth_params[n][2].grad.copy_(p.grad.data * grad_params[n][0].data *
                                                     grad_params[n][2].data)
                    
                        if args.per_params_alpha == 1:
                            per_params_alpha[n].grad.copy_(torch.sum(p.grad.data * grad_params[n][3].data * 
                                    per_params_z_grad[n].data))
                        if args.per_layer_alpha == 1:
                            per_layer_alpha.grad[get_num_layer_for_vit(n)] += torch.sum(p.grad.data * grad_params[n][3].data *
                                    per_layer_z_grad[ind].data)
        sum_l0_pen = 0
        for i in range(total_layers):
                    if l0_pen[i] != 0:
                        sum_l0_pen += (sparsity_pen[i] * l0_pen[i]).sum()
        sum_l0_pen.sum().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            #torch.nn.utils.clip_grad_norm_(list(parameter_group_vars.values()), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(alpha_params, args.max_grad_norm)
        
        optimizer.step()
        alpha_optimizer.step()
        scheduler.step()
        scheduler2.step()  # Update learning rate schedule
        model.zero_grad()
        
        params_norm = [0, 0, 0, 0, 0, 0]
        exp_z = 0
        for n, p in depth_params.items():
                    params_norm[0] += p[2].sum().item()
                    params_norm[1] += p[2].norm().item()**2
                    params_norm[2] += p[2].grad.norm().item()**2
                    params_norm[3] += torch.sigmoid(p[2]).sum().item()
                    params_norm[4] += p[2].numel()
                    # params_norm[5] += (grad_params[n][1] > 0).float().sum().item()
                    if args.per_params_alpha == 1:
                        exp_z += (torch.sigmoid(p[2]).sum() * torch.sigmoid(per_params_alpha[n])).item()
                    else:
                        exp_z += torch.sigmoid(p[2]).sum().item()

                    p[1].grad.zero_()
                    p[2].grad.zero_()

        mean_exp_z = exp_z / params_norm[4]
        if args.per_params_alpha == 1:
                for n,p in per_params_alpha.items():
                    p.grad.zero_()
                    
        
        
        if (step + 1)% 100 == 0:
                    print("outdated average prob: %.4f, new average prob: %.4f, (!)empirical prob: %.4f, alpha_norm: %.4f, alpha_grad_norm: %.8f, alpha_avg: %.4f, l0_pen: %.2f, \n" %
                          (params_norm[3]/params_norm[4], mean_exp_z, nonzero_params/params_norm[4],
                           params_norm[1]**0.5, params_norm[2]**0.5, params_norm[0]/params_norm[4],
                           l0_pen_sum))

        metric_logger.update(**metrics)
        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)
        metric_logger.update(loss_mask=sum_l0_pen.sum().item())
       
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        

        if log_images and pred_images is None and utils.is_main_process():
            # Just log images of first batch
            pred_images = {task: v.detach().cpu().float() for task, v in preds.items()}
            gt_images = {task: v.detach().cpu().float() for task, v in input_dict.items()}
            gt_images.update({task: v.detach().cpu().float() for task, v in tasks_dict.items() if task not in gt_images})

        if log_writer is not None:
            log_writer.update(
                {
                    'loss': loss_value,
                    'loss_mask': sum_l0_pen.sum().item(),
                    'lr': max_lr,
                    'weight_decay': weight_decay_value,
                    'grad_norm': grad_norm,
                }
            )
            log_writer.set_step()
    if log_images and utils.is_main_process():
            prefix = 'train/img'
            log_taskonomy_wandb(pred_images, gt_images, prefix=prefix, image_count=8)

        # gather the stats from all processes
        
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    train_stats={'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if epoch % args.eval_freq == 0:
            log_images = args.log_wandb and args.log_images_wandb and (epoch % args.log_images_freq == 0)
            val_stats = evaluate(model=model, tasks_loss_fn=tasks_loss_fn, data_loader=testloader,
                                 device=device, epoch=epoch, in_domains=args.in_domains, log_images=log_images,
                                 mode='val', return_all_layers=return_all_layers, standardize_depth=args.standardize_depth)
            
            if epoch==0:
                log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                         **{f'val/{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
                
            if val_stats["mean"] < min_val_loss:
                min_val_loss = val_stats["mean"]
                if args.output_dir and args.save_ckpt:
                   
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    ) 

                    info_dict = {"model": model.state_dict(), "depth_params": depth_params} #, "z": z.detach(), "nonzero_params": nonzero_params}
                    torch.save(info_dict, os.path.join(args.output_dir, "checkpoint-last-info.pt"))
                    

                print(f'New best val loss: {min_val_loss:.3f}')

                log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                         **{f'val/{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
    else:
                log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

    if log_writer is not None:
            log_writer.update(log_stats)

    if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

