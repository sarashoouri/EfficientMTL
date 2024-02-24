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
sys.path.append('./Codes/')
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
import pickle
import shutil
import tempfile
import torch.distributed as dist

from utils.pascal_context import PASCALContext

import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil
import pdb
import torchvision
from pascal_utils import transforms

DOMAIN_CONF = {
    'rgb': {
        'channels': 3,
        'stride_level': 1,
        'aug_type': 'image',
        'input_adapter': partial(PatchedInputAdapter, num_channels=3),
    },
    'depth': {
        'channels': 1,
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
    
    
    parser = argparse.ArgumentParser('MultiMAE semantic segmentation fine-tuning script', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # Task parameters
    parser.add_argument('--in_domains', default='rgb', type=str,
                        help='Input domain names, separated by hyphen')
    parser.add_argument('--standardize_depth', action='store_true')
    parser.add_argument('--no_standardize_depth', action='store_false', dest='standardize_depth')
    parser.set_defaults(standardize_depth=True)
    parser.add_argument('--use_mask_valid', action='store_true')
    parser.add_argument('--no_mask_valid', action='store_false', dest='use_mask_valid')
    parser.set_defaults(use_mask_valid=False)
    parser.add_argument('--load_pseudo_depth', action='store_true')
    parser.add_argument('--no_load_pseudo_depth', action='store_false', dest='load_pseudo_depth')
    parser.set_defaults(load_pseudo_depth=False)

    # Model parameters
    parser.add_argument('--model', default='multivit_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--num_global_tokens', default=1, type=int,
                        help='number of global tokens to add to encoder')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='base patch size for image-like modalities')
    parser.add_argument('--input_size', default=512, type=int,
                        help='images input size for backbone')
    parser.add_argument('--drop_path_encoder', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--learnable_pos_emb', action='store_true',
                        help='Makes the positional embedding learnable')
    parser.add_argument('--no_learnable_pos_emb', action='store_false', dest='learnable_pos_emb')
    parser.set_defaults(learnable_pos_emb=False)

    parser.add_argument('--output_adapter', type=str, default='convnext',
                        choices=['segmenter', 'convnext', 'dpt'],
                        help='One of [segmenter,  convnext, dpt] (default: convnext)')
    parser.add_argument('--decoder_dim', default=3072, type=int,
                        help='Token dimension for the decoder layers, for convnext and segmenter adapters')
    parser.add_argument('--decoder_depth', default=2, type=int,
                        help='Depth of decoder (for convnext and segmenter adapters')
    parser.add_argument('--drop_path_decoder', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--decoder_preds_per_patch', type=int, default=16,
                        help='Predictions per patch for convnext adapter')
    parser.add_argument('--decoder_interpolate_mode', type=str, default='bilinear',
                        choices=['bilinear', 'nearest'], help='for convnext adapter')
    parser.add_argument('--decoder_main_tasks', type=str, default='rgb',
                        help='for convnext adapter, separate tasks with a hyphen')

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
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")
    parser.add_argument('--decoder_decay', type=float, default=None,
                        help='decoder weight decay')
    parser.add_argument('--no_lr_scale_list', type=str, default='',
                        help='Weights that should not be affected by layer decay rate, separated by hyphen.')

    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=0.0, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (0.0)')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA')

    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--aug_name', type=str, default='simple',
                        choices=['simple'],
                        help='One of [simple] (default: simple)')

    # Finetuning parameters
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--num_classes', default=150, type=str, help='number of semantic classes')
    parser.add_argument('--dataset_name', default='ade20k', type=str, help='dataset name for plotting')
    parser.add_argument('--data_path', default=data_constants.ADE_TRAIN_PATH, type=str, help='dataset path')
    parser.add_argument('--eval_data_path', default=data_constants.ADE_VAL_PATH, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--test_data_path', default=None, type=str,
                        help='dataset path for testing')
    parser.add_argument('--max_val_images', default=None, type=int,
                        help='maximum number of validation images. (default: None)')
    parser.add_argument('--eval_freq', default=1, type=int, help="frequency of evaluation")
    parser.add_argument('--seg_reduce_zero_label', action='store_true',
                        help='set label 0 to ignore, reduce all other labels by 1')
    parser.add_argument('--seg_use_void_label', action='store_true', help='label border as void instead of ignore')

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

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--no_fp16', action='store_false', dest='fp16')
    parser.set_defaults(fp16=True)

    # Wandb logging
    parser.add_argument('--log_wandb', default=False, action='store_true',
                        help='log training and validation metrics to wandb')
    parser.add_argument('--wandb_project', default=None, type=str,
                        help='log training and validation metrics to wandb')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='user or team name of wandb')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='run name on wandb')
    parser.add_argument('--log_images_wandb', action='store_true')
    parser.add_argument('--log_images_freq', default=5, type=int,
                        help="Frequency of image logging (in epochs)")
    parser.add_argument('--show_user_warnings', default=False, action='store_true')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    #
   

    return parser

sys.argv = ['--config-file','my_config']
args = get_args().parse_args("")


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
                    loss_scaler, max_norm: float = 0, log_writer=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, in_domains=None, fp16=True,
                    return_all_layers=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    for step, (x, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
        }

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        if 'pseudo_semseg' in tasks_dict and 'semseg' in in_domains:
            psemseg  = tasks_dict['pseudo_semseg']
            psemseg[psemseg > COCO_SEMSEG_NUM_CLASSES - 1] = COCO_SEMSEG_NUM_CLASSES
            input_dict['semseg'] = psemseg
        
        # Forward + backward
        with torch.cuda.amp.autocast(enabled=fp16):
            preds = model(input_dict, return_all_layers=return_all_layers)
            seg_pred, seg_gt = preds['semseg'], tasks_dict['semseg']
            loss = criterion(seg_pred, seg_gt)

        loss_value = loss.item()

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        if fp16:
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        # Metrics and logging
        metric_logger.update(loss=loss_value)
        if fp16:
            metric_logger.update(loss_scale=loss_scale_value)
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
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(
                {
                    'loss': loss_value,
                    'lr': max_lr,
                    'weight_decay': weight_decay_value,
                    'grad_norm': grad_norm,
                }
            )
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, epoch, in_domains, num_classes, dataset_name,
             log_images=False, mode='val', fp16=True, return_all_layers=False):
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

    seg_preds = []
    seg_gts = []

    rgb_gts = None
    seg_preds_with_void = None
    if log_images:
        rgb_gts = []
        seg_preds_with_void = []
        depth_gts = []

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

        if 'pseudo_semseg' in tasks_dict and 'semseg' in in_domains:
            psemseg  = tasks_dict['pseudo_semseg']
            psemseg[psemseg == 254] = COCO_SEMSEG_NUM_CLASSES
            input_dict['semseg'] = psemseg

        # Forward + backward
        with torch.cuda.amp.autocast(enabled=fp16):
            preds = model(input_dict['image'].float(), return_all_layers=return_all_layers)
            seg_pred, seg_gt = preds['semseg'], tasks_dict['semseg'][:,0,:,:].long()
            loss = criterion(seg_pred, seg_gt)

        loss_value = loss.item()
        # If there is void, exclude it from the preds and take second highest class
        seg_pred_argmax = seg_pred[:, :num_classes].argmax(dim=1)
        seg_preds.extend(list(seg_pred_argmax.cpu().numpy()))
        seg_gts.extend(list(seg_gt.cpu().numpy()))

        if log_images:
            rgb_gts.extend(tasks_dict['rgb'].cpu().unbind(0))
            seg_preds_with_void.extend(list(seg_pred.argmax(dim=1).cpu().numpy()))
            if 'depth' in tasks_dict:
                depth_gts.extend(tasks_dict['depth'].cpu().unbind(0))

        metric_logger.update(loss=loss_value)

    # Do before metrics so that void is not replaced
    if log_images and utils.is_main_process():
        prefix = 'val/img' if mode == 'val' else 'test/img'
        log_semseg_wandb(rgb_gts, seg_preds_with_void, seg_gts, depth_gts=depth_gts, dataset_name=dataset_name, prefix=prefix)
    print("computing in score")
    print(device)
    scores = compute_metrics_distributed(seg_preds, seg_gts, size=len(data_loader.dataset), num_classes=num_classes,
                                         device=device, ignore_index=utils.SEG_IGNORE_INDEX )

    for k, v in scores.items():
        metric_logger.update(**{f"{k}": v})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(f'* mIoU {metric_logger.mean_iou.global_avg:.3f} aAcc {metric_logger.pixel_accuracy.global_avg:.3f} '
          f'Acc {metric_logger.mean_accuracy.global_avg:.3f} Loss {metric_logger.loss.global_avg:.3f}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_metrics_distributed(seg_preds, seg_gts, size, num_classes, device, ignore_index=utils.SEG_IGNORE_INDEX, dist_on='cpu'):

    # Replace void by ignore in gt (void is never counted in mIoU)
    for seg_gt in seg_gts:
        # Void label is equal to num_classes
        seg_gt[seg_gt == num_classes] = ignore_index

    # Collect metrics from all devices

    if dist_on == 'cpu':
        all_seg_preds = collect_results_cpu(seg_preds, size, tmpdir=None)
        all_seg_gts = collect_results_cpu(seg_gts, size, tmpdir=None)
    elif dist_on == 'gpu':
        world_size = utils.get_world_size()
        all_seg_preds = [None for _ in range(world_size)]
        all_seg_gts = [None for _ in range(world_size)]
        # gather all result part
        dist.all_gather_object(all_seg_preds, seg_preds)
        dist.all_gather_object(all_seg_gts, seg_gts)

    ret_metrics_mean = torch.zeros(3, dtype=float, device=device)

    if utils.is_main_process():
        ordered_seg_preds = [result for result_part in all_seg_preds for result in result_part]
        ordered_seg_gts = [result for result_part in all_seg_gts for result in result_part]

        ret_metrics = mean_iou(results=ordered_seg_preds,
                               gt_seg_maps=ordered_seg_gts,
                               num_classes=num_classes,
                               ignore_index=ignore_index)

        ret_metrics_mean = torch.tensor(
            [
                np.round(np.nanmean(ret_metric.astype(float)) * 100, 2)
                for ret_metric in ret_metrics
            ],
            dtype=float,
            device=device,
        )
        # cat_iou = ret_metrics[2]

    # broadcast metrics from 0 to all nodes
    dist.broadcast(ret_metrics_mean, 0)
    pix_acc, mean_acc, miou = ret_metrics_mean
    ret = dict(pixel_accuracy=pix_acc, mean_accuracy=mean_acc, mean_iou=miou)
    del all_seg_gts
    del all_seg_preds
    return ret


args.finetune='/nfs/turbo/coe-hunseok/sshoouri/Codes/base_weight/mae-b_dec512d8b_1600e_multivit-c477195b.pth'
args.input_size=512
args.num_classes=21
args.dist_eval=True
args.seg_reduce_zero_label=True
args.eval_freq=5
args.find_unused_params=False
args.batch_size=4
args.epochs = 400
args.lr=2e-5
args.weight_decay=1e-6

args.warmup_epochs=1
args.wandb_project='multimae-finetune-semseg'
args.output_dir='./pascal_finetune/main_seg/step1'
if os.path.exists(args.output_dir)==False:
          os.mkdir(args.output_dir)

args.dist_on_itp=True
utils.init_distributed_mode(args)
device = torch.device(args.device)
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)

cudnn.benchmark = True
#args.in_domains = args.in_domains.split('-')
#args.in_domains = ['rgb' ,'rgb_one_after' ,'rgb_one_before']
args.in_domains =['rgb']
args.out_domains = ['semseg']
args.all_domains = list(set(args.in_domains) | set(args.out_domains))
if args.use_mask_valid:
        args.all_domains.append('mask_valid')
if 'rgb' not in args.all_domains:
        args.all_domains.append('rgb')
args.num_classes_with_void = args.num_classes + 1 if args.seg_use_void_label else args.num_classes
additional_targets = {domain: DOMAIN_CONF[domain]['aug_type'] for domain in args.all_domains}
args.in_domains = ['rgb']
args.out_domains = ['semseg']
args.all_domains = list(set(args.in_domains) | set(args.out_domains))
additional_targets={}
additional_targets = {domain: DOMAIN_CONF[domain]['aug_type'] for domain in args.all_domains}
num_tasks = utils.get_world_size()
global_rank = utils.get_rank()
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
                                          do_semseg=True,
                                          do_edge=False,
                                          do_normals=False,
                                          do_sal=False,
                                          do_human_parts=False,
                                          overfit=False)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank,shuffle=True,drop_last=True)
trainloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,
                             num_workers=1, collate_fn=collate_mil, pin_memory=True, sampler=train_sampler)

test_dataset=PASCALContext(db_paths['PASCALContext'], download=False, split=['val'], transform=valid_transforms, retname=True,
                                          do_semseg=True,
                                          do_edge=False,
                                          do_normals=False,
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


print("Sampler_train = %s" % str(train_sampler))
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
        'convnext': partial(ConvNeXtAdapter, preds_per_patch=args.decoder_preds_per_patch, depth=args.decoder_depth,
                            interpolate_mode=args.decoder_interpolate_mode, main_tasks=args.decoder_main_tasks.split('-')),
        'dpt': partial(DPTOutputAdapter, stride_level=1, main_tasks=args.decoder_main_tasks.split('-'), head_type='semseg'),
}

output_adapters = {
        'semseg': adapters_dict[args.output_adapter](
            num_classes=args.num_classes_with_void,
            embed_dim=args.decoder_dim, patch_size=args.patch_size,
        ),
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

        # Interpolate position embedding
        interpolate_pos_embed_multimae(model, checkpoint_model)

        # Load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

model.to(device)
print(device)
model_without_ddp = model
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model = %s" % str(model_without_ddp))
print('number of params: {} M'.format(n_parameters / 1e6))

total_batch_size = args.batch_size * utils.get_world_size()
num_training_steps_per_epoch = len(trainloader) // total_batch_size

print("LR = %.8f" % args.lr)
print("Batch size = %d" % total_batch_size)
print("Number of training steps = %d" % num_training_steps_per_epoch)
print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))
num_layers = model_without_ddp.get_num_layers()
if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
else:
        assigner = None

if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

skip_weight_decay_list = model.no_weight_decay()
print("Skip weight decay list: ", skip_weight_decay_list)
args.distributed=True
if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
optimizer = create_optimizer(args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
loss_scaler = NativeScaler(enabled=args.fp16)

print("Use step level LR & WD scheduler!")
lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
)
if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
wd_schedule_values = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

criterion = torch.nn.CrossEntropyLoss(ignore_index=utils.SEG_IGNORE_INDEX)

print("criterion = %s" % str(criterion))

    # Specifies if transformer encoder should only return last layer or all layers for DPT
return_all_layers = args.output_adapter in ['dpt']
args.eval=False
utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
args.eval=0
if args.eval:
        val_stats = evaluate(model=model, criterion=criterion, data_loader=data_loader_val,
                             device=device, epoch=-1, in_domains=args.in_domains,
                             num_classes=args.num_classes, dataset_name=args.dataset_name, mode='val',
                             fp16=args.fp16, return_all_layers=return_all_layers)
        print(f"Performance of the network on the {len(dataset_val)} validation images")
        miou, a_acc, acc, loss = val_stats['mean_iou'], val_stats['pixel_accuracy'], val_stats['mean_accuracy'], val_stats['loss']
        print(f'* mIoU {miou:.3f} aAcc {a_acc:.3f} Acc {acc:.3f} Loss {loss:.3f}')
        exit(0)
print(f"Start training for {args.epochs} epochs")
start_time = time.time()
max_miou = 0.0


total_layers=13
get_num_layer=assigner.get_layer_id

args.act_diff_l0_pen_per_layer="2e-10,"*4+"2e-9,"*7+"1e-8"
import copy
args.save_ckpt_freq=5000
args.eval_freq=5
print(model.device)

if global_rank == 0 and args.log_wandb:
        log_writer = utils.WandbLogger(args)
else:
        log_writer = None
for epoch in range(args.start_epoch, args.epochs):
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

        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
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

        with torch.cuda.amp.autocast(enabled=args.fp16):
            preds = model(input_dict['image'].float(), return_all_layers=return_all_layers)
            seg_pred, seg_gt = preds['semseg'], tasks_dict['semseg'][:,0,:,:].long()
            loss = criterion(seg_pred, seg_gt)
        loss_value = loss.item()
        
        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                                parameters=model.parameters(), create_graph=is_second_order)
        if args.fp16:
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        # Metrics and logging
        metric_logger.update(loss=loss.item())
       
        if args.fp16:
            metric_logger.update(loss_scale=loss_scale_value)
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
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(
                {
                    'loss_whole': loss,
                    'lr': max_lr,
                    'weight_decay': weight_decay_value,
                    'grad_norm': grad_norm,
                }
            )
            log_writer.set_step()
        
    train_stats={'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

    if testloader is not None and (epoch % args.eval_freq == 0 or epoch == args.epochs - 1):
            log_images = args.log_wandb and args.log_images_wandb and (epoch % args.log_images_freq == 0)
            val_stats = evaluate(model=model, criterion=criterion, data_loader=testloader,
                                 device=device, epoch=epoch, in_domains=args.in_domains,
                                 num_classes=args.num_classes, log_images=log_images, 
                                 dataset_name=args.dataset_name, mode='val', fp16=args.fp16,
                                 return_all_layers=return_all_layers)
            if max_miou < val_stats["mean_iou"]:
                max_miou = val_stats["mean_iou"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best")
                print(f'best Max mIoU: {max_miou:.3f}')

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

    
