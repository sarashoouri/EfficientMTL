import argparse
import datetime
import json
import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Iterable
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
import cv2
import glob
import math
import pickle
import shutil
import tempfile
import torchvision
import pdb
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.functional import F
from multimae import multimae
from multimae.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from multimae.output_adapters import ConvNeXtAdapter, DPTOutputAdapter, SegmenterMaskTransformerAdapter
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import create_model
from utils.data_constants import COCO_SEMSEG_NUM_CLASSES, NYU_MEAN, NYU_STD
from utils.datasets_semseg import build_semseg_dataset, simple_transform 
from utils.dataset_regression import build_regression_dataset, build_regression_dataset_normal
from utils.dist import collect_results_cpu
from utils.log_images import log_semseg_wandb
from utils.optim_factory import LayerDecayValueAssigner, create_optimizer
from utils.pos_embed import interpolate_pos_embed_multimae
from utils.semseg_metrics import mean_iou
from utils.pascal_context import PASCALContext
from utils.custom_collate import collate_mil
from pascal_utils import transforms
from utils.mypath import db_paths, PROJECT_ROOT_DIR

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


args.finetune='./Codes/base_weight/mae-b_dec512d8b_1600e_multivit-c477195b.pth'
args.input_size=512
args.dist_eval=True
args.seg_reduce_zero_label=True
args.eval_freq=5
args.find_unused_params=False
args.batch_size=25
args.lr=0.0001
args.weight_decay=1e-6
args.warmup_epochs=1
args.wandb_project='multimae-finetune-semseg'
args.output_dir='./pascal_finetune/normal/step0'

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
        
#device = torch.device('cuda:0')
model_without_ddp = model
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model = %s" % str(model_without_ddp))
print('number of params: {} M'.format(n_parameters / 1e6))

return_all_layers = args.output_adapter in ['dpt']

tasks_loss_fn = {
            'normal': L1Loss(normalize=False, ignore_index=255)
}
    
num_layers = model.get_num_layers()
total_batch_size = args.batch_size * utils.get_world_size()
num_training_steps_per_epoch = len(trainloader) // total_batch_size

print("LR = %.8f" % args.lr)
print("Batch size = %d" % total_batch_size)
print("Number of training steps = %d" % num_training_steps_per_epoch)
print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

if args.layer_decay < 1.0:
        # idx=0: input adapters, idx>0: transformer layers 
        layer_decay_values = list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
        if not args.scale_input_lr:
            layer_decay_values[0] = 1.0
        assigner = LayerDecayValueAssigner(layer_decay_values)
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

get_num_layer=assigner.get_layer_id if assigner is not None else None
get_layer_scale=assigner.get_scale if assigner is not None else None


optimizer = create_optimizer(args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
loss_scaler = NativeScaler(enabled=False)

print("Use step level LR & WD scheduler!")
lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
)
if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # Specifies if transformer encoder should only return last layer or all layers for DPT
return_all_layers = args.output_adapter in ['dpt']

utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

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

print(f"Start training for {args.epochs} epochs")
start_time = time.time()
min_val_loss = np.inf

get_num_layer=assigner.get_layer_id 

args.act_diff_l0_pen_per_layer="2e-10,"*4+"2e-9,"*7+"1e-8"
import copy
args.save_ckpt_freq=50000
args.eval_freq=5
standardize_depth =True
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

        # Mask invalid input values
        for task in input_dict:
            if task in ['rgb']:
                continue
            channels = input_dict[task].shape[1]

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

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                                parameters=model.parameters(), create_graph=is_second_order)
        # loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(**metrics)
        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)
        # metric_logger.update(loss_scale=loss_scale_value)
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
        #metric_logger.update(grad_norm=grad_norm)

        if log_images and pred_images is None and utils.is_main_process():
            # Just log images of first batch
            pred_images = {task: v.detach().cpu().float() for task, v in preds.items()}
            gt_images = {task: v.detach().cpu().float() for task, v in input_dict.items()}
            gt_images.update({task: v.detach().cpu().float() for task, v in tasks_dict.items() if task not in gt_images})

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
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best")
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

