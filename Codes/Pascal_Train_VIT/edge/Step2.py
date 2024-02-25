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
import math
from typing import Callable, Iterable, Tuple
from torch.optim import Optimizer
from torch.optim.
class BalancedBinaryCrossEntropyLoss(nn.Module):
    """
    Balanced binary cross entropy loss with ignore regions.
    """
    def __init__(self, pos_weight=None, ignore_index=255):
        super().__init__()
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

    def forward(self, output, label, reduction='mean'):

        mask = (label != self.ignore_index)
        masked_label = torch.masked_select(label, mask)
        masked_output = torch.masked_select(output, mask)

        # weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            w = num_labels_neg / num_total
            if w == 1.0:
                return 0
        else:
            w = torch.as_tensor(self.pos_weight, device=output.device)
        factor = 1. / (1 - w)

        loss = nn.functional.binary_cross_entropy_with_logits(
            masked_output,
            masked_label,
            pos_weight=w*factor,
            reduction=reduction)
        loss /= factor
        return loss
    
DOMAIN_CONF = {
    'rgb': {
        'channels': 3,
        'stride_level': 1,
        'aug_type': 'image',
        'input_adapter': partial(PatchedInputAdapter, num_channels=3),
    },
    'edge': {
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
    parser.add_argument('--decoder_dim', default=3072, type=int,
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

args.input_size=512
args.num_classes=21
args.dist_eval=True
args.seg_reduce_zero_label=True
args.eval_freq=5
args.find_unused_params=False
args.batch_size=6
args.lr=2e-5
args.weight_decay=1e-6
args.warmup_epochs=1
args.wandb_project='multimae-finetune-semseg'
args.output_dir='./pascal_finetune/edge/step2_sara_m_one'
if os.path.exists(args.output_dir)==False:
     os.mkdir(args.output_dir)
args.dist_on_itp=True

args.dist_on_itp=True
local_rank = int(os.environ.get('OMPI_COMM_WORLD_RANK') or 0)
world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE') or 1)
gpu=local_rank
torch.cuda.set_device(gpu)
dist_backend = 'nccl'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '42681'

dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])

torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url,
                                         world_size=world_size, rank=local_rank)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
torch.distributed.barrier()
setup_for_distributed(local_rank == 0)
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
args.out_domains = ['edge']
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


train_dataset = PASCALContext(db_paths['PASCALContext'], download=False, split=['train'], transform=train_transforms, retname=True,
                                          do_semseg=False,
                                          do_edge=True,
                                          do_normals=False,
                                          do_sal=False,
                                          do_human_parts=False,
                                          overfit=False)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank,shuffle=True,drop_last=True)
trainloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,
                             num_workers=1, collate_fn=collate_mil, pin_memory=True, sampler=train_sampler)
test_dataset=PASCALContext(db_paths['PASCALContext'], download=False, split=['val'], transform=valid_transforms, retname=True,
                                          do_semseg=False,
                                          do_edge=True,
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
            num_classes=1,
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

model.to(device)
model.zero_grad()
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

model_without_ddp = model
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model = %s" % str(model_without_ddp))
print('number of params: {} M'.format(n_parameters / 1e6))

return_all_layers = args.output_adapter in ['dpt']
model_without_ddp = model
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model = %s" % str(model_without_ddp))
print('number of params: {} M'.format(n_parameters / 1e6))

return_all_layers = args.output_adapter in ['dpt']
class SaliencyMeter(object):
    def __init__(self, ignore_index=255, threshold_step=None, beta_squared=1):
        self.ignore_index = ignore_index
        self.beta_squared = beta_squared
        self.thresholds = torch.arange(threshold_step, 1, threshold_step)
        self.true_positives = torch.zeros(len(self.thresholds))
        self.predicted_positives = torch.zeros(len(self.thresholds))
        self.actual_positives = torch.zeros(len(self.thresholds))

    @torch.no_grad()
    def update(self, preds, target):
        """
        Update state with predictions and targets.
        Args:
            preds: Predictions from model [B, H, W]
            target: Ground truth values
        """
        preds = preds.float() / 255.

        if target.shape[1] == 1:
            target = target.squeeze(1)

        assert preds.shape == target.shape

        if len(preds.shape) == len(target.shape) + 1:
            assert preds.shape[1] == 2
            # two class probabilites
            preds = nn.functional.softmax(preds, dim=1)[:, 1, :, :]
        else:
            # squash logits into probabilities
            preds = torch.sigmoid(preds)

        if not len(preds.shape) == len(target.shape):
            raise ValueError("preds and target must have same number of dimensions, or preds one more")

        valid_mask = (target != self.ignore_index)

        for idx, thresh in enumerate(self.thresholds):
            # threshold probablities
            f_preds = (preds >= thresh).long()
            f_target = target.long()

            f_preds = torch.masked_select(f_preds, valid_mask)
            f_target = torch.masked_select(f_target, valid_mask)

            self.true_positives[idx] += torch.sum(f_preds * f_target).cpu()
            self.predicted_positives[idx] += torch.sum(f_preds).cpu()
            self.actual_positives[idx] += torch.sum(f_target).cpu()


    def get_score(self, verbose=False):    
        """
        Computes F-scores over state and returns the max.
        """
        precision = self.true_positives.float() / self.predicted_positives
        recall = self.true_positives.float() / self.actual_positives

        num = (1 + self.beta_squared) * precision * recall
        denom = self.beta_squared * precision + recall

        # For the rest we need to take care of instances where the denom can be 0
        # for some classes which will produce nans for that class
        fscore = num / denom
        fscore[fscore != fscore] = 0

        eval_result = {'maxF': fscore.max().item()}
        return eval_result
    
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
    performance_meter=SaliencyMeter(ignore_index=255, threshold_step=0.05, beta_squared=0.3)
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
                task: tasks_loss_fn[task](preds[task].float(), tasks_dict['edge'])
                for task in preds
            }
            loss = 50*sum(task_losses.values())
                
        

        loss_value = loss.item()
        task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}
        #metrics = normal_metrics(preds['normal'], tasks_dict['normals'], mask_valid=None)

        #metric_logger.update(**metrics)
        
        output = preds['edge'].permute(0, 2, 3, 1)
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)))
        targets = tasks_dict['edge'] 
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
    metric_logger.update(maxF=res['maxF'])
    metric_logger.synchronize_between_processes()

    print(f'* Loss {metric_logger.loss.global_avg:.3f}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
        model_without_ddp = model.module

get_num_layer=assigner.get_layer_id if assigner is not None else None
get_layer_scale=assigner.get_scale if assigner is not None else None
saved=torch.load('./pascal_finetune/edge/step1_sara_m_one/checkpoint-last-info.pt',map_location=model.device)
nonzero_mask = {}
model_keys = model.state_dict().keys()
n_and_p = list(model.named_parameters())
model_device = n_and_p[0][1].device
for n, p in model.named_parameters():
        print(n)
        extra = saved["model"][n].to(model_device) - saved["depth_params"][n][0].to(model_device)
        if n in model_keys:
            pass
        elif n.replace("module.", "") in model_keys:
            n = n.replace("module.", "")
        elif "module." + n in model_keys:
            n = "module." + n

        nonzero_mask[n] = extra != 0
        if getattr(nonzero_mask[n], "requires_grad"):
            nonzero_mask[n].requires_grad = False
            print("Nonzero mask requires grad set to False")
        nonzero_mask[n] = nonzero_mask[n].to(model_device)
        
for name,param in model.named_parameters():
       
        p0 = torch.zeros_like(param.data).copy_(param) #original BERT
        p1 = torch.zeros_like(param.data) #params to be fine-tuned
        
        p1.data.copy_(saved["depth_params"][name][1].data)
        p0.data.copy_(saved["depth_params"][name][0].data)
        
        p1.requires_grad = True

        p1.grad = torch.zeros_like(param.data)
        
        alpha = torch.zeros_like(param.data).to(device) + args.alpha_init
        alpha.data.copy_(saved["depth_params"][name][2].data)
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
            layer_id = get_num_layer(name)
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

args.warmup_epoch=t_total
print("Use step level LR & WD scheduler!")

scheduler=PolynomialLR(optimizer, 120000, gamma=0.9, min_lr=0)
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
min_val_loss = 0.0

args.eval_freq=4
args.distributed=False
args.concrete_lower=-1.5
args.concrete_upper=1.5
args.per_layer_alpha=0
args.max_grad_norm=10
max_norm=args.clip_grad
args.gradient_accumulation_steps =1
args.save_ckp=2
return_all_layers = args.output_adapter in ['dpt']

tasks_loss_fn = {
            'edge':  BalancedBinaryCrossEntropyLoss(pos_weight=0.95, ignore_index=255)
}

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

non_zero_weight=0
whole_weight=0
for n in nonzero_mask:
    if 'output_adapters' in n: # only changes in the output_adapter
        continue
    whole_weight+=nonzero_mask[n].numel()
    non_zero_weight+=torch.sum(nonzero_mask[n]).item()
print('sparsity ratio in encoder: {} M'.format((1-non_zero_weight/whole_weight)*100))
non_zero_weight=0
whole_weight=0
weight_layer=[0]*14
number_layer=[0]*14
whole_layer=[0]*14
for n in nonzero_mask:
    #print(n)
    whole_weight+=nonzero_mask[n].numel()
    non_zero_weight+=torch.sum(nonzero_mask[n]).item()
    #print(nonzero_mask[n].numel())
    #print(torch.sum(nonzero_mask[n]).item())
    ind=get_num_layer_for_vit(n)
    #print(ind)
    whole_layer[ind]+=nonzero_mask[n].numel()

    weight_layer[ind]+=torch.sum(nonzero_mask[n]).item()

print('sparsity ratio in whole model: {} M'.format((1-non_zero_weight/whole_weight)*100))     

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
        
        # Mask invalid input values
        for task in input_dict:
            if task in ['rgb']:
                continue
            channels = input_dict[task].shape[1]
        nonzero_params = 0
        grad_params = {}
        
        for n, p in model.named_parameters():
                if n not in depth_params:
                    print(" n not in depth_params")
                    embed()
                assert(n in depth_params)
                if "output_adapters" in n:
                    nonzero_params += p.numel()
                    
                    p.data.copy_(depth_params[n][0].data + depth_params[n][1].data)
                else:
                    
                    z = nonzero_mask[n]
                    grad_params[n] = [depth_params[n][1] , z , torch.zeros_like(z)]
                    p.data.copy_(depth_params[n][0].data + z.data*depth_params[n][1].data)
                nonzero_params += (z>0).float().detach().sum().item()
        
        model.train()
        
        # Forward + backward
        with torch.cuda.amp.autocast(enabled=False):
            # print('=====> input_dict:', {k: v.shape for k, v in input_dict.items()})

            preds = model(input_dict['image'], return_all_layers=return_all_layers)
            task_losses = {
                task: tasks_loss_fn[task](preds[task].float(), tasks_dict['edge'])
                for task in preds
            }
            loss = 50*sum(task_losses.values())
            
        loss_value = loss.item()
        
        task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}
        #metrics = normal_metrics(preds['normal'], tasks_dict['normals'], mask_valid=None)
        
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
                        depth_params[n][1].grad.copy_(p.grad.data * grad_params[n][1].data)
                    
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(alpha_params, args.max_grad_norm)
        
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        
        params_norm = [0, 0, 0, 0, 0, 0]
        exp_z = 0
        for n, p in depth_params.items():
                    
            params_norm[4] += p[2].numel()
                    
        
        if (step + 1)% 100 == 0:
                     print("empirical prob:", nonzero_params/params_norm[4])
        
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)
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
            
            
            if val_stats['maxF'] > min_val_loss:
                min_val_loss = val_stats['maxF']
                if args.output_dir and args.save_ckpt:
                   
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    ) 

                    info_dict = {"model": model.state_dict(), "depth_params": depth_params} #, "z": z.detach(), "nonzero_params": nonzero_params}
                    torch.save(info_dict, os.path.join(args.output_dir, "checkpoint-last-info.pt"))
                    

                print(f'New best mIoU: {min_val_loss:.3f}')

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

