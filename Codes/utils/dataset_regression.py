# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on BEiT, timm, DINO, DeiT and MAE-priv code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------
from typing import Dict, Tuple
import numpy as np
import torch
import cv2
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except:
    print('albumentations not installed')
# import cv2
import torch.nn.functional as F

from utils import (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, NYU_MEAN,
                   NYU_STD, PAD_MASK_VALUE)
from utils.dataset_folder import ImageFolder, MultiTaskImageFolder





def simple_transform(train: bool,
                     additional_targets: Dict[str, str],
                     input_size: int =512,
                     pad_value: Tuple[int, int, int] = (128, 128, 128),
                     pad_mask_value: int =PAD_MASK_VALUE):
    """Default transform for semantic segmentation, applied on all modalities

    During training:
        1. Random horizontal Flip
        2. Rescaling so that longest side matches input size
        3. Color jitter (for RGB-modality only)
        4. Large scale jitter (LSJ)
        5. Padding
        6. Random crop to given size
        7. Normalization with ImageNet mean and std dev

    During validation / test:
        1. Rescaling so that longest side matches given size
        2. Padding
        3. Normalization with ImageNet mean and std dev
     """

    if train:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.LongestMaxSize(max_size=input_size, p=1),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.5),  # Color jittering from MoCo-v3 / DINO
            A.RandomScale(scale_limit=(0.1 - 1, 2.0 - 1), p=1),  # This is LSJ (0.1, 2.0)
            A.PadIfNeeded(min_height=input_size, min_width=input_size,
                          position=A.augmentations.PadIfNeeded.PositionType.TOP_LEFT,
                          border_mode=cv2.BORDER_CONSTANT,
                          value=pad_value, mask_value=pad_mask_value),
            A.RandomCrop(height=input_size, width=input_size, p=1),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    else:
        transform = A.Compose([
            A.LongestMaxSize(max_size=input_size, p=1),
            A.PadIfNeeded(min_height=input_size, min_width=input_size,
                          position=A.augmentations.PadIfNeeded.PositionType.TOP_LEFT,
                          border_mode=cv2.BORDER_CONSTANT,
                          value=pad_value, mask_value=pad_mask_value),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    return transform

def nyu_transform(train, additional_targets, input_size=512, color_aug=False):
    if train:
        augs = [
            A.SmallestMaxSize(max_size=input_size, p=1),
            A.HorizontalFlip(p=0.5),
        ]
        if color_aug: augs += [
                # Color jittering from BYOL https://arxiv.org/pdf/2006.07733.pdf
                A.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                A.ToGray(p=0.3),
            ]
        augs += [
            A.RandomCrop(height=input_size, width=input_size, p=1),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ]

        transform = A.Compose(augs, additional_targets=additional_targets)

    else:
        transform = A.Compose([
            A.SmallestMaxSize(max_size=input_size, p=1),
            A.CenterCrop(height=input_size, width=input_size),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    return transform


def simple_regression_transform(train, additional_targets, input_size=512, pad_value=(128, 128, 128), pad_mask_value=PAD_MASK_VALUE):

    if train:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.LongestMaxSize(max_size=input_size, p=1),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.5),  # Color jittering from MoCo-v3 / DINO
            A.RandomScale(scale_limit=(0.1 - 1, 2.0 - 1), p=1),  # This is LSJ (0.1, 2.0)
            A.PadIfNeeded(min_height=input_size, min_width=input_size,
                          position=A.augmentations.PadIfNeeded.PositionType.TOP_LEFT,
                          border_mode=cv2.BORDER_CONSTANT,
                          value=pad_value, mask_value=pad_mask_value),
            A.RandomCrop(height=input_size, width=input_size, p=1),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    else:
        transform = A.Compose([
            A.LongestMaxSize(max_size=input_size, p=1),
            A.PadIfNeeded(min_height=input_size, min_width=input_size,
                          position=A.augmentations.PadIfNeeded.PositionType.TOP_LEFT,
                          border_mode=cv2.BORDER_CONSTANT,
                          value=pad_value, mask_value=pad_mask_value),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    return transform


class DataAugmentationForRegression(object):

    def __init__(self, transform, mask_value=0.0):
        self.transform = transform
        self.mask_value = mask_value

    def __call__(self, task_dict):

        # Need to replace rgb key to image
        task_dict['image'] = task_dict.pop('rgb')
        # Convert to np.array
        task_dict = {k: np.array(v) for k, v in task_dict.items()}

        task_dict = self.transform(**task_dict)

        task_dict['depth'] = (task_dict['depth'].float() - NYU_MEAN)/NYU_STD

        # And then replace it back to rgb
        task_dict['rgb'] = task_dict.pop('image')

        

        for task in task_dict:
            if task in ['depth']:
                img = task_dict[task]
                if 'mask_valid' in task_dict:
                    task_dict['mask_valid'] = (task_dict['mask_valid'] == 255)[None]
                    mask_valid = task_dict['mask_valid'].squeeze()
                    img[~mask_valid] = self.mask_value
                task_dict[task] = img.unsqueeze(0)
            elif task in ['rgb']:
                task_dict[task] = task_dict[task].to(torch.float)

        return task_dict


class DataAugmentationForRegression_normal(object):

    def __init__(self, transform, mask_value=0.0):
        self.transform = transform
        self.mask_value = mask_value

    def __call__(self, task_dict):

        # Need to replace rgb key to image
        task_dict['image'] = task_dict.pop('rgb')
        # Convert to np.array
        task_dict = {k: np.array(v) for k, v in task_dict.items()}

        task_dict = self.transform(**task_dict)
        task_dict['normal']=task_dict['normal'].permute( 2,0,1)
        #task_dict['depth'] = (task_dict['depth'].float() - NYU_MEAN)/NYU_STD
        task_dict['normal'] =2*task_dict['normal']/255.-1

        # And then replace it back to rgb
        task_dict['rgb'] = task_dict.pop('image')

        

        for task in task_dict:
            if task in ['normal']:
                img = task_dict[task]
                if 'mask_valid' in task_dict:
                    task_dict['mask_valid'] = (task_dict['mask_valid'] == 255)[None]
                    mask_valid = task_dict['mask_valid'].squeeze()
                    img[~mask_valid] = self.mask_value
                #task_dict[task] = img.unsqueeze(0)
            elif task in ['rgb']:
                task_dict[task] = task_dict[task].to(torch.float)

        return task_dict


def build_regression_dataset(args, data_path, transform, max_images=None):
    transform = DataAugmentationForRegression(transform=transform)

    return MultiTaskImageFolder(data_path, args.all_domains,args.in_domains, transform=transform, prefixes=None, max_images=max_images)



def build_regression_dataset_normal(args, data_path, transform, max_images=None):
    transform = DataAugmentationForRegression_normal(transform=transform)

    return MultiTaskImageFolder(data_path, args.all_domains,args.in_domains, transform=transform, prefixes=None, max_images=max_images)



class DataAugmentationForRegression_Edge(object):

    def __init__(self, transform, mask_value=0.0):
        self.transform = transform
        self.mask_value = mask_value

    def __call__(self, task_dict):

        # Need to replace rgb key to image
        task_dict['image'] = task_dict.pop('rgb')
        # Convert to np.array
        task_dict = {k: np.array(v) for k, v in task_dict.items()}

        task_dict = self.transform(**task_dict)

        #task_dict['depth'] = (task_dict['depth'].float() - NYU_MEAN)/NYU_STD
        task_dict['edge']=task_dict['edge']/255.
        # And then replace it back to rgb
        task_dict['rgb'] = task_dict.pop('image')

        

        for task in task_dict:
            if task in ['edge']:
                img = task_dict[task]
                if 'mask_valid' in task_dict:
                    task_dict['mask_valid'] = (task_dict['mask_valid'] == 255)[None]
                    mask_valid = task_dict['mask_valid'].squeeze()
                    img[~mask_valid] = self.mask_value
                task_dict[task] = img.unsqueeze(0)
            elif task in ['rgb']:
                task_dict[task] = task_dict[task].to(torch.float)

        return task_dict


def build_regression_dataset_edge(args, data_path, transform, max_images=None):
    transform = DataAugmentationForRegression_Edge(transform=transform)

    return MultiTaskImageFolder(data_path, args.all_domains,args.in_domains, transform=transform, prefixes=None, max_images=max_images)    



class DataAugmentationForRegression_2(object):

    def __init__(self, transform, mask_value=0.0):
        self.transform = transform
        self.mask_value = mask_value

    def __call__(self, task_dict):

        # Need to replace rgb key to image
        task_dict['image'] = task_dict.pop('rgb')
        # Convert to np.array
        task_dict = {k: np.array(v) for k, v in task_dict.items()}

        task_dict = self.transform(**task_dict)

        #task_dict['depth'] = (task_dict['depth'].float() - NYU_MEAN)/NYU_STD
        task_dict['depth'] =task_dict['depth'] /1000
        # And then replace it back to rgb
        task_dict['rgb'] = task_dict.pop('image')

        

        for task in task_dict:
            if task in ['depth']:
                img = task_dict[task]
                if 'mask_valid' in task_dict:
                    task_dict['mask_valid'] = (task_dict['mask_valid'] == 255)[None]
                    mask_valid = task_dict['mask_valid'].squeeze()
                    img[~mask_valid] = self.mask_value
                task_dict[task] = img.unsqueeze(0)
            elif task in ['rgb']:
                task_dict[task] = task_dict[task].to(torch.float)

        return task_dict
    
    
def build_regression_dataset2(args, data_path, transform, max_images=None):
    transform = DataAugmentationForRegression_2(transform=transform)

    return MultiTaskImageFolder(data_path, args.all_domains,args.in_domains, transform=transform, prefixes=None, max_images=max_images)


