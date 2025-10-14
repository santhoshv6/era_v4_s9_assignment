"""
ImageNet data transforms for ResNet50 from-scratch training.
Includes strong augmentation needed to reach 81% top-1 accuracy.
"""

import torch
from torchvision import transforms
from typing import Tuple


# ImageNet statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int = 224, 
                        strong_aug: bool = True,
                        auto_augment: bool = False) -> transforms.Compose:
    """Get training transforms for ImageNet.
    
    Args:
        img_size: Target image size (usually 224)
        strong_aug: Whether to use strong augmentation for from-scratch training
        auto_augment: Whether to use AutoAugment (requires timm)
    
    Returns:
        Training transform pipeline
    """
    transform_list = []
    
    # Basic geometric transforms
    transform_list.extend([
        transforms.RandomResizedCrop(
            img_size, 
            scale=(0.08, 1.0),  # Standard ImageNet scale range
            ratio=(3./4., 4./3.),  # Standard aspect ratio range
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    
    # Strong augmentation for from-scratch training
    if strong_aug:
        transform_list.extend([
            # Color jittering - crucial for from-scratch training
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4, 
                saturation=0.4,
                hue=0.1
            ),
            # Random erasing for regularization
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value=0,
                inplace=False
            ),
        ])
    else:
        # Light augmentation
        transform_list.extend([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    
    return transforms.Compose(transform_list)


def get_val_transforms(img_size: int = 224, 
                      crop_pct: float = 0.875) -> transforms.Compose:
    """Get validation transforms for ImageNet.
    
    Args:
        img_size: Target image size
        crop_pct: Center crop percentage
        
    Returns:
        Validation transform pipeline
    """
    resize_size = int(img_size / crop_pct)
    
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_test_transforms(img_size: int = 224) -> transforms.Compose:
    """Get test-time transforms (same as validation)."""
    return get_val_transforms(img_size)


def build_transforms(img_size: int = 224, 
                    strong_aug: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build train and validation transforms.
    
    Args:
        img_size: Target image size
        strong_aug: Whether to use strong augmentation for training
        
    Returns:
        Tuple of (train_transforms, val_transforms)
    """
    train_tfm = get_train_transforms(img_size, strong_aug=strong_aug)
    val_tfm = get_val_transforms(img_size)
    
    return train_tfm, val_tfm


# Mixup and CutMix will be implemented in the training script
# as they require label manipulation
