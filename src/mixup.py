"""
Mixup and CutMix implementations for advanced data augmentation.
These techniques are crucial for achieving 81% accuracy from scratch.
"""

import torch
import torch.nn.functional as F
import numpy as np


def mixup_data(x, y, alpha=1.0):
    """
    Mixup data augmentation.
    
    Args:
        x: Input batch
        y: Target labels
        alpha: Mixup parameter (default: 1.0)
    
    Returns:
        Mixed inputs, targets_a, targets_b, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss function.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: Original labels
        y_b: Mixed labels
        lam: Mixing parameter
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix data augmentation.
    
    Args:
        x: Input batch
        y: Target labels
        alpha: CutMix parameter (default: 1.0)
    
    Returns:
        Mixed inputs, targets_a, targets_b, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Generate random bounding box
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply CutMix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


class MixupCutmixCollator:
    """
    Collator that applies Mixup or CutMix with given probabilities.
    """
    
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5):
        """
        Args:
            mixup_alpha: Mixup alpha parameter
            cutmix_alpha: CutMix alpha parameter  
            prob: Probability of applying augmentation
            switch_prob: Probability of using Mixup vs CutMix
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob

    def __call__(self, batch):
        """Apply Mixup or CutMix to a batch."""
        x, y = batch
        
        if np.random.rand() < self.prob:
            if np.random.rand() < self.switch_prob:
                # Apply Mixup
                return mixup_data(x, y, self.mixup_alpha)
            else:
                # Apply CutMix
                return cutmix_data(x, y, self.cutmix_alpha)
        else:
            # No augmentation
            return x, y, y, 1.0


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2