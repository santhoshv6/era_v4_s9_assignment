"""
Utilities for ResNet50 ImageNet training from scratch.
Includes logging, metrics, checkpointing, and training helpers.
"""

import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, TextIO, Tuple, Union
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR


def seed_everything(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For competitive determinism at the cost of performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # For better performance with slight non-determinism
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state: Dict, is_best: bool, checkpoint_dir: str, filename: str = 'checkpoint.pth'):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_filepath)


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer=None, scheduler=None) -> Dict:
    """Load model checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load scheduler state if provided  
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return checkpoint


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def setup_logging(log_dir: str, log_level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('imagenet_training')
    logger.setLevel(log_level)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def setup_markdown_log(log_dir: str) -> TextIO:
    """Setup markdown log for epoch results."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'training_log.md')
    
    # Check if file exists to determine if we need header
    need_header = not os.path.exists(log_path)
    
    log_file = open(log_path, 'a', encoding='utf-8')
    
    if need_header:
        log_file.write('# ResNet50 ImageNet Training Log\n\n')
        log_file.write('Training from scratch to achieve 81% top-1 accuracy.\n\n')
        log_file.write('| Epoch | Phase | Loss | Top-1 Acc | Top-5 Acc | Time (s) | LR |\n')
        log_file.write('|-------|-------|------|-----------|-----------|----------|----|\n')
        log_file.flush()
    
    return log_file


def log_epoch_results(log_file: TextIO, epoch: int, phase: str, loss: float, 
                     top1_acc: float, top5_acc: float, epoch_time: float, lr: float):
    """Log epoch results to markdown file."""
    log_file.write(f'| {epoch:3d} | {phase:5s} | {loss:.4f} | {top1_acc:8.2f} | {top5_acc:8.2f} | {epoch_time:8.1f} | {lr:.6f} |\n')
    log_file.flush()


def format_time(seconds: float) -> str:
    """Format time in seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""
    
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, 
                 base_lr: float, warmup_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * (1 + math.cos(math.pi * progress)) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# Distributed training utilities
def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Get world size for distributed training."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get rank for distributed training."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_onecycle_scheduler(optimizer, max_lr: float, epochs: int, steps_per_epoch: int):
    """
    Create OneCycleLR scheduler for faster convergence.
    
    Args:
        optimizer: PyTorch optimizer
        max_lr: Maximum learning rate
        epochs: Total number of epochs
        steps_per_epoch: Number of steps per epoch
    
    Returns:
        OneCycleLR scheduler
    """
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # 30% of training for warmup
        anneal_strategy='cos',
        div_factor=25.0,  # max_lr/25 = initial_lr
        final_div_factor=10000.0,  # min_lr = max_lr/final_div_factor
    )


def reduce_across_processes(val):
    """Reduce value across all processes."""
    if not is_dist_avail_and_initialized():
        return val
    
    t = torch.tensor(val, device='cuda')
    dist.barrier()
    dist.all_reduce(t)
    return t.item() / get_world_size()


def is_main_process():
    """Check if this is the main process."""
    return get_rank() == 0


def reduce_dict(input_dict: Dict, average: bool = True):
    """Reduce dictionary values across all processes."""
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def save_config(config: Dict, save_path: str):
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)
