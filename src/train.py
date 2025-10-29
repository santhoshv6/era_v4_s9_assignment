"""
Train ResNet50 from scratch on ImageNet to achieve 81% top-1 accuracy.
This script implements all the necessary techniques for successful from-scratch training.
"""

import argparse
import os
import time
import warnings
from contextlib import suppress

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder  
from tqdm import tqdm  

# Export list for clean imports
__all__ = [
    'parse_args', 'main', 'train_epoch', 'validate'
]

from torchvision import datasets
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import get_model, get_model_info
from .transforms import build_transforms
from .utils import (
    AverageMeter, accuracy, seed_everything, save_checkpoint, load_checkpoint,
    get_device, format_time, is_main_process, setup_logging, setup_markdown_log,
    log_epoch_results, WarmupCosineScheduler, save_config
)
from .mixup import MixupCutmixCollator


def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet50 from scratch on ImageNet')
    
    # Data parameters
    parser.add_argument('--data', type=str, required=True,
                       help='Path to ImageNet dataset root (containing train/ and val/ folders)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Input image size')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of total epochs to run')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Mini-batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.5,
                       help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Number of warmup epochs')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing parameter')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model architecture')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate')
    
    # Optimization
    parser.add_argument('--amp', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--clip-grad', type=float, default=None,
                       help='Gradient clipping max norm')
    
    # Regularization
    parser.add_argument('--strong-aug', action='store_true', default=True,
                       help='Use strong data augmentation')
    
    # Logging and checkpointing  
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Path to save outputs')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='How often to log training stats')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Resume training
    parser.add_argument('--resume', type=str, default='',
                       help='Path to checkpoint to resume from')
    parser.add_argument('--start-epoch', type=int, default=0,
                       help='Manual epoch number (useful on restarts)')

    # New argument: LR Finder
    parser.add_argument('--lr-finder', action='store_true',
                        help='Run Learning Rate Finder before training')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model on validation set only')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    seed_everything(args.seed)
    device = get_device()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    markdown_log = setup_markdown_log(args.output_dir)
    
    # Save configuration
    config = vars(args)
    save_config(config, os.path.join(args.output_dir, 'config.json'))
    
    logger.info(f"Training configuration: {config}")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = get_model(num_classes=1000, model_name=args.model, dropout=args.dropout)
    model = model.to(device)
    
    # Log model info
    model_info = get_model_info(model)
    logger.info(f"Model: {model_info}")
    
    # Create data loaders
    train_transforms, val_transforms = build_transforms(
        img_size=args.img_size, 
        strong_aug=args.strong_aug
    )
    
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data, 'train'),
        transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data, 'val'),
        transform=val_transforms
    )
    
    # If LR Finder flag use a small subset for fast LR search
    if args.lr_finder:
        subset_size = 2000
        subset_indices = list(range(min(subset_size, len(train_dataset))))
        train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    logger.info(f"Number of classes: {len(train_dataset.classes)}")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Define optimizer
    # Use very small lr initially if LR Finder flag set
    init_lr = 1e-7 if args.lr_finder else args.lr
    optimizer = optim.SGD(
        model.parameters(),
        lr=init_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    
    # Define scheduler (for main training only)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
        warmup_lr=0.0
    )
    
    # AMP scaler
    scaler = GradScaler(enabled=args.amp)
    
    # Resume from checkpoint if specified (only for normal training)
    start_epoch = args.start_epoch
    best_acc1 = 0.0
    
    if args.resume and not args.lr_finder:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint '{args.resume}'")
            torch.serialization.add_safe_globals({'WarmupCosineScheduler': WarmupCosineScheduler})
            checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_acc1 = checkpoint.get('best_acc1', 0.0)
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logger.info(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch-1})")
        else:
            logger.warning(f"No checkpoint found at '{args.resume}'")
    
    # Run LR Finder if flagged
    if args.lr_finder:
        logger.info("Starting LR Finder run...")
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
        lr_finder.plot(log_lr=True, skip_start=10, skip_end=5)
        plot_path = os.path.join(args.output_dir, "lr_finder_plot.png")
        lr_finder.plot(log_lr=True, skip_start=10, skip_end=5).savefig(plot_path)
        logger.info(f"LR Finder completed. Plot saved at {plot_path}.")
        lr_finder.reset()
        return

    # Evaluate only
    if args.evaluate:
        logger.info("Evaluating model...")
        validate(val_loader, model, criterion, device, args, logger)
        return
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        
        # Update learning rate
        lr = scheduler.step(epoch)
        logger.info(f"Epoch {epoch}: Learning rate = {lr:.6f}")
        
        # Train for one epoch
        train_stats = train_epoch(
            train_loader, model, criterion, optimizer, scaler, 
            device, epoch, args, logger
        )
        
        # Evaluate on validation set
        val_stats = validate(val_loader, model, criterion, device, args, logger)
        
        # Log to markdown
        log_epoch_results(
            markdown_log, epoch, 'train', 
            train_stats['loss'], train_stats['top1'], train_stats['top5'],
            train_stats['time'], lr
        )
        log_epoch_results(
            markdown_log, epoch, 'val',
            val_stats['loss'], val_stats['top1'], val_stats['top5'], 
            val_stats['time'], lr
        )
        
        # Remember best accuracy and save checkpoint
        is_best = val_stats['top1'] > best_acc1
        best_acc1 = max(val_stats['top1'], best_acc1)
        
        if is_main_process():
            checkpoint_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler,
                'scaler': scaler.state_dict(),
                'best_acc1': best_acc1,
                'args': args,
            }
            
            # Save checkpoint
            if (epoch + 1) % args.save_freq == 0 or is_best:
                save_checkpoint(
                    checkpoint_state, is_best, args.output_dir,
                    filename=f'checkpoint_epoch_{epoch}.pth'
                )
            
            # Always save latest
            save_checkpoint(checkpoint_state, False, args.output_dir, 'checkpoint_latest.pth')
        
        logger.info(f"Epoch {epoch} completed. Best Acc@1: {best_acc1:.2f}%")
    
    markdown_log.close()
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_acc1:.2f}%")


def train_epoch(loader, model, criterion, optimizer, scaler, device, epoch, args, logger):
    """Train for one epoch."""
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f') 
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.train()
    
    epoch_start = time.time()
    end = time.time()
    
    # Use tqdm progress bar over the loader
    loop = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch} Training')
    
    for i, (images, target) in loop:
        # Measure data loading time
        data_time.update(time.time() - end)
        
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with AMP
        with autocast(enabled=args.amp):
            output = model(images)
            loss = criterion(output, target)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if args.clip_grad is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update tqdm postfix to show current losses and accuracies
        loop.set_postfix(loss=losses.avg, acc1=top1.avg, acc5=top5.avg)
        
        # Log progress
        if i % args.log_interval == 0:
            logger.info(
                f'Epoch: [{epoch}][{i}/{len(loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'
            )
    
    epoch_time = time.time() - epoch_start
    logger.info(f'Train Epoch {epoch}: Loss {losses.avg:.4f} Acc@1 {top1.avg:.2f} Acc@5 {top5.avg:.2f} Time {epoch_time:.1f}s')
    
    return {
        'loss': losses.avg,
        'top1': top1.avg, 
        'top5': top5.avg,
        'time': epoch_time
    }


def validate(loader, model, criterion, device, args, logger):
    """Validate model."""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.eval()
    
    val_start = time.time()
    with torch.no_grad():
        end = time.time()
        loop = tqdm(enumerate(loader), total=len(loader), desc='Validation')
        for i, (images, target) in loop:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Compute output
            with autocast(enabled=args.amp):
                output = model(images)
                loss = criterion(output, target)
            
            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Update tqdm postfix with validation stats
            loop.set_postfix(loss=losses.avg, acc1=top1.avg, acc5=top5.avg)
    
    val_time = time.time() - val_start
    logger.info(f'Validation: Loss {losses.avg:.4f} Acc@1 {top1.avg:.2f} Acc@5 {top5.avg:.2f} Time {val_time:.1f}s')
    
    return {
        'loss': losses.avg,
        'top1': top1.avg,
        'top5': top5.avg, 
        'time': val_time
    }


if __name__ == '__main__':
    main()
