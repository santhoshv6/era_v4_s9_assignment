"""
Production-ready ResNet50 ImageNet training script.
Combines EMA + SWA strategies for optimal 81% accuracy.

Usage:
    python train.py --data /path/to/imagenet --output-dir ./outputs
"""

import argparse
import os
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm import tqdm

from src.model import get_model
from src.transforms import build_transforms
from src.utils import (
    AverageMeter, accuracy, seed_everything, save_checkpoint, 
    get_device, format_time, setup_logging
)
from src.mixup import MixupCutmixCollator, mixup_criterion
from src.ema import EMAModel


def parse_args():
    parser = argparse.ArgumentParser(description='ResNet50 ImageNet Training')
    
    # Required
    parser.add_argument('--data', type=str, required=True,
                       help='Path to ImageNet dataset (with train/ and val/ folders)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=120,
                       help='Total training epochs (default: 120)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing (default: 0.1)')
    
    # Strategy configuration
    parser.add_argument('--ema-epochs', type=int, default=100,
                       help='Use EMA for first N epochs (default: 100)')
    parser.add_argument('--swa-epochs', type=int, default=20,
                       help='Use SWA for last N epochs (default: 20)')
    parser.add_argument('--ema-decay', type=float, default=0.9999,
                       help='EMA decay rate (default: 0.9999)')
    parser.add_argument('--swa-lr', type=float, default=0.01,
                       help='SWA learning rate (default: 0.01)')
    
    # Augmentation
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                       help='Mixup alpha (default: 0.2)')
    parser.add_argument('--mixup-prob', type=float, default=0.8,
                       help='Mixup probability (default: 0.8)')
    
    # System
    parser.add_argument('--workers', type=int, default=8,
                       help='Data loading workers (default: 8)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory (default: ./outputs)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--amp', action='store_true', default=True,
                       help='Use mixed precision (default: True)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default='',
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def build_optimizer_scheduler(model, args):
    """Build optimizer and cosine scheduler with warmup"""
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=0.001
    )
    
    return optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, epoch, args, 
                scaler=None, ema_model=None, swa_model=None):
    """Single training epoch"""
    model.train()
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    
    # Determine strategy
    use_ema = epoch < args.ema_epochs
    use_swa = epoch >= (args.epochs - args.swa_epochs)
    strategy = "EMA" if use_ema else "SWA" if use_swa else "Base"
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}/{args.epochs} ({strategy})')
    
    for images, targets in pbar:
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass
        if args.amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        # Backward pass
        if args.amp and scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Update models
        if use_ema and ema_model:
            ema_model.update(model)
        elif use_swa and swa_model:
            swa_model.update_parameters(model)
        
        # Metrics
        acc1, _ = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        
        pbar.set_postfix({
            'Loss': f'{losses.avg:.3f}',
            'Acc': f'{top1.avg:.1f}%'
        })
    
    return losses.avg, top1.avg


def validate(model, val_loader, criterion, args):
    """Validation"""
    model.eval()
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for images, targets in pbar:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            if args.amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            acc1, _ = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            
            pbar.set_postfix({
                'Loss': f'{losses.avg:.3f}',
                'Acc': f'{top1.avg:.1f}%'
            })
    
    return losses.avg, top1.avg


def main():
    args = parse_args()
    
    # Setup
    seed_everything(args.seed)
    device = get_device()
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Logging
    logger = setup_logging(str(output_dir))
    logger.info("🚀 Starting ResNet50 ImageNet Training")
    logger.info(f"Device: {device}")
    logger.info(f"Strategy: EMA (epochs 1-{args.ema_epochs}) + SWA (last {args.swa_epochs})")
    logger.info(f"Target: 81% top-1 accuracy")
    
    # Data loading
    train_transform, val_transform = build_transforms(
        img_size=224,
        strong_aug=True
    )
    
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data, 'train'),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data, 'val'),
        transform=val_transform
    )
    
    # Mixup collator
    mixup_collator = MixupCutmixCollator(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=0.0,  # Only mixup
        prob=args.mixup_prob
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=mixup_collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    logger.info(f"Dataset loaded: {len(train_dataset):,} train, {len(val_dataset):,} val images")
    
    # Model
    model = get_model(num_classes=1000, model_name='resnet50', dropout=0.0)
    model = model.cuda()
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Optimizer and scheduler
    optimizer, scheduler = build_optimizer_scheduler(model, args)
    
    # EMA model
    ema_model = EMAModel(model, decay=args.ema_decay)
    
    # SWA model
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
    
    # Mixed precision
    scaler = GradScaler() if args.amp else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc1 = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu')
            
            # Load model state
            model.load_state_dict(checkpoint['model'])
            
            # Load training state
            start_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Load EMA state if available
            if 'ema_model' in checkpoint and ema_model:
                ema_model.load_state_dict(checkpoint['ema_model'])
                logger.info("EMA model state restored")
            
            # Load SWA state if available
            if 'swa_model' in checkpoint and swa_model:
                swa_model.load_state_dict(checkpoint['swa_model'])
                logger.info("SWA model state restored")
            
            # Load scheduler state
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            
            # Load scaler state
            if 'scaler' in checkpoint and scaler:
                scaler.load_state_dict(checkpoint['scaler'])
            
            logger.info(f"Resumed from epoch {start_epoch-1}, best accuracy: {best_acc1:.2f}%")
        else:
            logger.warning(f"No checkpoint found at '{args.resume}'")
    
    # Training loop
    start_time = time.time()
    
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        
        # Learning rate scheduling
        if epoch >= (args.epochs - args.swa_epochs):
            swa_scheduler.step()
            current_lr = swa_scheduler.get_last_lr()[0]
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        
        logger.info(f'\nEpoch {epoch+1:3d}/{args.epochs} - LR: {current_lr:.6f}')
        
        # Training
        train_loss, train_acc1 = train_epoch(
            model, train_loader, mixup_criterion, optimizer, epoch, args,
            scaler=scaler, ema_model=ema_model, swa_model=swa_model
        )
        
        # Validation with appropriate model
        if epoch < args.ema_epochs:
            # Use EMA model
            val_loss, val_acc1 = validate(ema_model.model, val_loader, criterion, args)
            eval_model = ema_model.model
            model_type = "EMA"
        elif epoch >= (args.epochs - args.swa_epochs):
            # Use SWA model
            if epoch == (args.epochs - args.swa_epochs):
                logger.info("Updating SWA batch norm statistics...")
                update_bn(train_loader, swa_model, device=torch.device('cuda'))
            val_loss, val_acc1 = validate(swa_model, val_loader, criterion, args)
            eval_model = swa_model
            model_type = "SWA"
        else:
            # Use base model
            val_loss, val_acc1 = validate(model, val_loader, criterion, args)
            eval_model = model
            model_type = "Base"
        
        # Logging
        elapsed = time.time() - start_time
        logger.info(f'Train: {train_loss:.3f} loss, {train_acc1:.2f}% acc')
        logger.info(f'Val:   {val_loss:.3f} loss, {val_acc1:.2f}% acc ({model_type})')
        logger.info(f'Best:  {max(best_acc1, val_acc1):.2f}% | Elapsed: {format_time(elapsed)}')
        
        # Save best model
        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            
            # Prepare checkpoint with all necessary states
            checkpoint_state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc1': best_acc1,
                'model_type': model_type
            }
            
            # Add EMA state if available
            if ema_model:
                checkpoint_state['ema_model'] = ema_model.state_dict()
            
            # Add SWA state if available
            if swa_model and epoch >= (args.epochs - args.swa_epochs):
                checkpoint_state['swa_model'] = swa_model.state_dict()
            
            # Add scaler state if using mixed precision
            if scaler:
                checkpoint_state['scaler'] = scaler.state_dict()
            
            save_checkpoint(checkpoint_state, True, str(output_dir), 'best_model.pth')
        
        # Milestone checks
        if epoch == 80:  # 81st epoch
            logger.info(f'🎯 Milestone: Epoch 81 = {val_acc1:.2f}% (target: >75%)')
        if epoch == 89:  # 90th epoch  
            logger.info(f'🎯 Milestone: Epoch 90 = {val_acc1:.2f}% (target: >77%)')
    
    # Final results
    total_time = time.time() - start_time
    logger.info(f'\n🎉 Training Complete!')
    logger.info(f'Best Accuracy: {best_acc1:.2f}%')
    logger.info(f'Total Time: {format_time(total_time)}')
    logger.info(f'Output saved to: {output_dir}')
    
    if best_acc1 >= 81.0:
        logger.info('✅ SUCCESS: Achieved 81%+ target!')
    elif best_acc1 >= 79.0:
        logger.info('🟡 GOOD: Close to 81% target')
    else:
        logger.info('🔴 BELOW TARGET: Check hyperparameters')


if __name__ == '__main__':
    main()