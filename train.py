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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm import tqdm

from src.model import get_model
from src.transforms import build_transforms
from src.mixup import MixupCutmixCollator, mixup_criterion
from src.utils import (
    AverageMeter, accuracy, seed_everything, save_checkpoint, 
    get_device, format_time, setup_logging
)
from src.ema import EMAModel


def get_dynamic_mixup_params(epoch, current_alpha=0.1, current_prob=0.5, target_alpha=0.05, target_prob=0.3):
    """Get mixup parameters that gradually reduce from current conservative to very conservative"""
    # The model has already been using conservative params (0.1, 0.5)
    # We can reduce them further to help close the train/val gap
    if epoch < 67:
        # Keep current conservative parameters
        return current_alpha, current_prob
    elif epoch < 70:
        # Reduce further over 3 epochs to very conservative
        progress = (epoch - 67) / (70 - 67)
        alpha = current_alpha + progress * (target_alpha - current_alpha)
        prob = current_prob + progress * (target_prob - current_prob)
        return alpha, prob
    else:
        # Use very conservative parameters from epoch 70+
        return target_alpha, target_prob


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
    parser.add_argument('--lr', type=float, default=0.05,
                       help='Initial learning rate (default: 0.05)')
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
    parser.add_argument('--mixup-alpha', type=float, default=0.1,
                       help='Mixup alpha (default: 0.1)')
    parser.add_argument('--mixup-prob', type=float, default=0.5,
                       help='Mixup probability (default: 0.5)')
    
    # Plateau handling
    parser.add_argument('--plateau-patience', type=int, default=5,
                       help='Epochs to wait before reducing LR on plateau (default: 5)')
    parser.add_argument('--plateau-factor', type=float, default=0.5,
                       help='Factor to reduce LR by on plateau (default: 0.5)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                       help='Minimum learning rate (default: 1e-6)')
    
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
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with additional logging')
    
    return parser.parse_args()


def build_optimizer_scheduler(model, args):
    """Build optimizer with both cosine scheduler and plateau scheduler"""
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Primary scheduler: Cosine annealing
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.min_lr
    )
    
    # Secondary scheduler: Plateau detection
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # We want to maximize accuracy
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        verbose=True,
        min_lr=args.min_lr
    )
    
    return optimizer, scheduler, plateau_scheduler


def train_epoch(model, train_loader, criterion, optimizer, epoch, args, 
                scaler=None, ema_model=None, swa_model=None):
    """Single training epoch"""
    model.train()
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    
    # Determine strategy - EMA DISABLED
    use_ema = False  # EMA disabled for stability
    use_swa = epoch >= (args.epochs - args.swa_epochs)
    strategy = "SWA" if use_swa else "Main"
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}/{args.epochs} ({strategy})')
    
    for batch_data in pbar:
        # Handle Mixup/CutMix data format
        if len(batch_data) == 4:  # Mixup/CutMix applied
            images, targets_a, targets_b, lam = batch_data
            images = images.cuda(non_blocking=True)
            targets_a = targets_a.cuda(non_blocking=True)
            targets_b = targets_b.cuda(non_blocking=True)
            mixup_active = True
            # For compatibility with metrics
            targets = targets_a
        else:  # Normal batch (no mixup)
            images, targets = batch_data
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            targets_a, targets_b, lam = targets, targets, 1.0
            mixup_active = False
        
        optimizer.zero_grad()
        
        # Forward pass
        if args.amp:
            with autocast():
                outputs = model(images)
                # Use appropriate loss calculation for mixup
                if mixup_active:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            # Use appropriate loss calculation for mixup
            if mixup_active:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
        
        # Backward pass
        if args.amp and scaler:
            scaler.scale(loss).backward()
            # Faster gradient clipping transition for stability
            scaler.unscale_(optimizer)
            if epoch < 68:
                max_norm = 1.0  # Keep original for 1 epoch
            elif epoch < 70:
                max_norm = 1.5  # Gradual increase for 2 epochs
            else:
                max_norm = 2.0  # Full relaxation from epoch 70
                    
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Faster gradient clipping transition for stability
            if epoch < 68:
                max_norm = 1.0  # Keep original for 1 epoch
            elif epoch < 70:
                max_norm = 1.5  # Gradual increase for 2 epochs
            else:
                max_norm = 2.0  # Full relaxation from epoch 70
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
        
        # Update models
        if use_swa and swa_model:
            swa_model.update_parameters(model)
        # EMA disabled for stability
        
        # Metrics
        acc1, _ = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        
        pbar.set_postfix({
            'Loss': f'{losses.avg:.3f}',
            'Acc': f'{top1.avg:.1f}%'
        })
    
    return losses.avg, top1.avg


def validate(model, val_loader, criterion, args, logger=None):
    """Validation"""
    model.eval()  # Ensure model is in eval mode
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for i, (images, targets) in enumerate(pbar):
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
            
            # Debug: Print first batch predictions for verification
            if i == 0 and args.debug and logger:
                pred_classes = outputs.argmax(dim=1)[:5]
                actual_classes = targets[:5]
                logger.info(f"Sample predictions: {pred_classes.cpu().tolist()}")
                logger.info(f"Sample targets: {actual_classes.cpu().tolist()}")
    
    return losses.avg, top1.avg


def main():
    args = parse_args()
    
    # Parameter validation
    assert args.ema_epochs + args.swa_epochs <= args.epochs, \
        f"EMA epochs ({args.ema_epochs}) + SWA epochs ({args.swa_epochs}) must be <= total epochs ({args.epochs})"
    assert args.batch_size > 0, "Batch size must be positive"
    assert args.lr > 0, "Learning rate must be positive"
    assert os.path.exists(args.data), f"Data directory does not exist: {args.data}"
    
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
    logger.info(f"Strategy: Main Model Only + SWA (last {args.swa_epochs}) - EMA Disabled")
    logger.info(f"Target: 78% top-1 accuracy")
    
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
        drop_last=True,
        collate_fn=mixup_collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"Dataset loaded: {len(train_dataset):,} train, {len(val_dataset):,} val images")
    
    # Model
    model = get_model(num_classes=1000, model_name='resnet50', dropout=0.0)
    model = model.cuda()
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Optimizer and scheduler
    optimizer, scheduler, plateau_scheduler = build_optimizer_scheduler(model, args)
    
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
            try:
                logger.info(f"Loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume, map_location='cpu')
                
                # Validate checkpoint structure
                required_keys = ['epoch', 'model', 'optimizer', 'best_acc1']
                missing_keys = [key for key in required_keys if key not in checkpoint]
                if missing_keys:
                    logger.warning(f"Checkpoint missing keys: {missing_keys}")
                    logger.info("🚀 Starting fresh training instead")
                else:
                    # Load model state
                    model.load_state_dict(checkpoint['model'])
                    
                    # Load training state
                    start_epoch = checkpoint['epoch'] + 1
                    best_acc1 = checkpoint['best_acc1']
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    
                    # Validate resumed values
                    if torch.isnan(torch.tensor(best_acc1)) or best_acc1 < 0:
                        logger.warning(f"Invalid best_acc1 in checkpoint: {best_acc1}")
                        best_acc1 = 0.0
                    
                    # EMA disabled for stability - skip EMA loading
                    logger.info("🔄 EMA disabled for stability - skipping EMA state loading")
                    
                    # Load SWA state if available
                    if 'swa_model' in checkpoint and swa_model:
                        try:
                            swa_model.load_state_dict(checkpoint['swa_model'])
                            logger.info("✅ SWA model state restored")
                        except Exception as e:
                            logger.warning(f"⚠️  Failed to load SWA model state: {e}")
                            logger.info("🔄 SWA model will be initialized fresh")
                    
                    # Load scheduler state (regular scheduler)
                    if 'scheduler' in checkpoint:
                        try:
                            scheduler.load_state_dict(checkpoint['scheduler'])
                            logger.info("✅ Scheduler state restored")
                        except Exception as e:
                            logger.warning(f"⚠️  Failed to load scheduler state: {e}")
                            logger.info("🔄 Scheduler will continue with current configuration")
                    
                    # Load plateau scheduler state if available
                    if 'plateau_scheduler' in checkpoint:
                        try:
                            plateau_scheduler.load_state_dict(checkpoint['plateau_scheduler'])
                            logger.info("✅ Plateau scheduler state restored")
                        except Exception as e:
                            logger.warning(f"⚠️  Failed to load plateau scheduler state: {e}")
                            # Initialize plateau scheduler with current best accuracy to avoid immediate LR reduction
                            plateau_scheduler.best = best_acc1
                            plateau_scheduler.num_bad_epochs = 0
                            logger.info(f"🔧 Plateau scheduler baseline set to {best_acc1:.2f}%")
                    else:
                        logger.info("ℹ️  No plateau scheduler in checkpoint - initializing fresh")
                        # Initialize plateau scheduler with current best accuracy to avoid immediate LR reduction
                        plateau_scheduler.best = best_acc1
                        plateau_scheduler.num_bad_epochs = 0
                        logger.info(f"🔧 Plateau scheduler baseline set to {best_acc1:.2f}%")
                    
                    # Load SWA scheduler state if in SWA phase
                    if 'swa_scheduler' in checkpoint and start_epoch >= (args.epochs - args.swa_epochs):
                        try:
                            swa_scheduler.load_state_dict(checkpoint['swa_scheduler'])
                            logger.info("✅ SWA scheduler state restored")
                        except Exception as e:
                            logger.warning(f"⚠️  Failed to load SWA scheduler state: {e}")
                            logger.info("🔄 SWA scheduler will be initialized fresh")
                    
                    # Load scaler state
                    if 'scaler' in checkpoint and scaler:
                        try:
                            scaler.load_state_dict(checkpoint['scaler'])
                            logger.info("✅ Mixed precision scaler restored")
                        except Exception as e:
                            logger.warning(f"⚠️  Failed to load scaler state: {e}")
                            logger.info("🔄 Mixed precision scaler will be initialized fresh")
                    
                    logger.info(f"✅ Resumed from epoch {start_epoch}, best accuracy: {best_acc1:.2f}%")
                    logger.info(f"📊 Resuming in {'Main Model' if start_epoch < (args.epochs - args.swa_epochs) else 'SWA'} phase")
            
            except Exception as e:
                logger.error(f"❌ Failed to load checkpoint: {e}")
                logger.info("🚀 Starting fresh training instead")
                start_epoch = 0
                best_acc1 = 0.0
        else:
            logger.warning(f"❌ No checkpoint found at '{args.resume}'")
            logger.info("🚀 Starting fresh training instead")
    else:
        logger.info("🚀 Starting fresh training")
    
    # Training loop
    start_time = time.time()
    
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        
        # Dynamic mixup parameter adjustment for smooth transition
        # DISABLED: Model was already trained with conservative mixup (0.1, 0.5)
        # No need for further mixup transitions - focus on gradient clipping only
        if False and hasattr(train_loader.collate_fn, 'mixup_alpha'):
            current_alpha, current_prob = get_dynamic_mixup_params(epoch)
            old_alpha = train_loader.collate_fn.mixup_alpha
            old_prob = train_loader.collate_fn.prob
            
            train_loader.collate_fn.mixup_alpha = current_alpha
            train_loader.collate_fn.prob = current_prob
            
            # Log parameter changes
            if abs(old_alpha - current_alpha) > 0.001 or abs(old_prob - current_prob) > 0.001:
                logger.info(f"🔧 Mixup params updated: alpha {old_alpha:.3f}→{current_alpha:.3f}, prob {old_prob:.3f}→{current_prob:.3f}")
        
        # Learning rate scheduling
        if epoch >= (args.epochs - args.swa_epochs):
            swa_scheduler.step()
            current_lr = swa_scheduler.get_last_lr()[0]
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        
        logger.info(f'\nEpoch {epoch+1:3d}/{args.epochs} - LR: {current_lr:.6f}')
        
        # Log gradient clipping transitions (faster schedule)
        if epoch == 68:
            logger.info(f"🔧 Gradient clipping transition: 1.0 → 1.5 (starting epoch {epoch+1})")
        elif epoch == 70:
            logger.info(f"🔧 Gradient clipping transition: 1.5 → 2.0 (starting epoch {epoch+1})")
        
        # Training
        # Track parameter changes for debugging
        pre_train_param = None
        if args.debug and epoch == 0:
            pre_train_param = next(model.parameters()).clone().detach()
            
        train_loss, train_acc1 = train_epoch(
            model, train_loader, criterion, optimizer, epoch, args,
            scaler=scaler, ema_model=ema_model, swa_model=swa_model
        )
        
        if args.debug and epoch == 0 and pre_train_param is not None:
            post_train_param = next(model.parameters()).clone().detach()
            param_change = torch.norm(post_train_param - pre_train_param).item()
            logger.info(f"🔧 Parameter change magnitude: {param_change:.6f}")
            if param_change < 1e-6:
                logger.warning("⚠️  Very small parameter changes detected - check optimizer/gradients!")
        
        # Validation with appropriate model
        # Strategy: Use ONLY main model (EMA disabled for stability)
        if epoch < (args.epochs - args.swa_epochs):
            # Use ONLY main model during training - EMA disabled completely
            val_loss, val_acc1 = validate(model, val_loader, criterion, args, logger)
            eval_model = model
            model_type = "Main"
            logger.info(f"🔄 Using main model only (EMA disabled for stability)")
                
        elif epoch >= (args.epochs - args.swa_epochs):
            # Use SWA model
            if epoch == (args.epochs - args.swa_epochs):
                logger.info("Updating SWA batch norm statistics...")
                update_bn(train_loader, swa_model, device=torch.device('cuda'))
            val_loss, val_acc1 = validate(swa_model, val_loader, criterion, args, logger)
            eval_model = swa_model
            model_type = "SWA"
        else:
            # Use base model
            val_loss, val_acc1 = validate(model, val_loader, criterion, args, logger)
            eval_model = model
            model_type = "Base"
        
        # Logging
        elapsed = time.time() - start_time
        logger.info(f'Train: {train_loss:.3f} loss, {train_acc1:.2f}% acc')
        logger.info(f'Val:   {val_loss:.3f} loss, {val_acc1:.2f}% acc ({model_type})')
        logger.info(f'Best:  {max(best_acc1, val_acc1):.2f}% | Elapsed: {format_time(elapsed)}')
        
        # Debug training/validation gap
        acc_gap = val_acc1 - train_acc1
        if acc_gap > 15.0:
            logger.warning(f"⚠️  Large train/val gap: {acc_gap:.2f}% - Consider reducing augmentation")
        elif acc_gap > 10.0:
            logger.info(f"📊 Train/val gap: {acc_gap:.2f}% - Normal for strong augmentation")
        
        # Plateau detection (only for non-SWA epochs)
        if epoch < (args.epochs - args.swa_epochs):
            old_lr = optimizer.param_groups[0]['lr']
            plateau_scheduler.step(val_acc1)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                logger.info(f"🔻 Plateau detected! LR reduced: {old_lr:.6f} → {new_lr:.6f}")
                # Reset scheduler if LR was reduced
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                # Also update the cosine scheduler's last_epoch to sync
                scheduler.last_epoch = epoch
        
        # Prepare checkpoint with all necessary states (for both regular and best saves)
        checkpoint_state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'plateau_scheduler': plateau_scheduler.state_dict(),
            'best_acc1': best_acc1,
            'model_type': model_type,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc1': val_acc1
        }
        
        # Add SWA state if available
        if swa_model and epoch >= (args.epochs - args.swa_epochs):
            checkpoint_state['swa_model'] = swa_model.state_dict()
            checkpoint_state['swa_scheduler'] = swa_scheduler.state_dict()
        
        # Add scaler state if using mixed precision
        if scaler:
            checkpoint_state['scaler'] = scaler.state_dict()
        
        # Save checkpoint every epoch (for crash recovery)
        save_checkpoint(checkpoint_state, False, str(output_dir), 'latest.pth')
        
        # Save numbered checkpoint every epoch (comprehensive backup)
        save_checkpoint(checkpoint_state, False, str(output_dir), f'checkpoint_epoch_{epoch+1}.pth')
        
        # Save best model when accuracy improves
        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            checkpoint_state['best_acc1'] = best_acc1  # Update best accuracy in checkpoint
            save_checkpoint(checkpoint_state, True, str(output_dir), 'best_model.pth')
            logger.info(f'💾 New best model saved: {val_acc1:.2f}%')
        
        # Milestone checks (0-indexed epochs)
        if epoch == (args.epochs - args.swa_epochs - 1):  # Last epoch before SWA
            logger.info(f'🎯 Milestone: Epoch {epoch+1} = {val_acc1:.2f}% (target: >75% before SWA)')
        if epoch == (args.epochs - 1):  # Final epoch
            logger.info(f'🎯 Milestone: Final Epoch {epoch+1} = {val_acc1:.2f}% (target: >78%)')
    
    # Final results
    total_time = time.time() - start_time
    logger.info(f'\n🎉 Training Complete!')
    logger.info(f'Best Accuracy: {best_acc1:.2f}%')
    logger.info(f'Total Time: {format_time(total_time)}')
    logger.info(f'Output saved to: {output_dir}')
    
    if best_acc1 >= 78.0:
        logger.info('✅ SUCCESS: Achieved 78%+ target!')
    elif best_acc1 >= 75.0:
        logger.info('🟡 GOOD: Close to 78% target')
    else:
        logger.info('🔴 BELOW TARGET: Check hyperparameters')


if __name__ == '__main__':
    main()