"""
Production-ready ResNet50 ImageNet training script.
Simplified and stabilized for consistent training from epoch 72+.

Usage:
    python train.py --data /path/to/imagenet --output-dir ./outputs --resume ./outputs/latest.pth
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
from src.mixup import MixupCutmixCollator, mixup_criterion
from src.utils import (
    AverageMeter, accuracy, seed_everything, save_checkpoint,
    get_device, format_time, setup_logging
)
from src.ema import EMAModel


def parse_args():
    parser = argparse.ArgumentParser(description='ResNet50 ImageNet Training')

    # Required
    parser.add_argument('--data', type=str, required=True,
                        help='Path to ImageNet dataset (with train/ and val/ folders)')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=90,
                        help='Total training epochs (default: 90)')
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
    parser.add_argument('--swa-epochs', type=int, default=10,
                        help='Use SWA for last N epochs (default: 10)')
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

    # Debug mode
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional logging')

    return parser.parse_args()


def build_optimizer_scheduler(model, args):
    """Build optimizer and cosine scheduler"""
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
                scaler=None, swa_model=None, logger=None):
    """Single training epoch with simplified gradient clipping"""
    model.train()
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')

    use_swa = epoch >= (args.epochs - args.swa_epochs)
    strategy = "SWA" if use_swa else "Main"

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}/{args.epochs} ({strategy})')

    for i, batch_data in enumerate(pbar):
        # Handle Mixup/CutMix data format
        if len(batch_data) == 4:  # Mixup/CutMix applied
            images, targets_a, targets_b, lam = batch_data
            images = images.cuda(non_blocking=True)
            targets_a = targets_a.cuda(non_blocking=True)
            targets_b = targets_b.cuda(non_blocking=True)
            mixup_active = True
            targets = targets_a  # For metrics
        else:  # Normal batch
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
                if mixup_active:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            if mixup_active:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)

        # Debug: Print first batch predictions for verification
        if i == 0 and args.debug and logger:
            pred_classes = outputs.argmax(dim=1)[:5]
            actual_classes = targets[:5]
            logger.info(f"Sample predictions: {pred_classes.cpu().tolist()}")
            logger.info(f"Sample targets: {actual_classes.cpu().tolist()}")

        # Safety check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"⚠️ NaN/Inf loss detected at epoch {epoch+1}, batch {i}") # type: ignore
            logger.error("Skipping this batch to prevent divergence") # type: ignore
            optimizer.zero_grad()
            continue

        # Backward pass with gradient clipping
        if args.amp and scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Update SWA model if in SWA phase
        if use_swa and swa_model:
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


def validate(model, val_loader, criterion, args, logger=None):
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

    # Validation
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
    logger.info("Target: 78% top-1 accuracy")

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
    optimizer, scheduler = build_optimizer_scheduler(model, args)

    # SWA model
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)

    # Mixed precision - ALWAYS INITIALIZE FRESH (don't load from checkpoint)
    scaler = GradScaler() if args.amp else None

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc1 = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            try:
                logger.info(f"Loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume, map_location='cpu')

                # Validate checkpoint
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

                    # Load SWA state if available
                    if 'swa_model' in checkpoint and swa_model:
                        try:
                            swa_model.load_state_dict(checkpoint['swa_model'])
                            logger.info("✅ SWA model state restored")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load SWA model state: {e}")

                    # Load scheduler state - SIMPLIFIED APPROACH
                    if 'scheduler' in checkpoint:
                        try:
                            # Always load the scheduler state first
                            scheduler.load_state_dict(checkpoint['scheduler'])

                            # Check if total epochs changed
                            checkpoint_epochs = checkpoint.get('total_epochs', 90)
                            if checkpoint_epochs != args.epochs:
                                # Adjust T_max for new total epochs
                                scheduler.T_max = args.epochs
                                logger.info(f"✅ Scheduler loaded and T_max adjusted: {checkpoint_epochs} → {args.epochs}")

                                # CRITICAL: Reset optimizer momentum buffers to prevent divergence
                                logger.info("🔄 Resetting optimizer momentum buffers for stability")
                                momentum_reset_count = 0
                                for param in model.parameters():
                                    if param in optimizer.state:
                                        param_state = optimizer.state[param]
                                        if 'momentum_buffer' in param_state:
                                            param_state['momentum_buffer'].zero_()
                                            momentum_reset_count += 1
                                logger.info(f"✅ Reset {momentum_reset_count} momentum buffers")
                            else:
                                logger.info("✅ Scheduler state restored (epochs unchanged)")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load scheduler state: {e}")
                    else:
                        logger.warning("⚠️ No scheduler state in checkpoint")

                    # Load SWA scheduler if in SWA phase
                    if 'swa_scheduler' in checkpoint and start_epoch >= (args.epochs - args.swa_epochs):
                        try:
                            swa_scheduler.load_state_dict(checkpoint['swa_scheduler'])
                            logger.info("✅ SWA scheduler state restored")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load SWA scheduler state: {e}")

                    # IMPORTANT: Do NOT load scaler state - reinitialize fresh
                    logger.info("🔄 Mixed precision scaler initialized fresh (not loaded from checkpoint)")

                    # Fallbacking to original augmentation parameters values
                    logger.info("🔄 Augmentation parameters fallback to original values of 0.2 and 0.8")

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

    # Track parameter changes for debugging
    pre_train_param = None
    if args.debug and start_epoch >= 0:
        pre_train_param = next(model.parameters()).clone().detach()

    for epoch in range(start_epoch, args.epochs):
        # Get current LR before training
        if epoch >= (args.epochs - args.swa_epochs):
            current_lr = swa_scheduler.get_last_lr()[0]
        else:
            current_lr = scheduler.get_last_lr()[0]


        logger.info(f'\nEpoch {epoch+1:3d}/{args.epochs} - LR: {current_lr:.6f}')

        # Training
        train_loss, train_acc1 = train_epoch(
            model, train_loader, criterion, optimizer, epoch, args,
            scaler=scaler, swa_model=swa_model, logger=logger
        )

        # Check parameter changes for first epoch only
        if args.debug and epoch == start_epoch and pre_train_param is not None:
            post_train_param = next(model.parameters()).clone().detach()
            param_change = torch.norm(post_train_param - pre_train_param).item()
            logger.info(f"🔧 Parameter change magnitude: {param_change:.6f}")
            if param_change < 1e-6:
                logger.warning("⚠️  Very small parameter changes detected - check optimizer/gradients!")

        # Validation with appropriate model
        if epoch < (args.epochs - args.swa_epochs):
            # Use main model
            val_loss, val_acc1 = validate(model, val_loader, criterion, args, logger=logger)
            eval_model = model
            model_type = "Main"
        
        elif epoch < args.epochs - 1:
            # SWA accumulation (epochs 91-99) - validate main model
            val_loss, val_acc1 = validate(model, val_loader, criterion, args, logger=logger)
            eval_model = model
            model_type = "Main (SWA accumulating)"
            logger.info("📊 SWA model weights being averaged - final evaluation at epoch 100")
    
        else:
            # Final epoch (100) - update BN once and validate SWA
            logger.info("=" * 80)
            logger.info("🎯 Final SWA Batch Normalization Update")
            logger.info("=" * 80)
            update_bn(train_loader, swa_model, device=torch.device('cuda'))
            
            val_loss, val_acc1 = validate(swa_model, val_loader, criterion, args, logger=logger)
            eval_model = swa_model
            model_type = "SWA (final)"
            
            logger.info("=" * 80)
            logger.info(f"🎉 Final SWA Model Accuracy: {val_acc1:.2f}%")
            logger.info("=" * 80)

        # Logging
        elapsed = time.time() - start_time
        logger.info(f'Train: {train_loss:.3f} loss, {train_acc1:.2f}% acc')
        logger.info(f'Val:   {val_loss:.3f} loss, {val_acc1:.2f}% acc ({model_type})')
        logger.info(f'Best:  {max(best_acc1, val_acc1):.2f}% | Elapsed: {format_time(elapsed)}')

        # Log train/val gap
        acc_gap = val_acc1 - train_acc1
        if acc_gap > 15:
            logger.info(f'📊 Train/val gap: {acc_gap:.2f}% - High gap (strong augmentation)')
        else:
            logger.info(f'📊 Train/val gap: {acc_gap:.2f}% - Normal for strong augmentation')

        # Prepare checkpoint
        checkpoint_state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc1': best_acc1,
            'total_epochs': args.epochs,  
            'model_type': model_type,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc1': val_acc1
        }

        # Add SWA state if available
        if swa_model and epoch >= (args.epochs - args.swa_epochs):
            checkpoint_state['swa_model'] = swa_model.state_dict()
            checkpoint_state['swa_scheduler'] = swa_scheduler.state_dict()

        # Add scaler state (for reference, though we won't load it)
        if scaler:
            checkpoint_state['scaler'] = scaler.state_dict()

        # Save checkpoint every epoch
        save_checkpoint(checkpoint_state, False, str(output_dir), 'latest.pth')

        # Save numbered checkpoint
        save_checkpoint(checkpoint_state, False, str(output_dir), f'checkpoint_epoch_{epoch+1}.pth')

        # Save best model
        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            checkpoint_state['best_acc1'] = best_acc1
            save_checkpoint(checkpoint_state, True, str(output_dir), 'best_model.pth')
            logger.info(f'💾 New best model saved: {val_acc1:.2f}%')

        # Step scheduler AFTER epoch completes
        if epoch >= (args.epochs - args.swa_epochs):
            swa_scheduler.step()
        else:
            scheduler.step()

    # Final results
    total_time = time.time() - start_time
    logger.info(f'\n🎉 Training Complete!')
    logger.info(f'Best Accuracy: {best_acc1:.2f}%')
    logger.info(f'Total Time: {format_time(total_time)}')
    logger.info(f'Output saved to: {output_dir}')

    if best_acc1 >= 78.0:
        logger.info('✅ SUCCESS: Achieved 78% target!')
    elif best_acc1 >= 76.0:
        logger.info('🟡 CLOSE: Almost at target')
    else:
        logger.info('🔴 BELOW TARGET: Continue training or adjust configuration')


if __name__ == '__main__':
    main()
