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
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with additional logging')
    
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
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    
                    # Load EMA state if available with decay override
                    if 'ema_model' in checkpoint and ema_model:
                        ema_checkpoint = checkpoint['ema_model']
                        old_decay = ema_checkpoint.get('decay', 0.999)
                        
                        # Load EMA state
                        ema_model.load_state_dict(ema_checkpoint)
                        logger.info("✅ EMA model state restored")
                        
                        # Only override decay if significantly different (don't interfere with internal logic)
                        if abs(old_decay - args.ema_decay) > 0.001:
                            logger.info(f"🔧 EMA decay overridden: {old_decay:.4f} → {args.ema_decay:.4f}")
                            ema_model.decay = args.ema_decay
                        
                        # Smart EMA health check - only reinitialize if actually broken
                        logger.info("🔍 Performing EMA health check...")
                        try:
                            ema_model.model.eval()
                            with torch.no_grad():
                                # Get a small validation batch for health check
                                val_iter = iter(val_loader)
                                batch_data = next(val_iter)
                                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                                    images = batch_data[0].to(device)
                                    targets = batch_data[1].to(device)
                                    
                                    # Test EMA predictions on real validation data
                                    test_images = images[:min(32, images.size(0))]
                                    ema_outputs = ema_model.model(test_images)
                                    ema_probs = torch.softmax(ema_outputs, dim=1)
                                    ema_confidence = ema_probs.max(dim=1)[0].mean().item()
                                    
                                    logger.info(f"📊 EMA health check - Average confidence: {ema_confidence:.4f}")
                                    logger.info(f"📊 EMA has {ema_model.num_updates} accumulated updates")
                                    
                                    # Updated health check for new adaptive decay EMA
                                    if ema_confidence < 0.01 or ema_model.num_updates < 100:
                                        logger.warning("⚠️  EMA model appears corrupted or insufficient updates")
                                        logger.info("🔄 Reinitializing EMA from current main model...")
                                        ema_model = EMAModel(model, decay=args.ema_decay)
                                        # Force some rapid updates to jumpstart EMA
                                        logger.info("🚀 Performing rapid EMA initialization...")
                                        for _ in range(20):
                                            ema_model.update(model)
                                        logger.info("✅ EMA model reinitialized with healthy weights")
                                    else:
                                        logger.info("✅ EMA model health check passed - keeping existing EMA")
                                else:
                                    # Fallback: skip health check if batch structure is unexpected
                                    logger.warning("⚠️  Unexpected validation batch structure - skipping detailed health check")
                                    logger.info("✅ EMA model health check skipped - keeping existing EMA")
                                
                        except Exception as e:
                            logger.warning(f"⚠️  EMA health check failed: {e}")
                            logger.info("🔄 Reinitializing EMA from current main model as fallback...")
                            ema_model = EMAModel(model, decay=args.ema_decay)
                            # Force some rapid updates to jumpstart EMA
                            for _ in range(20):
                                ema_model.update(model)
                            logger.info("✅ EMA model reinitialized with healthy weights")
                    
                    # Load SWA state if available
                    if 'swa_model' in checkpoint and swa_model:
                        swa_model.load_state_dict(checkpoint['swa_model'])
                        logger.info("✅ SWA model state restored")
                    
                    # Load scheduler state (regular scheduler)
                    if 'scheduler' in checkpoint:
                        scheduler.load_state_dict(checkpoint['scheduler'])
                        logger.info("✅ Scheduler state restored")
                    
                    # Load SWA scheduler state if in SWA phase
                    if 'swa_scheduler' in checkpoint and start_epoch >= (args.epochs - args.swa_epochs):
                        swa_scheduler.load_state_dict(checkpoint['swa_scheduler'])
                        logger.info("✅ SWA scheduler state restored")
                    
                    # Load scaler state
                    if 'scaler' in checkpoint and scaler:
                        scaler.load_state_dict(checkpoint['scaler'])
                        logger.info("✅ Mixed precision scaler restored")
                    
                    logger.info(f"✅ Resumed from epoch {start_epoch}, best accuracy: {best_acc1:.2f}%")
                    logger.info(f"📊 Resuming in {'EMA' if start_epoch < args.ema_epochs else 'SWA' if start_epoch >= (args.epochs - args.swa_epochs) else 'Base'} phase")
            
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
        
        # Learning rate scheduling
        if epoch >= (args.epochs - args.swa_epochs):
            swa_scheduler.step()
            current_lr = swa_scheduler.get_last_lr()[0]
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        
        logger.info(f'\nEpoch {epoch+1:3d}/{args.epochs} - LR: {current_lr:.6f}')
        
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
        # Strategy: Use EMA after minimal warmup (new adaptive decay makes EMA ready much faster)
        if epoch < args.ema_epochs:
            # During EMA phase: validate with both models to track progress
            main_val_loss, main_val_acc1 = validate(model, val_loader, criterion, args, logger)
            
            # With new adaptive decay, EMA is ready much sooner (just need basic warmup)
            ema_ready = (ema_model.num_updates > 500 and epoch >= 2)  # Much earlier readiness
            
            if ema_ready:
                ema_val_loss, ema_val_acc1 = validate(ema_model.model, val_loader, criterion, args, logger)
                
                # More lenient performance check - EMA can temporarily underperform during adaptation
                if ema_val_acc1 >= (main_val_acc1 - 10.0):  # Allow 10% gap during adaptation
                    val_loss, val_acc1 = ema_val_loss, ema_val_acc1
                    eval_model = ema_model.model
                    model_type = "EMA"
                    logger.info(f"📊 EMA updates: {ema_model.num_updates}")
                    logger.info(f"✅ Using EMA model (EMA: {ema_val_acc1:.2f}% vs Main: {main_val_acc1:.2f}%)")
                else:
                    val_loss, val_acc1 = main_val_loss, main_val_acc1
                    eval_model = model
                    model_type = "Main (EMA adapting)"
                    logger.info(f"🔄 Using main model - EMA adapting (EMA: {ema_val_acc1:.2f}% vs Main: {main_val_acc1:.2f}%)")
                    logger.info(f"📊 EMA updates: {ema_model.num_updates}")
            else:
                val_loss, val_acc1 = main_val_loss, main_val_acc1
                eval_model = model
                model_type = "Main (EMA warmup)"
                warmup_reason = f"updates: {ema_model.num_updates}" if ema_model.num_updates <= 500 else f"epoch: {epoch+1}/3"
                logger.info(f"🔄 Using main model for validation (EMA warmup - {warmup_reason})")
                
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
        
        # Prepare checkpoint with all necessary states (for both regular and best saves)
        checkpoint_state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc1': best_acc1,
            'model_type': model_type,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc1': val_acc1
        }
        
        # Add EMA state if available
        if ema_model:
            checkpoint_state['ema_model'] = ema_model.state_dict()
        
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
        if epoch == 79:  # This is epoch 80 (EMA phase end)
            logger.info(f'🎯 Milestone: Epoch 80 = {val_acc1:.2f}% (target: >75%)')
        if epoch == 89:  # This is epoch 90 (training end)
            logger.info(f'🎯 Milestone: Epoch 90 = {val_acc1:.2f}% (target: >78%)')
    
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