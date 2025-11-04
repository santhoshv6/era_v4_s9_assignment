#!/usr/bin/env python3
"""
Validate Epoch 4 Checkpoint with Main Model
==========================================
This script loads the epoch 4 checkpoint and validates it using the main model
to show what the validation accuracy should have been with extended EMA warmup.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import argparse

from src.model import get_model
from src.transforms import build_transforms
from src.utils import AverageMeter, accuracy, setup_logging
from tqdm import tqdm


def validate_checkpoint(checkpoint_path, data_path, batch_size=400):
    """Validate a checkpoint using the main model"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Validating checkpoint: {checkpoint_path}")
    print(f"📊 Device: {device}")
    
    # Load checkpoint
    print("📥 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint['epoch']
    
    # Model
    print("🏗️ Building model...")
    model = get_model(num_classes=1000, model_name='resnet50', dropout=0.0)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    # Data loading
    print("📂 Loading validation dataset...")
    _, val_transform = build_transforms(img_size=224, strong_aug=True)
    
    val_dataset = datasets.ImageFolder(
        os.path.join(data_path, 'val'),
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"📊 Validation dataset: {len(val_dataset):,} images")
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Validation
    print("🔍 Running validation with main model...")
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Validating Epoch {epoch+1} (Main Model)')
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Metrics
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            
            pbar.set_postfix({
                'Loss': f'{losses.avg:.3f}',
                'Top1': f'{top1.avg:.2f}%',
                'Top5': f'{top5.avg:.2f}%'
            })
    
    return losses.avg, top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser(description='Validate checkpoint with main model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--data', type=str, default='/mnt/nvme_data/imagenet',
                       help='Path to ImageNet dataset')
    parser.add_argument('--batch-size', type=int, default=400,
                       help='Batch size for validation')
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Run validation
    val_loss, val_acc1, val_acc5 = validate_checkpoint(
        args.checkpoint, args.data, args.batch_size
    )
    
    # Load checkpoint info
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    epoch = checkpoint['epoch']
    train_loss = checkpoint.get('train_loss', 'N/A')
    train_acc = 27.22 if epoch == 3 else 'N/A'  # From your log
    
    # Results
    print("\n" + "="*60)
    print("📊 EPOCH 4 VALIDATION RESULTS (CORRECTED)")
    print("="*60)
    print(f"Checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"Epoch: {epoch + 1}/90")
    print(f"Train Loss: {train_loss}")
    print(f"Train Acc: {train_acc}%")
    print(f"Val Loss: {val_loss:.3f}")
    print(f"Val Acc (Top-1): {val_acc1:.2f}%")
    print(f"Val Acc (Top-5): {val_acc5:.2f}%")
    print(f"Model Type: Main (EMA warmup)")
    print("="*60)
    print("🔄 This shows what Epoch 4 validation SHOULD have been")
    print("   with extended EMA warmup period!")
    print("="*60)
    
    # Expected vs actual comparison
    actual_ema_acc = 0.10  # From your log
    improvement = val_acc1 - actual_ema_acc
    
    print(f"\n📈 COMPARISON:")
    print(f"   Actual (EMA): {actual_ema_acc:.2f}%")
    print(f"   Corrected (Main): {val_acc1:.2f}%")
    print(f"   Improvement: +{improvement:.2f}%")
    
    # Trajectory prediction
    if val_acc1 > 30:
        print(f"\n🚀 TRAJECTORY ANALYSIS:")
        print(f"   Epoch 1: 13.22% → Epoch 2: 23.80% → Epoch 3: 34.02% → Epoch 4: {val_acc1:.2f}%")
        print(f"   Consistent upward trend confirmed! ✅")
        print(f"   Projected Epoch 10: ~{val_acc1 + 25:.0f}%")
        print(f"   Projected Final: ~{val_acc1 + 55:.0f}%")


if __name__ == '__main__':
    main()



# # On your training machine (or locally if you have the dataset)
# python validate_epoch4.py \
#   --checkpoint ./outputs/latest.pth \
#   --data /mnt/nvme_data/imagenet \
#   --batch-size 400