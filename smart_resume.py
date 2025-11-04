#!/usr/bin/env python3
"""
Smart resume script that finds the best checkpoint to resume from.
Handles multiple checkpoint types: latest.pth, best_model.pth, numbered checkpoints.
"""

import torch
import os
import sys
from pathlib import Path
import glob

def find_all_checkpoints():
    """Find all available checkpoints in outputs directory."""
    checkpoint_dir = Path('./outputs')
    
    if not checkpoint_dir.exists():
        return {}
    
    checkpoints = {}
    
    # Check for latest checkpoint (most recent epoch)
    latest_path = checkpoint_dir / 'latest.pth'
    if latest_path.exists():
        try:
            checkpoint = torch.load(latest_path, map_location='cpu')
            checkpoints['latest'] = {
                'path': str(latest_path),
                'epoch': checkpoint['epoch'],
                'accuracy': checkpoint['best_acc1'],
                'val_acc': checkpoint.get('val_acc1', 0.0),
                'type': 'Latest (auto-save)'
            }
        except Exception as e:
            print(f"⚠️  Could not load latest.pth: {e}")
    
    # Check for best model
    best_path = checkpoint_dir / 'best_model.pth'
    if best_path.exists():
        try:
            checkpoint = torch.load(best_path, map_location='cpu')
            checkpoints['best'] = {
                'path': str(best_path),
                'epoch': checkpoint['epoch'],
                'accuracy': checkpoint['best_acc1'],
                'val_acc': checkpoint.get('val_acc1', 0.0),
                'type': 'Best model'
            }
        except Exception as e:
            print(f"⚠️  Could not load best_model.pth: {e}")
    
    # Check for numbered checkpoints
    numbered_checkpoints = glob.glob(str(checkpoint_dir / 'checkpoint_epoch_*.pth'))
    for ckpt_path in numbered_checkpoints:
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            epoch_num = checkpoint['epoch']
            checkpoints[f'epoch_{epoch_num}'] = {
                'path': ckpt_path,
                'epoch': checkpoint['epoch'],
                'accuracy': checkpoint['best_acc1'],
                'val_acc': checkpoint.get('val_acc1', 0.0),
                'type': f'Epoch {epoch_num + 1} checkpoint'
            }
        except Exception as e:
            print(f"⚠️  Could not load {ckpt_path}: {e}")
    
    return checkpoints

def analyze_checkpoints():
    """Analyze all checkpoints and recommend the best one to resume from."""
    checkpoints = find_all_checkpoints()
    
    if not checkpoints:
        print("❌ No checkpoints found")
        print("🚀 Starting fresh training:")
        print_fresh_command()
        return
    
    print("✅ Found checkpoints:")
    print("=" * 80)
    
    latest_epoch = -1
    best_accuracy = -1
    recommended_checkpoint = None
    
    # Display all checkpoints
    for key, info in checkpoints.items():
        status = ""
        if info['epoch'] > latest_epoch:
            latest_epoch = info['epoch']
            recommended_checkpoint = info
        if info['accuracy'] > best_accuracy:
            best_accuracy = info['accuracy']
        
        print(f"📁 {info['type']}")
        print(f"   Path: {info['path']}")
        print(f"   Epoch: {info['epoch'] + 1}/90")
        print(f"   Best Accuracy: {info['accuracy']:.3f}%")
        print(f"   Current Val Acc: {info['val_acc']:.3f}%")
        print()
    
    print("=" * 80)
    
    if recommended_checkpoint:
        print(f"🎯 RECOMMENDED: Resume from latest checkpoint")
        print(f"   Epoch: {recommended_checkpoint['epoch'] + 1}/90")
        print(f"   Best Accuracy: {recommended_checkpoint['accuracy']:.3f}%")
        print(f"   Progress: {(recommended_checkpoint['epoch'] + 1)/90*100:.1f}%")
        
        remaining_epochs = 90 - (recommended_checkpoint['epoch'] + 1)
        print(f"   Remaining: {remaining_epochs} epochs")
        
        print("\n🔄 Resume command:")
        print_resume_command(recommended_checkpoint['path'])
        
        # Check for potential issues
        if recommended_checkpoint['val_acc'] == 0.0 or recommended_checkpoint['accuracy'] < 1.0:
            print("\n⚠️  WARNING: Low accuracy detected - training might have issues")
            print("   Consider starting fresh with different hyperparameters")
    
def print_fresh_command():
    """Print command for fresh training."""
    print("""
python train.py \\
  --data /mnt/nvme_data/imagenet \\
  --output-dir ./outputs \\
  --epochs 90 \\
  --batch-size 256 \\
  --lr 0.1 \\
  --ema-epochs 80 \\
  --swa-epochs 10 \\
  --workers 8 \\
  --amp \\
  2>&1 | tee training.log
""")

def print_resume_command(checkpoint_path):
    """Print command for resuming training."""
    print(f"""
python train.py \\
  --data /mnt/nvme_data/imagenet \\
  --output-dir ./outputs \\
  --epochs 90 \\
  --batch-size 256 \\
  --lr 0.1 \\
  --ema-epochs 80 \\
  --swa-epochs 10 \\
  --workers 8 \\
  --amp \\
  --resume {checkpoint_path} \\
  2>&1 | tee -a training.log
""")

def check_checkpoint_health(checkpoint_path):
    """Check if checkpoint contains valid data."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"🏥 Checkpoint Health Check: {checkpoint_path}")
        print("-" * 50)
        
        # Check basic structure
        required_keys = ['epoch', 'model', 'optimizer', 'best_acc1']
        for key in required_keys:
            if key in checkpoint:
                print(f"✅ {key}: Present")
            else:
                print(f"❌ {key}: Missing")
        
        # Check for NaN values in accuracy
        if 'best_acc1' in checkpoint:
            acc = checkpoint['best_acc1']
            if torch.isnan(torch.tensor(acc)) or acc == float('inf'):
                print(f"❌ best_acc1: Contains NaN/Inf ({acc})")
            else:
                print(f"✅ best_acc1: Valid ({acc:.3f}%)")
        
        # Check val accuracy if available
        if 'val_acc1' in checkpoint:
            val_acc = checkpoint['val_acc1']
            if torch.isnan(torch.tensor(val_acc)) or val_acc == float('inf'):
                print(f"❌ val_acc1: Contains NaN/Inf ({val_acc})")
            else:
                print(f"✅ val_acc1: Valid ({val_acc:.3f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint corrupted: {e}")
        return False

if __name__ == "__main__":
    print("🔍 SMART CHECKPOINT ANALYZER")
    print("=" * 80)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--health-check':
        # Health check mode
        checkpoints = find_all_checkpoints()
        for key, info in checkpoints.items():
            check_checkpoint_health(info['path'])
            print()
    else:
        # Normal analysis mode
        analyze_checkpoints()