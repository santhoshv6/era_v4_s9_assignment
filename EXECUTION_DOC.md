# 📋 EXECUTION DOCUMENT
## ResNet50 ImageNet Training - Complete Execution Guide

**Date**: November 2, 2025  
**Target**: 81% Top-1 Accuracy on ImageNet  
**Budget**: ~$25 on AWS EC2 Spot Instances  
**Timeline**: 4-5 days (96 hours training)

---

## 📊 PROJECT OVERVIEW

### 🎯 **Objective**
Train ResNet50 from scratch on full ImageNet dataset to achieve 81% top-1 accuracy using AWS EC2 spot instances with optimal cost efficiency.

### 🔬 **Strategy**
- **Training Technique**: EMA (epochs 1-100) + SWA (epochs 101-120)
- **Learning Rate**: Cosine Annealing with warmup
- **Augmentation**: Progressive Mixup + RandAugment
- **Optimization**: Mixed Precision + Gradient Clipping

### 💰 **Budget Analysis**
- **Instance**: g4dn.2xlarge (T4 GPU, 32GB RAM)
- **Spot Price**: ~$0.264/hour
- **Training Time**: 96 hours
- **Total Cost**: $25.34 + storage (~$26 total)

---

## 🚀 EXECUTION PHASES

### **PHASE 1: ENVIRONMENT SETUP**

#### ✅ **Step 1.1: Connect to EC2 Instance**
```bash
# Connect via SSH (replace with your instance details)
ssh -i your-key.pem ubuntu@your-instance-ip

# Verify instance details
lscpu | grep "Model name"
nvidia-smi  # Should show GPU details
df -h       # Check available storage
```

#### ✅ **Step 1.2: System Update**
```bash
# Update system packages
sudo apt update -y
sudo apt upgrade -y

# Install essential packages
sudo apt install -y build-essential git wget curl htop nvtop tree unzip python3-pip
```

#### ✅ **Step 1.3: Install Miniconda**
```bash
# Download and install Miniconda
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Add to PATH and initialize
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda init bash

# Restart shell or source again
source ~/.bashrc
```

#### ✅ **Step 1.4: Create Python Environment**
```bash
# Create PyTorch environment
conda create -n pytorch_env python=3.9 -y
conda activate pytorch_env

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install additional packages
pip install datasets huggingface_hub tqdm pillow numpy scikit-learn tensorboard

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

#### ✅ **Step 1.5: Setup Storage**
```bash
# Check for NVMe SSD
lsblk

# If you have additional NVMe storage, mount it
sudo mkdir -p /mnt/nvme_data

# For additional NVMe device (if available)
# sudo mkfs.ext4 /dev/nvme1n1
# sudo mount /dev/nvme1n1 /mnt/nvme_data
# sudo chown $USER:$USER /mnt/nvme_data

# If using root storage only
sudo chown $USER:$USER /mnt/nvme_data
```

#### ✅ **Step 1.6: Install tmux**
```bash
# Install tmux for session management
sudo apt install tmux -y

# Verify tmux installation
tmux -V
```

### **PHASE 2: PROJECT SETUP & DATA PREPARATION**

#### ✅ **Step 2.1: Setup Project Directory**
```bash
# Create project directory
mkdir -p /mnt/nvme_data/imagenet_training
cd /mnt/nvme_data/imagenet_training

# Create subdirectories
mkdir -p outputs logs checkpoints

# Upload your training code here (using scp or git)
# You should have: train.py, src/, requirements.txt, download_imagenet_hf.py
```

#### ✅ **Step 2.2: Login to Hugging Face**
```bash
# Install Hugging Face CLI if not already installed
pip install huggingface_hub[cli]

# Login to Hugging Face (you'll need your token)
huggingface-cli login
# Enter your HF token when prompted
```

#### ✅ **Step 2.3: Download ImageNet Dataset**
```bash
# Navigate to project directory
cd /mnt/nvme_data/imagenet_training

# Start ImageNet download (this takes 1-3 hours)
python download_imagenet_hf.py \
  --output-dir /mnt/nvme_data/imagenet \
  --num-workers 8 \
  --chunk-size 1000 \
  --quality 95 \
  --verify

# For faster download (if system can handle it)
# python download_imagenet_hf.py \
#   --output-dir /mnt/nvme_data/imagenet \
#   --num-workers 16 \
#   --chunk-size 2000 \
#   --skip-existing

# Monitor download progress
# The script shows progress bars and statistics
```

#### ✅ **Step 2.4: Verify Dataset**
```bash
# Check dataset structure
ls /mnt/nvme_data/imagenet/
# Should show: train/ and val/

# Count classes and images
echo "Train classes: $(ls /mnt/nvme_data/imagenet/train/ | wc -l)"
echo "Val classes: $(ls /mnt/nvme_data/imagenet/val/ | wc -l)"
echo "Train images: $(find /mnt/nvme_data/imagenet/train/ -name "*.JPEG" | wc -l)"
echo "Val images: $(find /mnt/nvme_data/imagenet/val/ -name "*.JPEG" | wc -l)"

# Check total size
du -sh /mnt/nvme_data/imagenet
# Should be ~150GB

# Quick test on a few images
ls /mnt/nvme_data/imagenet/train/ | head -5
ls /mnt/nvme_data/imagenet/train/$(ls /mnt/nvme_data/imagenet/train/ | head -1)/ | head -3
```

#### ✅ **Step 2.5: Pre-Flight Checklist**
```bash
# Complete system verification before training
echo "🔍 PRE-FLIGHT CHECKLIST"
echo "======================="

# 1. Check GPU
nvidia-smi && echo "✅ GPU detected" || echo "❌ GPU not found"

# 2. Check disk space (need >200GB free)
DISK_FREE=$(df /mnt/nvme_data | tail -1 | awk '{print $4}')
if [ $DISK_FREE -gt 200000000 ]; then
    echo "✅ Sufficient disk space: $(df -h /mnt/nvme_data | tail -1 | awk '{print $4}') available"
else
    echo "❌ Insufficient disk space: $(df -h /mnt/nvme_data | tail -1 | awk '{print $4}') available"
fi

# 3. Check conda environment
conda activate pytorch_env && echo "✅ Conda environment activated" || echo "❌ Conda environment issue"

# 4. Check Python imports
python -c "
import torch
import torchvision
from src.model import get_model
from src.transforms import build_transforms
from src.mixup import MixupCutmixCollator
from src.ema import EMAModel
from src.utils import AverageMeter, accuracy
print('✅ All imports successful')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
" 2>/dev/null && echo "✅ Python environment ready" || echo "❌ Python import issues"

# 5. Check ImageNet dataset
if [ -d "/mnt/nvme_data/imagenet/train" ] && [ -d "/mnt/nvme_data/imagenet/val" ]; then
    TRAIN_CLASSES=$(ls /mnt/nvme_data/imagenet/train/ | wc -l)
    VAL_CLASSES=$(ls /mnt/nvme_data/imagenet/val/ | wc -l)
    if [ $TRAIN_CLASSES -eq 1000 ] && [ $VAL_CLASSES -eq 1000 ]; then
        echo "✅ ImageNet dataset complete (1000 classes each)"
    else
        echo "❌ ImageNet dataset incomplete: $TRAIN_CLASSES train, $VAL_CLASSES val classes"
    fi
else
    echo "❌ ImageNet dataset not found"
fi

# 6. Check HuggingFace authentication
huggingface-cli whoami >/dev/null 2>&1 && echo "✅ HuggingFace authenticated" || echo "❌ HuggingFace login required"

# 7. Test training script syntax
python -m py_compile train.py && echo "✅ Training script syntax valid" || echo "❌ Training script has syntax errors"

echo "======================="
echo "🚀 Ready for training if all items show ✅"
echo "Fix any ❌ issues before proceeding to Phase 3"
```

### **PHASE 3: TRAINING EXECUTION**

### **PHASE 3: TRAINING SETUP & EXECUTION**

#### ✅ **Step 3.1: Create tmux Sessions**
```bash
# Create main training session
tmux new-session -d -s training

# Create monitoring session
tmux new-session -d -s monitoring

# Create dashboard session for comprehensive monitoring
tmux new-session -d -s dashboard

# List all sessions
tmux list-sessions
```

#### ✅ **Step 3.2: Setup Training Environment**
```bash
# Attach to training session
tmux attach-session -t training

# Navigate to project directory
cd /mnt/nvme_data/imagenet_training

# Activate conda environment
conda activate pytorch_env

# Verify everything is ready
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
"

# Check training script
ls -la train.py src/
```

#### ✅ **Step 3.3: Start Training**
```bash
# In the training tmux session, start training
python train.py \
  --data /mnt/nvme_data/imagenet \
  --output-dir ./outputs \
  --epochs 120 \
  --batch-size 256 \
  --ema-epochs 100 \
  --swa-epochs 20 \
  2>&1 | tee training.log

# The training will show progress and save logs
# Detach from session: Ctrl+B, then D
# Training continues in background
```

#### ✅ **Step 3.4: Setup Monitoring Dashboard**
```bash
# Detach from training session (Ctrl+B, D)
# Attach to dashboard session
tmux attach-session -t dashboard

# Create 4-pane monitoring dashboard
tmux split-window -h    # Split horizontally
tmux split-window -v    # Split top-right vertically
tmux select-pane -t 0   # Select top-left
tmux split-window -v    # Split top-left vertically

# Setup each pane:
# Pane 0 (top-left): Training logs
tail -f /mnt/nvme_data/imagenet_training/training.log

# Switch to pane 1 (bottom-left): Ctrl+B, arrow keys
# Pane 1: GPU monitoring
watch -n 5 nvidia-smi

# Switch to pane 2 (top-right)
# Pane 2: System resources
htop

# Switch to pane 3 (bottom-right)
# Pane 3: Training progress
watch -n 60 'cd /mnt/nvme_data/imagenet_training && python -c "
import torch
import os
if os.path.exists(\"./outputs/best_model.pth\"):
    checkpoint = torch.load(\"./outputs/best_model.pth\", map_location=\"cpu\")
    print(f\"Best Accuracy: {checkpoint[\"best_acc1\"]:.2f}%\")
    print(f\"Epoch: {checkpoint[\"epoch\"]}\")
    print(f\"Model Type: {checkpoint.get(\"model_type\", \"Unknown\")}\")
else:
    print(\"No checkpoint found yet\")
"'

# Detach from dashboard: Ctrl+B, D
```

### **PHASE 4: MILESTONE TRACKING**

#### 🎯 **Critical Milestones**

| Epoch | Target Accuracy | Expected Time | Status Check |
|-------|----------------|---------------|--------------|
| 10    | ~30%           | 8 hours       | ⏳ Initial learning |
| 30    | ~55%           | 24 hours      | 📈 Active training |
| 50    | ~68%           | 40 hours      | 🎯 Mid-training |
| **81** | **>75%**       | **65 hours**  | 🚨 **CRITICAL** |
| **90** | **>77%**       | **72 hours**  | 🚨 **MILESTONE** |
| 100   | ~79%           | 80 hours      | 🔄 EMA→SWA transition |
| **120** | **>81%**       | **96 hours**  | 🏆 **TARGET** |

#### ✅ **Step 4.1: Setup Milestone Monitoring in tmux**
```bash
# Create dedicated monitoring session
tmux new-session -d -s milestones

# Attach to milestones session
tmux attach-session -t milestones

# Set up milestone checking script
cat > check_milestones.sh << 'EOF'
#!/bin/bash
while true; do
    if [ -f "./outputs/best_model.pth" ]; then
        ACCURACY=$(python -c "
import torch
checkpoint = torch.load('./outputs/best_model.pth', map_location='cpu')
print(f'{checkpoint[\"best_acc1\"]:.2f}')
")
        EPOCH=$(python -c "
import torch
checkpoint = torch.load('./outputs/best_model.pth', map_location='cpu')
print(f'{checkpoint[\"epoch\"]}')
")
        
        echo "$(date): Epoch $EPOCH - Accuracy: $ACCURACY%"
        
        # Check milestones
        if [ "$EPOCH" -eq 81 ] && [ $(echo "$ACCURACY >= 75" | bc -l) -eq 1 ]; then
            echo "🎉 MILESTONE: Epoch 81 target achieved ($ACCURACY% >= 75%)"
            # Send notification to training session
            tmux send-keys -t training "echo 'MILESTONE 81 ACHIEVED!'" Enter
        fi
        
        if [ "$EPOCH" -eq 90 ] && [ $(echo "$ACCURACY >= 77" | bc -l) -eq 1 ]; then
            echo "🎉 MILESTONE: Epoch 90 target achieved ($ACCURACY% >= 77%)"
            tmux send-keys -t training "echo 'MILESTONE 90 ACHIEVED!'" Enter
        fi
        
        if [ $(echo "$ACCURACY >= 81" | bc -l) -eq 1 ]; then
            echo "🏆 SUCCESS: 81% TARGET ACHIEVED! ($ACCURACY%)"
            echo "Training can be stopped early if desired."
            tmux send-keys -t training "echo 'TARGET 81% ACHIEVED!'" Enter
        fi
    fi
    sleep 3600  # Check every hour
done
EOF

chmod +x check_milestones.sh

# Run milestone checker in tmux
./check_milestones.sh

# Detach and let it run: Ctrl+B, then D
```

### **PHASE 5: COST & RESOURCE MONITORING**

#### ✅ **Step 5.1: Cost Tracking Commands**
```bash
# Check current instance uptime
uptime -p

# Calculate training cost estimate
python -c "
import subprocess
import time
from datetime import datetime, timedelta

# Manual cost calculation
print('=' * 50)
print('💰 COST TRACKING')
print('=' * 50)

# Get uptime
result = subprocess.run(['uptime', '-p'], capture_output=True, text=True)
print(f'Instance uptime: {result.stdout.strip()}')

# Manual cost input (since uptime parsing can be complex)
try:
    hours = float(input('Enter current runtime hours: '))
    
    hourly_cost = 0.264  # g4dn.2xlarge spot instance
    current_cost = hours * hourly_cost
    total_budget = 25.34  # For 96 hours
    remaining = total_budget - current_cost
    
    print(f'Current cost: \${current_cost:.2f}')
    print(f'Total budget: \${total_budget:.2f}')
    print(f'Remaining: \${remaining:.2f}')
    
    # Progress estimate
    progress = (hours / 96) * 100
    print(f'Time progress: {progress:.1f}%')
    
    if remaining < 5:
        print('⚠️  WARNING: Low budget remaining!')
    elif progress > 80:
        print('⏰ Training should be completing soon')
    else:
        print('✅ Budget and progress on track')
        
except ValueError:
    print('Please enter a valid number for hours')
"

# Quick cost check script
cat > cost_check.py << 'EOF'
import time
import os
from datetime import datetime

def cost_check():
    print("💰 QUICK COST CHECK")
    print("=" * 30)
    
    # Check training start time from log
    if os.path.exists('training.log'):
        with open('training.log', 'r') as f:
            first_line = f.readline()
        print(f"Training started: {first_line[:19] if first_line else 'Unknown'}")
    
    # Manual runtime entry
    try:
        hours = float(input("Enter runtime hours: "))
        cost = hours * 0.264
        print(f"Estimated cost: ${cost:.2f}")
        print(f"Budget remaining: ${25.34 - cost:.2f}")
    except:
        print("Invalid input")

if __name__ == "__main__":
    cost_check()
EOF

# Run cost check
python cost_check.py
```

#### ✅ **Step 5.2: Resource Monitoring Commands**
```bash
# GPU utilization monitoring
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

# Continuous GPU monitoring (run in separate tmux pane)
watch -n 10 "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"

# System resources
free -h  # Memory usage
df -h    # Disk usage
iostat 1 3  # I/O statistics

# Training process monitoring
ps aux | grep python | grep train
pstree -p $(pgrep -f train.py)  # Process tree

# Check for any errors in system logs
dmesg | tail -20
journalctl -u nvidia-persistenced --no-pager -n 10
```

#### ✅ **Step 5.3: Training Health Checks**
```bash
# Comprehensive health check script
cat > health_check.py << 'EOF'
import torch
import psutil
import subprocess
import json
import os
from datetime import datetime

def health_check():
    print("🏥 TRAINING HEALTH CHECK")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # GPU Health
    try:
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            
            # Memory usage
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"📊 GPU Memory Used: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
        else:
            print("❌ CUDA not available!")
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
    
    # System Resources
    print(f"\n🖥️  System Resources:")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"RAM Usage: {psutil.virtual_memory().percent}%")
    print(f"Disk Usage: {psutil.disk_usage('/').percent}%")
    
    # Training Process
    print(f"\n🔄 Training Process:")
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        if 'python' in proc.info['name'] and proc.info['cpu_percent'] > 10:
            print(f"Process {proc.info['pid']}: CPU {proc.info['cpu_percent']:.1f}%, RAM {proc.info['memory_percent']:.1f}%")
    
    # Check log for errors
    print(f"\n📋 Recent Log Status:")
    if os.path.exists('training.log'):
        with open('training.log', 'r') as f:
            lines = f.readlines()
        
        # Check last 10 lines for errors
        recent_lines = lines[-10:] if len(lines) >= 10 else lines
        error_count = sum(1 for line in recent_lines if 'error' in line.lower() or 'failed' in line.lower())
        
        if error_count > 0:
            print(f"⚠️  Found {error_count} potential errors in recent logs")
        else:
            print("✅ No recent errors detected")
            
        # Show last log entry
        if lines:
            print(f"Last log entry: {lines[-1].strip()}")
    else:
        print("❌ No training log found")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    health_check()
EOF

# Run health check
python health_check.py

# Schedule regular health checks in tmux
tmux new-window -t monitoring -n health
tmux send-keys -t monitoring:health "
cd /mnt/nvme_data/imagenet_training
while true; do
    python health_check.py
    echo 'Next health check in 2 hours...'
    sleep 7200
done
" Enter
```

#### ✅ **Step 5.3: tmux Session Overview**
```bash
# Your complete tmux setup:

# Main sessions:
tmux list-sessions
# Should show:
# training: 1 windows    (main training process)
# monitoring: 1 windows  (log monitoring)  
# milestones: 1 windows  (milestone tracking)
# dashboard: 1 windows   (4-pane overview)

# Quick session switching:
tmux attach -t training    # Check training progress
tmux attach -t monitoring  # View logs and GPU
tmux attach -t milestones  # Check milestone alerts  
tmux attach -t dashboard   # Full monitoring dashboard

# Emergency commands:
tmux kill-session -t training  # Stop training
tmux kill-server              # Kill all sessions
```

### **PHASE 6: COMPLETION & RESULTS**

#### ✅ **Step 6.1: Safe Training Completion in tmux**
```bash
# Attach to training session to check completion
tmux attach-session -t training

# If training is complete, check results
python -c "
import torch
checkpoint = torch.load('./outputs/best_model.pth', map_location='cpu')
final_acc = checkpoint['best_acc1']
final_epoch = checkpoint['epoch']

print(f'🎯 FINAL RESULTS:')
print(f'Best Accuracy: {final_acc:.2f}%')
print(f'Final Epoch: {final_epoch}')
print(f'Model Type: {checkpoint.get(\"model_type\", \"Unknown\")}')

if final_acc >= 81.0:
    print('✅ SUCCESS: 81% target achieved!')
elif final_acc >= 79.0:
    print('🟡 GOOD: Close to target')
else:
    print('🔴 BELOW TARGET: Needs investigation')
"

# Stop training manually if needed (only if you want early stopping)
# Ctrl+C in the training session, then:
# tmux send-keys -t training C-c
```

#### ✅ **Step 6.2: Download Results**
```bash
# Package results for download
tar -czf imagenet_results.tar.gz \
  outputs/ \
  training.log \
  milestones.log \
  cost_monitor.sh

# Download to local machine
scp -i your-key.pem ubuntu@your-instance-ip:imagenet_results.tar.gz .
```

#### ✅ **Step 6.3: Clean tmux Cleanup**
```bash
# Before terminating instance, properly clean up tmux sessions

# Stop training gracefully
tmux send-keys -t training C-c  # Send Ctrl+C to training session

# Wait for graceful shutdown (model will save current state)
sleep 30

# Kill all monitoring sessions
tmux kill-session -t monitoring
tmux kill-session -t milestones  
tmux kill-session -t dashboard
tmux kill-session -t auto_restart

# Finally kill training session
tmux kill-session -t training

# Or kill all sessions at once
tmux kill-server

# Terminate instance to stop billing
aws ec2 terminate-instances --instance-ids your-instance-id
```

---

## �️ TMUX QUICK REFERENCE

### **Essential tmux Commands**
```bash
# Session Management
tmux new-session -d -s name    # Create detached session
tmux attach-session -t name    # Attach to session
tmux detach-session            # Detach (or Ctrl+B, D)
tmux list-sessions             # List all sessions
tmux kill-session -t name      # Kill specific session
tmux kill-server               # Kill all sessions

# Window Management (within session)
tmux new-window                # Create new window (or Ctrl+B, C)
tmux select-window -t 0        # Switch to window 0 (or Ctrl+B, 0)
tmux rename-window name        # Rename current window (or Ctrl+B, ,)

# Pane Management (within window)
tmux split-window -h           # Split horizontally (or Ctrl+B, %)
tmux split-window -v           # Split vertically (or Ctrl+B, ")
tmux select-pane -t 0          # Select pane 0 (or Ctrl+B, arrow keys)
tmux resize-pane -D 5          # Resize pane down 5 lines

# Send commands to specific session/window/pane
tmux send-keys -t session:window.pane "command" Enter
```

### **Training-Specific tmux Workflow**
```bash
# 1. Setup phase
tmux new-session -d -s training      # Main training
tmux new-session -d -s monitoring    # Log monitoring  
tmux new-session -d -s milestones    # Milestone tracking
tmux new-session -d -s dashboard     # Full dashboard

# 2. Start training
tmux attach -t training
# (start training command, then Ctrl+B, D to detach)

# 3. Monitor progress
tmux attach -t dashboard             # View 4-pane dashboard

# 4. Check specific logs
tmux attach -t monitoring            # View training logs

# 5. Emergency stop
tmux send-keys -t training C-c       # Send Ctrl+C to training

# 6. Cleanup
tmux kill-server                     # Kill all sessions
```

### **Pro Tips for Long Training**
```bash
# Always work in tmux when training on EC2
# Sessions survive SSH disconnections
# Can reconnect from anywhere

# Monitor from multiple terminals:
ssh -i key.pem ubuntu@instance-ip "tmux attach -t dashboard"

# Send commands remotely:
ssh -i key.pem ubuntu@instance-ip "tmux send-keys -t training 'echo HELLO' Enter"

# Check if training is still running:
ssh -i key.pem ubuntu@instance-ip "tmux capture-pane -t training -p | grep python"
```

---

## � **CHECKPOINT RECOVERY & RESUME TRAINING**

### **Automatic Resume Detection**
```bash
# Create resume training script
cat > resume_training.py << 'EOF'
import torch
import os
import json
import argparse
from pathlib import Path

def find_latest_checkpoint():
    """Find the most recent checkpoint to resume from."""
    checkpoint_dir = Path('./outputs')
    
    if not checkpoint_dir.exists():
        print("❌ No outputs directory found - starting fresh training")
        return None, 0
    
    # Look for best model checkpoint
    best_checkpoint = checkpoint_dir / 'best_model.pth'
    if best_checkpoint.exists():
        try:
            checkpoint = torch.load(best_checkpoint, map_location='cpu')
            epoch = checkpoint['epoch']
            accuracy = checkpoint['best_acc1']
            print(f"✅ Found checkpoint: Epoch {epoch}, Accuracy {accuracy:.2f}%")
            return str(best_checkpoint), epoch
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
    
    print("❌ No valid checkpoint found - starting fresh")
    return None, 0

def check_training_completeness():
    """Check if training was completed or interrupted."""
    checkpoint_path, last_epoch = find_latest_checkpoint()
    
    if checkpoint_path is None:
        return "fresh", 0, "No previous training found"
    
    if last_epoch >= 120:
        return "complete", last_epoch, f"Training already complete at epoch {last_epoch}"
    elif last_epoch >= 100:
        swa_progress = last_epoch - 100
        return "swa_interrupted", last_epoch, f"SWA interrupted at epoch {last_epoch} ({swa_progress}/20 SWA epochs)"
    else:
        return "ema_interrupted", last_epoch, f"EMA interrupted at epoch {last_epoch} ({last_epoch}/100 EMA epochs)"

def main():
    status, epoch, message = check_training_completeness()
    
    print("=" * 60)
    print("🔍 TRAINING STATUS CHECK")
    print("=" * 60)
    print(f"Status: {status}")
    print(f"Last Epoch: {epoch}")
    print(f"Message: {message}")
    print("=" * 60)
    
    if status == "complete":
        print("🎉 Training is already complete!")
        print("To restart training, delete ./outputs/ directory")
        return
    elif status == "fresh":
        print("🚀 Ready to start fresh training")
        print("Run: python train.py --data /mnt/nvme_data/imagenet --output-dir ./outputs --epochs 120 --batch-size 256 --ema-epochs 100 --swa-epochs 20")
    else:
        print("🔄 Ready to resume training from checkpoint")
        print("Run: python train.py --data /mnt/nvme_data/imagenet --output-dir ./outputs --epochs 120 --batch-size 256 --ema-epochs 100 --swa-epochs 20 --resume ./outputs/best_model.pth")
    
    print("\n📋 Resume command ready to copy-paste above ☝️")

if __name__ == "__main__":
    main()
EOF

# Run the checkpoint checker
python resume_training.py
```

### **Manual Resume After Instance Interruption**
```bash
# Step 1: Check if previous training data exists
ls -la /mnt/nvme_data/imagenet_training/

# Step 2: Navigate to training directory
cd /mnt/nvme_data/imagenet_training

# Step 3: Check for existing checkpoints
python resume_training.py

# Step 4: Verify checkpoint integrity
python -c "
import torch
import os

if os.path.exists('./outputs/best_model.pth'):
    try:
        checkpoint = torch.load('./outputs/best_model.pth', map_location='cpu')
        print('✅ Checkpoint is valid')
        print(f'   Epoch: {checkpoint[\"epoch\"]}')
        print(f'   Accuracy: {checkpoint[\"best_acc1\"]:.2f}%')
        print(f'   Model type: {checkpoint.get(\"model_type\", \"Unknown\")}')
        
        # Check if optimizer state exists
        if 'optimizer' in checkpoint:
            print('✅ Optimizer state found')
        else:
            print('⚠️  No optimizer state - will restart optimizer')
            
        # Check if scheduler state exists  
        if 'scheduler' in checkpoint:
            print('✅ Scheduler state found')
        else:
            print('⚠️  No scheduler state - will restart scheduler')
            
    except Exception as e:
        print(f'❌ Checkpoint corrupted: {e}')
        print('   Recommendation: Start fresh training')
else:
    print('❌ No checkpoint found')
"

# Step 5: Resume training with proper command
# (Copy the exact command from resume_training.py output)
```

### **Resume Training Commands**
```bash
# For EMA phase interruption (epochs 1-100)
python train.py \
  --data /mnt/nvme_data/imagenet \
  --output-dir ./outputs \
  --epochs 120 \
  --batch-size 256 \
  --ema-epochs 100 \
  --swa-epochs 20 \
  --resume ./outputs/best_model.pth \
  2>&1 | tee -a training_resumed.log

# For SWA phase interruption (epochs 101-120)  
python train.py \
  --data /mnt/nvme_data/imagenet \
  --output-dir ./outputs \
  --epochs 120 \
  --batch-size 256 \
  --ema-epochs 100 \
  --swa-epochs 20 \
  --resume ./outputs/best_model.pth \
  2>&1 | tee -a training_resumed.log

# The training script automatically detects which phase to resume
```

### **Post-Interruption Checklist**
```bash
# 1. Verify data integrity
du -sh /mnt/nvme_data/imagenet
python -c "
import os
train_path = '/mnt/nvme_data/imagenet/train'
val_path = '/mnt/nvme_data/imagenet/val'
if os.path.exists(train_path) and os.path.exists(val_path):
    train_classes = len(os.listdir(train_path))
    val_classes = len(os.listdir(val_path))
    print(f'✅ Dataset intact: {train_classes} train classes, {val_classes} val classes')
else:
    print('❌ Dataset missing - need to re-download')
"

# 2. Verify GPU availability
nvidia-smi

# 3. Check disk space
df -h

# 4. Verify conda environment
conda activate pytorch_env
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 5. Setup fresh tmux sessions for resumed training
tmux new-session -d -s training_resumed
tmux new-session -d -s monitoring_resumed

# 6. Start monitoring first
tmux attach-session -t monitoring_resumed
tail -f training_resumed.log

# 7. Start resumed training (in training_resumed session)
tmux attach-session -t training_resumed
# Run the resume command from step above
```

### **Spot Instance Interruption Recovery**
```bash
# If spot instance was terminated and you're starting a new one:

# 1. Launch new g4dn.2xlarge spot instance
# 2. Attach the existing EBS volume with your data
# 3. Mount the volume
sudo mkdir -p /mnt/nvme_data
sudo mount /dev/nvme1n1 /mnt/nvme_data  # Adjust device as needed

# 4. Verify data preservation
ls -la /mnt/nvme_data/imagenet_training/
ls -la /mnt/nvme_data/imagenet_training/outputs/

# 5. If data is intact, continue with resume process above
cd /mnt/nvme_data/imagenet_training
python resume_training.py

# 6. If data is lost, restart from Phase 2 (data preparation)
```

### **Emergency Checkpoint Backup**
```bash
# Before any risky operations, backup your checkpoint
mkdir -p ./checkpoint_backups
cp ./outputs/best_model.pth ./checkpoint_backups/backup_$(date +%Y%m%d_%H%M%S).pth
cp ./outputs/training_history.json ./checkpoint_backups/ 2>/dev/null || echo "No history file"

# List backups
ls -la ./checkpoint_backups/

# Restore from backup if needed
cp ./checkpoint_backups/backup_YYYYMMDD_HHMMSS.pth ./outputs/best_model.pth
```

---

## �🚨 **TROUBLESHOOTING GUIDE**

### **Issue 1: Out of Memory**
```bash
# Reduce batch size
python train.py --batch-size 128  # instead of 256
```

### **Issue 2: Spot Instance Interruption with tmux Recovery**
```bash
# Auto-restart script with tmux management
cat > auto_restart.sh << 'EOF'
#!/bin/bash

# Function to restart training in tmux
restart_training() {
    echo "Restarting training after interruption..."
    
    # Kill existing training session if it exists
    tmux kill-session -t training 2>/dev/null
    
    # Create new training session
    tmux new-session -d -s training
    
    # Navigate to project directory and activate environment
    tmux send-keys -t training "cd /mnt/nvme_data/imagenet_training" Enter
    tmux send-keys -t training "conda activate pytorch_env" Enter
    
    # Check for existing checkpoint and resume
    if [ -f "./outputs/best_model.pth" ]; then
        echo "Found checkpoint, resuming training..."
        tmux send-keys -t training "python train.py --resume ./outputs/best_model.pth --data /mnt/nvme_data/imagenet 2>&1 | tee -a training.log" Enter
    else
        echo "No checkpoint found, starting fresh..."
        tmux send-keys -t training "python train.py --data /mnt/nvme_data/imagenet 2>&1 | tee training.log" Enter
    fi
}

# Main restart loop
while true; do
    # Check if training session exists and is running
    if ! tmux has-session -t training 2>/dev/null; then
        restart_training
    fi
    
    # Check if training process is actually running
    if tmux has-session -t training 2>/dev/null; then
        # Check if Python process is running in the session
        PYTHON_RUNNING=$(tmux capture-pane -t training -p | grep -c "python train.py")
        if [ $PYTHON_RUNNING -eq 0 ]; then
            echo "Training process not found, restarting..."
            restart_training
        fi
    fi
    
    sleep 300  # Check every 5 minutes
done
EOF

chmod +x auto_restart.sh

# Run the auto-restart in its own tmux session
tmux new-session -d -s auto_restart
tmux send-keys -t auto_restart "./auto_restart.sh" Enter
```

### **Issue 3: Low Accuracy at Milestones**
```bash
# Check dataset integrity
find /mnt/nvme_data/imagenet/train/ -name "*.JPEG" | wc -l

# Check GPU utilization (should be >90%)
nvidia-smi

# Check learning rate schedule
grep "LR:" training.log | tail -10
```

---

## ✅ SUCCESS CRITERIA

**Training is considered successful if:**
- [x] Reaches >75% accuracy by epoch 81
- [x] Reaches >77% accuracy by epoch 90  
- [x] Achieves >81% final accuracy
- [x] Completes within $30 budget
- [x] No major interruptions

**Final deliverables:**
- [x] Model weights (best_model.pth)
- [x] Training logs (training.log)
- [x] Accuracy progression data
- [x] Cost breakdown report

---

## 📞 EXECUTION CHECKLIST

### Pre-Training ☐
- [ ] EC2 instance launched and configured
- [ ] Environment setup completed
- [ ] ImageNet dataset downloaded and verified
- [ ] Training script tested on sample data

### During Training ☐
- [ ] Training started successfully
- [ ] Monitoring scripts running
- [ ] Milestone tracking active
- [ ] Cost monitoring in place

### Post-Training ☐
- [ ] Results documented
- [ ] Model files downloaded
- [ ] Instance terminated
- [ ] Final cost calculated

---

**📅 Expected Completion**: November 6-7, 2025  
**🎯 Target Achievement**: 81% ImageNet Top-1 Accuracy  
**💰 Budget**: $25-30 total cost**