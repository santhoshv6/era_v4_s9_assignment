# 📋 EXECUTION DOCUMENT
## ResNet50 ImageNet Training - Complete Execution Guide

**Date**: November 4, 2025 (**Updated for g5.2xlarge**)  
**Target**: 81% Top-1 Accuracy on ImageNet  
**Budget**: ~$25 on AWS EC2 Spot Instances  
**Timeline**: 4-5 days (96 hours training)

**🔄 Latest Updates:**
- ✅ **CUDA 12.1** support for g5.2xlarge instances
- ✅ **Streamlined setup** - single command environment installation  
- ✅ **Fixed repository cloning** to `/home/ubuntu/resent50_training`
- ✅ **Optimized PyTorch** installation for A10G GPU

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
- **Recommended Instance**: g4dn.2xlarge (T4 GPU, 32GB RAM)
- **Spot Price**: ~$0.264/hour
- **Training Time**: 96 hours
- **Total Cost**: $25.34 + storage (~$26 total)

### 🖥️ **INSTANCE COMPARISON: g4dn.2xlarge vs g5.2xlarge**

| **Specification** | **g4dn.2xlarge** | **g5.2xlarge** |
|-------------------|-------------------|----------------|
| **GPU** | NVIDIA T4 (16GB) | NVIDIA A10G (24GB) |
| **GPU Memory** | 16GB GDDR6 | 24GB GDDR6 |
| **CUDA Version** | 11.8 | **12.1** |
| **vCPUs** | 8 | 8 |
| **RAM** | 32GB | 32GB |
| **Network** | Up to 25 Gbps | Up to 10 Gbps |
| **Storage** | NVMe SSD | NVMe SSD |

#### **⚠️ IMPORTANT: g5.2xlarge Updates**
- **CUDA 12.1**: g5.2xlarge instances come with newer CUDA 12.1 (not 11.8)
- **PyTorch Index**: Use `cu121` for PyTorch installation on g5.2xlarge
- **Performance**: Same training code, better GPU performance with A10G

#### **💵 COST ANALYSIS**

| **Instance** | **Spot Price/hr** | **96hr Total** | **Performance** | **Time to 81%** |
|--------------|-------------------|----------------|-----------------|------------------|
| **g4dn.2xlarge** | ~$0.264 | **$25.34** | 100% baseline | 96 hours |
| **g5.2xlarge** | ~$0.544 | **$52.22** | ~140% faster | ~69 hours |

#### **🎯 RECOMMENDATION: g4dn.2xlarge**

**Choose g4dn.2xlarge for the following reasons:**

✅ **Budget Compliance**: $25.34 total cost vs $52.22 for g5  
✅ **Proven Performance**: T4 handles ResNet50 + batch size 256 comfortably  
✅ **Sufficient Memory**: 16GB GPU memory is adequate for ImageNet training  
✅ **Spot Availability**: g4dn instances have better spot availability  
✅ **Cost Efficiency**: 2x cheaper with only ~30% longer training time  

#### **⚠️ When to Consider g5.2xlarge:**

- Budget allows $50+ spend
- Need faster results (69 hours vs 96 hours)
- Planning multiple training runs
- Experimenting with larger batch sizes (512+)

#### **🧮 DETAILED COST BREAKDOWN (g4dn.2xlarge)**

```
Compute Cost:
- 96 hours × $0.264/hour = $25.34

Storage Cost:
- 200GB EBS gp3: ~$2/month (prorated ~$0.50 for 4 days)
- 150GB ImageNet: Already included in EBS

Total Estimated Cost: $25.84
Safety Buffer: $4.16 (for price fluctuations)
**TOTAL BUDGET: $30**
```

#### **📊 PERFORMANCE EXPECTATIONS**

**g4dn.2xlarge Performance:**
- Batch size: 256 (optimal for T4)
- Memory usage: ~14GB/16GB (safe margin)
- Training speed: ~8-10 epochs/day
- Total time: 96 hours for 120 epochs
- Expected accuracy: 81%+ with EMA+SWA

---

## � **DIRECTORY STRUCTURE & DATA SETUP**

### **✅ RECOMMENDED SETUP (Separation of Code and Data)**

```bash
# Training code location (where you run commands from)
/home/ubuntu/resent50_training/
├── train.py                    # Main training script
├── src/                        # Source code modules
├── outputs/                    # Training outputs & checkpoints
├── requirements.txt            # Dependencies
└── logs/                       # Training logs

# Data location (NVMe storage for speed)
/mnt/nvme_data/imagenet/
├── train/                      # Training images (1.3M images)
│   ├── n01440764/
│   ├── n01443537/
│   └── ... (1000 classes)
└── val/                        # Validation images (50K images)
    ├── n01440764/
    ├── n01443537/
    └── ... (1000 classes)
```

### **🎯 WHY THIS SETUP IS OPTIMAL:**

✅ **Code Portability**: Training code in home directory is easy to backup/version  
✅ **Data Performance**: ImageNet on NVMe for maximum I/O speed  
✅ **Space Management**: Separates large data from lightweight code  
✅ **Flexibility**: Can point to data from any training directory  

### **📋 SETUP COMMANDS**

```bash
# 1. Create training directory in home
cd /home/ubuntu
mkdir -p resent50_training
cd resent50_training

# 2. Copy/clone your training code here
# (Your code should already be here based on your question)

# 3. Verify data location
ls -la /mnt/nvme_data/imagenet/
# Should show: train/ and val/ directories

# 4. Check you're in the right place for training
pwd
# Should show: /home/ubuntu/resent50_training

# 5. Ready to run training!
```

---

## �🚀 EXECUTION PHASES

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

#### ✅ **Step 1.2: Python Environment Setup (One Command)**
```bash
# 🚀 STREAMLINED SETUP FOR G5.2XLARGE - ONE-STEP INSTALLATION
# This script sets up everything needed for ResNet50 training

# Update system and install Python essentials
sudo apt update -y && sudo apt install -y python3-venv python3-pip python3-dev build-essential

# Create and activate virtual environment
python3 -m venv pytorch_env
source pytorch_env/bin/activate

# Upgrade pip for faster installs
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support (latest available for g5.2xlarge)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all training dependencies in one go (Python 3.12 + PyArrow compatible versions)
pip install \
    "datasets>=2.15.0" \
    "huggingface-hub>=0.18.0" \
    "pyarrow>=14.0.0" \
    tqdm==4.66.1 \
    pillow==10.0.1 \
    "numpy>=1.26.0" \
    "scikit-learn>=1.3.2" \
    tensorboard==2.14.1 \
    psutil==5.9.5 \
    "matplotlib>=3.8.0"

# Verify complete installation
echo "🔍 Verifying installation..."
python -c "
import torch, torchvision, datasets, numpy, sklearn, tensorboard, psutil
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
print(f'✅ All packages successfully installed!')
"

# Add activation to .bashrc for convenience
echo "source ~/pytorch_env/bin/activate" >> ~/.bashrc

echo "🎉 Environment setup complete! PyTorch with CUDA 12.1 is ready for g5.2xlarge training."
```

#### ⚠️ **TROUBLESHOOTING: PyArrow/Datasets Compatibility Issues**
```bash
# If you get "AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'" error:
# This is a PyArrow version compatibility issue with datasets library

# SOLUTION 1: Update to compatible versions (recommended)
pip uninstall datasets pyarrow huggingface-hub -y
pip install "datasets>=2.15.0" "pyarrow>=14.0.0" "huggingface-hub>=0.18.0"

# SOLUTION 2: Force reinstall with latest versions
pip install --upgrade --force-reinstall datasets pyarrow huggingface-hub

# SOLUTION 3: Use specific compatible versions
pip install datasets==2.16.0 pyarrow==15.0.0 huggingface-hub==0.20.0

# SOLUTION 4: If issues persist, use conda for these packages
pip uninstall datasets pyarrow huggingface-hub -y
conda install -c conda-forge datasets pyarrow huggingface_hub

# Verify the fix
python -c "
try:
    from datasets import load_dataset
    print('✅ Datasets library working correctly')
    # Quick test load
    print('✅ Testing dataset loading...')
    import datasets
    print(f'✅ Datasets version: {datasets.__version__}')
    import pyarrow
    print(f'✅ PyArrow version: {pyarrow.__version__}')
except Exception as e:
    print(f'❌ Error: {e}')
"
```

#### ⚠️ **TROUBLESHOOTING: PyTorch Version Issues**
```bash
# If you get "No matching distribution found for torch==2.1.0" error:
# This happens because PyTorch 2.1.0 is not available for CUDA 12.1

# SOLUTION 1: Install latest available PyTorch for CUDA 12.1 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# SOLUTION 2: Check available versions first
# Visit: https://download.pytorch.org/whl/cu121/
# Available versions: 2.2.0+cu121, 2.2.1+cu121, 2.2.2+cu121, 2.3.0+cu121, 2.3.1+cu121, 2.4.0+cu121, 2.4.1+cu121, 2.5.0+cu121, 2.5.1+cu121

# SOLUTION 3: Install specific stable version
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# SOLUTION 4: If CUDA 12.1 continues to have issues, try CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
else:
    print('❌ CUDA not available - check installation')
"
```

#### ⚠️ **TROUBLESHOOTING: Python 3.12 Compatibility Issues**
```bash
# If you encounter numpy build errors with Python 3.12:
# Error: "AttributeError: module 'pkgutil' has no attribute 'ImpImporter'"

# SOLUTION 1: Use compatible package versions (recommended)
pip install numpy>=1.26.0  # Python 3.12 compatible
pip install scikit-learn>=1.3.2  # Updated for Python 3.12

# SOLUTION 2: If still having issues, use binary wheels only
pip install --only-binary=all numpy scikit-learn

# SOLUTION 3: Alternative - use conda for problematic packages
# (Only if pip continues to fail)
conda install numpy scikit-learn -c conda-forge

# SOLUTION 4: Verify your environment is clean
pip list | grep -E "(numpy|scipy|scikit)"
# If you see conflicting versions, reinstall:
pip uninstall numpy scikit-learn -y
pip install numpy>=1.26.0 scikit-learn>=1.3.2

# Final verification after fixing
python -c "
import numpy as np
import sklearn
print(f'✅ NumPy: {np.__version__}')
print(f'✅ Scikit-learn: {sklearn.__version__}')
print(f'✅ Python 3.12 compatibility: OK')
"
```

#### ✅ **Step 1.3: Setup Storage for ImageNet**
```bash
# Check available storage
df -h
lsblk

# Create directory for ImageNet data on fastest storage
sudo mkdir -p /mnt/nvme_data
sudo chown $USER:$USER /mnt/nvme_data

# For g5.2xlarge with additional NVMe (if needed)
# sudo mkfs.ext4 /dev/nvme1n1  # Only if you have additional NVMe
# sudo mount /dev/nvme1n1 /mnt/nvme_data  
# sudo chown $USER:$USER /mnt/nvme_data

# Verify storage setup
echo "📊 Storage verification:"
df -h /mnt/nvme_data
ls -la /mnt/nvme_data

#### ✅ **Step 1.4: Create Python Virtual Environment**
```bash
# Create Python virtual environment
python3 -m venv pytorch_env

# Activate the environment
source pytorch_env/bin/activate

# Verify activation (prompt should show (pytorch_env))
#### ✅ **Step 1.3: Setup Storage for ImageNet**
```bash
# Check available storage
df -h
lsblk

# Create directory for ImageNet data on fastest storage
sudo mkdir -p /mnt/nvme_data
sudo chown $USER:$USER /mnt/nvme_data

# For g5.2xlarge with additional NVMe (if needed)
# sudo mkfs.ext4 /dev/nvme1n1  # Only if you have additional NVMe
# sudo mount /dev/nvme1n1 /mnt/nvme_data  
# sudo chown $USER:$USER /mnt/nvme_data

# Verify storage setup
echo "📊 Storage verification:"
df -h /mnt/nvme_data
ls -la /mnt/nvme_data
```

#### ✅ **Step 1.4: Final Environment Verification**
```bash
# 🔍 COMPREHENSIVE VERIFICATION FOR G5.2XLARGE

echo "🔍 Environment Verification Summary:"
echo "=================================="

# Python environment
echo "✅ Python environment:"
which python
python --version

# PyTorch and CUDA
echo "✅ PyTorch and GPU:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
    print(f'CUDA version: {torch.version.cuda}')
"

# Package versions
echo "✅ Key packages:"
python -c "
import datasets, numpy, sklearn, tensorboard
print(f'datasets: {datasets.__version__}')
print(f'numpy: {numpy.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print(f'tensorboard: {tensorboard.__version__}')
"

# System resources
echo "✅ System resources:"
echo "CPU cores: $(nproc)"
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"

echo "🎉 Environment ready for ResNet50 training on g5.2xlarge!"

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

#### ✅ **Step 2.1: Setup Project Directory & Clone Repository**
```bash
### **PHASE 2: PROJECT SETUP**

#### ✅ **Step 2.1: Setup Training Directory and Clone Repository**
```bash
# Create training directory in home (as discussed)
cd /home/ubuntu
mkdir -p resent50_training
cd resent50_training

# Install git if not already available
sudo apt install git -y

# Clone the latest training repository
git clone https://github.com/santhoshv6/era_v4_s9_assignment.git .

# Alternative methods if above fails:
# Method 1: Clone to temp directory then copy
# git clone https://github.com/santhoshv6/era_v4_s9_assignment.git temp_repo
# cp -r temp_repo/* .
# rm -rf temp_repo/

# Method 2: Download as zip if git fails
# wget https://github.com/santhoshv6/era_v4_s9_assignment/archive/refs/heads/master.zip
# unzip master.zip
# cp -r era_v4_s9_assignment-master/* .
# rm -rf era_v4_s9_assignment-master/ master.zip

# Verify all essential files are present
echo "📋 Verifying repository contents:"
ls -la
# Should show: train.py, src/, requirements.txt, final_code_download/, EXECUTION_DOC.md, etc.

# Activate virtual environment (if not already active)
source ~/pytorch_env/bin/activate

# Install any additional dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "📦 Installing additional dependencies..."
    pip install -r requirements.txt
else
    echo "ℹ️  No requirements.txt found - using environment packages"
fi

# Verify Python packages
echo "🔍 Verifying key packages:"
python -c "
import torch, torchvision, datasets
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA: {torch.cuda.is_available()}')
print(f'✅ Datasets: {datasets.__version__}')
"
mkdir -p outputs logs checkpoints
```

#### ✅ **Step 2.2: Login to Hugging Face**
```bash
# Install Hugging Face CLI if not already installed
pip install huggingface_hub[cli]

# Login to Hugging Face (you'll need your token)
huggingface-cli login
# Enter your HF token when prompted
```

#### ✅ **Step 2.2b: Verify Repository Setup**
```bash
# Verify all critical files are present
cd /mnt/nvme_data/imagenet_training

echo "🔍 Verifying repository files..."
echo "=================================="

# Check main training files
[ -f "train.py" ] && echo "✅ train.py found" || echo "❌ train.py missing"
[ -f "final_code_download/final_fast_download.py" ] && echo "✅ final_fast_download.py found" || echo "❌ final_fast_download.py missing"
[ -f "requirements.txt" ] && echo "✅ requirements.txt found" || echo "❌ requirements.txt missing"
[ -f "EXECUTION_DOC.md" ] && echo "✅ EXECUTION_DOC.md found" || echo "❌ EXECUTION_DOC.md missing"

# Check src modules
[ -d "src" ] && echo "✅ src/ directory found" || echo "❌ src/ directory missing"
[ -f "src/model.py" ] && echo "✅ src/model.py found" || echo "❌ src/model.py missing"
[ -f "src/utils.py" ] && echo "✅ src/utils.py found" || echo "❌ src/utils.py missing"
[ -f "src/transforms.py" ] && echo "✅ src/transforms.py found" || echo "❌ src/transforms.py missing"
[ -f "src/mixup.py" ] && echo "✅ src/mixup.py found" || echo "❌ src/mixup.py missing"
[ -f "src/ema.py" ] && echo "✅ src/ema.py found" || echo "❌ src/ema.py missing"

# Test Python imports
echo ""
echo "🧪 Testing Python imports..."
python -c "
import sys
sys.path.append('src')
try:
    from src.model import get_model
    from src.transforms import build_transforms
    from src.mixup import MixupCutmixCollator
    from src.ema import EMAModel
    from src.utils import AverageMeter, accuracy
    print('✅ All core modules import successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

# Test training script syntax
python -m py_compile train.py && echo "✅ train.py syntax valid" || echo "❌ train.py syntax errors"

echo "=================================="
echo "Repository setup verification complete!"
echo "Fix any ❌ issues before proceeding to data download."
```

#### ✅ **Step 2.3: Download ImageNet Dataset with Optimized Script**
```bash
# Navigate to training directory (where you cloned the repo)
cd /home/ubuntu/resent50_training

# Use the optimized final_fast_download.py script
# This script is specifically optimized for g5.2xlarge with:
# - Auto-optimization for A10G GPU
# - CUDA 12.1 support  
# - GPU batch processing
# - Duplicate prevention
# - Resume capability

# Create main data_download session
tmux new-session -d -s data_download

# Attach to data_download session
tmux attach-session -t data_download


# RECOMMENDED: Auto-optimized download (best for g5.2xlarge)
python final_code_download/final_fast_download.py \
  --output-dir /mnt/nvme_data/imagenet

# Alternative: Manual optimization (if you want specific settings)
python final_code_download/final_fast_download.py \
  --output-dir /mnt/nvme_data/imagenet \
  --num-workers 20 \
  --chunk-size 2500 \
  --quality 90

# For fresh start (clears cache and starts clean)
python final_code_download/final_fast_download.py \
  --output-dir /mnt/nvme_data/imagenet \
  --fresh-start

# Expected performance on g5.2xlarge:
# - Speed: 120-180 images/second
# - Time: ~2-3 hours for full ImageNet
# - Memory: Efficient GPU and RAM usage
```

### **🚀 Why final_fast_download.py is Optimized:**

✅ **Hardware Auto-Detection**: Automatically detects A10G GPU and optimizes settings  
✅ **CUDA 12.1 Support**: Built for g5.2xlarge CUDA version  
✅ **GPU Batch Processing**: Processes up to 20 images simultaneously on GPU  
✅ **Duplicate Prevention**: Deterministic filenames prevent duplicate downloads  
✅ **Resume Capability**: Intelligent resume without re-downloading existing files  
✅ **Real-time Monitoring**: Live performance metrics and resource usage  
✅ **Memory Management**: Automatic GPU cache clearing and memory optimization  
✅ **Error Recovery**: Handles corrupted files and network interruptions  

### **📊 Performance Comparison:**

| Method | Speed (img/sec) | Time (Full ImageNet) | Features |
|--------|----------------|---------------------|----------|
| **final_fast_download.py** | 120-180 | **2-3 hours** | ✅ All optimizations |
| Old scripts | 40-80 | 6-8 hours | ❌ Basic functionality |

### **💡 Usage Tips:**

```bash
# Check download progress (in another terminal)
cd /home/ubuntu/resent50_training
python -c "
import os
train_count = len([f for f in os.listdir('/mnt/nvme_data/imagenet/train') if os.path.isdir(f'/mnt/nvme_data/imagenet/train/{f}')])
val_count = len([f for f in os.listdir('/mnt/nvme_data/imagenet/val') if os.path.isdir(f'/mnt/nvme_data/imagenet/val/{f}')]) if os.path.exists('/mnt/nvme_data/imagenet/val') else 0
print(f'Progress: {train_count}/1000 train classes, {val_count}/1000 val classes')
"

# If download gets interrupted, simply re-run the same command
# The script will automatically resume from where it left off
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

# 3. Check virtual environment
source ~/pytorch_env/bin/activate && echo "✅ Virtual environment activated" || echo "❌ Virtual environment issue"

# 4. Check git repository status
cd /home/ubuntu/resent50_training
git status >/dev/null 2>&1 && echo "✅ Git repository properly cloned" || echo "❌ Git repository issue"
git log --oneline -1 && echo "✅ Latest commit verified" || echo "❌ Git log issue"

# 5. Check Python imports
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

# 6. Check ImageNet dataset
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

# 7. Check HuggingFace authentication
huggingface-cli whoami >/dev/null 2>&1 && echo "✅ HuggingFace authenticated" || echo "❌ HuggingFace login required"

# 8. Test training script syntax
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

# Activate virtual environment
source pytorch_env/bin/activate

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
# ✅ RUNNING FROM: /home/ubuntu/resent50_training
# ✅ DATA LOCATION: /mnt/nvme_data/imagenet

# Optimized for g5.2xlarge (A10G GPU - 24GB VRAM)
python train.py \
  --data /mnt/nvme_data/imagenet \
  --output-dir ./outputs \
  --epochs 90 \
  --batch-size 400 \
  --lr 0.156 \
  --ema-epochs 80 \
  --swa-epochs 10 \
  --workers 8 \
  --amp \
  2>&1 | tee training.log

# The training will show progress and save logs
# Outputs will be saved to: /home/ubuntu/resent50_training/outputs/
# Logs will be saved to: /home/ubuntu/resent50_training/training.log
# Detach from session: Ctrl+B, then D
# Training continues in background
```

### **🔍 VERIFY YOUR SETUP BEFORE TRAINING**

```bash
# 1. Confirm you're in the training directory
pwd
# Expected: /home/ubuntu/resent50_training

# 2. Check training script exists
ls -la train.py
# Expected: Should show train.py file

# 3. Verify data is accessible
ls -la /mnt/nvme_data/imagenet/
# Expected: Should show train/ and val/ directories

# 4. Check data counts (optional, takes a few minutes)
find /mnt/nvme_data/imagenet/train -name "*.JPEG" | wc -l
# Expected: ~1,281,167 (training images)

find /mnt/nvme_data/imagenet/val -name "*.JPEG" | wc -l  
# Expected: ~50,000 (validation images)

# 5. Test data loading (quick test)
python -c "
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
print('Testing data loading...')
dataset = datasets.ImageFolder('/mnt/nvme_data/imagenet/train', 
                              transform=transforms.ToTensor())
print(f'✅ Dataset loaded: {len(dataset)} images')
print(f'✅ Classes: {len(dataset.classes)}')
"
```

#### ✅ **Step 3.4: Setup Comprehensive Monitoring Dashboard**

```bash
# Detach from training session (Ctrl+B, D)
# Create dedicated monitoring dashboard with 4 panes
tmux new-session -d -s dashboard

# Split into 4 panes for comprehensive monitoring
tmux send-keys -t dashboard 'cd /home/ubuntu/resent50_training' C-m
tmux split-window -h -t dashboard    # Split horizontally
tmux split-window -v -t dashboard:0.1    # Split right pane vertically
tmux select-pane -t dashboard:0.0   # Select top-left
tmux split-window -v -t dashboard    # Split left pane vertically

# Pane 0 (top-left): Real-time training logs
tmux send-keys -t dashboard:0.0 'tail -f training.log' C-m

# Pane 1 (bottom-left): GPU monitoring with memory usage
tmux send-keys -t dashboard:0.1 'watch -n 3 "nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits"' C-m

# Pane 2 (top-right): System resources
tmux send-keys -t dashboard:0.2 'htop' C-m

# Pane 3 (bottom-right): Training progress and milestones
tmux send-keys -t dashboard:0.3 'watch -n 60 "echo \"=== TRAINING PROGRESS ===\"; if [ -f ./outputs/best_model.pth ]; then python3 -c \"import torch; c=torch.load('"'"'./outputs/best_model.pth'"'"', map_location='"'"'cpu'"'"'); print(f'"'"'Current Epoch: {c[\"epoch\"]}'"'"'); print(f'"'"'Best Accuracy: {c[\"best_acc1\"]:.3f}%'"'"'); print(f'"'"'Strategy: {\"EMA\" if c[\"epoch\"] <= 80 else \"SWA\"}'"'"'); print(f'"'"'GPU Memory: ~{400*64/1024:.1f}GB (Batch 400)'"'"');\"; else echo \"No checkpoint yet - training starting...\"; fi; echo \"\"; echo \"=== MILESTONES ===\"; echo \"Epoch 20: ~45% (Expected)\"; echo \"Epoch 40: ~65% (Target)\"; echo \"Epoch 60: ~72% (Good)\"; echo \"Epoch 80: >75% (CRITICAL)\"; echo \"Epoch 90: >78% (SUCCESS)\""' C-m

# Attach to dashboard to view all monitoring
echo "✅ Dashboard created! Attach with: tmux attach-session -t dashboard"
echo "💡 Navigate between panes: Ctrl+B + Arrow Keys"
echo "🔄 Detach anytime: Ctrl+B + D"
```

### **🛠️ TROUBLESHOOTING PATH ISSUES**

```bash
# Common issue: "Dataset not found" errors
# Solution: Always use absolute paths

# ❌ WRONG (relative path)
python train.py --data ./imagenet

# ✅ CORRECT (absolute path)  
python train.py --data /mnt/nvme_data/imagenet

# Check if data path is accessible from training directory
cd /home/ubuntu/resent50_training
ls -la /mnt/nvme_data/imagenet/
# Should work from any directory

# If you get permission errors:
sudo chown -R ubuntu:ubuntu /mnt/nvme_data/imagenet/
sudo chmod -R 755 /mnt/nvme_data/imagenet/

# Test PyTorch can access the data:
python -c "
import os
data_path = '/mnt/nvme_data/imagenet'
train_path = os.path.join(data_path, 'train') 
val_path = os.path.join(data_path, 'val')
print(f'Train exists: {os.path.exists(train_path)}')
print(f'Val exists: {os.path.exists(val_path)}')
print(f'Train classes: {len(os.listdir(train_path))}')
print(f'Val classes: {len(os.listdir(val_path))}')
"
```
```bash
# Attach to dashboard session
tmux attach-session -t dashboard
```

### **PHASE 4: MILESTONE TRACKING**

#### 🎯 **Critical Milestones (90 Epochs - g5.2xlarge Optimized)**

| Epoch | Target Accuracy | Expected Time | Strategy | Status Check |
|-------|----------------|---------------|----------|--------------|
| 10    | ~35%           | 2.5 hours     | EMA      | ⏳ Initial learning |
| 20    | ~50%           | 5 hours       | EMA      | 📈 Early progress |
| 40    | ~65%           | 10 hours      | EMA      | 📊 Mid-training |
| 60    | ~72%           | 15 hours      | EMA      | 🎯 Strong progress |
| **80** | **>75%**       | **20 hours**  | **EMA**  | 🚨 **CRITICAL MILESTONE** |
| **85** | **>77%**       | **21.5 hours** | **SWA**  | 🔄 **EMA→SWA Transition** |
| **90** | **>78%**       | **22.5 hours** | **SWA**  | 🏆 **TARGET ACHIEVED** |

#### ✅ **Step 4.1: Setup Automated Milestone Monitoring**
```bash
# Create milestone monitoring script
cat > monitor_milestones.sh << 'EOF'
#!/bin/bash

echo "🎯 MILESTONE MONITORING STARTED"
echo "================================"

while true; do
    if [ -f ./outputs/best_model.pth ]; then
        # Extract current progress
        PROGRESS=$(python3 -c "
import torch
checkpoint = torch.load('./outputs/best_model.pth', map_location='cpu')
epoch = checkpoint['epoch']
accuracy = checkpoint['best_acc1']
strategy = 'EMA' if epoch <= 80 else 'SWA'
print(f'{epoch},{accuracy:.3f},{strategy}')
")
        
        EPOCH=$(echo $PROGRESS | cut -d',' -f1)
        ACC=$(echo $PROGRESS | cut -d',' -f2)
        STRATEGY=$(echo $PROGRESS | cut -d',' -f3)
        
        # Clear screen and show status
        clear
        echo "🚀 RESNET50 IMAGENET TRAINING - BATCH 400"
        echo "=========================================="
        echo "📅 $(date)"
        echo "🔢 Current Epoch: $EPOCH/90"
        echo "🎯 Best Accuracy: $ACC%"
        echo "⚙️  Strategy: $STRATEGY"
        echo ""
        
        # Check milestones
        if (( $(echo "$EPOCH >= 80" | bc -l) )) && (( $(echo "$ACC >= 75.0" | bc -l) )); then
            echo "✅ MILESTONE 80: ACHIEVED ($ACC% >= 75%)"
        elif (( $(echo "$EPOCH >= 80" | bc -l) )); then
            echo "⚠️  MILESTONE 80: BEHIND TARGET ($ACC% < 75%)"
        fi
        
        if (( $(echo "$EPOCH >= 90" | bc -l) )) && (( $(echo "$ACC >= 78.0" | bc -l) )); then
            echo "🏆 FINAL TARGET: ACHIEVED ($ACC% >= 78%)"
            echo "🎉 TRAINING SUCCESS!"
            break
        elif (( $(echo "$EPOCH >= 90" | bc -l) )); then
            echo "🟡 FINAL TARGET: CLOSE ($ACC% approaching 78%)"
        fi
        
        echo ""
        echo "⏱️  Progress Estimator:"
        echo "   • Time per epoch: ~15 minutes"
        echo "   • Remaining epochs: $((90 - EPOCH))"
        echo "   • Est. completion: $((15 * (90 - EPOCH) / 60)) hours"
        
    else
        echo "⏳ Waiting for training to start..."
    fi
    
    sleep 300  # Check every 5 minutes
done
EOF

chmod +x monitor_milestones.sh

# Run milestone monitoring in background
tmux new-session -d -s milestones './monitor_milestones.sh'
echo "✅ Milestone monitoring started in background"
echo "📊 View with: tmux attach-session -t milestones"
```
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

## 🔄 **CHECKPOINT RECOVERY & RESUME TRAINING**

### **✅ Step 5.1: Smart Resume Training**
```bash
# Create intelligent resume script for your 400 batch training
cat > smart_resume.py << 'EOF'
#!/usr/bin/env python3
import torch
import os
import sys
from pathlib import Path

def analyze_checkpoint():
    """Analyze existing checkpoint and provide resume command."""
    checkpoint_path = Path('./outputs/best_model.pth')
    
    if not checkpoint_path.exists():
        print("❌ No checkpoint found")
        print("🚀 Starting fresh training:")
        print_fresh_command()
        return
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        epoch = checkpoint['epoch']
        accuracy = checkpoint['best_acc1']
        
        print(f"✅ Found checkpoint:")
        print(f"   📊 Epoch: {epoch}/90")
        print(f"   🎯 Best Accuracy: {accuracy:.3f}%")
        print(f"   ⚙️  Strategy: {'EMA' if epoch <= 80 else 'SWA'}")
        
        if epoch >= 90:
            print("🏆 Training already complete!")
            print(f"   Final accuracy: {accuracy:.3f}%")
            return
            
        remaining = 90 - epoch
        print(f"\n🔄 Resume training ({remaining} epochs remaining):")
        print_resume_command()
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("🚀 Starting fresh training:")
        print_fresh_command()

def print_fresh_command():
    """Print command for fresh training."""
    print("""
python train.py \\
  --data /mnt/nvme_data/imagenet \\
  --output-dir ./outputs \\
  --epochs 90 \\
  --batch-size 400 \\
  --lr 0.156 \\
  --ema-epochs 80 \\
  --swa-epochs 10 \\
  --workers 8 \\
  --amp \\
  2>&1 | tee training.log
""")

def print_resume_command():
    """Print command for resuming training."""
    print("""
python train.py \\
  --data /mnt/nvme_data/imagenet \\
  --output-dir ./outputs \\
  --epochs 90 \\
  --batch-size 400 \\
  --lr 0.156 \\
  --ema-epochs 80 \\
  --swa-epochs 10 \\
  --workers 8 \\
  --amp \\
  --resume ./outputs/best_model.pth \\
  2>&1 | tee -a training.log
""")

if __name__ == "__main__":
    analyze_checkpoint()
EOF

chmod +x smart_resume.py

# Run the analysis
python3 smart_resume.py
```

### **✅ Step 5.2: Manual Resume Commands**

#### **Resume from Interruption:**
```bash
# Check current status first
ls -la ./outputs/
python3 -c "
import torch
if os.path.exists('./outputs/best_model.pth'):
    c = torch.load('./outputs/best_model.pth', map_location='cpu')
    print(f'Last epoch: {c[\"epoch\"]}, Accuracy: {c[\"best_acc1\"]:.3f}%')
else:
    print('No checkpoint found')
"

# Resume training with exact same parameters
python train.py \
  --data /mnt/nvme_data/imagenet \
  --output-dir ./outputs \
  --epochs 90 \
  --batch-size 400 \
  --lr 0.156 \
  --ema-epochs 80 \
  --swa-epochs 10 \
  --workers 8 \
  --amp \
  --resume ./outputs/best_model.pth \
  2>&1 | tee -a training.log
```

#### **Resume with Different Parameters (if needed):**
```bash
# If you need to change batch size due to memory issues
python train.py \
  --data /mnt/nvme_data/imagenet \
  --output-dir ./outputs \
  --epochs 90 \
  --batch-size 320 \
  --lr 0.125 \
  --ema-epochs 80 \
  --swa-epochs 10 \
  --workers 8 \
  --amp \
  --resume ./outputs/best_model.pth \
  2>&1 | tee -a training.log

# Or extend training beyond 90 epochs if needed
python train.py \
  --data /mnt/nvme_data/imagenet \
  --output-dir ./outputs \
  --epochs 100 \
  --batch-size 400 \
  --lr 0.156 \
  --ema-epochs 80 \
  --swa-epochs 20 \
  --workers 8 \
  --amp \
  --resume ./outputs/best_model.pth \
  2>&1 | tee -a training.log
```

---

## 🚨 **TROUBLESHOOTING GUIDE**

### **Issue 1: Out of Memory (OOM)**
```bash
# If batch size 400 causes OOM, reduce to 320
python train.py \
  --data /mnt/nvme_data/imagenet \
  --output-dir ./outputs \
  --epochs 90 \
  --batch-size 320 \
  --lr 0.125 \
  --ema-epochs 80 \
  --swa-epochs 10 \
  --workers 8 \
  --amp \
  --resume ./outputs/best_model.pth \
  2>&1 | tee -a training.log

# Or further reduce to 256 if still having issues
python train.py \
  --data /mnt/nvme_data/imagenet \
  --output-dir ./outputs \
  --epochs 90 \
  --batch-size 256 \
  --lr 0.1 \
  --ema-epochs 80 \
  --swa-epochs 10 \
  --workers 8 \
  --amp \
  --resume ./outputs/best_model.pth \
  2>&1 | tee -a training.log
```

### **Issue 2: Training Process Killed**
```bash
# Check if training is still running
tmux has-session -t training
ps aux | grep "python train.py"

# Force kill any stuck processes
pkill -9 -f "train.py"

# Resume training
python3 smart_resume.py
```

### **Issue 3: Low Accuracy at Milestones**
```bash
# Check dataset integrity
find /mnt/nvme_data/imagenet/train/ -name "*.JPEG" | wc -l  # Should be ~1,281,167
find /mnt/nvme_data/imagenet/val/ -name "*.JPEG" | wc -l    # Should be ~50,000

# Check GPU utilization (should be >95%)
nvidia-smi

# Check training logs for issues
tail -100 training.log | grep -E "(Loss|Acc|Error)"
```

### **Issue 4: Spot Instance Interruption**
```bash
# If instance is interrupted, just restart training on new instance
# Data is preserved on EBS, training will resume automatically
python3 smart_resume.py
```

---

## ✅ **SUCCESS CRITERIA**

**Training is considered successful if:**
- [x] Reaches >75% accuracy by epoch 80 (end of EMA phase)
- [x] Reaches >78% accuracy by epoch 90 (target completion)
- [x] Completes within 22-24 hours on g5.2xlarge
- [x] Stays within budget (~$30-40 for training)
- [x] No major data loss or corruption

**Final deliverables:**
- [x] Model weights (`best_model.pth`)
- [x] Training logs (`training.log`)
- [x] Milestone achievement report
- [x] GPU utilization analysis

---

## 📋 **EXECUTION CHECKLIST**

### **Pre-Training** ☐
- [ ] g5.2xlarge spot instance launched
- [ ] Python environment configured (Python 3.12 + PyTorch cu121)
- [ ] ImageNet dataset downloaded to `/mnt/nvme_data/imagenet`
- [ ] Training script syntax validated
- [ ] tmux sessions created (training, dashboard, milestones)

### **During Training** ☐
- [ ] Training started with batch size 400
- [ ] 4-pane monitoring dashboard active
- [ ] Milestone tracking running
- [ ] GPU utilization >95%, memory usage ~20GB/23GB
- [ ] Progress: Epoch 80 >75%, Epoch 90 >78%

### **Post-Training** ☐
- [ ] Final accuracy >78% achieved
- [ ] Model checkpoint downloaded
- [ ] Training logs preserved
- [ ] Instance terminated properly
- [ ] Total cost documented

---

**📅 Target Completion**: 22-24 hours from start  
**🎯 Success Metric**: >78% ImageNet Top-1 Accuracy at 90 epochs  
**💰 Expected Cost**: $30-40 (g5.2xlarge spot pricing)  
**🚀 Strategy**: 80 epochs EMA + 10 epochs SWA with batch size 400**