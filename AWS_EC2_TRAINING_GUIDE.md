# AWS EC2 GPU Training Guide for ResNet50 ImageNet 🚀

## Your AWS Configuration
- **Service**: EC2 Spot Instances  
- **Region**: US East (Northern Virginia)
- **Instance Types**: All G and VT Spot Instance Requests
- **vCPU Limit**: 8 vCPUs

---

# Section 1: Kaggle Replication on EC2 (Free Tier Testing) 🧪

## Overview
This section replicates your Kaggle environment on EC2 without consuming significant AWS credits. Perfect for validating your training pipeline before scaling to full ImageNet.

## Target Specifications
- **Goal**: Replicate Kaggle sample training environment
- **Dataset**: ImageNet sample (100 classes, ~5K images)
- **Expected Accuracy**: 60-70% (same as Kaggle)
- **Training Time**: 2-4 hours
- **Estimated Cost**: $0.50-$1.50 (minimal credit usage)

## 🎯 Optimal Instance for Testing

## 🎯 Optimal Instance for Testing

### **Recommended Instance: g4dn.xlarge (FREE TIER FRIENDLY)**

| Specification | Value | Why This Matters |
|---------------|-------|------------------|
| **vCPUs** | 4 | Half of your limit, minimal cost |
| **Memory** | 16 GB | Sufficient for sample dataset |
| **GPU** | 1x NVIDIA T4 (16GB) | Same as Kaggle P100 performance |
| **Storage** | 125 GB NVMe SSD | Fast I/O for sample data |
| **Network** | Up to 25 Gbps | Fast downloads |
| **On-Demand Price** | $0.526/hour | Standard pricing |
| **Spot Price** | $0.113-0.151/hour | **~75% savings** |

### **Why g4dn.xlarge for Testing:**
- ✅ **Minimal Cost**: Only $0.15/hour spot pricing
- ✅ **Quick Validation**: 2-4 hour runs cost under $1
- ✅ **Identical Environment**: Same GPU performance as Kaggle
- ✅ **Safe Testing**: Won't exhaust free credits

## 💰 Section 1 Cost Analysis

```
Dataset: ImageNet Sample (100 classes, ~5K images)
Training Duration: 2-4 hours (15-20 epochs)
Spot Price: $0.113-0.151/hour

Total Cost for Testing:
- Conservative: 4 hours × $0.151 = $0.60
- Optimistic: 2 hours × $0.113 = $0.23
- Realistic: 3 hours × $0.132 = $0.40

Additional Costs:
- Storage (50GB EBS): ~$0.25
- Data Transfer: ~$0.10 (sample download)
- Total Testing Cost: $0.50-$1.00
```

## 🛠️ Section 1: Quick Setup Guide

### **Phase 1A: Launch Test Instance**

#### Step 1: Create Minimal Spot Request
```bash
# Create test-spot-request.json
cat > test-spot-request.json << 'EOF'
{
  "ImageId": "ami-0c02fb55956c7d316",
  "InstanceType": "g4dn.xlarge",
  "KeyName": "your-key-pair-name",
  "SecurityGroupIds": ["sg-your-security-group"],
  "BlockDeviceMappings": [
    {
      "DeviceName": "/dev/xvda",
      "Ebs": {
        "VolumeSize": 50,
        "VolumeType": "gp3",
        "DeleteOnTermination": true
      }
    }
  ]
}
EOF

# Launch small test instance
aws ec2 request-spot-instances \
  --spot-price "0.20" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://test-spot-request.json \
  --region us-east-1
```

#### Step 2: Quick Environment Setup
```bash
# Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install essentials
sudo apt update
sudo apt install -y nvidia-driver-470 python3-pip git
sudo reboot

# After reboot, setup Python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install tqdm numpy matplotlib Pillow
```

#### Step 3: Download Sample Dataset
```bash
# Create sample ImageNet (replicating Kaggle)
mkdir -p ~/imagenet_sample

# Download ImageNet sample (100 classes)
wget https://github.com/fastai/imagenette/releases/download/v2/imagenette2-320.tgz
tar -xzf imagenette2-320.tgz
mv imagenette2-320 ~/imagenet_sample/

# Organize like full ImageNet structure
cd ~/imagenet_sample
mkdir -p train val
mv imagenette2-320/train/* train/
mv imagenette2-320/val/* val/
```

#### Step 4: Run Test Training
```bash
# Download your training code
git clone https://github.com/your-username/resnet50-imagenet-scratch.git
cd resnet50-imagenet-scratch

# Run quick test (15 epochs, same as Kaggle)
python -m src.train \
  --data ~/imagenet_sample \
  --epochs 15 \
  --batch-size 32 \
  --lr 0.1 \
  --workers 4 \
  --amp \
  --output-dir ./test_outputs \
  2>&1 | tee test_training.log

# Expected result: 60-70% accuracy in 2-4 hours
```

#### Step 5: Validate and Cleanup
```bash
# Check final accuracy
tail -20 test_training.log

# Download results
scp -i your-key.pem ubuntu@your-instance-ip:~/resnet50-imagenet-scratch/test_outputs ./

# Terminate instance to stop charges
aws ec2 terminate-instances --instance-ids i-your-instance-id
```

---

# Section 2: Full ImageNet Training for 81% Target 🎯

## Overview
This section provides the complete setup for training ResNet50 from scratch on full ImageNet dataset to achieve 81% top-1 accuracy with optimized cost-performance balance.

## Target Specifications
- **Goal**: 81%+ top-1 accuracy on ImageNet 1K
- **Dataset**: Full ImageNet (1000 classes, 1.3M images)
- **Training Time**: 60-80 hours
- **Estimated Cost**: $15-25 total

## 🎯 Optimal Instance for Production Training

### **Recommended Instance: g4dn.2xlarge (OPTIMAL BALANCE)**

| Specification | Value | Why This Matters |
|---------------|-------|------------------|
| **vCPUs** | 8 | Exactly matches your limit |
| **Memory** | 32 GB | Required for full ImageNet dataset |
| **GPU** | 1x NVIDIA T4 (16GB) | Sufficient VRAM for batch_size=64 |
| **Storage** | 225 GB NVMe SSD | Fast I/O for 1.3M images |
| **Network** | Up to 25 Gbps | Fast dataset downloads |
| **On-Demand Price** | $0.752/hour | Premium pricing |
| **Spot Price** | $0.226-0.301/hour | **~70% savings** |

### **Cost-Performance Analysis:**

| Option | Instance | GPU | Training Time | Spot Cost | Total Cost | Speed/Cost Ratio |
|--------|----------|-----|---------------|-----------|------------|------------------|
| **Balanced** | g4dn.2xlarge | T4 | 70 hours | $0.264/hr | **$18.48** | **Best Value** |
| Budget | g4dn.xlarge | T4 | 140 hours | $0.132/hr | $18.48 | Slower |
| Performance | p3.2xlarge | V100 | 45 hours | $0.918/hr | $41.31 | Expensive |

**Recommendation: g4dn.2xlarge offers the optimal balance of speed and cost.**

## 💰 Section 2 Cost Analysis

```
Dataset: Full ImageNet (1000 classes, 1.3M images)
Training Duration: 60-80 hours (100 epochs)
Spot Price: $0.226-0.301/hour (average $0.264/hour)

Total Cost Estimate:
- Conservative: 80 hours × $0.301 = $24.08
- Optimistic: 60 hours × $0.226 = $13.56
- Realistic: 70 hours × $0.264 = $18.48

Additional Costs:
- Storage (500GB EBS): ~$2.50
- Data Transfer: ~$1-2 (ImageNet download)
- Total Project Cost: $15-25
```

### **Cost Optimization Strategies:**
1. **Spot Instances**: 60-70% savings vs On-Demand
2. **Off-Peak Training**: Lower weekend spot prices
3. **Checkpoint Frequency**: Save every 2 epochs for preemption recovery
4. **Storage Management**: Delete intermediate data after training

## 🛠️ Section 2: Production Setup Guide

### **Phase 2A: Pre-Setup Preparation**
## 🛠️ Section 2: Production Setup Guide

### **Phase 2A: Pre-Setup Preparation**

#### Step 1: Verify AWS Configuration
```bash
# Check your current vCPU limits
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-34B43A08 \
  --region us-east-1

# Check current spot pricing for g4dn.2xlarge
aws ec2 describe-spot-price-history \
  --instance-types g4dn.2xlarge \
  --product-descriptions "Linux/UNIX" \
  --region us-east-1 \
  --max-items 5
```

#### Step 2: Create IAM Role for EC2
```bash
# Create role with necessary permissions
aws iam create-role \
  --role-name EC2-ImageNet-Role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {"Service": "ec2.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }
    ]
  }'

# Attach policies for S3 access and CloudWatch logging
aws iam attach-role-policy \
  --role-name EC2-ImageNet-Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name EC2-ImageNet-Role \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy

# Create instance profile
aws iam create-instance-profile --instance-profile-name EC2-ImageNet-Profile
aws iam add-role-to-instance-profile \
  --instance-profile-name EC2-ImageNet-Profile \
  --role-name EC2-ImageNet-Role
```

### **Phase 2B: Instance Launch and Setup**

#### Step 3: Create Production Spot Instance
```bash
# Create production-spot-request.json
cat > production-spot-request.json << 'EOF'
{
  "ImageId": "ami-0c02fb55956c7d316",
  "InstanceType": "g4dn.2xlarge",
  "KeyName": "your-key-pair-name",
  "SecurityGroupIds": ["sg-your-security-group"],
  "SubnetId": "subnet-your-subnet-id",
  "IamInstanceProfile": {
    "Name": "EC2-ImageNet-Profile"
  },
  "BlockDeviceMappings": [
    {
      "DeviceName": "/dev/xvda",
      "Ebs": {
        "VolumeSize": 500,
        "VolumeType": "gp3",
        "Iops": 3000,
        "Throughput": 125,
        "DeleteOnTermination": true,
        "Encrypted": false
      }
    }
  ],
  "UserData": "IyEvYmluL2Jhc2gKYXB0IHVwZGF0ZSAteQphcHQgaW5zdGFsbCAteSBweXRob24zLXBpcCBnaXQgaHRvcCBud3JpdGU="
}
EOF

# Launch production spot instance
aws ec2 request-spot-instances \
  --spot-price "0.35" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://production-spot-request.json \
  --region us-east-1

# Monitor spot request status
aws ec2 describe-spot-instance-requests \
  --region us-east-1 \
  --query 'SpotInstanceRequests[?State==`active`].[SpotInstanceRequestId,InstanceId,State,Status]' \
  --output table
```

#### Step 4: Connect and Initial Setup
```bash
# Find your instance IP
INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
  --region us-east-1 \
  --query 'SpotInstanceRequests[?State==`active`].InstanceId' \
  --output text)

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"

# Connect via SSH
ssh -i your-key.pem ubuntu@$PUBLIC_IP
```

### **Phase 2C: Production Environment Setup**

#### Step 5: Install NVIDIA Drivers and CUDA
```bash
# Update system and install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake git wget curl htop nvtop

# Install NVIDIA drivers
sudo apt install -y nvidia-driver-470
sudo reboot

# After reboot, verify GPU
nvidia-smi

# Install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit

# Configure environment
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
```

#### Step 6: Setup Python Environment
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Create dedicated environment
conda create -n resnet50-production python=3.10 -y
conda activate resnet50-production

# Install PyTorch with CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu118

# Install training dependencies
pip install tqdm numpy matplotlib Pillow tensorboard wandb
pip install timm torchinfo scipy scikit-learn

# Verify GPU setup
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'GPU Count: {torch.cuda.device_count()}')
print(f'GPU Name: {torch.cuda.get_device_name(0)}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

### **Phase 2D: Dataset Preparation**

#### Step 7: Download Full ImageNet Dataset
```bash
# Create optimized data directory on fast NVMe storage
sudo mkdir -p /mnt/nvme_data
sudo chown ubuntu:ubuntu /mnt/nvme_data
cd /mnt/nvme_data

# Method 1: Academic Torrents (Recommended)
pip install academictorrents
python -c "
import academictorrents as at
# Download ImageNet 2012 training set
at.get('CLS-LOC', datastore='/mnt/nvme_data')
"

# Method 2: Direct download (if you have ImageNet credentials)
# Place your downloaded ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar here

# Extract and organize training data
mkdir -p imagenet/{train,val}
cd imagenet

# Extract training data
tar -xf ../ILSVRC2012_img_train.tar -C train/
cd train
for f in *.tar; do
  d=$(basename "$f" .tar)
  mkdir -p "$d"
  tar -xf "$f" -C "$d"
  rm "$f"
done
cd ..

# Extract and organize validation data
tar -xf ../ILSVRC2012_img_val.tar -C val/

# Download validation ground truth and organize
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
chmod +x valprep.sh
cd val && ../valprep.sh && cd ..

# Verify dataset structure
echo "Training classes: $(ls train/ | wc -l)"
echo "Training images: $(find train/ -name "*.JPEG" | wc -l)"
echo "Validation classes: $(ls val/ | wc -l)"
echo "Validation images: $(find val/ -name "*.JPEG" | wc -l)"
```

#### Step 8: Setup Training Code
```bash
# Clone training repository
cd /home/ubuntu
git clone https://github.com/your-username/resnet50-imagenet-scratch.git
cd resnet50-imagenet-scratch

# Verify all modules are present
ls -la src/
# Expected: model.py, transforms.py, utils.py, train.py, mixup.py, gradcam.py

# Create output directory
mkdir -p outputs/{checkpoints,logs,results}
```

### **Phase 2E: Production Training Execution**

#### Step 9: Launch Full Training
```bash
# Create comprehensive training script
cat > start_production_training.sh << 'EOF'
#!/bin/bash
set -e

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# Activate environment
source ~/miniconda3/bin/activate resnet50-production

# Training configuration
DATA_DIR="/mnt/nvme_data/imagenet"
OUTPUT_DIR="./outputs"
EPOCHS=100
BATCH_SIZE=64
LR=0.1
WORKERS=8

# Comprehensive training command
python -m src.train \
  --data "$DATA_DIR" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --weight-decay 2e-4 \
  --momentum 0.9 \
  --warmup-epochs 5 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.4 \
  --cutmix-alpha 1.0 \
  --workers $WORKERS \
  --amp \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-freq 2 \
  --print-freq 100 \
  --save-best-only \
  --resume-latest \
  2>&1 | tee "$OUTPUT_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"
EOF

chmod +x start_production_training.sh

# Start training in persistent session
screen -S production_training
./start_production_training.sh

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r production_training
```

#### Step 10: Monitoring and Management
```bash
# Real-time monitoring script
cat > monitor_training.sh << 'EOF'
#!/bin/bash
while true; do
  clear
  echo "=== GPU Status ==="
  nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
  
  echo -e "\n=== Training Progress ==="
  tail -10 outputs/logs/training_*.log | grep -E "(Epoch|Best|Acc|Loss)"
  
  echo -e "\n=== System Resources ==="
  echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
  echo "Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
  echo "Disk: $(df -h /mnt/nvme_data | awk 'NR==2 {print $3 "/" $2 " (" $5 " used)"}')"
  
  echo -e "\n=== Spot Instance Status ==="
  curl -s http://169.254.169.254/latest/meta-data/spot/instance-action || echo "No interruption notice"
  
  sleep 30
done
EOF

chmod +x monitor_training.sh

# Run monitoring in separate screen
screen -S monitoring
./monitor_training.sh
```

### **Phase 2F: Checkpointing and Recovery**

#### Step 11: Automated Recovery System
```bash
# Create recovery script for spot interruptions
cat > recovery_system.sh << 'EOF'
#!/bin/bash

# Find latest checkpoint
LATEST_CHECKPOINT=$(find outputs/checkpoints -name "checkpoint_epoch_*.pth" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

if [ -f "$LATEST_CHECKPOINT" ]; then
    EPOCH=$(echo "$LATEST_CHECKPOINT" | grep -o 'epoch_[0-9]*' | grep -o '[0-9]*')
    echo "Found checkpoint from epoch $EPOCH: $LATEST_CHECKPOINT"
    
    # Resume training
    source ~/miniconda3/bin/activate resnet50-production
    python -m src.train \
      --data "/mnt/nvme_data/imagenet" \
      --epochs 100 \
      --batch-size 64 \
      --lr 0.1 \
      --weight-decay 2e-4 \
      --momentum 0.9 \
      --warmup-epochs 5 \
      --label-smoothing 0.1 \
      --mixup-alpha 0.4 \
      --cutmix-alpha 1.0 \
      --workers 8 \
      --amp \
      --output-dir "./outputs" \
      --checkpoint-freq 2 \
      --resume "$LATEST_CHECKPOINT" \
      2>&1 | tee -a "./outputs/logs/recovery_$(date +%Y%m%d_%H%M%S).log"
else
    echo "No checkpoint found, starting fresh training"
    ./start_production_training.sh
fi
EOF

chmod +x recovery_system.sh

# Setup automatic checkpoint backup to S3 (optional)
cat > backup_checkpoints.sh << 'EOF'
#!/bin/bash
if [ -f outputs/checkpoints/checkpoint_epoch_latest.pth ]; then
    aws s3 cp outputs/checkpoints/ s3://your-bucket/resnet50-checkpoints/ --recursive
    echo "Checkpoints backed up to S3"
fi
EOF

chmod +x backup_checkpoints.sh

# Add to crontab for hourly backups
echo "0 * * * * /home/ubuntu/resnet50-imagenet-scratch/backup_checkpoints.sh" | crontab -
```

## 📊 Expected Training Metrics and Timeline

### **Training Progress Milestones:**

| Epoch Range | Expected Top-1 Acc | Time Elapsed | GPU Utilization | Memory Usage |
|-------------|-------------------|--------------|-----------------|--------------|
| 1-10 | 15-30% | 6-8 hours | >95% | ~14GB/16GB |
| 11-25 | 30-55% | 15-20 hours | >95% | ~14GB/16GB |
| 26-50 | 55-70% | 30-40 hours | >95% | ~14GB/16GB |
| 51-75 | 70-78% | 45-60 hours | >95% | ~14GB/16GB |
| 76-100 | 78-82% | 60-80 hours | >95% | ~14GB/16GB |

### **Key Performance Indicators:**

- **Target Accuracy**: 81.0% top-1, 95.5% top-5
- **Training Speed**: ~1.5-2.0 hours per epoch
- **GPU Utilization**: Should maintain >95%
- **Memory Usage**: ~14GB out of 16GB available
- **Convergence**: Loss should decrease steadily, accuracy should plateau around epoch 80

## 🚨 Troubleshooting Guide

### **Common Issues and Solutions:**

#### **1. GPU Out of Memory**
```bash
# Reduce batch size
python -m src.train --batch-size 32  # Instead of 64

# Enable gradient accumulation
python -m src.train --batch-size 32 --accumulate-grad-batches 2
```

#### **2. Slow Training Speed**
```bash
# Check data loading bottleneck
python -c "
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Test data loading speed
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.ImageFolder('/mnt/nvme_data/imagenet/train', transform=transform)
loader = DataLoader(dataset, batch_size=64, num_workers=8, pin_memory=True)

start_time = time.time()
for i, (data, target) in enumerate(loader):
    if i >= 100:  # Test 100 batches
        break
elapsed = time.time() - start_time
print(f'Data loading speed: {100 * 64 / elapsed:.1f} samples/sec')
"

# Optimize if needed
# - Increase num_workers to 16
# - Move data to faster storage
# - Use pin_memory=True
```

#### **3. Spot Instance Interruption**
```bash
# Check interruption warning
curl -s http://169.254.169.254/latest/meta-data/spot/instance-action

# If interruption notice received, run emergency checkpoint
pkill -TERM -f "src.train"  # Graceful shutdown
sleep 30  # Wait for checkpoint save

# After new instance launch
./recovery_system.sh
```

#### **4. Training Not Converging**
```bash
# Check learning rate schedule
grep -E "(lr|LR)" outputs/logs/training_*.log

# Verify data augmentation is working
python -c "
from src.transforms import get_train_transforms
transform = get_train_transforms()
print('Transforms:', transform)
"

# Check loss values
grep "Loss" outputs/logs/training_*.log | tail -20
```

## 💰 Cost Optimization Tips

### **1. Spot Price Monitoring**
```bash
# Check current spot prices
aws ec2 describe-spot-price-history \
  --instance-types g4dn.2xlarge \
  --product-descriptions "Linux/UNIX" \
  --region us-east-1 \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S)

# Switch regions if prices are high
aws ec2 describe-spot-price-history \
  --instance-types g4dn.2xlarge \
  --product-descriptions "Linux/UNIX" \
  --region us-west-2
```

### **2. Training Schedule Optimization**
- **Best Times**: Weekends and off-peak hours (typically 20% cheaper)
- **Avoid**: Monday mornings and business hours (high demand)
- **Monitor**: Set up price alerts for optimal training windows

### **3. Resource Management**
```bash
# Clean up unnecessary files
rm -rf /tmp/*
rm -rf ~/.cache/pip/*

# Compress logs periodically
gzip outputs/logs/training_*.log

# Use smaller checkpoint frequency for final epochs
# Change --checkpoint-freq from 2 to 5 after epoch 80
```

## 📋 Success Checklist

### **Pre-Training Verification:**
- [ ] g4dn.2xlarge spot instance running
- [ ] NVIDIA drivers and CUDA 11.8 installed
- [ ] PyTorch 2.0.1 with CUDA support verified
- [ ] Full ImageNet dataset (1.3M images) downloaded and organized
- [ ] All training modules present in src/ directory
- [ ] Output directories created with proper permissions
- [ ] Screen sessions configured for persistent training

### **During Training Monitoring:**
- [ ] GPU utilization consistently >95%
- [ ] Memory usage stable around 14GB
- [ ] Training loss decreasing steadily
- [ ] Validation accuracy improving
- [ ] Checkpoints being saved every 2 epochs
- [ ] No spot interruption warnings
- [ ] Log files being written correctly

### **Post-Training Validation:**
- [ ] Final top-1 accuracy ≥81%
- [ ] Final top-5 accuracy ≥95%
- [ ] Training completed within budget (<$25)
- [ ] Final model weights saved
- [ ] Training logs and metrics exported
- [ ] Instance terminated to stop charges

## 🎯 Expected Outcomes

### **Success Metrics:**
- **Accuracy**: 81.0-82.5% top-1 on ImageNet validation
- **Training Time**: 60-80 hours total
- **Total Cost**: $15-25 (including all AWS charges)
- **Model Size**: ~98MB (25.6M parameters)
- **Inference Speed**: ~50ms per image on T4 GPU

### **Deliverables:**
1. Trained ResNet50 model achieving 81%+ accuracy
2. Complete training logs and metrics
3. Model checkpoints and final weights
4. Performance analysis and visualizations
5. Cost breakdown and optimization recommendations

---

**🎉 You now have a complete guide to train ResNet50 from scratch on AWS EC2!**

**Section 1** gets you started with minimal cost ($0.50-$1.00) to validate your setup.
**Section 2** provides the full production pipeline to achieve 81% accuracy for under $25.

Both sections are optimized for your 8 vCPU limit and provide maximum cost-efficiency while maintaining training quality. 🚀
```bash
# Create role with necessary permissions
aws iam create-role \
  --role-name EC2-ImageNet-Role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {"Service": "ec2.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }
    ]
  }'

# Attach policies
aws iam attach-role-policy \
  --role-name EC2-ImageNet-Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### **Phase 2: Launch EC2 Instance**

#### Step 3: Create Spot Instance Request
```bash
# Create spot-instance-request.json
cat > spot-instance-request.json << 'EOF'
{
  "ImageId": "ami-0c02fb55956c7d316",
  "InstanceType": "g4dn.2xlarge",
  "KeyName": "your-key-pair-name",
  "SecurityGroupIds": ["sg-your-security-group"],
  "SubnetId": "subnet-your-subnet-id",
  "IamInstanceProfile": {
    "Name": "EC2-ImageNet-Role"
  },
  "BlockDeviceMappings": [
    {
      "DeviceName": "/dev/xvda",
      "Ebs": {
        "VolumeSize": 500,
        "VolumeType": "gp3",
        "DeleteOnTermination": true
      }
    }
  ],
  "UserData": "IyEvYmluL2Jhc2gKYXB0IHVwZGF0ZSAteQphcHQgaW5zdGFsbCAteSBweXRob24zLXBpcCBnaXQgaHRvcCBud3JpdGU="
}
EOF

# Launch spot instance
aws ec2 request-spot-instances \
  --spot-price "0.35" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://spot-instance-request.json \
  --region us-east-1
```

#### Step 4: Connect to Instance
```bash
# Find your instance IP
aws ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running" \
  --query 'Reservations[*].Instances[*].[InstanceId,PublicIpAddress]' \
  --output table

# Connect via SSH
ssh -i your-key.pem ubuntu@your-instance-ip
```

### **Phase 3: Environment Setup**

#### Step 5: Install NVIDIA Drivers & CUDA
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers
sudo apt install -y nvidia-driver-470
sudo reboot

# After reboot, verify GPU
nvidia-smi

# Install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Step 6: Install Python Environment
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init
source ~/.bashrc

# Create environment
conda create -n resnet50 python=3.10 -y
conda activate resnet50

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies
pip install tqdm numpy matplotlib Pillow tensorboard wandb
```

#### Step 7: Verify GPU Setup
```bash
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'GPU Count: {torch.cuda.device_count()}')
print(f'GPU Name: {torch.cuda.get_device_name(0)}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

### **Phase 4: Dataset Preparation**

#### Step 8: Download ImageNet Dataset
```bash
# Create data directory
mkdir -p /home/ubuntu/imagenet

# Option 1: Download from Academic Torrents (Recommended)
pip install academictorrents
python -c "
import academictorrents as at
at.get('CLS-LOC/b5d6675b1ac005fa3f8c8ad9-ee70c7a20d2ff1f7'
at.get('CLS-LOC/b5d6675b1ac005fa3f8c8ad9-ee70c7a20d2ff1f7', datastore='/home/ubuntu/imagenet')
"

# Option 2: Download directly (if you have ImageNet account)
# wget --user=your_username --password=your_password \
#   http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar
#   http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar

# Extract and organize
cd /home/ubuntu/imagenet
tar -xf ILSVRC2012_img_train.tar
tar -xf ILSVRC2012_img_val.tar

# Organize training data
mkdir train
mv ILSVRC2012_img_train.tar train/
cd train
tar -xf ILSVRC2012_img_train.tar
rm ILSVRC2012_img_train.tar

# Extract individual class folders
for f in *.tar; do
  d=$(basename "$f" .tar)
  mkdir "$d"
  tar -xf "$f" -C "$d"
  rm "$f"
done

# Organize validation data (use provided script)
cd ../
python organize_val_data.py
```

#### Step 9: Download Training Code
```bash
# Clone your repository
cd /home/ubuntu
git clone https://github.com/your-username/resnet50-imagenet-scratch.git
cd resnet50-imagenet-scratch

# Verify all files are present
ls -la src/
# Should see: model.py, transforms.py, utils.py, train.py, mixup.py
```

### **Phase 5: Training Execution**

#### Step 10: Start Training with Monitoring
```bash
# Create training script
cat > start_training.sh << 'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Start training with comprehensive logging
python -m src.train \
  --data /home/ubuntu/imagenet \
  --epochs 100 \
  --batch-size 64 \
  --lr 0.1 \
  --weight-decay 2e-4 \
  --momentum 0.9 \
  --warmup-epochs 5 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.4 \
  --cutmix-alpha 1.0 \
  --workers 8 \
  --amp \
  --output-dir ./outputs \
  --checkpoint-freq 2 \
  --print-freq 100 \
  2>&1 | tee training.log
EOF

chmod +x start_training.sh

# Start training in screen session (survives disconnection)
screen -S training
./start_training.sh

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r training
```

#### Step 11: Monitoring & Checkpointing
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training progress
tail -f training.log

# Check checkpoints
ls -la outputs/checkpoints/

# Monitor disk space
df -h

# Check spot instance status
aws ec2 describe-spot-instance-requests \
  --region us-east-1 \
  --query 'SpotInstanceRequests[?State==`active`].[InstanceId,SpotPrice,Status]'
```

### **Phase 6: Spot Instance Preemption Handling**

#### Step 12: Automatic Checkpoint Recovery
```bash
# Create recovery script
cat > resume_training.sh << 'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Find latest checkpoint
LATEST_CHECKPOINT=$(ls -t outputs/checkpoints/checkpoint_epoch_*.pth | head -n 1)

if [ -f "$LATEST_CHECKPOINT" ]; then
    echo "Resuming from $LATEST_CHECKPOINT"
    python -m src.train \
      --data /home/ubuntu/imagenet \
      --epochs 100 \
      --batch-size 64 \
      --lr 0.1 \
      --weight-decay 2e-4 \
      --momentum 0.9 \
      --warmup-epochs 5 \
      --label-smoothing 0.1 \
      --mixup-alpha 0.4 \
      --cutmix-alpha 1.0 \
      --workers 8 \
      --amp \
      --output-dir ./outputs \
      --checkpoint-freq 2 \
      --resume "$LATEST_CHECKPOINT" \
      2>&1 | tee -a training.log
else
    echo "No checkpoint found, starting fresh training"
    ./start_training.sh
fi
EOF

chmod +x resume_training.sh
```

#### Step 13: Setup Spot Instance Interruption Handling
```bash
# Install AWS CLI and setup metadata monitoring
pip install boto3

# Create interruption monitor
cat > monitor_spot.py << 'EOF'
import requests
import time
import subprocess
import sys

def check_spot_interruption():
    try:
        response = requests.get(
            'http://169.254.169.254/latest/meta-data/spot/instance-action',
            timeout=1
        )
        return response.status_code == 200
    except:
        return False

def save_emergency_checkpoint():
    print("Spot instance interruption detected! Saving emergency checkpoint...")
    # Send SIGTERM to training process to trigger checkpoint save
    subprocess.run(['pkill', '-TERM', '-f', 'src.train'])
    time.sleep(30)  # Give time for checkpoint save

if __name__ == "__main__":
    while True:
        if check_spot_interruption():
            save_emergency_checkpoint()
            sys.exit(0)
        time.sleep(5)
EOF

# Run monitor in background
nohup python monitor_spot.py &
```

## 📊 Expected Training Progress

### **Timeline Expectations:**
- **Epochs 1-20**: Rapid accuracy improvement (20% → 50%)
- **Epochs 21-50**: Steady improvement (50% → 70%)
- **Epochs 51-80**: Gradual improvement (70% → 78%)
- **Epochs 81-100**: Fine-tuning (78% → 81%)

### **Key Milestones:**
| Epoch | Expected Top-1 Acc | Time Elapsed | Notes |
|-------|-------------------|--------------|-------|
| 5 | 25-35% | 3-4 hours | Warmup complete |
| 20 | 50-60% | 12-16 hours | Model learning features |
| 40 | 65-72% | 24-32 hours | Good progress |
| 60 | 72-77% | 36-48 hours | Approaching target |
| 80 | 77-80% | 48-64 hours | Close to goal |
| 100 | 80-82% | 60-80 hours | Target achieved |

## 🚨 Troubleshooting Guide

### **Common Issues & Solutions:**

#### GPU Out of Memory
```bash
# Reduce batch size
python -m src.train --batch-size 32  # Instead of 64

# Enable gradient checkpointing
python -m src.train --gradient-checkpointing
```

#### Slow Data Loading
```bash
# Increase workers
python -m src.train --workers 16  # Instead of 8

# Move data to instance storage
sudo mkfs.ext4 /dev/nvme1n1
sudo mount /dev/nvme1n1 /mnt/fast_storage
cp -r /home/ubuntu/imagenet /mnt/fast_storage/
```

#### Spot Instance Interruption
```bash
# Check if training can resume
./resume_training.sh

# If no checkpoint, restart completely
./start_training.sh
```

#### High Costs
```bash
# Check current pricing
aws ec2 describe-spot-price-history \
  --instance-types g4dn.2xlarge \
  --product-descriptions "Linux/UNIX" \
  --region us-east-1

# Consider smaller instance
aws ec2 request-spot-instances --instance-type g4dn.xlarge
```

## 🎯 Success Metrics

### **Training Success Indicators:**
- ✅ **Top-1 Accuracy > 81%** on ImageNet validation
- ✅ **Top-5 Accuracy > 95%** on ImageNet validation  
- ✅ **Training Loss < 0.5** in final epochs
- ✅ **Validation Loss < 1.0** in final epochs
- ✅ **No overfitting** (train-val gap < 5%)

### **Cost Success Indicators:**
- ✅ **Total Cost < $25** for complete training
- ✅ **Cost per % accuracy < $0.30** 
- ✅ **Spot instance savings > 60%**

## 📋 Final Checklist

### **Before Starting Training:**
- [ ] vCPU limit verified (8 vCPUs)
- [ ] Spot instance pricing checked
- [ ] IAM roles configured
- [ ] Security groups allow SSH
- [ ] Key pair created and downloaded
- [ ] ImageNet dataset access confirmed

### **During Training:**
- [ ] Monitor GPU utilization (should be >90%)
- [ ] Check training logs every 2-4 hours
- [ ] Verify checkpoints are being saved
- [ ] Monitor spot pricing trends
- [ ] Watch for interruption warnings

### **After Training:**
- [ ] Final accuracy > 81%
- [ ] Save final model weights
- [ ] Download training logs
- [ ] Terminate instance to stop charges
- [ ] Clean up EBS volumes

## 💡 Pro Tips

1. **Start Training on Friday Evening**: Lower weekend spot prices
2. **Use Multiple Small Instances**: If g4dn.2xlarge unavailable, use 2x g4dn.xlarge
3. **Monitor Spot Pricing**: Switch regions if prices spike
4. **Backup Checkpoints**: Copy to S3 every 10 epochs
5. **Use Tensorboard**: Monitor training curves in real-time
6. **Test First**: Run 2-3 epochs on sample data to verify setup

---

**🎉 You're now ready to train ResNet50 from scratch on AWS EC2!**

Expected outcome: **81%+ top-1 accuracy** for under **$25** in cloud costs. 🚀