# ResNet50 ImageNet Training From Scratch 🚀

Training ResNet50 from scratch on ImageNet 1K to achieve **81% top-1 accuracy** - a feat accomplished by only ~10,000 people worldwide!

## 🎯 Assignment Overview

This project implements ResNet50 training from scratch (no pretrained weights) on ImageNet using:
- **Target**: 81% top-1 accuracy on ImageNet 1K validation set
- **Architecture**: ResNet50 with proper weight initialization
- **Training**: From random weights using advanced techniques
- **Scaling**: Sample validation on Kaggle → Full training on EC2

## 📁 Project Structure

```
├── src/
│   ├── model.py              # ResNet50 model implementation  
│   ├── transforms.py         # ImageNet data transforms & augmentation
│   ├── utils.py              # Training utilities, metrics, schedulers
│   ├── train.py              # Main training framework
│   ├── mixup.py              # Advanced augmentation (Mixup/CutMix)
│   ├── gradcam.py            # Model visualization tools
│   ├── enhanced_model.py     # Enhanced ResNet50 with stochastic depth
│   └── debug_synthetic_run.py # Synthetic data testing
├── imagenet_kaggle_notebook.ipynb  # Professional Kaggle notebook
├── logs/                     # Training logs and checkpoints
├── outputs/                  # Model outputs and results  
├── requirements.txt          # Python dependencies
└── README.md                # Documentation
```

## 🛠️ Model Architecture

**ResNet50 From Scratch**:
- Uses `torchvision.models.resnet50(weights=None)` - standard architecture, random initialization
- **25.6M parameters** total
- He initialization for Conv2D layers
- Proper BatchNorm initialization (weight=1, bias=0)
- No pretrained weights whatsoever

**Why ResNet50**: Proven architecture that can achieve 81% with proper training techniques.

## 📊 Model Architecture & Analysis

### Model Summary
| Component | Details |
|-----------|---------|
| **Architecture** | ResNet50 with Bottleneck blocks |
| **Total Parameters** | 25,557,032 |
| **Trainable Parameters** | 25,557,032 |
| **Model Size** | 97.5 MB |
| **Input Size** | 224×224×3 |
| **Output Classes** | 1000 (ImageNet) |
| **Approximate FLOPs** | 4.1 GFLOPs |

### Layer-wise Parameter Distribution
| Layer Type | Parameters | Percentage |
|------------|------------|-----------|
| **Final Classifier (fc)** | 2,049,000 | 8.0% |
| **Layer 4 Bottlenecks** | 14,942,720 | 58.4% |
| **Layer 3 Bottlenecks** | 6,039,552 | 23.6% |
| **Layer 2 Bottlenecks** | 1,512,448 | 5.9% |
| **Layer 1 Bottlenecks** | 379,392 | 1.5% |
| **Initial Conv + BN** | 9,472 | 0.04% |

### Receptive Field Analysis
| Layer | Kernel | Stride | Receptive Field | Output Size | Jump |
|-------|--------|--------|-----------------|-------------|------|
| Input | - | - | 1 | 224×224 | 1 |
| conv1 | 7×7 | 2 | 7 | 112×112 | 2 |
| maxpool | 3×3 | 2 | 11 | 56×56 | 4 |
| layer1 | 3×3 | 1 | 19 | 56×56 | 4 |
| layer2 | 3×3 | 2 | 27 | 28×28 | 8 |
| layer3 | 3×3 | 2 | 43 | 14×14 | 16 |
| layer4 | 3×3 | 2 | 75 | 7×7 | 32 |
| avgpool | 7×7 | 7 | 267 | 1×1 | 224 |

**Key Insights:**
- 🎯 **Final Receptive Field**: 267 pixels (119% of input image)
- ✅ **Full Coverage**: Receptive field covers entire 224×224 input
- 🔄 **Total Downsampling**: 32× (224→7 feature maps)
- 📊 **Feature Density**: 7×7×2048 = 100,352 features before classification

### Architecture Design Choices

**ImageNet-Specific Optimizations:**
- **7×7 Initial Conv**: Larger receptive field for high-resolution inputs
- **Stride-2 + MaxPool**: Aggressive early downsampling to manage computation
- **Bottleneck Blocks**: 1×1→3×3→1×1 design reduces parameters while maintaining capacity
- **Batch Normalization**: After every convolution for stable training
- **Global Average Pooling**: Replaces fully connected layers, reduces overfitting

**Training-from-Scratch Considerations:**
- **He Initialization**: Kaiming normal for ReLU networks
- **Zero-init Residual**: Last BN in each block initialized to zero
- **No Dropout**: ResNet50 typically doesn't use dropout (relies on residual connections)
- **Deep Architecture**: 50 layers provide sufficient capacity for ImageNet complexity

**Memory & Computation:**
- **Peak Memory**: ~8GB for batch_size=64 with mixed precision
- **Training Speed**: ~4.1 GFLOPs per forward pass
- **Gradient Memory**: ~2× model size during backpropagation

## 🧪 Quick Start: Kaggle Testing

### 1. Kaggle Setup
1. Create new Kaggle notebook with **GPU runtime** (T4/P100)
2. Upload the **`imagenet_kaggle_notebook.ipynb`** file  
3. Attach a small ImageNet subset dataset (2-10 classes, ~100 images each)
4. Enable internet if packages need installation

### 2. Expected Dataset Structure
```
/kaggle/input/imagenet-sample/
├── train/
│   ├── n01440764/  # Class folders with ImageNet naming
│   ├── n02123045/  
│   └── ...
└── val/
    ├── n01440764/  
    ├── n02123045/  
    └── ...
```

### 3. Run Notebook
- The notebook contains complete implementation with all required modules
- Tests synthetic data first, then real data if available
- Runs 5 epochs by default for quick validation
- Generates training logs in markdown format

### 4. Expected Output
```
📊 Training results after 5 epochs:
   - Training loss: ~6.0 → ~2.5 (decreasing)
   - Validation accuracy: 10% → 30%+ (increasing)
   - Checkpoints saved in ./outputs/
   - training_log.md with epoch-by-epoch results
```

## 🚀 Full Training: EC2 Setup

### Instance Requirements
- **Recommended**: `p3.8xlarge` (4x Tesla V100, 32 vCPUs, 244 GB RAM)
- **Alternative**: `p3.2xlarge` (1x Tesla V100) for single-GPU training
- **Storage**: 500GB+ EBS for ImageNet dataset
- **AMI**: Deep Learning AMI (Ubuntu) with PyTorch pre-installed

### EC2 Setup Commands
```bash
# 1. Launch instance and connect
ssh -i your-key.pem ubuntu@ec2-instance-ip

# 2. Clone repository
git clone https://github.com/your-username/resnet50-imagenet-scratch.git
cd resnet50-imagenet-scratch

# 3. Setup environment
conda activate pytorch_p310  # Or create new environment
pip install -r requirements.txt

# 4. Download ImageNet (if not already available)
# Note: You need ImageNet access - register at image-net.org
wget [ImageNet URL] -O imagenet.tar
tar -xf imagenet.tar

# 5. Verify structure
ls -la imagenet/
# Should show: train/ val/ (with 1000 class folders each)

# 6. Run full training
python -m src.train \
  --data ./imagenet \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.5 \
  --warmup-epochs 5 \
  --workers 16 \
  --amp \
  --output-dir ./outputs
```

### Multi-GPU Training (4x V100)
```bash
# Distributed training across 4 GPUs
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port=12345 \
  src/train.py \
  --data ./imagenet \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.5 \
  --warmup-epochs 5 \
  --workers 16 \
  --amp \
  --output-dir ./outputs
```

## 🏆 Model Comparison & Benchmarks

### ResNet Family Comparison
| Model | Parameters | FLOPs | Top-1 Acc* | Top-5 Acc* | Our Target |
|-------|------------|-------|-------------|-------------|------------|
| **ResNet50** | **25.6M** | **4.1G** | **76.1%** | **92.9%** | **🎯 81.0%** |
| ResNet34 | 21.8M | 3.7G | 73.3% | 91.4% | - |
| ResNet101 | 44.5M | 7.8G | 77.4% | 93.5% | - |
| ResNet152 | 60.2M | 11.6G | 78.3% | 94.1% | - |

*Standard ImageNet results with proper training

### Training Efficiency Analysis
| Metric | Value | Comparison |
|--------|-------|------------|
| **Parameters vs Accuracy** | 25.6M → 81% | Excellent efficiency |
| **FLOPs vs Accuracy** | 4.1G → 81% | Optimal for deployment |
| **Training Time** | ~12-48 hours | Reasonable on modern GPUs |
| **Memory Usage** | ~8GB (bs=64) | Fits on most modern GPUs |
| **Convergence Speed** | ~100 epochs | Standard for from-scratch |

## 📊 Training Hyperparameters for 81% Accuracy

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Epochs** | 100-200 | Start with 100, extend if needed |
| **Batch Size** | 256 (per GPU) | Scale with available GPUs |
| **Learning Rate** | 0.5 | For batch size 256; scale linearly |
| **LR Schedule** | Cosine + Warmup | 5-10 epoch warmup, then cosine decay |
| **Optimizer** | SGD + Nesterov | momentum=0.9, weight_decay=1e-4 |
| **Augmentation** | Strong | ColorJitter, RandomErasing, eventually Mixup |
| **Label Smoothing** | 0.1 | Regularization for from-scratch training |
| **Mixed Precision** | ✅ Enabled | Faster training, lower memory |

### Advanced Techniques for 81%
- **Mixup/CutMix**: Label and image mixing augmentation
- **EMA**: Exponential moving average of model weights  
- **AutoAugment**: Learned augmentation policies
- **Stochastic Depth**: Randomly skip residual blocks during training
- **Multi-Scale Training**: Vary input resolution during training

## 📈 Expected Training Progress

| Epoch Range | Expected Top-1 Accuracy | Notes |
|-------------|------------------------|-------|
| 1-10 | 5-25% | Initial learning, warmup phase |
| 11-30 | 25-50% | Rapid improvement |
| 31-60 | 50-70% | Steady progress |
| 61-100 | 70-76% | Baseline ResNet50 plateau |
| 100+ | 76-81% | Advanced techniques needed |

**To reach 81%**: Requires advanced augmentation (Mixup, CutMix), EMA, and potentially longer training.

## 📋 Assignment Deliverables Checklist

- [x] **Modular Code**: ✅ Separate model, transforms, utils, train modules
- [x] **Kaggle Notebook**: ✅ `imagenet_kaggle_notebook.ipynb` with full pipeline  
- [x] **Training Logs**: ✅ Markdown format with epoch-by-epoch results
- [ ] **EC2 Screenshot**: 📸 Evidence of EC2 training (TODO)
- [ ] **81% Accuracy**: 🎯 Target validation accuracy (TODO)
- [ ] **HuggingFace Space**: 🤗 Live inference application (TODO)
- [ ] **GitHub Repository**: 📱 Public repo with all code (TODO)

## 🏗️ Local Development Setup

```bash
# 1. Clone repository
git clone https://github.com/your-username/resnet50-imagenet-scratch.git
cd resnet50-imagenet-scratch

# 2. Create environment
conda create -n resnet50 python=3.10
conda activate resnet50

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test synthetic run (no data needed)
python -m src.debug_synthetic_run

# 5. Test with sample data
python -m src.train \
  --data /path/to/imagenet/sample \
  --epochs 2 \
  --batch-size 32 \
  --output-dir ./test_outputs
```

## 🔧 Technical Implementation Details

### Model Initialization
- **Conv2D**: He (Kaiming) normal initialization for ReLU networks
- **BatchNorm2D**: weight=1, bias=0 (standard)
- **Linear**: Normal distribution, std=0.01

### Data Pipeline
- **Training**: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomErasing
- **Validation**: Resize → CenterCrop → Normalize
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Training Loop
- **Mixed Precision**: Automatic Mixed Precision (AMP) for speed
- **Gradient Scaling**: Handle mixed precision gradients correctly
- **Checkpointing**: Save best model + latest model every N epochs
- **Logging**: Both console output and markdown file

## 🚨 Common Issues & Solutions

### Memory Issues
```bash
# Reduce batch size
--batch-size 128  # Instead of 256

# Reduce workers
--workers 4       # Instead of 16

# Enable gradient checkpointing (if implemented)
--gradient-checkpointing
```

### Slow Training
```bash
# Enable mixed precision
--amp

# Increase batch size (if memory allows)
--batch-size 512

# More workers (if CPU allows)
--workers 32
```

### Poor Convergence
```bash
# Longer warmup
--warmup-epochs 10

# Lower learning rate
--lr 0.1

# More regularization
--label-smoothing 0.2
```

## 🎯 Performance Targets

| Metric | Kaggle Demo | EC2 Full Training |
|--------|-------------|-------------------|
| **Runtime** | 5-30 minutes | 12-48 hours |
| **GPU Memory** | 4-6 GB | 15+ GB |
| **Top-1 Accuracy** | 30-60% (subset) | 81% (full ImageNet) |
| **Dataset Size** | 1K-10K images | 1.2M images |

## 📞 Support & Resources

- **ImageNet Access**: [image-net.org](http://image-net.org) registration required
- **Papers**: [Deep Residual Learning](https://arxiv.org/abs/1512.03385), [Bag of Tricks](https://arxiv.org/abs/1812.01187)
- **References**: [DAWNBench](https://dawn.cs.stanford.edu/benchmark/), [Papers With Code](https://paperswithcode.com/sota/image-classification-on-imagenet)

## 📜 License

MIT License - Feel free to use for educational purposes.

## 🎉 Achievement Unlock

Upon reaching 81% accuracy, you'll join an exclusive group of ~10,000 people worldwide who have successfully trained ImageNet from scratch! 🏆
