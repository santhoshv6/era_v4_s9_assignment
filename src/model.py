import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any
import math

# Export list for clean imports
__all__ = ['get_model', 'count_parameters', 'get_model_info']


def get_model(num_classes: int = 1000,
              model_name: str = 'resnet50',
              dropout: float = 0.0,
              device: Optional[str] = None) -> nn.Module:
    """Create a ResNet50 from scratch (no pretrained weights) with dynamic class support.
    
    This function automatically handles different dataset sizes:
    - Kaggle sample: ~20-100 classes  
    - Full ImageNet: 1000 classes
    - Custom datasets: any number of classes
    
    Designed for training from scratch to achieve 81% top-1 accuracy on ImageNet.

    Args:
        num_classes: Number of output classes (auto-detected from dataset)
        model_name: Model architecture (only 'resnet50' supported)
        dropout: Dropout rate before final classifier (usually 0.0 for ResNet50)
        device: Device to move model to ('cuda', 'cpu', or None for auto-detect)

    Returns:
        PyTorch model ready for from-scratch training with proper device placement
    """
    if model_name != 'resnet50':
        raise ValueError(f"Only ResNet50 supported, got {model_name}")

    print(f"🏗️ Creating ResNet50 from scratch for {num_classes} classes...")
    
    # Create ResNet50 with no pretrained weights - dynamic num_classes
    model = models.resnet50(weights=None, num_classes=num_classes)
    
    # Apply proper initialization for from-scratch training
    _init_weights(model)
    
    # Optional dropout before classifier
    if dropout > 0:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(  # type: ignore
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        # Re-initialize the new linear layer
        if isinstance(model.fc, nn.Sequential) and len(model.fc) > 1:
            linear_layer = model.fc[1]
            if isinstance(linear_layer, nn.Linear):
                nn.init.normal_(linear_layer.weight, 0, 0.01)
                if linear_layer.bias is not None:
                    nn.init.constant_(linear_layer.bias, 0)
    
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Move model to device
    model = model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model: ResNet50")
    print(f"📊 Parameters: {total_params:,}")  
    print(f"🎯 Classes: {num_classes}")
    print(f"💾 Device: {device}")
    
    return model


def _init_weights(model: nn.Module):
    """Apply He initialization for from-scratch ResNet training.
    
    This follows the initialization scheme from "Delving Deep into Rectifiers"
    which is crucial for training deep networks from scratch.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # He initialization for conv layers
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.BatchNorm2d):
            # BN layers: weight=1, bias=0
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.Linear):
            # Linear layers: normal init with small std
            nn.init.normal_(module.weight, 0, 0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get comprehensive model information."""
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
        'architecture': 'ResNet50'
    }


if __name__ == "__main__":
    # Test model creation
    model = get_model(num_classes=1000)
    info = get_model_info(model)
    print(f"Model: {info['architecture']}")
    print(f"Parameters: {info['trainable_parameters']/1e6:.2f}M")
    print(f"Model size: {info['model_size_mb']:.1f}MB")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {y.shape}")
