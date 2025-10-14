"""
GradCAM implementation for ResNet50 model visualization.
Helps understand what the model is learning during training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


class GradCAM:
    """
    GradCAM implementation for CNN models.
    """
    
    def __init__(self, model, target_layer_name: str):
        """
        Args:
            model: PyTorch model
            target_layer_name: Name of the target layer for GradCAM
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.handles = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        def forward_hook(module, input, output):
            self.activations = output
            
        # Find target layer and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                self.handles.append(module.register_forward_hook(forward_hook))
                self.handles.append(module.register_backward_hook(backward_hook))
                break
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input tensor of shape (1, C, H, W)
            class_idx: Target class index. If None, uses predicted class.
            
        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:], device=activations.device)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def visualize_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None, 
                      alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Visualize GradCAM overlay on input image.
        
        Args:
            input_tensor: Input tensor of shape (1, C, H, W)
            class_idx: Target class index
            alpha: Overlay alpha value
            colormap: OpenCV colormap
            
        Returns:
            Overlayed image as numpy array
        """
        # Generate CAM
        cam = self.generate_cam(input_tensor, class_idx)
        
        # Resize CAM to input size
        input_size = input_tensor.shape[-2:]
        cam_resized = cv2.resize(cam, input_size)
        
        # Convert input tensor to image
        img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        
        # Apply colormap to CAM
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        cam_colored = cam_colored / 255.0
        
        # Overlay
        overlayed = alpha * cam_colored + (1 - alpha) * np.expand_dims(img, -1)
        overlayed = np.clip(overlayed, 0, 1)
        
        return overlayed
    
    def cleanup(self):
        """Remove hooks."""
        for handle in self.handles:
            handle.remove()


def get_gradcam_target_layer(model_name: str = 'resnet50') -> str:
    """
    Get the appropriate target layer name for GradCAM based on model architecture.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Target layer name for GradCAM
    """
    if 'resnet' in model_name.lower():
        return 'layer4'  # Last convolutional layer before avg pool
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def visualize_predictions_with_gradcam(model, dataloader, device: str, num_samples: int = 8, 
                                     class_names: Optional[List[str]] = None):
    """
    Visualize model predictions with GradCAM for multiple samples.
    
    Args:
        model: Trained model
        dataloader: DataLoader
        device: Device to run on
        num_samples: Number of samples to visualize
        class_names: List of class names for display
    """
    model.eval()
    gradcam = GradCAM(model, get_gradcam_target_layer())
    
    fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 6))
    axes = axes.flatten()
    
    samples_processed = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            if samples_processed >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            
            for i in range(min(data.size(0), num_samples - samples_processed)):
                input_tensor = data[i:i+1]
                true_label = target[i].item()
                
                # Get prediction
                output = model(input_tensor)
                pred_label = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1)[0, pred_label].item()
                
                # Generate GradCAM
                cam_overlay = gradcam.visualize_cam(input_tensor, pred_label)
                
                # Plot
                ax = axes[samples_processed]
                ax.imshow(cam_overlay)
                ax.axis('off')
                
                # Title
                if class_names:
                    true_name = class_names[true_label]
                    pred_name = class_names[pred_label]
                    title = f'True: {true_name}\nPred: {pred_name} ({confidence:.2f})'
                else:
                    title = f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})'
                
                ax.set_title(title, fontsize=8)
                
                samples_processed += 1
                if samples_processed >= num_samples:
                    break
    
    plt.tight_layout()
    plt.savefig('gradcam_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    gradcam.cleanup()


# Test GradCAM functionality
if __name__ == "__main__":
    import torchvision.models as models
    
    # Create a test model
    model = models.resnet50(weights=None)
    model.eval()
    
    # Create test input
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # Initialize GradCAM
    gradcam = GradCAM(model, 'layer4')
    
    # Generate CAM
    cam = gradcam.generate_cam(input_tensor)
    print(f"✅ GradCAM generated successfully! CAM shape: {cam.shape}")
    
    # Cleanup
    gradcam.cleanup()
    print("✅ GradCAM test completed!")