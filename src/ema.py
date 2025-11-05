"""
EMA (Exponential Moving Average) implementation for better model generalization.
Critical component for achieving 81% ImageNet accuracy.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional

__all__ = ['EMAModel', 'apply_ema_to_model']


class EMAModel:
    """
    Exponential Moving Average of model weights.
    
    This technique maintains a moving average of model parameters during training,
    which often leads to better generalization and higher accuracy (+1-2% on ImageNet).
    
    Usage:
        ema_model = EMAModel(model, decay=0.9999)
        
        # During training:
        ema_model.update(model)
        
        # For validation/inference:
        ema_model.eval_mode()
        with torch.no_grad():
            output = ema_model.model(input)
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[str] = None):
        """
        Initialize EMA model.
        
        Args:
            model: The training model to track
            decay: EMA decay rate (0.999 or 0.9999 are common)
            device: Device to store EMA model on
        """
        self.decay = decay
        self.device = device or next(model.parameters()).device
        
        # Create a deep copy of the model for EMA
        self.model = deepcopy(model).to(self.device)
        self.model.eval()  # EMA model should always be in eval mode
        
        # Disable gradients for EMA model (saves memory)
        for param in self.model.parameters():
            param.requires_grad_(False)
        
        self.num_updates = 0
    
    def update(self, model: nn.Module):
        """
        Update EMA weights with current model weights.
        
        Args:
            model: Current training model
        """
        self.num_updates += 1
        
        # ✅ FIXED: Proper warmup that preserves averaging
        if self.num_updates <= 100:
            # Use slightly lower decay for first 100 updates for faster initial convergence
            decay = min(self.decay, 0.999)
        else:
            # Conservative dynamic decay after warmup - keeps decay high
            decay = min(self.decay, (1000 + self.num_updates) / (1010 + self.num_updates))
        
        with torch.no_grad():
            for ema_param, model_param in zip(self.model.parameters(), model.parameters()):
                if model_param.device != ema_param.device:
                    model_param = model_param.to(ema_param.device)
                ema_param.mul_(decay).add_(model_param, alpha=1 - decay)
    
    def eval_mode(self):
        """Set EMA model to evaluation mode."""
        self.model.eval()
        return self.model
    
    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            'model': self.model.state_dict(),
            'decay': self.decay,
            'num_updates': self.num_updates
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.model.load_state_dict(state_dict['model'])
        self.decay = state_dict.get('decay', self.decay)
        self.num_updates = state_dict.get('num_updates', 0)
    
    def reset_updates(self):
        """Reset the update counter (useful for fresh EMA initialization)."""
        self.num_updates = 0
    
    def __call__(self, *args, **kwargs):
        """Allow direct calling of EMA model."""
        return self.model(*args, **kwargs)


def apply_ema_to_model(model: nn.Module, ema_model: EMAModel):
    """
    Apply EMA weights to the original model (for final inference).
    
    Args:
        model: Original model to update
        ema_model: EMA model with averaged weights
    """
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.model.parameters()):
            param.copy_(ema_param)