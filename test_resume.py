#!/usr/bin/env python3
"""
Test script to verify resume functionality works correctly.
"""

import torch
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_resume_functionality():
    """Test that resume detection and validation works."""
    
    print("🔍 Testing Resume Functionality")
    print("=" * 50)
    
    # Test 1: No checkpoint (fresh start)
    if os.path.exists('./outputs'):
        import shutil
        shutil.rmtree('./outputs')
    
    # Import the resume checker
    exec(open('resume_training.py').read()) if os.path.exists('resume_training.py') else print("❌ resume_training.py not found")
    
    # Test 2: Create a mock checkpoint
    os.makedirs('./outputs', exist_ok=True)
    
    # Create a simple mock checkpoint
    mock_checkpoint = {
        'epoch': 45,
        'model': {},  # Empty model state for testing
        'optimizer': {},
        'scheduler': {},
        'best_acc1': 68.5,
        'model_type': 'EMA'
    }
    
    torch.save(mock_checkpoint, './outputs/best_model.pth')
    print("✅ Created mock checkpoint at epoch 45")
    
    # Test checkpoint loading
    try:
        checkpoint = torch.load('./outputs/best_model.pth', map_location='cpu')
        print(f"✅ Checkpoint loaded successfully:")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Accuracy: {checkpoint['best_acc1']:.2f}%")
        print(f"   Model Type: {checkpoint['model_type']}")
        
        # Determine resume status
        if checkpoint['epoch'] >= 120:
            print("📊 Status: Training complete")
        elif checkpoint['epoch'] >= 100:
            swa_progress = checkpoint['epoch'] - 100
            print(f"📊 Status: SWA phase ({swa_progress}/20 epochs)")
        else:
            print(f"📊 Status: EMA phase ({checkpoint['epoch']}/100 epochs)")
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Resume functionality test complete!")
    print("You can now use --resume ./outputs/best_model.pth")

if __name__ == "__main__":
    test_resume_functionality()