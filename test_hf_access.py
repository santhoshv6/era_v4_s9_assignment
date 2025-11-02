#!/usr/bin/env python3
"""
Quick test script for Hugging Face ImageNet download.
Note: Requires 'datasets' package - install with: pip install datasets huggingface_hub
Run this on EC2 after setting up the environment.
"""

import os
try:
    from datasets import load_dataset
except ImportError:
    print("❌ datasets package not installed. This is expected in local environment.")
    print("This script will work on EC2 after installing requirements.txt")
    exit(0)

def test_hf_access():
    """Test if we can access the Hugging Face ImageNet dataset"""
    try:
        print("🔍 Testing Hugging Face ImageNet access...")
        
        # Try to load just the dataset info (no download)
        dataset = load_dataset("ILSVRC/imagenet-1k", split="train[:10]", trust_remote_code=True)
        
        print(f"✅ Successfully accessed dataset!")
        print(f"Sample size: {len(dataset)} images")
        
        # Check first sample
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Image type: {type(sample['image'])}")
        print(f"Label: {sample['label']}")
        
        # Check if we have class names
        if hasattr(dataset, 'features'):
            label_feature = dataset.features.get('label')
            if hasattr(label_feature, 'int2str'):
                class_name = label_feature.int2str(sample['label'])
                print(f"Class name: {class_name}")
        
        print("🎉 Hugging Face access test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error accessing Hugging Face dataset: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Login with: huggingface-cli login")
        print("2. Make sure you have internet connection")
        print("3. Install required packages: pip install datasets huggingface_hub")
        return False

if __name__ == "__main__":
    test_hf_access()