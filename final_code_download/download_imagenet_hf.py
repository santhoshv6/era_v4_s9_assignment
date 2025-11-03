#!/usr/bin/env python3
"""
Optimized ImageNet-1K Download from Hugging Face
================================================
Downloads ImageNet-1K dataset from Hugging Face and organizes it into 
the standard ImageNet folder structure for PyTorch training.

Requirements:
    pip install datasets pillow tqdm

Usage:
    python download_imagenet_hf.py --output-dir /mnt/nvme_data/imagenet --num-workers 8
"""

import os
import argparse
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import time

import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Download ImageNet from Hugging Face')
    parser.add_argument('--output-dir', type=str, default='./imagenet',
                       help='Output directory for ImageNet dataset')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of parallel workers for processing')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Chunk size for batch processing')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG quality for saved images (1-100)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip images that already exist')
    parser.add_argument('--verify', action='store_true',
                       help='Verify downloaded images after processing')
    return parser.parse_args()


def setup_directories(output_dir):
    """Create the ImageNet directory structure"""
    train_dir = Path(output_dir) / 'train'
    val_dir = Path(output_dir) / 'val'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    return train_dir, val_dir


def get_class_mapping():
    """Get ImageNet class ID to folder name mapping"""
    # This will be populated when we load the dataset
    return {}


def save_image_optimized(args):
    """Optimized image saving function for multiprocessing"""
    image_data, save_path, quality, skip_existing = args
    
    try:
        # Skip if file exists and skip_existing is True
        if skip_existing and save_path.exists():
            return True, save_path, "skipped"
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert and save image
        if hasattr(image_data, 'convert'):
            # PIL Image
            if image_data.mode != 'RGB':
                image_data = image_data.convert('RGB')
            image_data.save(save_path, 'JPEG', quality=quality, optimize=True)
        else:
            # Numpy array or other format
            if isinstance(image_data, np.ndarray):
                image_data = Image.fromarray(image_data)
            if image_data.mode != 'RGB':
                image_data = image_data.convert('RGB')
            image_data.save(save_path, 'JPEG', quality=quality, optimize=True)
            
        return True, save_path, "saved"
        
    except Exception as e:
        return False, save_path, f"error: {e}"


def process_dataset_split(dataset, split_name, output_dir, args):
    """Process a dataset split (train or val) with optimized parallel processing"""
    
    print(f"\n🔄 Processing {split_name} split...")
    print(f"Total images: {len(dataset):,}")
    
    # Get unique labels and create class directories
    unique_labels = set()
    label_to_class = {}
    
    # First pass: collect all unique labels and their class names
    print("📋 Mapping classes...")
    for i, item in enumerate(tqdm(dataset, desc="Scanning classes", leave=False)):
        label = item['label']
        unique_labels.add(label)
        
        # Create class name (you might need to adjust this based on HF dataset structure)
        if hasattr(dataset, 'features') and hasattr(dataset.features['label'], 'int2str'):
            class_name = dataset.features['label'].int2str(label)
        else:
            class_name = f"class_{label:04d}"
            
        label_to_class[label] = class_name
        
        # Create class directory
        class_dir = Path(output_dir) / split_name / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Break early if just scanning
        if i > 100 and len(unique_labels) > 50:  # Quick scan for efficiency
            continue
    
    print(f"Found {len(unique_labels)} unique classes")
    
    # Prepare all save tasks
    save_tasks = []
    error_count = 0
    
    for i, item in enumerate(dataset):
        try:
            image = item['image']
            label = item['label']
            class_name = label_to_class.get(label, f"class_{label:04d}")
            
            # Generate filename
            filename = f"{split_name}_{i:08d}.JPEG"
            save_path = Path(output_dir) / split_name / class_name / filename
            
            save_tasks.append((image, save_path, args.quality, args.skip_existing))
            
        except Exception as e:
            error_count += 1
            if error_count < 10:  # Show first few errors
                print(f"⚠️  Error preparing item {i}: {e}")
    
    print(f"📦 Prepared {len(save_tasks):,} save tasks")
    if error_count > 0:
        print(f"⚠️  {error_count} items had preparation errors")
    
    # Process in chunks for memory efficiency
    chunk_size = args.chunk_size
    saved_count = 0
    skipped_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for i in range(0, len(save_tasks), chunk_size):
            chunk = save_tasks[i:i + chunk_size]
            chunk_end = min(i + chunk_size, len(save_tasks))
            
            print(f"🔄 Processing chunk {i//chunk_size + 1}/{(len(save_tasks) + chunk_size - 1)//chunk_size}")
            
            # Submit chunk for processing
            futures = [executor.submit(save_image_optimized, task) for task in chunk]
            
            # Collect results with progress bar
            for future in tqdm(futures, desc=f"Saving {split_name}", leave=False):
                try:
                    success, path, status = future.result(timeout=30)
                    if success:
                        if status == "saved":
                            saved_count += 1
                        elif status == "skipped":
                            skipped_count += 1
                    else:
                        error_count += 1
                        if error_count < 10:
                            print(f"⚠️  Error saving {path}: {status}")
                except Exception as e:
                    error_count += 1
                    if error_count < 10:
                        print(f"⚠️  Future error: {e}")
    
    print(f"✅ {split_name} split complete:")
    print(f"   Saved: {saved_count:,} images")
    print(f"   Skipped: {skipped_count:,} images")
    print(f"   Errors: {error_count:,} images")
    
    return saved_count, skipped_count, error_count


def verify_dataset(output_dir):
    """Verify the downloaded dataset structure and count"""
    print("\n🔍 Verifying dataset...")
    
    train_dir = Path(output_dir) / 'train'
    val_dir = Path(output_dir) / 'val'
    
    if not train_dir.exists() or not val_dir.exists():
        print("❌ Missing train or val directories")
        return False
    
    # Count classes and images
    train_classes = list(train_dir.iterdir())
    val_classes = list(val_dir.iterdir())
    
    train_images = sum(len(list(class_dir.glob('*.JPEG'))) for class_dir in train_classes if class_dir.is_dir())
    val_images = sum(len(list(class_dir.glob('*.JPEG'))) for class_dir in val_classes if class_dir.is_dir())
    
    print(f"📊 Dataset Statistics:")
    print(f"   Train classes: {len(train_classes)}")
    print(f"   Val classes: {len(val_classes)}")
    print(f"   Train images: {train_images:,}")
    print(f"   Val images: {val_images:,}")
    
    # Check expected counts
    if len(train_classes) == 1000 and len(val_classes) == 1000:
        print("✅ Class count looks correct (1000 each)")
    else:
        print("⚠️  Unexpected class count")
    
    # Estimate total size
    total_size = sum(f.stat().st_size for f in Path(output_dir).rglob('*.JPEG'))
    total_size_gb = total_size / (1024**3)
    print(f"   Total size: {total_size_gb:.1f} GB")
    
    return True


def main():
    args = parse_args()
    
    print("🚀 ImageNet-1K Download from Hugging Face")
    print("=" * 50)
    print(f"Output directory: {args.output_dir}")
    print(f"Workers: {args.num_workers}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"JPEG quality: {args.quality}")
    print(f"Skip existing: {args.skip_existing}")
    
    # Setup directories
    train_dir, val_dir = setup_directories(args.output_dir)
    print(f"✅ Created directories: {train_dir}, {val_dir}")
    
    start_time = time.time()
    
    try:
        print("\n📥 Loading dataset from Hugging Face...")
        print("Note: You may need to login with: huggingface-cli login")
        
        # Load dataset
        dataset = load_dataset("ILSVRC/imagenet-1k", trust_remote_code=True)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        total_saved = 0
        total_skipped = 0
        total_errors = 0
        
        # Process train split
        if 'train' in dataset:
            saved, skipped, errors = process_dataset_split(
                dataset['train'], 'train', args.output_dir, args
            )
            total_saved += saved
            total_skipped += skipped
            total_errors += errors
        
        # Process validation split
        if 'validation' in dataset:
            saved, skipped, errors = process_dataset_split(
                dataset['validation'], 'val', args.output_dir, args
            )
            total_saved += saved
            total_skipped += skipped
            total_errors += errors
        
        # Verify dataset if requested
        if args.verify:
            verify_dataset(args.output_dir)
        
        # Final statistics
        elapsed_time = time.time() - start_time
        print(f"\n🎉 Download Complete!")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        print(f"Total saved: {total_saved:,} images")
        print(f"Total skipped: {total_skipped:,} images")
        print(f"Total errors: {total_errors:,} images")
        
        if total_errors > 0:
            print(f"⚠️  {total_errors} images had errors - this is usually normal for large datasets")
        
        print(f"\n✅ ImageNet ready for training!")
        print(f"Use: python train.py --data {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during download: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check internet connection")
        print("3. Verify disk space (need ~150GB)")
        print("4. Try reducing --num-workers if memory issues")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())