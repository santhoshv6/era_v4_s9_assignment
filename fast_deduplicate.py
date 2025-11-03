#!/usr/bin/env python3
"""Fast parallel image deduplication for ImageNet"""

import hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

def compute_image_hash(image_path):
    """Fast hash using file content (no PIL decoding needed)"""
    try:
        # Just hash the file bytes - faster than decoding
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest(), image_path
    except:
        return None, image_path

def deduplicate_parallel(root_dir, num_workers=8):
    """Parallel deduplication using multiprocessing"""
    print(f"🔍 Scanning {root_dir} with {num_workers} workers...")
    
    # Get all image files
    image_files = list(Path(root_dir).rglob("*.JPEG"))
    total_images = len(image_files)
    print(f"Found {total_images:,} images")
    
    # Compute hashes in parallel
    hash_to_files = defaultdict(list)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compute_image_hash, img) for img in image_files]
        
        for future in tqdm(as_completed(futures), total=total_images, desc="Hashing"):
            img_hash, img_path = future.result()
            if img_hash:
                hash_to_files[img_hash].append(img_path)
    
    # Find duplicates
    duplicates_to_remove = []
    for img_hash, files in hash_to_files.items():
        if len(files) > 1:
            # Keep first, mark rest for deletion
            duplicates_to_remove.extend(files[1:])
    
    print(f"\n🗑️  Removing {len(duplicates_to_remove):,} duplicates...")
    
    # Delete duplicates
    for dup_file in tqdm(duplicates_to_remove, desc="Deleting"):
        try:
            dup_file.unlink()
        except:
            pass
    
    # Verify final count
    remaining = len(list(Path(root_dir).rglob("*.JPEG")))
    print(f"\n✅ Complete!")
    print(f"📊 Original: {total_images:,} images")
    print(f"📊 Removed: {len(duplicates_to_remove):,} duplicates")
    print(f"📊 Remaining: {remaining:,} images")
    
    expected = 1281167
    if remaining == expected:
        print(f"🎉 Perfect! Exactly {expected:,} images")
    else:
        diff = remaining - expected
        print(f"⚠️  Difference: {diff:+,} images from expected {expected:,}")
    
    return remaining

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='/mnt/nvme_data/imagenet/train')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    
    deduplicate_parallel(args.dir, args.workers)
