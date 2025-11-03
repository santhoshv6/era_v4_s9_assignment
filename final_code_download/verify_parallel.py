#!/usr/bin/env python3
"""
ULTRA-FAST image verification using aggressive optimizations
Optimized for NVMe SSD + multi-core CPU
"""

from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb check for speed
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys

def verify_image_batch(image_paths):
    """Verify a batch of images - returns list of corrupt paths"""
    corrupt = []
    for img_path in image_paths:
        try:
            # Ultra-fast verification - just check header, don't decode
            with open(img_path, 'rb') as f:
                # JPEG files start with FF D8 and end with FF D9
                header = f.read(2)
                if header != b'\xff\xd8':
                    corrupt.append(str(img_path))
                    continue
                
                # Seek to end and check marker
                f.seek(-2, os.SEEK_END)
                footer = f.read(2)
                if footer != b'\xff\xd9':
                    corrupt.append(str(img_path))
        except Exception:
            corrupt.append(str(img_path))
    
    return corrupt

def verify_dataset_ultrafast(dataset_dir, num_workers=16, batch_size=1000):
    """
    Ultra-fast verification using:
    - Binary header/footer checks (no image decoding!)
    - Large batch processing
    - More workers than cores (I/O bound)
    """
    image_dir = Path(dataset_dir)
    
    print(f"🚀 ULTRA-FAST VERIFICATION MODE")
    print(f"📁 Directory: {dataset_dir}")
    print(f"⚡ Workers: {num_workers} (I/O optimized)")
    print(f"📦 Batch size: {batch_size}")
    print()
    
    # Get all images
    print("🔍 Scanning for images...")
    all_images = list(image_dir.rglob('*.JPEG'))
    total_images = len(all_images)
    print(f"📊 Found {total_images:,} images\n")
    
    # Split into batches
    batches = [all_images[i:i+batch_size] 
               for i in range(0, len(all_images), batch_size)]
    
    print(f"🔥 Starting verification with {len(batches)} batches...")
    
    # Process in parallel
    all_corrupt = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches
        futures = {executor.submit(verify_image_batch, batch): len(batch) 
                   for batch in batches}
        
        # Collect results as they complete
        for future in as_completed(futures):
            batch_size_done = futures[future]
            corrupt_in_batch = future.result()
            all_corrupt.extend(corrupt_in_batch)
            completed += batch_size_done
            
            # Progress update every ~100k images
            if completed % 100000 < batch_size_done:
                print(f"  ✓ {completed:,} / {total_images:,} images checked "
                      f"({completed/total_images*100:.1f}%)")
    
    # Results
    verified = total_images - len(all_corrupt)
    print(f"\n{'='*70}")
    print(f"⚡ ULTRA-FAST VERIFICATION COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Verified:     {verified:,} images")
    print(f"❌ Corrupt:      {len(all_corrupt):,} images")
    print(f"📈 Success rate: {verified/total_images*100:.2f}%")
    
    if all_corrupt:
        print(f"\n⚠️ Corrupt images:")
        for img in all_corrupt[:10]:
            print(f"  {img}")
        if len(all_corrupt) > 10:
            print(f"  ... and {len(all_corrupt)-10} more")
        
        with open('corrupt_images.txt', 'w') as f:
            f.write('\n'.join(all_corrupt))
        print(f"\n💾 Full list saved to: corrupt_images.txt")
    else:
        print(f"\n🎉 All images are valid JPEG files!")
    
    return verified, all_corrupt

if __name__ == '__main__':
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Ultra-fast image verification')
    parser.add_argument('--dir', default='/mnt/nvme_data/imagenet/train',
                       help='Dataset directory')
    parser.add_argument('--workers', type=int, default=16,
                       help='Number of workers (default: 16 for I/O bound)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Images per batch (default: 1000)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    verify_dataset_ultrafast(args.dir, args.workers, args.batch_size)
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("🚀 Verification finished!")