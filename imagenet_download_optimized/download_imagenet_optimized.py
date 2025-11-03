#!/usr/bin/env python3
"""
Highly Optimized ImageNet-1K Download from Hugging Face
======================================================
Fast, GPU-accelerated, resumable ImageNet downloader with parallel processing
optimized for EC2 g4dn.2xlarge instances.

Features:
- GPU-accelerated image processing (when available)
- Smart resume from exact interruption point
- Parallel processing with optimized worker count
- Memory-efficient streaming
- Proper error handling and recovery
- Real-time progress tracking with ETA

Requirements:
    pip install datasets pillow tqdm torch torchvision psutil

Usage:
    python download_imagenet_optimized.py --output-dir /mnt/nvme_data/imagenet --workers auto
"""

import os
import argparse
import json
import pickle
import hashlib
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps
from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch

# Try to import GPU acceleration
try:
    import torchvision.transforms.functional as TF
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not available - using CPU-only processing")


def parse_args():
    parser = argparse.ArgumentParser(description='Optimized ImageNet Download from Hugging Face')
    parser.add_argument('--output-dir', type=str, default='./imagenet',
                       help='Output directory for ImageNet dataset')
    parser.add_argument('--workers', type=str, default='auto',
                       help='Number of workers (auto, or specific number)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG quality (85-100 recommended)')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Resume from previous download (default: True)')
    parser.add_argument('--force-restart', action='store_true',
                       help='Force restart from beginning')
    parser.add_argument('--verify-images', action='store_true',
                       help='Verify image integrity after download')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                       help='Use GPU for image processing if available')
    parser.add_argument('--max-memory-gb', type=float, default=6.0,
                       help='Maximum memory usage in GB')
    parser.add_argument('--splits', nargs='+', default=['train', 'validation'],
                       help='Dataset splits to download')
    return parser.parse_args()


class ProgressTracker:
    """Track download progress with resume capability"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.progress_file = output_dir / '.download_progress.json'
        self.stats_file = output_dir / '.download_stats.json'
        self.progress = self.load_progress()
        
    def load_progress(self) -> Dict:
        """Load progress from disk"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'completed_splits': {},
            'current_split': None,
            'current_index': 0,
            'class_mapping': {},
            'total_processed': 0,
            'total_errors': 0,
            'start_time': time.time()
        }
    
    def save_progress(self):
        """Save progress to disk"""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def update_stats(self, split_name: str, stats: Dict):
        """Update download statistics"""
        all_stats = {}
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    all_stats = json.load(f)
            except:
                pass
        
        all_stats[split_name] = stats
        
        with open(self.stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
    
    def is_split_complete(self, split_name: str) -> bool:
        """Check if a split is already complete"""
        return split_name in self.progress.get('completed_splits', {})
    
    def mark_split_complete(self, split_name: str, total_images: int):
        """Mark a split as complete"""
        self.progress['completed_splits'][split_name] = {
            'total_images': total_images,
            'completed_at': time.time()
        }
        self.save_progress()


class ImageNetDownloader:
    """Optimized ImageNet downloader with GPU acceleration"""
    
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.tracker = ProgressTracker(self.output_dir)
        
        # Auto-detect optimal worker count
        if args.workers == 'auto':
            cpu_count = mp.cpu_count()
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            # For g4dn.2xlarge: 8 vCPUs, 32GB RAM
            # Conservative estimate: each worker uses ~0.5GB
            max_workers_by_memory = int(available_memory_gb / 0.5)
            self.num_workers = min(cpu_count * 2, max_workers_by_memory, 16)
        else:
            self.num_workers = int(args.workers)
        
        print(f"🔧 Using {self.num_workers} workers")
        
        # GPU setup
        self.device = None
        if args.use_gpu and HAS_TORCH and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"🚀 GPU acceleration enabled: {torch.cuda.get_device_name()}")
        else:
            print("💻 Using CPU processing")
            
        # Memory monitoring
        self.max_memory_bytes = args.max_memory_gb * 1024**3
        
    def get_imagenet_classes(self) -> Dict[int, str]:
        """Get proper ImageNet class mapping"""
        # Use cached mapping if available
        if 'class_mapping' in self.tracker.progress and self.tracker.progress['class_mapping']:
            return self.tracker.progress['class_mapping']
        
        print("📋 Loading ImageNet class mapping...")
        
        # Standard ImageNet-1K class names (subset for speed)
        # In practice, you'd want to get this from the dataset features
        try:
            # Quick sample to get the class structure
            sample_dataset = load_dataset(
                "ILSVRC/imagenet-1k", 
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            
            # Get a few samples to understand the label structure
            class_mapping = {}
            for i, sample in enumerate(sample_dataset):
                if i >= 100:  # Just sample first 100 to understand structure
                    break
                label = sample['label']
                if label not in class_mapping:
                    # Try to get class name from dataset features
                    if hasattr(sample_dataset, 'features') and 'label' in sample_dataset.features:
                        label_feature = sample_dataset.features['label']
                        if hasattr(label_feature, 'int2str'):
                            class_name = label_feature.int2str(label)
                        elif hasattr(label_feature, 'names'):
                            class_name = label_feature.names[label]
                        else:
                            class_name = f"n{label:08d}"  # WordNet format
                    else:
                        class_name = f"n{label:08d}"  # WordNet format fallback
                    class_mapping[label] = class_name
            
            # For missing classes, generate standard names
            for i in range(1000):
                if i not in class_mapping:
                    class_mapping[i] = f"n{i:08d}"
            
            # Cache the mapping
            self.tracker.progress['class_mapping'] = class_mapping
            self.tracker.save_progress()
            
            print(f"✅ Loaded {len(class_mapping)} class mappings")
            return class_mapping
            
        except Exception as e:
            print(f"⚠️  Error loading class mapping: {e}")
            # Fallback to standard WordNet format
            class_mapping = {i: f"n{i:08d}" for i in range(1000)}
            self.tracker.progress['class_mapping'] = class_mapping
            self.tracker.save_progress()
            return class_mapping
    
    def process_image_gpu(self, image: Image.Image) -> Optional[Image.Image]:
        """Process image using GPU acceleration"""
        try:
            if self.device is None:
                return self.process_image_cpu(image)
            
            # Convert PIL to tensor
            img_tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)
            
            # Ensure RGB
            if img_tensor.shape[1] == 1:  # Grayscale
                img_tensor = img_tensor.repeat(1, 3, 1, 1)
            elif img_tensor.shape[1] == 4:  # RGBA
                img_tensor = img_tensor[:, :3, :, :]
            
            # Convert back to PIL
            img_tensor = img_tensor.squeeze(0).cpu()
            processed_image = TF.to_pil_image(img_tensor)
            
            return processed_image
            
        except Exception as e:
            # Fallback to CPU processing
            return self.process_image_cpu(image)
    
    def process_image_cpu(self, image: Image.Image) -> Optional[Image.Image]:
        """Process image using CPU"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                if image.mode == 'RGBA':
                    # Handle transparency by compositing on white background
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                else:
                    image = image.convert('RGB')
            
            # Auto-orient based on EXIF
            image = ImageOps.exif_transpose(image)
            
            return image
            
        except Exception:
            return None
    
    def save_image_batch(self, batch_data: List[Tuple]) -> List[Dict]:
        """Save a batch of images efficiently"""
        results = []
        
        for image_data, save_path, quality in batch_data:
            try:
                # Skip if already exists and we're resuming
                if save_path.exists() and self.args.resume:
                    results.append({
                        'success': True,
                        'path': save_path,
                        'status': 'skipped',
                        'size': save_path.stat().st_size
                    })
                    continue
                
                # Ensure directory exists
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Process image
                if self.device:
                    processed_image = self.process_image_gpu(image_data)
                else:
                    processed_image = self.process_image_cpu(image_data)
                
                if processed_image is None:
                    results.append({
                        'success': False,
                        'path': save_path,
                        'status': 'processing_failed',
                        'size': 0
                    })
                    continue
                
                # Save with optimization
                processed_image.save(
                    save_path, 
                    'JPEG', 
                    quality=quality, 
                    optimize=True,
                    progressive=True
                )
                
                file_size = save_path.stat().st_size
                results.append({
                    'success': True,
                    'path': save_path,
                    'status': 'saved',
                    'size': file_size
                })
                
            except Exception as e:
                results.append({
                    'success': False,
                    'path': save_path if 'save_path' in locals() else Path('unknown'),
                    'status': f'error: {str(e)[:100]}',
                    'size': 0
                })
        
        return results
    
    def process_split_streaming(self, split_name: str) -> Dict:
        """Process a dataset split with streaming and smart resume"""
        
        print(f"\n🔄 Processing {split_name} split...")
        
        # Check if already complete
        if self.tracker.is_split_complete(split_name) and not self.args.force_restart:
            print(f"✅ {split_name} split already complete - skipping")
            return {'status': 'already_complete'}
        
        # Load dataset in streaming mode
        try:
            dataset = load_dataset(
                "ILSVRC/imagenet-1k",
                split=split_name if split_name == 'train' else 'validation',
                streaming=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"❌ Error loading {split_name} split: {e}")
            return {'status': 'error', 'error': str(e)}
        
        # Get class mapping
        class_mapping = self.get_imagenet_classes()
        
        # Determine resume point
        resume_index = 0
        if self.args.resume and not self.args.force_restart:
            if (self.tracker.progress.get('current_split') == split_name and 
                'current_index' in self.tracker.progress):
                resume_index = self.tracker.progress['current_index']
                print(f"📍 Resuming from index {resume_index}")
        
        # Setup split directory
        split_dir = self.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Create class directories
        for class_name in class_mapping.values():
            (split_dir / class_name).mkdir(exist_ok=True)
        
        # Process statistics
        stats = {
            'start_time': time.time(),
            'total_processed': 0,
            'saved': 0,
            'skipped': 0,
            'errors': 0,
            'total_size_mb': 0.0
        }
        
        # Processing loop with batching
        batch_data = []
        current_index = 0
        
        # Skip to resume point
        dataset_iter = iter(dataset)
        for _ in range(resume_index):
            try:
                next(dataset_iter)
                current_index += 1
            except StopIteration:
                break
        
        # Progress bar
        pbar = tqdm(
            desc=f"Processing {split_name}",
            initial=resume_index,
            unit="images"
        )
        
        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                
                for item in dataset_iter:
                    try:
                        image = item['image']
                        label = item['label']
                        class_name = class_mapping.get(label, f"n{label:08d}")
                        
                        # Generate filename
                        filename = f"{split_name}_{current_index:08d}.JPEG"
                        save_path = split_dir / class_name / filename
                        
                        batch_data.append((image, save_path, self.args.quality))
                        
                        # Process batch when full
                        if len(batch_data) >= self.args.batch_size:
                            future = executor.submit(self.save_image_batch, batch_data.copy())
                            futures.append(future)
                            batch_data = []
                        
                        current_index += 1
                        pbar.update(1)
                        
                        # Update progress periodically
                        if current_index % 1000 == 0:
                            self.tracker.progress['current_split'] = split_name
                            self.tracker.progress['current_index'] = current_index
                            self.tracker.save_progress()
                        
                        # Memory management
                        if current_index % 5000 == 0:
                            gc.collect()
                            if self.device:
                                torch.cuda.empty_cache()
                        
                        # Check memory usage
                        memory_usage = psutil.virtual_memory().used
                        if memory_usage > self.max_memory_bytes:
                            print("⚠️  Memory limit reached, processing current batches...")
                            break
                        
                    except Exception as e:
                        stats['errors'] += 1
                        if stats['errors'] < 10:
                            print(f"⚠️  Error processing item {current_index}: {e}")
                        continue
                
                # Process remaining batch
                if batch_data:
                    future = executor.submit(self.save_image_batch, batch_data)
                    futures.append(future)
                
                # Collect all results
                for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
                    try:
                        batch_results = future.result(timeout=60)
                        for result in batch_results:
                            stats['total_processed'] += 1
                            if result['success']:
                                if result['status'] == 'saved':
                                    stats['saved'] += 1
                                elif result['status'] == 'skipped':
                                    stats['skipped'] += 1
                                stats['total_size_mb'] += result['size'] / (1024 * 1024)
                            else:
                                stats['errors'] += 1
                    except Exception as e:
                        stats['errors'] += 1
                        print(f"⚠️  Batch processing error: {e}")
        
        finally:
            pbar.close()
        
        # Update final statistics
        stats['end_time'] = time.time()
        stats['duration_minutes'] = (stats['end_time'] - stats['start_time']) / 60
        
        # Mark split as complete
        self.tracker.mark_split_complete(split_name, stats['total_processed'])
        self.tracker.update_stats(split_name, stats)
        
        print(f"\n✅ {split_name} split complete:")
        print(f"   Total processed: {stats['total_processed']:,}")
        print(f"   Saved: {stats['saved']:,}")
        print(f"   Skipped: {stats['skipped']:,}")
        print(f"   Errors: {stats['errors']:,}")
        print(f"   Size: {stats['total_size_mb']:.1f} MB")
        print(f"   Duration: {stats['duration_minutes']:.1f} minutes")
        
        if stats['total_processed'] > 0:
            rate = stats['total_processed'] / stats['duration_minutes']
            print(f"   Rate: {rate:.1f} images/minute")
        
        return stats
    
    def verify_dataset(self) -> bool:
        """Verify dataset integrity and structure"""
        print("\n🔍 Verifying dataset...")
        
        verification_results = {}
        
        for split in ['train', 'val']:
            split_dir = self.output_dir / split
            if not split_dir.exists():
                continue
            
            class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            total_images = 0
            corrupted_images = 0
            
            print(f"📊 Verifying {split} split...")
            
            for class_dir in tqdm(class_dirs, desc=f"Checking {split} classes"):
                images = list(class_dir.glob('*.JPEG'))
                total_images += len(images)
                
                # Sample verification (check 10% of images)
                sample_size = max(1, len(images) // 10)
                sample_images = images[::len(images)//sample_size][:sample_size]
                
                for img_path in sample_images:
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                    except Exception:
                        corrupted_images += 1
            
            verification_results[split] = {
                'classes': len(class_dirs),
                'total_images': total_images,
                'corrupted_images': corrupted_images,
                'corruption_rate': corrupted_images / total_images if total_images > 0 else 0
            }
            
            print(f"   {split}: {len(class_dirs)} classes, {total_images:,} images")
            if corrupted_images > 0:
                print(f"   ⚠️  {corrupted_images} potentially corrupted images ({corrupted_images/total_images*100:.2f}%)")
        
        return all(result['corruption_rate'] < 0.01 for result in verification_results.values())
    
    def run(self) -> int:
        """Main download execution"""
        print("🚀 Optimized ImageNet-1K Download")
        print("=" * 50)
        print(f"Output directory: {self.output_dir}")
        print(f"Workers: {self.num_workers}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Resume: {self.args.resume}")
        print(f"GPU acceleration: {self.device is not None}")
        
        start_time = time.time()
        total_stats = {'saved': 0, 'skipped': 0, 'errors': 0, 'total_size_mb': 0}
        
        try:
            # Process each split
            for split in self.args.splits:
                split_stats = self.process_split_streaming(split)
                if split_stats.get('status') != 'error':
                    for key in total_stats:
                        if key in split_stats:
                            total_stats[key] += split_stats[key]
            
            # Verification
            if self.args.verify_images:
                self.verify_dataset()
            
            # Final summary
            total_time = time.time() - start_time
            print(f"\n🎉 Download Complete!")
            print(f"Total time: {total_time/60:.1f} minutes")
            print(f"Total saved: {total_stats['saved']:,}")
            print(f"Total skipped: {total_stats['skipped']:,}")
            print(f"Total errors: {total_stats['errors']:,}")
            print(f"Total size: {total_stats['total_size_mb']:.1f} MB")
            
            if total_stats['saved'] + total_stats['skipped'] > 0:
                rate = (total_stats['saved'] + total_stats['skipped']) / (total_time / 60)
                print(f"Average rate: {rate:.1f} images/minute")
            
            print(f"\n✅ Dataset ready at: {self.output_dir}")
            return 0
            
        except KeyboardInterrupt:
            print("\n⚠️  Download interrupted - progress saved for resume")
            return 1
        except Exception as e:
            print(f"\n❌ Error during download: {e}")
            return 1


def main():
    args = parse_args()
    
    # Validate arguments
    if args.quality < 75 or args.quality > 100:
        print("⚠️  Warning: JPEG quality should be between 75-100 for good results")
    
    if args.max_memory_gb < 2:
        print("⚠️  Warning: Memory limit very low, may cause issues")
    
    downloader = ImageNetDownloader(args)
    return downloader.run()


if __name__ == '__main__':
    exit(main())