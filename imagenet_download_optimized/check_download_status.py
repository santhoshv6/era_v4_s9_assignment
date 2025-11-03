#!/usr/bin/env python3
"""
ImageNet Download Status Checker and Debugger
=============================================
Analyzes the current state of ImageNet download and provides debugging information.

Usage:
    python check_download_status.py [--output-dir ./imagenet]
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Check ImageNet download status')
    parser.add_argument('--output-dir', type=str, default='./imagenet',
                       help='ImageNet output directory to check')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed class-by-class breakdown')
    parser.add_argument('--fix-empty-dirs', action='store_true',
                       help='Remove empty class directories')
    return parser.parse_args()


class DownloadStatusChecker:
    """Check and analyze ImageNet download status"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.progress_file = self.output_dir / '.download_progress.json'
        self.stats_file = self.output_dir / '.download_stats.json'
    
    def load_progress_info(self):
        """Load progress and stats information"""
        progress = {}
        stats = {}
        
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading progress file: {e}")
        
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    stats = json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading stats file: {e}")
        
        return progress, stats
    
    def analyze_directory_structure(self):
        """Analyze the current directory structure"""
        analysis = {
            'splits': {},
            'total_images': 0,
            'total_size_gb': 0.0,
            'empty_dirs': [],
            'issues': []
        }
        
        if not self.output_dir.exists():
            analysis['issues'].append(f"Output directory does not exist: {self.output_dir}")
            return analysis
        
        for split in ['train', 'val', 'validation']:
            split_dir = self.output_dir / split
            if not split_dir.exists():
                continue
            
            split_analysis = {
                'classes': 0,
                'images': 0,
                'size_gb': 0.0,
                'empty_classes': [],
                'class_distribution': Counter()
            }
            
            class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            split_analysis['classes'] = len(class_dirs)
            
            for class_dir in class_dirs:
                images = list(class_dir.glob('*.JPEG')) + list(class_dir.glob('*.jpg'))
                image_count = len(images)
                
                if image_count == 0:
                    split_analysis['empty_classes'].append(class_dir.name)
                    analysis['empty_dirs'].append(str(class_dir))
                else:
                    # Calculate size for a sample of images
                    sample_size = min(10, image_count)
                    sample_images = images[:sample_size]
                    total_sample_size = sum(img.stat().st_size for img in sample_images)
                    avg_image_size = total_sample_size / sample_size if sample_size > 0 else 0
                    estimated_class_size = avg_image_size * image_count
                    
                    split_analysis['size_gb'] += estimated_class_size / (1024**3)
                
                split_analysis['images'] += image_count
                split_analysis['class_distribution'][image_count] += 1
            
            analysis['splits'][split] = split_analysis
            analysis['total_images'] += split_analysis['images']
            analysis['total_size_gb'] += split_analysis['size_gb']
        
        return analysis
    
    def check_resume_capability(self, progress):
        """Check if resume will work correctly"""
        issues = []
        
        if not progress:
            return ["No progress file found - will start from beginning"]
        
        current_split = progress.get('current_split')
        current_index = progress.get('current_index', 0)
        
        if current_split:
            split_dir = self.output_dir / current_split
            if not split_dir.exists():
                issues.append(f"Current split directory missing: {split_dir}")
            
            class_mapping = progress.get('class_mapping', {})
            if not class_mapping:
                issues.append("Class mapping missing - may cause incorrect class structure")
            
            # Check if class directories exist
            missing_classes = []
            for label, class_name in class_mapping.items():
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    missing_classes.append(class_name)
            
            if missing_classes:
                issues.append(f"Missing class directories: {len(missing_classes)} classes")
        
        return issues
    
    def identify_issues(self, analysis, progress):
        """Identify potential issues with the download"""
        issues = []
        
        # Check for empty directories
        if analysis['empty_dirs']:
            issues.append(f"Found {len(analysis['empty_dirs'])} empty class directories")
        
        # Check for incomplete splits
        for split_name, split_info in analysis['splits'].items():
            if split_name == 'train' and split_info['images'] < 1200000:  # Expected ~1.28M
                issues.append(f"Train split appears incomplete: {split_info['images']:,} images")
            elif split_name in ['val', 'validation'] and split_info['images'] < 45000:  # Expected ~50K
                issues.append(f"Validation split appears incomplete: {split_info['images']:,} images")
            
            if split_info['classes'] != 1000:
                issues.append(f"{split_name} split has {split_info['classes']} classes (expected 1000)")
        
        # Check resume issues
        resume_issues = self.check_resume_capability(progress)
        issues.extend(resume_issues)
        
        return issues
    
    def print_detailed_breakdown(self, analysis):
        """Print detailed class-by-class breakdown"""
        print("\n📊 Detailed Class Distribution:")
        print("=" * 60)
        
        for split_name, split_info in analysis['splits'].items():
            print(f"\n{split_name.upper()} Split:")
            print(f"Total classes: {split_info['classes']}")
            print(f"Total images: {split_info['images']:,}")
            print(f"Size: {split_info['size_gb']:.1f} GB")
            
            if split_info['empty_classes']:
                print(f"\nEmpty classes ({len(split_info['empty_classes'])}):")
                for i, class_name in enumerate(split_info['empty_classes'][:10]):
                    print(f"  {class_name}")
                if len(split_info['empty_classes']) > 10:
                    print(f"  ... and {len(split_info['empty_classes']) - 10} more")
            
            # Distribution of images per class
            if split_info['class_distribution']:
                print(f"\nImages per class distribution:")
                for image_count, class_count in sorted(split_info['class_distribution'].items()):
                    if image_count == 0:
                        print(f"  {class_count} classes with {image_count} images (EMPTY)")
                    else:
                        print(f"  {class_count} classes with {image_count} images")
    
    def fix_empty_directories(self, analysis):
        """Remove empty class directories"""
        if not analysis['empty_dirs']:
            print("✅ No empty directories to fix")
            return
        
        print(f"🔧 Removing {len(analysis['empty_dirs'])} empty directories...")
        
        removed_count = 0
        for empty_dir in analysis['empty_dirs']:
            try:
                empty_path = Path(empty_dir)
                if empty_path.exists() and empty_path.is_dir():
                    # Double-check it's empty
                    if not any(empty_path.iterdir()):
                        empty_path.rmdir()
                        removed_count += 1
                        print(f"  Removed: {empty_path}")
            except Exception as e:
                print(f"  ⚠️  Error removing {empty_dir}: {e}")
        
        print(f"✅ Removed {removed_count} empty directories")
    
    def print_recommendations(self, issues, analysis, progress):
        """Print recommendations for fixing issues"""
        print("\n💡 Recommendations:")
        print("=" * 40)
        
        if not issues:
            print("✅ No issues detected! Download appears to be working correctly.")
            return
        
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        
        print("\n🔧 Suggested Actions:")
        
        if any("empty" in issue.lower() for issue in issues):
            print("• Run with --fix-empty-dirs to clean up empty directories")
            print("• This is likely caused by the original code's class mapping issue")
        
        if any("incomplete" in issue.lower() for issue in issues):
            print("• Resume the download using the optimized script:")
            print(f"  python download_imagenet_optimized.py --output-dir {self.output_dir} --resume")
        
        if any("class mapping" in issue.lower() for issue in issues):
            print("• The class mapping was corrupted. Use --force-restart with optimized script:")
            print(f"  python download_imagenet_optimized.py --output-dir {self.output_dir} --force-restart")
        
        print("\n🚀 For fastest download on EC2 g4dn.2xlarge:")
        print("bash setup_and_download.sh --output-dir /mnt/nvme_data/imagenet")
    
    def run(self, args):
        """Main analysis function"""
        print("🔍 ImageNet Download Status Check")
        print("=" * 50)
        print(f"Checking directory: {self.output_dir}")
        
        # Load progress information
        progress, stats = self.load_progress_info()
        
        # Analyze directory structure
        analysis = self.analyze_directory_structure()
        
        # Print basic statistics
        print(f"\n📊 Current Status:")
        print(f"Total images: {analysis['total_images']:,}")
        print(f"Estimated size: {analysis['total_size_gb']:.1f} GB")
        print(f"Splits found: {list(analysis['splits'].keys())}")
        
        for split_name, split_info in analysis['splits'].items():
            print(f"  {split_name}: {split_info['classes']} classes, {split_info['images']:,} images")
        
        # Print progress information
        if progress:
            print(f"\n📋 Progress Information:")
            current_split = progress.get('current_split', 'None')
            current_index = progress.get('current_index', 0)
            completed_splits = progress.get('completed_splits', {})
            
            print(f"Current split: {current_split}")
            print(f"Current index: {current_index:,}")
            print(f"Completed splits: {list(completed_splits.keys())}")
            
            if progress.get('start_time'):
                elapsed_hours = (time.time() - progress['start_time']) / 3600
                print(f"Running time: {elapsed_hours:.1f} hours")
        
        # Print statistics from previous runs
        if stats:
            print(f"\n📈 Previous Run Statistics:")
            for split_name, split_stats in stats.items():
                if isinstance(split_stats, dict):
                    saved = split_stats.get('saved', 0)
                    errors = split_stats.get('errors', 0)
                    duration = split_stats.get('duration_minutes', 0)
                    print(f"  {split_name}: {saved:,} saved, {errors} errors, {duration:.1f} min")
        
        # Identify and print issues
        issues = self.identify_issues(analysis, progress)
        
        if issues:
            print(f"\n⚠️  Issues Detected ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print(f"\n✅ No issues detected!")
        
        # Detailed breakdown if requested
        if args.detailed:
            self.print_detailed_breakdown(analysis)
        
        # Fix empty directories if requested
        if args.fix_empty_dirs:
            print()
            self.fix_empty_directories(analysis)
        
        # Print recommendations
        self.print_recommendations(issues, analysis, progress)


def main():
    args = parse_args()
    checker = DownloadStatusChecker(args.output_dir)
    checker.run(args)


if __name__ == '__main__':
    main()