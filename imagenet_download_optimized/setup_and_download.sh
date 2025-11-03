#!/bin/bash
"""
EC2 g4dn.2xlarge Setup and ImageNet Download Script
==================================================
Optimized setup script for downloading ImageNet on AWS EC2 g4dn.2xlarge spot instances.

This script:
1. Sets up the environment optimally for g4dn.2xlarge
2. Configures storage and memory settings
3. Runs the optimized ImageNet download
4. Provides monitoring and recovery options

Usage:
    bash setup_and_download.sh
    
Or with custom settings:
    bash setup_and_download.sh --output-dir /mnt/nvme_data/imagenet --workers 12
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings for g4dn.2xlarge
DEFAULT_OUTPUT_DIR="/mnt/nvme_data/imagenet"
DEFAULT_WORKERS="auto"
DEFAULT_BATCH_SIZE="100"
DEFAULT_MAX_MEMORY_GB="28"  # Leave 4GB for system

# Parse command line arguments
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
WORKERS="$DEFAULT_WORKERS"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
MAX_MEMORY_GB="$DEFAULT_MAX_MEMORY_GB"
FORCE_RESTART=""
VERIFY_IMAGES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-memory-gb)
            MAX_MEMORY_GB="$2"
            shift 2
            ;;
        --force-restart)
            FORCE_RESTART="--force-restart"
            shift
            ;;
        --verify)
            VERIFY_IMAGES="--verify-images"
            shift
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_system() {
    print_status "Checking system configuration..."
    
    # Check if running on g4dn.2xlarge
    INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
    if [[ "$INSTANCE_TYPE" == "g4dn.2xlarge" ]]; then
        print_success "Running on g4dn.2xlarge instance"
    else
        print_warning "Not running on g4dn.2xlarge (detected: $INSTANCE_TYPE)"
    fi
    
    # Check available memory
    TOTAL_MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    print_status "Total memory: ${TOTAL_MEMORY_GB}GB"
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
        print_success "GPU detected: $GPU_INFO"
    else
        print_warning "NVIDIA GPU not detected"
    fi
    
    # Check available disk space
    DISK_SPACE_GB=$(df -BG "$OUTPUT_DIR" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//' || echo "0")
    if [[ $DISK_SPACE_GB -lt 200 ]]; then
        print_warning "Low disk space: ${DISK_SPACE_GB}GB available. ImageNet needs ~150GB"
    else
        print_success "Sufficient disk space: ${DISK_SPACE_GB}GB available"
    fi
}

setup_storage() {
    print_status "Setting up storage..."
    
    # Create output directory
    sudo mkdir -p "$OUTPUT_DIR"
    sudo chown $USER:$USER "$OUTPUT_DIR"
    
    # Check if NVMe SSD is mounted
    if [[ "$OUTPUT_DIR" == "/mnt/nvme_data"* ]]; then
        if ! mountpoint -q /mnt/nvme_data; then
            print_status "Setting up NVMe SSD..."
            
            # Find NVMe device
            NVME_DEVICE=$(lsblk -no NAME,TYPE | grep disk | grep nvme | head -1 | awk '{print $1}')
            if [[ -n "$NVME_DEVICE" ]]; then
                sudo mkdir -p /mnt/nvme_data
                
                # Format if not already formatted
                if ! sudo blkid /dev/$NVME_DEVICE; then
                    print_status "Formatting NVMe device /dev/$NVME_DEVICE..."
                    sudo mkfs.ext4 /dev/$NVME_DEVICE
                fi
                
                # Mount
                sudo mount /dev/$NVME_DEVICE /mnt/nvme_data
                sudo chown $USER:$USER /mnt/nvme_data
                
                print_success "NVMe SSD mounted at /mnt/nvme_data"
            else
                print_warning "NVMe device not found"
            fi
        else
            print_success "NVMe SSD already mounted"
        fi
    fi
    
    # Create final output directory
    mkdir -p "$OUTPUT_DIR"
}

setup_python_environment() {
    print_status "Setting up Python environment..."
    
    # Update system packages
    sudo apt update
    
    # Install Python and pip if not available
    if ! command -v python3 &> /dev/null; then
        sudo apt install -y python3 python3-pip
    fi
    
    # Install required packages
    pip3 install --upgrade pip
    
    if [[ -f "requirements_optimized.txt" ]]; then
        pip3 install -r requirements_optimized.txt
        print_success "Installed packages from requirements_optimized.txt"
    else
        # Install essential packages directly
        pip3 install torch torchvision datasets huggingface_hub pillow tqdm psutil numpy
        print_success "Installed essential packages"
    fi
}

setup_huggingface() {
    print_status "Setting up Hugging Face authentication..."
    
    if [[ -z "${HF_TOKEN}" ]]; then
        print_warning "HF_TOKEN environment variable not set"
        print_status "You may need to login manually: huggingface-cli login"
    else
        echo "$HF_TOKEN" | huggingface-cli login --token
        print_success "Logged in to Hugging Face"
    fi
}

optimize_system() {
    print_status "Optimizing system settings..."
    
    # Increase file descriptor limits
    echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
    echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
    
    # Optimize TCP settings for faster downloads
    sudo sysctl -w net.core.rmem_max=16777216
    sudo sysctl -w net.core.wmem_max=16777216
    sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 16777216"
    sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"
    
    # Set CPU governor to performance
    sudo cpupower frequency-set -g performance 2>/dev/null || true
    
    print_success "System optimizations applied"
}

run_download() {
    print_status "Starting ImageNet download..."
    
    # Build command
    CMD="python3 download_imagenet_optimized.py"
    CMD="$CMD --output-dir '$OUTPUT_DIR'"
    CMD="$CMD --workers $WORKERS"
    CMD="$CMD --batch-size $BATCH_SIZE"
    CMD="$CMD --max-memory-gb $MAX_MEMORY_GB"
    CMD="$CMD --use-gpu"
    
    if [[ -n "$FORCE_RESTART" ]]; then
        CMD="$CMD $FORCE_RESTART"
    fi
    
    if [[ -n "$VERIFY_IMAGES" ]]; then
        CMD="$CMD $VERIFY_IMAGES"
    fi
    
    print_status "Running command: $CMD"
    
    # Create a screen session for background running
    SCREEN_SESSION="imagenet_download"
    
    if screen -list | grep -q "$SCREEN_SESSION"; then
        print_status "Attaching to existing screen session..."
        screen -r "$SCREEN_SESSION"
    else
        print_status "Starting new screen session: $SCREEN_SESSION"
        print_status "You can detach with Ctrl+A then D, and reattach with: screen -r $SCREEN_SESSION"
        sleep 3
        screen -S "$SCREEN_SESSION" bash -c "$CMD; echo 'Download completed. Press any key to exit.'; read"
    fi
}

monitor_progress() {
    print_status "You can monitor progress in several ways:"
    echo "1. Reattach to screen session: screen -r imagenet_download"
    echo "2. Check disk usage: watch -n 30 'df -h $OUTPUT_DIR'"
    echo "3. Monitor system resources: htop"
    echo "4. Check progress file: cat $OUTPUT_DIR/.download_progress.json"
    echo "5. View logs: tail -f $OUTPUT_DIR/download.log"
}

cleanup_on_interrupt() {
    print_warning "Script interrupted. Cleaning up..."
    exit 1
}

# Set up signal handlers
trap cleanup_on_interrupt SIGINT SIGTERM

main() {
    echo "🚀 EC2 g4dn.2xlarge ImageNet Download Setup"
    echo "=" * 50
    echo "Output directory: $OUTPUT_DIR"
    echo "Workers: $WORKERS"
    echo "Batch size: $BATCH_SIZE"
    echo "Max memory: ${MAX_MEMORY_GB}GB"
    echo ""
    
    check_system
    setup_storage
    setup_python_environment
    setup_huggingface
    optimize_system
    
    print_success "Setup complete! Starting download..."
    run_download
    
    print_success "Setup script finished!"
    monitor_progress
}

# Run main function
main "$@"