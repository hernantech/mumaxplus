#!/bin/bash
#
# HeFFTe Installation Script for mumax+ Multi-GPU PoC
#
# This script downloads and builds HeFFTe with CUDA and MPI support.
#
# Usage:
#   ./setup_heffte.sh [install_prefix]
#
# Examples:
#   ./setup_heffte.sh                    # Install to /opt/heffte
#   ./setup_heffte.sh $HOME/heffte       # Install to ~/heffte
#

set -e

# Configuration
HEFFTE_VERSION="v2.4.0"
INSTALL_PREFIX="${1:-/opt/heffte}"
BUILD_DIR="/tmp/heffte_build"

# Detect CUDA architectures
detect_cuda_arch() {
    if command -v nvidia-smi &> /dev/null; then
        # Get compute capability from nvidia-smi
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo "Detected GPU: $gpu_name"

        # Map common GPUs to architectures
        case "$gpu_name" in
            *"V100"*)     echo "70" ;;
            *"A100"*)     echo "80" ;;
            *"A10"*)      echo "86" ;;
            *"RTX 3090"*) echo "86" ;;
            *"RTX 3080"*) echo "86" ;;
            *"RTX 3070"*) echo "86" ;;
            *"RTX 4090"*) echo "89" ;;
            *"RTX 4080"*) echo "89" ;;
            *"H100"*)     echo "90" ;;
            *)            echo "70;80;86" ;;  # Default to multiple
        esac
    else
        echo "70;80;86"  # Default
    fi
}

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."

    local missing=""

    if ! command -v nvcc &> /dev/null; then
        missing="$missing nvcc(CUDA)"
    fi

    if ! command -v mpicc &> /dev/null; then
        missing="$missing mpicc(MPI)"
    fi

    if ! command -v cmake &> /dev/null; then
        missing="$missing cmake"
    fi

    if ! command -v git &> /dev/null; then
        missing="$missing git"
    fi

    if [ -n "$missing" ]; then
        echo "ERROR: Missing required tools:$missing"
        echo ""
        echo "Install them with:"
        echo "  Ubuntu/Debian: sudo apt install cmake git"
        echo "  CUDA: https://developer.nvidia.com/cuda-downloads"
        echo "  MPI:  sudo apt install libopenmpi-dev openmpi-bin"
        exit 1
    fi

    echo "All prerequisites found."
}

# Main installation
main() {
    echo "========================================"
    echo "HeFFTe Installation for mumax+ PoC"
    echo "========================================"
    echo "Version: $HEFFTE_VERSION"
    echo "Install prefix: $INSTALL_PREFIX"
    echo ""

    check_prerequisites

    CUDA_ARCH=$(detect_cuda_arch)
    echo "CUDA Architectures: $CUDA_ARCH"
    echo ""

    # Create build directory
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Clone HeFFTe
    echo "Cloning HeFFTe..."
    git clone --depth 1 --branch "$HEFFTE_VERSION" https://github.com/icl-utk-edu/heffte.git
    cd heffte

    # Create build directory
    mkdir build && cd build

    # Configure
    echo "Configuring HeFFTe..."
    cmake \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DCMAKE_BUILD_TYPE=Release \
        -DHeffte_ENABLE_CUDA=ON \
        -DHeffte_ENABLE_MPI=ON \
        -DHeffte_ENABLE_FFTW=OFF \
        -DHeffte_ENABLE_ROCM=OFF \
        -DHeffte_ENABLE_ONEAPI=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        ..

    # Build
    echo "Building HeFFTe..."
    make -j$(nproc)

    # Install
    echo "Installing HeFFTe..."
    if [ -w "$INSTALL_PREFIX" ] || [ -w "$(dirname $INSTALL_PREFIX)" ]; then
        make install
    else
        echo "Need sudo for installation to $INSTALL_PREFIX"
        sudo make install
    fi

    # Cleanup
    cd /
    rm -rf "$BUILD_DIR"

    echo ""
    echo "========================================"
    echo "HeFFTe installed successfully!"
    echo "========================================"
    echo ""
    echo "To use with the mumax+ PoC:"
    echo ""
    echo "  cd distributed_poc"
    echo "  mkdir build && cd build"
    echo "  cmake -DHEFFTE_DIR=$INSTALL_PREFIX .."
    echo "  make"
    echo "  mpirun -np 2 ./heffte_poc"
    echo ""
    echo "Add to your environment (optional):"
    echo ""
    echo "  export HEFFTE_DIR=$INSTALL_PREFIX"
    echo "  export LD_LIBRARY_PATH=\$HEFFTE_DIR/lib:\$LD_LIBRARY_PATH"
    echo ""
}

main "$@"
