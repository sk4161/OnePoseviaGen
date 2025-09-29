#!/bin/bash

set -e  # Exit immediately if a command fails

echo "🚀 Starting setup for OnePoseviaGen..."

# 检查 Python 版本是否为 3.11
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
if [[ "$PYTHON_VERSION" != 3.11* ]]; then
    echo "⚠️  Warning: This project is tested with Python 3.11, but current version is $PYTHON_VERSION"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Setup canceled."
        exit 1
    fi
fi

# Step 1: Install PyTorch
echo "📦 Installing PyTorch with CUDA support..."
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install dependencies
echo "📦 Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Step 3: Install Boost and Eigen via Conda
echo "📦 Installing Boost and Eigen via Conda..."
conda install -c conda-forge boost -y
conda install -c conda-forge eigen=3.4.0 -y

# Step 4: Clone and install extensions
echo "📦 Cloning and installing external extensions..."
mkdir -p tmp/extensions

# DiffOctreeRaster
echo "🛠️ Installing diffoctreerast..."
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git tmp/extensions/diffoctreerast
pip install tmp/extensions/diffoctreerast

# Mip-Splatting
echo "🛠️ Installing mip-splatting and diff-gaussian-rasterization..."
git clone https://github.com/autonomousvision/mip-splatting.git tmp/extensions/mip-splatting
pip install tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# PyTorch3D
echo "🛠️ Installing PyTorch3D..."
pip install git+https://github.com/facebookresearch/pytorch3d.git

# Step 5: Build F-Pose
echo "🛠️ Building F-Pose..."
cd oneposeviagen/fpose/fpose
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
cd ../../..

# Step 6: Install packages in development mode
echo "🛠️ Installing local packages in editable mode..."

cd oneposeviagen

# Install fpose
echo "📦 Installing fpose..."
cd fpose
pip install -e .
cd ..

# Install SAM2-in-video
echo "📦 Installing SAM2-in-video..."
cd SAM2-in-video
pip install -e .
cd ..

# Install Trellis
echo "📦 Installing Trellis..."
cd trellis
pip install -e .
cd ..

# Install Amodal3r
echo "📦 Installing Trellis..."
cd Amodal3R
pip install -e .
cd ..

# Install SpaTrackerV2
echo "📦 Installing SpaTrackerV2..."
cd SpaTrackerV2
pip install -e .
cd ..

# Step 7: Download pretrained weights
echo "📦 Downloading pretrained weights..."
python oneposeviagen/scripts/download_weights.py

echo "🎉 Setup completed successfully!"
