#!/bin/bash
# Fix Jetson PyTorch GPU for JetPack 6.x (R36.x, Ubuntu 22.04 aarch64)
set -e

TORCH_WHL="$HOME/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"

# ── Step 1: Fix libcusparseLt ────────────────────────────────────────────── #
echo "[1/4] Fixing libcusparseLt..."

# Search existing installation
LIB=$(find /usr/local/cuda* /usr/lib/aarch64-linux-gnu /usr/lib /opt/nvidia \
      -name "libcusparseLt.so*" 2>/dev/null | head -1)

if [ -n "$LIB" ]; then
    DIR=$(dirname "$LIB")
    echo "  Found: $LIB"
    # Make sure .so.0 symlink exists
    if [ ! -f "$DIR/libcusparseLt.so.0" ]; then
        sudo ln -sf "$LIB" "$DIR/libcusparseLt.so.0"
    fi
    grep -qF "$DIR" /etc/ld.so.conf.d/cuda.conf 2>/dev/null || \
        echo "$DIR" | sudo tee -a /etc/ld.so.conf.d/cuda-jetson.conf
    sudo ldconfig
else
    echo "  Not found — adding NVIDIA apt repo for Ubuntu 22.04 arm64..."
    KEYRING="/tmp/cuda-keyring_1.1-1_all.deb"
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb" -O "$KEYRING"
    sudo dpkg -i "$KEYRING"
    sudo apt-get update -qq
    sudo apt-get install -y libcusparselt0 libcusparselt-dev
fi

# Always add /usr/local/cuda/lib64 to ldconfig
if [ -d /usr/local/cuda/lib64 ]; then
    echo "/usr/local/cuda/lib64" | sudo tee /etc/ld.so.conf.d/cuda-jetson.conf > /dev/null
    sudo ldconfig
fi

# ── Step 2: Reinstall torch ──────────────────────────────────────────────── #
echo "[2/4] Installing torch..."
pip install --force-reinstall "$TORCH_WHL"

# ── Step 3: Install torchvision (build from source, matches torch 2.5) ───── #
echo "[3/4] Installing torchvision..."
pip install torchvision --no-build-isolation 2>/dev/null || \
pip install "torchvision>=0.20.0" --no-deps --extra-index-url https://download.pytorch.org/whl/cpu || \
echo "  torchvision not critical for YOLO — skipping"

# ── Step 4: Verify ───────────────────────────────────────────────────────── #
echo "[4/4] Verifying..."
python3 - <<'EOF'
import torch, sys
avail = torch.cuda.is_available()
print(f"CUDA available : {avail}")
if avail:
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory
    print(f"VRAM           : {mem/1e9:.1f} GB")
    print(f"torch version  : {torch.__version__}")
    # Quick tensor test on GPU
    x = torch.randn(3, 3, device='cuda')
    print(f"Tensor test    : OK  shape={x.shape}")
else:
    print("ERROR: CUDA still not available")
    # Check why
    try:
        torch.cuda.init()
    except Exception as e:
        print(f"  Reason: {e}")
    sys.exit(1)
EOF
