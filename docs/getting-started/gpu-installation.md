# GPU Installation Guide

Guide for installing and using faiss-go with GPU (CUDA) support.

---

## Overview

faiss-go supports GPU acceleration through CUDA, providing **10-100x faster** similarity search on NVIDIA GPUs. However, GPU support requires:

1. **NVIDIA GPU** with CUDA support (Compute Capability ≥ 7.0 recommended)
2. **CUDA Toolkit** installed (version 11.0 or later)
3. **Building FAISS from source** with GPU enabled

> **Important**: Pre-built static libraries and amalgamation are CPU-only. GPU support requires building FAISS with CUDA enabled.

---

## Prerequisites

### 1. NVIDIA GPU

Check your GPU and CUDA support:

```bash
# Check if NVIDIA GPU is available
nvidia-smi

# Should show your GPU, e.g.:
# Tesla V100, RTX 3090, A100, etc.
```

### 2. CUDA Toolkit

Install CUDA Toolkit (version 11.0+):

#### Ubuntu/Debian

```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA Toolkit (adjust version as needed)
sudo apt-get install -y cuda-toolkit-12-3

# Verify installation
nvcc --version
```

#### macOS

> **Note**: macOS does not support CUDA (Apple Silicon and Intel Macs with AMD GPUs are not compatible). GPU support is only available on Linux and Windows with NVIDIA GPUs.

#### Windows

1. Download CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
2. Run the installer
3. Add CUDA to PATH:
   ```powershell
   setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"
   setx PATH "%PATH%;%CUDA_PATH%\bin"
   ```

---

## Building FAISS with GPU Support

### Ubuntu/Debian (Recommended)

```bash
# 1. Install dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    libgomp1 \
    libomp-dev \
    cuda-toolkit-12-3

# 2. Download and build FAISS
cd /tmp
wget https://github.com/facebookresearch/faiss/archive/refs/tags/v1.8.0.tar.gz
tar -xzf v1.8.0.tar.gz
cd faiss-1.8.0

# 3. Configure with GPU support
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DFAISS_ENABLE_C_API=ON \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" \
    -DCMAKE_INSTALL_PREFIX=/usr/local

# Note: Adjust CUDA architectures for your GPU:
# - 70: Tesla V100
# - 75: RTX 20xx (Turing)
# - 80: A100
# - 86: RTX 30xx (Ampere)
# - 89: RTX 40xx (Ada Lovelace)
# - 90: H100

# 4. Build and install
cmake --build build -j$(nproc)
sudo cmake --install build
sudo ldconfig

# 5. Verify installation
ls -la /usr/local/lib/libfaiss.so
ls -la /usr/local/lib/libfaiss_gpu.so
```

### Windows

```powershell
# 1. Install vcpkg and dependencies
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
.\vcpkg install openblas:x64-windows

# 2. Download FAISS
cd C:\
git clone https://github.com/facebookresearch/faiss.git
cd faiss
git checkout v1.8.0

# 3. Build with GPU support
mkdir build
cd build
cmake .. ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DFAISS_ENABLE_GPU=ON ^
    -DFAISS_ENABLE_PYTHON=OFF ^
    -DFAISS_ENABLE_C_API=ON ^
    -DBUILD_TESTING=OFF ^
    -DBUILD_SHARED_LIBS=ON ^
    -DCMAKE_CUDA_ARCHITECTURES="86;89" ^
    -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake ^
    -DCMAKE_INSTALL_PREFIX="C:\faiss"

cmake --build . --config Release -j8
cmake --install . --config Release

# Add to PATH
setx PATH "%PATH%;C:\faiss\bin"
```

---

## Building faiss-go with GPU Support

Once FAISS with GPU is installed:

### Set Environment Variables

#### Linux

```bash
export CGO_CFLAGS="-I/usr/local/include -DFAISS_GPU"
export CGO_LDFLAGS="-L/usr/local/lib -lfaiss -lfaiss_gpu -lcudart -lcublas -lgomp -lstdc++ -lm -lopenblas"
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

Add to `~/.bashrc` or `~/.zshrc` for persistence:

```bash
echo 'export CGO_CFLAGS="-I/usr/local/include -DFAISS_GPU"' >> ~/.bashrc
echo 'export CGO_LDFLAGS="-L/usr/local/lib -lfaiss -lfaiss_gpu -lcudart -lcublas -lgomp -lstdc++ -lm -lopenblas"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### Windows

```powershell
setx CGO_CFLAGS "-IC:\faiss\include -DFAISS_GPU"
setx CGO_LDFLAGS "-LC:\faiss\lib -lfaiss -lfaiss_gpu -lcudart -lcublas"
setx PATH "%PATH%;C:\faiss\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"
```

### Build Your Go Project

```bash
# Get faiss-go
go get github.com/NerdMeNot/faiss-go

# Build your project
cd your-project
go build -v

# Run tests (GPU tests will run if CUDA is available)
go test -v ./...
```

---

## Usage Example

```go
package main

import (
    "fmt"
    "log"

    "github.com/NerdMeNot/faiss-go"
)

func main() {
    // Create GPU resources
    res, err := faiss.NewStandardGpuResources()
    if err != nil {
        log.Fatal("GPU not available:", err)
    }
    defer res.Close()

    // Set GPU memory (optional)
    res.SetTempMemory(512 * 1024 * 1024) // 512MB

    // Create CPU index
    dim := 128
    cpuIndex, err := faiss.NewIndexFlatL2(dim)
    if err != nil {
        log.Fatal(err)
    }

    // Transfer to GPU (device 0)
    gpuIndex, err := faiss.IndexCpuToGpu(res, 0, cpuIndex)
    if err != nil {
        log.Fatal(err)
    }
    defer gpuIndex.Close()

    // Add vectors (example data)
    vectors := make([]float32, 1000*dim)
    for i := range vectors {
        vectors[i] = float32(i % 100)
    }

    err = gpuIndex.Add(vectors)
    if err != nil {
        log.Fatal(err)
    }

    // Search on GPU
    queries := vectors[:10*dim] // First 10 vectors
    k := 5
    distances, indices, err := gpuIndex.Search(queries, k)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Found %d results\n", len(indices))
    fmt.Printf("Top result: index=%d, distance=%f\n", indices[0], distances[0])

    // Transfer back to CPU if needed
    cpuIndex2, err := faiss.IndexGpuToCpu(gpuIndex)
    if err != nil {
        log.Fatal(err)
    }
    defer cpuIndex2.Close()
}
```

---

## Multi-GPU Support

Use all available GPUs:

```go
// Transfer to all GPUs
gpuIndex, err := faiss.IndexCpuToAllGpus(res, cpuIndex)
if err != nil {
    log.Fatal(err)
}
defer gpuIndex.Close()

// Check number of GPUs
numGPUs, err := faiss.GetNumGpus()
fmt.Printf("Using %d GPUs\n", numGPUs)
```

---

## Troubleshooting

### CUDA Not Found

```
Error: CUDA not available
```

**Solution**:
1. Verify CUDA installation: `nvcc --version`
2. Check GPU availability: `nvidia-smi`
3. Ensure CUDA libraries are in `LD_LIBRARY_PATH`:
   ```bash
   export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
   ```

### Library Not Found

```
Error: cannot open shared object file: libfaiss_gpu.so
```

**Solution**:
1. Verify FAISS GPU library is installed:
   ```bash
   ls -la /usr/local/lib/libfaiss_gpu.so
   ```
2. Update library cache:
   ```bash
   sudo ldconfig
   ```
3. Add to `LD_LIBRARY_PATH`:
   ```bash
   export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
   ```

### Build Tags Conflict

If you get errors about `nogpu` build tag:

```bash
# Don't use -tags=nogpu when building with GPU support
go build  # Correct
go build -tags=nogpu  # Wrong - this disables GPU
```

### CUDA Architecture Mismatch

```
Error: no kernel image available for execution
```

**Solution**: Rebuild FAISS with your GPU's compute capability:

```bash
# Find your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Rebuild with correct architecture (e.g., 86 for RTX 3090)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="86" ...
```

---

## Performance Tips

1. **Batch Queries**: GPU is optimized for batch operations (1000+ queries)
2. **GPU Memory**: Set appropriate temp memory based on your GPU VRAM
3. **Transfer Once**: Keep index on GPU, avoid frequent CPU↔GPU transfers
4. **Multiple GPUs**: For very large datasets, use `IndexCpuToAllGpus`
5. **Async Operations**: CUDA operations are asynchronous by default

---

## Build Modes Comparison

| Build Mode | GPU Support | CUDA Required | Library |
|------------|-------------|---------------|---------|
| **Default (source)** | ❌ No | No | CPU-only amalgamation |
| **Pre-built libs** (`-tags=faiss_use_lib`) | ❌ No | No | CPU-only static libs |
| **GPU Build** (this guide) | ✅ Yes | Yes | Build FAISS with GPU |

To use GPU, you **must** build FAISS from source with `FAISS_ENABLE_GPU=ON`.

---

## See Also

- [Installation Guide](installation.md) - CPU-only installation
- [Choosing an Index](choosing-an-index.md) - GPU index selection
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [FAISS GPU Documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU)
