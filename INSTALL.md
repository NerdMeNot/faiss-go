# Installation Guide

Complete installation instructions for faiss-go with the C++ bridge.

---

## Quick Install (Recommended Platforms)

### macOS (Homebrew)

```bash
# Install FAISS
brew install faiss

# Install faiss-go
go get github.com/NerdMeNot/faiss-go

# Build and test
cd your-project
go build
go test
```

**Done!** FAISS is now embedded in your Go binary.

### Ubuntu/Debian

```bash
# Install FAISS and dependencies
sudo apt-get update
sudo apt-get install -y libfaiss-dev libopenblas-dev

# Install faiss-go
go get github.com/NerdMeNot/faiss-go

# Build and test
cd your-project
go build
go test
```

### Fedora/RHEL/CentOS

```bash
# Install FAISS and dependencies
sudo dnf install -y faiss-devel openblas-devel

# Install faiss-go
go get github.com/NerdMeNot/faiss-go

# Build
cd your-project
go build
```

---

## Detailed Installation by Platform

### macOS

#### Option 1: Homebrew (Easiest)

```bash
# Install FAISS
brew install faiss

# Verify installation
brew list faiss
# Should show:
#   /opt/homebrew/include/faiss/...
#   /opt/homebrew/lib/libfaiss.dylib
```

#### Option 2: Build from Source

```bash
# Install dependencies
brew install cmake openblas

# Clone and build FAISS
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build -j
sudo cmake --install build
```

#### Troubleshooting macOS

**Error: "library not found for -lfaiss"**
```bash
# Add library path
export LIBRARY_PATH="/opt/homebrew/lib:$LIBRARY_PATH"
export CPATH="/opt/homebrew/include:$CPATH"

# Or install to /usr/local instead
brew install faiss --prefix=/usr/local
```

**Error: "ld: framework not found Accelerate"**

This shouldn't happen on modern macOS, but if it does:
```bash
# Ensure Xcode Command Line Tools are installed
xcode-select --install
```

---

### Linux (Ubuntu/Debian)

#### Option 1: Package Manager (If Available)

```bash
# Ubuntu 22.04+ has FAISS in repos
sudo apt-get update
sudo apt-get install -y libfaiss-dev libopenblas-dev

# Verify
pkg-config --modversion faiss
```

#### Option 2: Build from Source (Universal)

```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libgomp1

# Clone FAISS
git clone https://github.com/facebookresearch/faiss.git
cd faiss

# Build (CPU-only, no Python)
cmake -B build \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local

# Compile (use all cores)
cmake --build build -j$(nproc)

# Install
sudo cmake --install build

# Update library cache
sudo ldconfig
```

#### Verify Installation

```bash
# Check libraries
ldconfig -p | grep faiss
# Should show: libfaiss.so

# Check headers
ls /usr/local/include/faiss/
# Should show: Index.h, IndexFlat.h, etc.

# Test with pkg-config
pkg-config --cflags --libs faiss
```

---

### Linux (Fedora/RHEL/CentOS)

```bash
# Install build dependencies
sudo dnf install -y \
    gcc-c++ \
    cmake \
    openblas-devel \
    libgomp

# Build FAISS from source (same as Ubuntu)
git clone https://github.com/facebookresearch/faiss.git
cd faiss

cmake -B build \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
sudo cmake --install build
sudo ldconfig
```

---

### Windows

#### Option 1: vcpkg (Recommended)

```powershell
# Install vcpkg (if not already)
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install FAISS
.\vcpkg install faiss:x64-windows

# Set environment variables
$env:CGO_CPPFLAGS="-IC:\path\to\vcpkg\installed\x64-windows\include"
$env:CGO_LDFLAGS="-LC:\path\to\vcpkg\installed\x64-windows\lib"
```

#### Option 2: Build from Source (MSYS2)

```bash
# In MSYS2 MinGW64 shell
pacman -S \
    mingw-w64-x86_64-gcc \
    mingw-w64-x86_64-cmake \
    mingw-w64-x86_64-openblas

# Build FAISS
git clone https://github.com/facebookresearch/faiss.git
cd faiss

cmake -B build -G "MSYS Makefiles" \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build -j
cmake --install build
```

---

## Installing faiss-go

Once FAISS is installed:

```bash
# Get faiss-go
go get github.com/NerdMeNot/faiss-go

# Create a test program
cat > test.go << 'EOF'
package main

import (
    "fmt"
    "github.com/NerdMeNot/faiss-go"
)

func main() {
    index, err := faiss.NewIndexFlatL2(128)
    if err != nil {
        panic(err)
    }
    defer index.Close()

    fmt.Printf("Created index with dimension %d\n", index.D())
    fmt.Println(faiss.GetBuildInfo())
}
EOF

# Build and run
go run test.go
```

**Expected output:**
```
Created index with dimension 128
faiss-go 0.1.0-alpha (FAISS 1.8.0)
Build: source
Compiler: GCC/Clang (from amalgamation build)
Platform: darwin/arm64
BLAS: Accelerate Framework + OpenBLAS
```

---

## Docker Installation

### Using Pre-built Image (Coming Soon)

```dockerfile
FROM golang:1.21

# Install FAISS
RUN apt-get update && apt-get install -y libfaiss-dev

# Copy your application
WORKDIR /app
COPY . .

# Build
RUN go build -o myapp

CMD ["./myapp"]
```

### Build Your Own

```dockerfile
FROM golang:1.21 AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        cmake \
        libopenblas-dev \
        git

# Build FAISS from source
RUN git clone https://github.com/facebookresearch/faiss.git /tmp/faiss && \
    cd /tmp/faiss && \
    cmake -B build \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j && \
    cmake --install build && \
    ldconfig

# Build your application
WORKDIR /app
COPY . .
RUN go build -o myapp

# Runtime image
FROM debian:bookworm-slim
RUN apt-get update && \
    apt-get install -y libopenblas0 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/libfaiss.so /usr/local/lib/
COPY --from=builder /app/myapp /usr/local/bin/
RUN ldconfig

CMD ["myapp"]
```

---

## Verification

After installation, verify everything works:

```bash
# Clone examples
git clone https://github.com/NerdMeNot/faiss-go.git
cd faiss-go/examples

# Run basic example
go run basic_search.go

# Run all tests
cd ..
go test -v ./...
```

---

## Troubleshooting

### Common Issues

#### "cannot find -lfaiss"

**Cause:** FAISS library not in linker path

**Solution:**
```bash
# Linux
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
sudo ldconfig

# macOS
export DYLD_LIBRARY_PATH="/usr/local/lib:$DYLD_LIBRARY_PATH"

# Or install to standard location
sudo cmake --install build --prefix=/usr
```

#### "faiss/Index.h: No such file or directory"

**Cause:** FAISS headers not in include path

**Solution:**
```bash
# Linux
export CPATH="/usr/local/include:$CPATH"

# macOS
export CPATH="/opt/homebrew/include:$CPATH"

# Or set CGO flags explicitly
export CGO_CPPFLAGS="-I/usr/local/include"
export CGO_LDFLAGS="-L/usr/local/lib"
```

#### Build takes forever / runs out of memory

**Cause:** Compiling large C++ codebase

**Solution:**
```bash
# Limit parallel jobs
cmake --build build -j2  # Use only 2 cores

# Or use pre-built libraries (if available)
brew install faiss  # macOS
apt-get install libfaiss-dev  # Ubuntu
```

#### "undefined reference to `faiss::...`"

**Cause:** Linking order issue

**Solution:**
```bash
# Ensure -lfaiss comes before -lopenblas
export CGO_LDFLAGS="-lfaiss -lopenblas -lgfortran"
```

### Platform-Specific Issues

**macOS Apple Silicon (M1/M2):**
```bash
# Use Homebrew ARM64 paths
export CPATH="/opt/homebrew/include"
export LIBRARY_PATH="/opt/homebrew/lib"
```

**Ubuntu 20.04 (FAISS not in repos):**
```bash
# Must build from source
# Follow "Build from Source" instructions above
```

**Windows CGO issues:**
```powershell
# Use MSYS2 paths in environment
$env:CGO_ENABLED="1"
$env:CC="gcc"
$env:CXX="g++"
```

---

## Performance Tuning

### OpenMP Threads

```bash
# Set number of threads for FAISS operations
export OMP_NUM_THREADS=8

# Or in Go code:
// Not directly exposed yet, but FAISS respects OMP_NUM_THREADS
```

### BLAS Backend

For best performance on Intel CPUs, use MKL instead of OpenBLAS:

```bash
# Install MKL
# Ubuntu
apt-get install intel-mkl

# macOS
brew install intel-mkl

# Rebuild FAISS with MKL
cmake -B build -DBLA_VENDOR=Intel10_64lp
```

---

## Upgrading

To upgrade to a newer version of faiss-go:

```bash
go get -u github.com/NerdMeNot/faiss-go
go mod tidy
```

To upgrade FAISS:

```bash
# macOS
brew upgrade faiss

# Ubuntu (from source)
cd faiss
git pull
cmake --build build -j
sudo cmake --install build
sudo ldconfig
```

---

## Uninstallation

```bash
# Remove faiss-go from your project
go mod edit -droprequire github.com/NerdMeNot/faiss-go

# Remove FAISS
# macOS
brew uninstall faiss

# Linux
sudo rm -rf /usr/local/include/faiss
sudo rm -f /usr/local/lib/libfaiss.*
sudo ldconfig
```

---

## Getting Help

- 📖 [Documentation](https://pkg.go.dev/github.com/NerdMeNot/faiss-go)
- ❓ [FAQ](docs/FAQ.md)
- 🐛 [Report Issues](https://github.com/NerdMeNot/faiss-go/issues)
- 💬 [Discussions](https://github.com/NerdMeNot/faiss-go/discussions)

---

**Installation successful?** Check out the [Quick Start Guide](docs/QUICKSTART.md)!
