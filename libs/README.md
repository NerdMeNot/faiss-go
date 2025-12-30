# Pre-built FAISS Libraries

This directory contains pre-compiled static FAISS libraries for different platforms.

📖 **[Complete Documentation: docs/STATIC-BUILDS.md](../docs/STATIC-BUILDS.md)**

## Available Platforms

| Platform | Architecture | Build Mode | BLAS | Size | Runtime Deps |
|----------|-------------|------------|------|------|--------------|
| `linux_amd64` | x86_64 | Unified | OpenBLAS 0.3.30 (merged) | ~45MB | gomp, gfortran |
| `linux_arm64` | ARM64 | Unified | OpenBLAS 0.3.30 (merged) | ~45MB | gomp, gfortran |
| `darwin_amd64` | x86_64 | Standard | Accelerate Framework | ~9MB | Accelerate, libomp |
| `darwin_arm64` | ARM64 (M1/M2) | Standard | Accelerate Framework | ~9MB | Accelerate, libomp |
| `windows_amd64` | x86_64 | Unified | OpenBLAS 0.3.30 (merged) | ~45MB | gomp, gfortran |

## Build Modes

### Unified Build (Linux/Windows)
- **What**: FAISS + OpenBLAS merged into single `libfaiss.a`
- **Size**: ~45MB
- **Runtime Dependencies**: gomp (OpenMP), gfortran (Fortran runtime)
- **Use case**: Production, distribution, Docker containers

### Standard Build (macOS)
- **What**: FAISS linked against system Accelerate framework
- **Size**: ~9MB
- **Runtime Dependencies**: Accelerate (always available), libomp
- **Use case**: All macOS deployments (Accelerate cannot be statically linked)

**Why macOS is smaller:** Apple's Accelerate framework is a system library that cannot be merged into static builds. It's always available on macOS and highly optimized for Apple Silicon.

## File Structure

Each platform directory contains:
- `libfaiss.a` (or `.lib` on Windows) - Static FAISS library
  - **Unified builds**: Includes OpenBLAS merged in
  - **Standard builds**: FAISS only
- `libfaiss_c.a` - C API wrapper library
- `build_info.json` - Build metadata (includes build mode)
- `checksums.txt` - SHA256 checksums for verification
- `include/` - C API header files

## Usage

These libraries are automatically used when building with the `faiss_use_lib` tag:

```bash
go build -tags=faiss_use_lib,nogpu
```

**For Linux/Windows unified builds:**
- Requires: gomp (OpenMP) and gfortran (Fortran runtime)
- Install on Linux: `apt-get install libgomp1 libgfortran5`

**For macOS:**
- Ensure libomp is installed:
```bash
brew install libomp
```

## Building Libraries

### Using GitHub Actions (Recommended)

Trigger the workflow via GitHub UI:
1. Go to Actions → "Build Static Libraries"
2. Click "Run workflow"
3. Select FAISS version and platforms
4. Download artifacts when complete

### Local Build

**Standard build:**
```bash
./scripts/build_static_lib.sh linux-amd64 v1.13.2
```

**Unified build (Linux/Windows):**
```bash
./scripts/build_static_lib.sh linux-amd64 v1.13.2 --unified
```

**Prerequisites for unified builds:**
- CMake, Git, GCC/Clang
- gfortran (for OpenBLAS)
- Build time: ~30-40 minutes

See [docs/STATIC-BUILDS.md](../docs/STATIC-BUILDS.md) for complete build instructions.

## Size Information

| Platform | Library | Size | Notes |
|----------|---------|------|-------|
| Linux (unified) | libfaiss.a | ~45MB | Includes OpenBLAS |
| macOS (standard) | libfaiss.a | ~9MB | Uses Accelerate |
| Windows (unified) | faiss.lib | ~45MB | Includes OpenBLAS |

## Verification

Each library includes SHA256 checksums in `checksums.txt` for verification:

```bash
cd libs/linux_amd64
sha256sum -c checksums.txt
```

## License

The pre-built libraries are compiled from:
- FAISS (MIT License) - Copyright Meta Platforms, Inc.
- OpenBLAS (BSD License) - Copyright OpenBLAS developers

See LICENSE files in each directory for full license text.
