# FAISS Amalgamated Source

This directory contains the amalgamated FAISS source code for compilation.

> **Note**: This amalgamation is **CPU-only**. For GPU support, use the main faiss-go build which includes CUDA support. See the main [README](../README.md) for GPU usage.

## Files

- `faiss.cpp` - Amalgamated FAISS C++ implementation (~10-15 MB)
- `faiss.h` - FAISS C API header file

## Generating the Amalgamation

The amalgamation files are generated using the script in `../scripts/generate_amalgamation.sh`.

To regenerate (only needed when updating FAISS version):

```bash
cd scripts
./generate_amalgamation.sh
```

This will:
1. Clone the FAISS repository
2. Configure the build for CPU-only
3. Generate amalgamated source files
4. Copy them to this directory

## Current FAISS Version

**Version**: v1.8.0 (will be updated after generation)

**Features included in this amalgamation**:
- CPU-only indexes
- L2 and Inner Product metrics
- Flat indexes
- IVF indexes
- Product Quantization
- HNSW (Hierarchical Navigable Small World)

**Features excluded from this amalgamation**:
- GPU support (CUDA) - *Use main faiss-go build for GPU support*
- Python bindings
- Unit tests
- Benchmarks

> **GPU Support**: The main faiss-go library supports GPU acceleration through CUDA. This amalgamation is CPU-only for simplified compilation and distribution. To use GPU features, build without the amalgamation using the standard build process.

## Build Requirements

When compiling from this amalgamated source, you need:
- C++17 compatible compiler
- BLAS/LAPACK implementation (OpenBLAS recommended)
- OpenMP support (usually included with compiler)

## Size Optimization

The amalgamation is optimized for:
- Minimal external dependencies
- Fast compilation
- Small binary size
- Easy distribution
