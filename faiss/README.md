# FAISS Amalgamated Source

This directory contains the amalgamated FAISS source code for compilation.

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

**Features included**:
- CPU-only indexes
- L2 and Inner Product metrics
- Flat indexes
- IVF indexes
- Product Quantization
- HNSW (Hierarchical Navigable Small World)

**Features excluded**:
- GPU support (CUDA)
- Python bindings
- Unit tests
- Benchmarks

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
