# Changelog

All notable changes to faiss-go will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Initial Release

Go bindings for FAISS (Facebook AI Similarity Search).

### Features

- **Pre-built static libraries** for Linux and macOS (AMD64, ARM64)
- **Zero dependencies** on supported platforms
- **All major index types** via IndexFactory:
  - Flat (exact search)
  - IVF (inverted file)
  - HNSW (graph-based)
  - PQ (product quantization)
  - LSH (locality-sensitive hashing)
  - Scalar Quantizer
  - Composite indexes (IVF+PQ, PCA+IVF, etc.)
- **Core operations**: Add, Search, Train, Save/Load
- **Custom IDs** via IndexIDMap
- **Search refinement** via IndexRefine
- **Metrics**: L2 (Euclidean) and Inner Product
- **Comprehensive test suite** across Go 1.21-1.25

### Notes

- Use `IndexFactory()` for all index creation (recommended)
- See [LIMITATIONS.md](LIMITATIONS.md) for known limitations
