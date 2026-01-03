# Changelog

All notable changes to faiss-go will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Go bindings for FAISS vector similarity search
- Pre-built static libraries for 5 platforms:
  - Linux AMD64
  - Linux ARM64
  - macOS Intel (AMD64)
  - macOS Apple Silicon (ARM64)
  - Windows AMD64
- Index types via IndexFactory:
  - Flat (exact search)
  - IVF (inverted file)
  - HNSW (graph-based)
  - PQ (product quantization)
  - LSH (locality-sensitive hashing)
  - Scalar Quantizer
  - Composite indexes (IVF+PQ, PCA+IVF, etc.)
- Core operations:
  - Add vectors
  - Search (k-nearest neighbors)
  - Train (for IVF, PQ indexes)
  - Save/Load indexes
- IndexIDMap for custom ID mapping
- IndexRefine for search result refinement
- Support for L2 and Inner Product metrics
- Comprehensive test suite
- CI testing across Go 1.21-1.25
- Documentation and examples

### Notes

- Recommended to use `IndexFactory()` for all index creation
- See [LIMITATIONS.md](LIMITATIONS.md) for current limitations
