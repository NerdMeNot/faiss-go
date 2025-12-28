# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation overhaul with 40+ documentation files
- Complete README.md with motivation, use cases, and production examples
- Detailed getting-started guides (installation, quickstart, choosing an index, migration)
- Enhanced package-level godoc for better pkg.go.dev presentation
- Documentation navigation hub (docs/README.md)
- Migration guide from Python FAISS

### Changed
- Reorganized documentation structure into logical sections
- Improved code examples and usage patterns

## [0.1.0-alpha] - 2025-01-XX

### Added
- Initial public alpha release
- Complete FAISS bindings with 18+ index types
- Flat indexes (IndexFlatL2, IndexFlatIP)
- IVF indexes (IndexIVFFlat, IndexIVFPQ, IndexIVFScalarQuantizer)
- HNSW indexes (IndexHNSWFlat)
- Product Quantization (IndexPQ, IndexPQFastScan)
- Scalar Quantization indexes
- GPU indexes (GpuIndexFlat, GpuIndexIVFFlat)
- OnDisk indexes for billion-scale datasets
- ID mapping (IndexIDMap)
- Training API for indexes that require it
- Serialization (save/load indexes)
- Range search functionality
- Vector reconstruction
- Clustering (Kmeans)
- Preprocessing transforms (PCA, OPQ, Random Rotation)
- Index factory pattern
- Comprehensive test suite with recall validation
- Stress tests for scale (1K to 10M+ vectors)
- CI/CD pipelines (Linux, macOS, linting, coverage)
- Pre-built static libraries for fast builds
- Amalgamated source build mode
- Platform support: Linux (x64, ARM64), macOS (Intel, Apple Silicon), Windows (x64)

### Documentation
- README.md with quick start
- INSTALL.md with platform-specific instructions
- QUICKSTART.md tutorial
- CONTRIBUTING.md for developers
- TESTING.md with comprehensive test strategy
- TEST_EXECUTION_GUIDE.md
- VERSIONING.md for release process
- API_COMPLETENESS.md tracking Python parity
- FAQ.md
- Multiple example programs

## Project Milestones

### Completed
- ✅ 100% Python FAISS feature parity
- ✅ All major index types implemented
- ✅ Comprehensive testing infrastructure
- ✅ Multi-platform support
- ✅ Pre-built libraries for fast builds
- ✅ Production-ready documentation

### In Progress
- 🚧 Additional real-world examples
- 🚧 Performance benchmarking suite
- 🚧 GPU CI testing

### Planned
- 📋 v1.0.0 stable release
- 📋 Additional index types (Binary indexes, LSH variants)
- 📋 Enhanced GPU support
- 📋 Performance optimization tools
- 📋 Vector database integrations
- 📋 Distributed search examples

---

## Version History Legend

- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` for vulnerability fixes

---

## Links

- [Latest Release](https://github.com/NerdMeNot/faiss-go/releases/latest)
- [All Releases](https://github.com/NerdMeNot/faiss-go/releases)
- [Roadmap](https://github.com/NerdMeNot/faiss-go/discussions)
- [Report Issues](https://github.com/NerdMeNot/faiss-go/issues)

[Unreleased]: https://github.com/NerdMeNot/faiss-go/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/NerdMeNot/faiss-go/releases/tag/v0.1.0-alpha
