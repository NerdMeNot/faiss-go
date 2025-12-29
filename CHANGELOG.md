# Changelog

All notable changes to faiss-go will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Initial Release Features

**Core Functionality:**
- 20+ FAISS index types with full Go bindings
- Support for exact and approximate nearest neighbor search
- Vector add, search, train operations
- Index serialization (save/load to disk)
- Range search support
- Custom ID mapping via IndexIDMap
- K-means clustering
- Preprocessing transforms (PCA, OPQ, normalization)

**Build System (Game-Changer!):**
- 🚀 Pre-built static libraries for 5 platforms
  - Linux AMD64 / ARM64
  - macOS Intel / Apple Silicon
  - Windows AMD64
- **30-second builds** vs 15-30 minute FAISS compilation
- Auto-detection: Uses pre-built libs automatically on supported platforms
- Fallback: System FAISS mode for other platforms

**Index Types:**
- **Exact**: IndexFlatL2, IndexFlatIP
- **Fast Approximate**: IndexIVFFlat, IndexHNSW, IndexLSH
- **Compressed**: IndexPQ, IndexScalarQuantizer
- **Hybrid**: IndexIVFPQ, IndexIVFScalarQuantizer
- **GPU**: GpuIndexFlat, GpuIndexIVFFlat (optional, requires CUDA)
- **Billion-Scale**: IndexIVFFlatOnDisk, IndexIVFPQOnDisk
- **Composite**: IndexIDMap, IndexShards, IndexRefine, IndexPreTransform
- **Binary**: IndexBinaryFlat, IndexBinaryIVF, IndexBinaryHash
- **SIMD Optimized**: IndexPQFastScan, IndexIVFPQFastScan

**Testing & Quality:**
- Comprehensive CI: 11 parallel jobs
- Multi-version: Go 1.21, 1.22, 1.23, 1.24, 1.25
- Multi-platform: Ubuntu + macOS
- Multi-arch: AMD64 + ARM64
- Test types: Unit tests, integration tests, benchmarks
- Recall validation tests
- Performance regression detection

**Documentation:**
- Comprehensive README with quickstart
- Contributing guide
- API reference (coming soon)
- Examples (coming soon)
- Build modes guide
- Testing documentation
- Troubleshooting guide (coming soon)
- FAQ (coming soon)

**Developer Experience:**
- Type-safe Go API
- Idiomatic error handling
- Memory-safe resource management
- Clear documentation
- Working examples

---

## Development Philosophy

faiss-go is built with quality-first principles:

1. **Pre-built binaries** - Developer time is valuable; don't waste it waiting for compilation
2. **Comprehensive testing** - Test across 5 Go versions, 2 OSes, 2 architectures
3. **Type safety** - Leverage Go's type system for compile-time guarantees
4. **Clear documentation** - Make it easy to get started and debug issues
5. **Idiomatic Go** - Feel natural to Go developers, not a direct Python port

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to contribute to faiss-go.

---

## Version History

### Upcoming v0.1.0 (Target: Q1 2026)

**First stable release** will include:
- ✅ All core FAISS index types
- ✅ Pre-built binaries for 5 platforms
- ✅ Comprehensive documentation
- ✅ Full test coverage
- ✅ Production-ready API (no breaking changes after 1.0)

---

## Support

- 🐛 **Report bugs**: [GitHub Issues](https://github.com/NerdMeNot/faiss-go/issues)
- 💬 **Ask questions**: [GitHub Discussions](https://github.com/NerdMeNot/faiss-go/discussions)
- 📖 **Read docs**: [Documentation](README.md)

---

_This changelog follows semantic versioning. Pre-1.0 versions may include breaking changes between minor versions._
