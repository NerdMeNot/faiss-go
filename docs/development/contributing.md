# Contributing to faiss-go

Thank you for your interest in contributing to faiss-go! This document provides guidelines and instructions for contributing.

## 🚀 Development Setup

### Prerequisites

1. **Go 1.21 or later**
   ```bash
   go version
   ```

2. **C++17 Compiler**
   - Linux: GCC 7+ or Clang 5+
   - macOS: Xcode Command Line Tools
   - Windows: MSYS2 or Visual Studio 2019+

3. **BLAS Library**
   - Linux: `sudo apt-get install libopenblas-dev`
   - macOS: `brew install openblas`
   - Windows: Install via vcpkg or build from source

4. **Development Tools**
   ```bash
   # Install pre-commit hooks
   go install golang.org/x/tools/cmd/goimports@latest
   go install honnef.co/go/tools/cmd/staticcheck@latest
   ```

### Clone and Build

```bash
# Clone the repository
git clone https://github.com/NerdMeNot/faiss-go.git
cd faiss-go

# Build from source (first time will take 5-10 minutes)
go build -v ./...

# Run tests
go test -v ./...

# Run with pre-built libraries (faster for iteration)
go test -tags=faiss_use_lib -v ./...
```

## 📁 Project Structure

```
faiss-go/
├── faiss/                  # FAISS amalgamated source
│   ├── faiss.cpp          # Generated amalgamation
│   └── faiss.h            # C API header
├── libs/                   # Pre-built static libraries
│   └── [platform]/        # Platform-specific builds
├── scripts/               # Build and maintenance scripts
│   ├── generate_amalgamation.sh
│   ├── build_static_libs.sh
│   └── update_faiss.sh
├── examples/              # Usage examples
├── docs/                  # Additional documentation
├── faiss.go              # Main Go API
├── faiss_source.go       # CGO for source build
├── faiss_lib.go          # CGO for pre-built libs
├── index.go              # Index implementations
├── index_flat.go         # Flat index types
└── *_test.go             # Tests
```

## 🔨 Building the Amalgamation

The FAISS amalgamation is generated from the official FAISS repository.

### Generate/Update Amalgamation

```bash
cd scripts
./generate_amalgamation.sh [version]

# Example: Update to FAISS v1.8.0
./generate_amalgamation.sh v1.8.0

# Or use latest main branch
./generate_amalgamation.sh
```

This script:
1. Clones FAISS at the specified version
2. Configures CMake for CPU-only build
3. Generates amalgamated `faiss.cpp` and `faiss.h`
4. Copies files to `faiss/` directory
5. Updates version information

### Testing the Amalgamation

```bash
# Test that it compiles
go build -v ./...

# Run all tests
go test -v -race ./...

# Run benchmarks
go test -bench=. -benchmem ./...
```

## 📦 Building Pre-compiled Libraries

For maintainers building the pre-compiled static libraries:

### Build for Specific Platform

```bash
cd scripts
./build_static_libs.sh linux_amd64
```

### Build for All Platforms

```bash
cd scripts
./build_static_libs.sh all
```

This requires:
- Docker (for cross-compilation)
- ~10 GB disk space
- ~30-60 minutes build time

### Verify Built Libraries

```bash
# Check library symbols
nm -g libs/linux_amd64/libfaiss.a | grep faiss_

# Verify checksums
cd libs/linux_amd64
sha256sum -c checksums.txt

# Test integration
go test -tags=faiss_use_lib -v ./...
```

## 🧪 Testing

### Run Tests

```bash
# All tests
go test -v ./...

# With race detector
go test -race -v ./...

# Specific package
go test -v ./index

# With coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

### Benchmarks

```bash
# Run all benchmarks
go test -bench=. -benchmem ./...

# Specific benchmark
go test -bench=BenchmarkIndexFlat -benchmem ./...

# Compare performance
go test -bench=. -benchmem ./... > new.txt
git checkout main
go test -bench=. -benchmem ./... > old.txt
benchcmp old.txt new.txt
```

### Test Matrix

We test on:
- Go versions: 1.21, 1.22, 1.23
- Platforms: linux/amd64, linux/arm64, darwin/amd64, darwin/arm64
- Build modes: source, pre-built libs
- Compilers: GCC, Clang, MSVC

## 📝 Code Style

### Go Code

- Follow standard Go formatting: `gofmt -s`
- Use `goimports` for import organization
- Run `staticcheck` before committing
- Write idiomatic Go code (see [Effective Go](https://go.dev/doc/effective_go))

```bash
# Format code
gofmt -s -w .
goimports -w .

# Lint
staticcheck ./...
go vet ./...
```

### C/C++ Code

For any C/C++ wrapper code:
- Follow the existing FAISS style
- Use C++17 features where appropriate
- Keep wrapper code minimal
- Document any platform-specific code

### Documentation

- Document all exported functions, types, and constants
- Include usage examples in doc comments
- Update README.md for significant changes
- Add examples to `examples/` for new features

## 🔄 Pull Request Process

1. **Fork and Branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make Changes**
   - Write code
   - Add tests
   - Update documentation

3. **Test Locally**
   ```bash
   go test -v ./...
   go test -race ./...
   gofmt -s -w .
   staticcheck ./...
   ```

4. **Commit**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

   Commit message format:
   - `feat: Add new index type`
   - `fix: Correct search results ordering`
   - `docs: Update installation instructions`
   - `test: Add benchmarks for IVF index`
   - `chore: Update FAISS to v1.8.0`

5. **Push and Create PR**
   ```bash
   git push origin feature/my-new-feature
   ```
   Then create a pull request on GitHub.

6. **PR Requirements**
   - All tests pass
   - Code coverage doesn't decrease
   - Documentation is updated
   - Follows code style guidelines
   - Includes relevant tests

## 🐛 Reporting Issues

When reporting issues, please include:

1. **Environment**
   - OS and version
   - Go version (`go version`)
   - Compiler version (`gcc --version` or `clang --version`)
   - BLAS library and version

2. **Build Information**
   - Build mode (source or pre-built)
   - Build tags used
   - Full error output

3. **Reproduction**
   - Minimal code example
   - Steps to reproduce
   - Expected vs actual behavior

4. **Additional Context**
   - Stack traces
   - Logs
   - Screenshots (if relevant)

## 🎯 Areas for Contribution

### High Priority
- [ ] Additional index types (IVF, PQ, HNSW)
- [ ] Serialization/deserialization
- [ ] Batch operations
- [ ] Error handling improvements
- [ ] Performance benchmarks
- [ ] Windows testing and support

### Medium Priority
- [ ] Index training API
- [ ] Metric/distance function options
- [ ] Index parameter tuning helpers
- [ ] More comprehensive examples
- [ ] Integration tests

### Low Priority
- [ ] GPU support (future)
- [ ] Advanced index types
- [ ] Custom distance metrics
- [ ] Distributed search (future)

## 📚 Resources

- [FAISS Documentation](https://faiss.ai/)
- [FAISS GitHub Wiki](https://github.com/facebookresearch/faiss/wiki)
- [CGO Documentation](https://pkg.go.dev/cmd/cgo)
- [Go Wiki: CGO](https://github.com/golang/go/wiki/cgo)

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 💬 Questions?

- Open an issue for questions
- Check existing issues and discussions
- Read the FAQ in the README

Thank you for contributing! 🎉
