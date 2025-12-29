# Contributing to faiss-go

Thank you for your interest in contributing to faiss-go! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Build Modes](#build-modes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Project Structure](#project-structure)
- [CI/CD Pipeline](#cicd-pipeline)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

---

## How Can I Contribute?

### Reporting Bugs

**Before submitting a bug report:**
1. Check the [FAQ](docs/faq.md) and [Troubleshooting Guide](docs/troubleshooting.md)
2. Search [existing issues](https://github.com/NerdMeNot/faiss-go/issues) to avoid duplicates
3. Ensure you're using a [supported Go version](README.md#requirements) (1.21+)

**When submitting a bug report, include:**
- **Go version**: Output of `go version`
- **Platform**: OS, architecture (e.g., "Ubuntu 22.04 AMD64")
- **Build mode**: Pre-built binaries or system FAISS
- **Minimal reproduction**: Smallest code that reproduces the issue
- **Expected vs actual behavior**
- **Stack trace or error messages**

### Suggesting Enhancements

We welcome feature requests! Please:
1. Check [existing issues](https://github.com/NerdMeNot/faiss-go/issues) first
2. Explain the **use case** (not just the solution)
3. Describe **expected behavior**
4. Consider if it should be in faiss-go or upstream FAISS

### Contributing Code

We love pull requests! Areas where contributions are especially welcome:
- **Bug fixes** - Always appreciated
- **Documentation** - Improve clarity, fix typos, add examples
- **Tests** - Increase coverage, add edge cases
- **Performance** - Optimizations with benchmarks
- **New index types** - Bindings for FAISS indexes we haven't wrapped yet
- **Examples** - Real-world use cases

---

## Development Setup

### Prerequisites

- **Go 1.21+** (we test on 1.21-1.25)
- **Git**
- **Build tools** (for system FAISS mode):
  - Linux: `build-essential`, `cmake`
  - macOS: Xcode Command Line Tools
  - Windows: MinGW or Visual Studio

### Quick Setup (Pre-built Binaries)

```bash
# Clone the repository
git clone https://github.com/NerdMeNot/faiss-go.git
cd faiss-go

# Build (uses pre-built static libraries)
go build -tags=nogpu ./...

# Run tests
go test -tags=nogpu -v ./...

# Run benchmarks
go test -tags=nogpu -bench=. -benchtime=1s ./...
```

**That's it!** Pre-built binaries make development setup instant.

### Alternative: System FAISS Mode

If you need to test against system FAISS or a custom build:

```bash
# Install FAISS
# Ubuntu/Debian:
sudo apt-get install libfaiss-dev libopenblas-dev

# macOS:
brew install faiss openblas

# Build with system FAISS
go build -tags=faiss_use_system,nogpu ./...

# Test
go test -tags=faiss_use_system,nogpu -v ./...
```

See [docs/build-modes.md](docs/build-modes.md) for detailed explanation.

---

## Build Modes

faiss-go supports two build modes:

### 1. Static Libraries (Default) ⚡

```bash
go build -tags=nogpu
```

- **Fast**: 30-second builds
- **No dependencies**: Works out of the box
- **5 platforms pre-built**: Linux/macOS/Windows AMD64/ARM64
- **Best for**: Development, CI, most users

### 2. System FAISS (Fallback) 🔧

```bash
go build -tags=faiss_use_system,nogpu
```

- **Flexible**: Use your own FAISS build
- **Any platform**: Works anywhere FAISS can be installed
- **Best for**: Custom FAISS configs, unsupported platforms

**Note**: The `nogpu` tag disables GPU support (required unless you have CUDA setup).

See [docs/build-modes.md](docs/build-modes.md) for comprehensive details.

---

## Testing

We maintain high test coverage and quality standards.

### Running Tests

```bash
# Quick tests (recommended during development)
go test -tags=nogpu -short ./...

# Full test suite
go test -tags=nogpu -v ./...

# With coverage
go test -tags=nogpu -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Specific package
go test -tags=nogpu -v ./test/recall

# Specific test
go test -tags=nogpu -v -run TestIndexFlatL2_Search
```

### Running Benchmarks

```bash
# All benchmarks
go test -tags=nogpu -bench=. -benchtime=1s ./...

# Specific benchmark
go test -tags=nogpu -bench=BenchmarkIndexFlatL2 -benchtime=5s

# With memory profiling
go test -tags=nogpu -bench=. -benchmem ./...
```

### Test Types

1. **Unit Tests** (`*_test.go`)
   - Test individual functions
   - Fast, isolated
   - Run with: `go test -short`

2. **Integration Tests** (`test/` directory)
   - Test real FAISS operations
   - Recall validation
   - Run with: `go test ./test/...`

3. **Benchmarks** (`benchmark_test.go`)
   - Performance regression detection
   - Run with: `go test -bench=.`

### Writing Tests

**Good test checklist:**
- ✅ Descriptive test names: `TestIndexFlatL2_SearchReturnsNearestNeighbors`
- ✅ Table-driven where appropriate
- ✅ Test both success and error cases
- ✅ Clean up resources (use `defer index.Close()`)
- ✅ Use `-short` flag for slow tests: `if testing.Short() { t.Skip() }`
- ✅ Validate results (recall, distances, etc.)

**Example:**

```go
func TestIndexFlatL2_Add(t *testing.T) {
    tests := []struct {
        name      string
        dimension int
        numVectors int
        wantErr   bool
    }{
        {"normal", 128, 1000, false},
        {"empty", 128, 0, false},
        {"wrong dimension", 128, 1000, true}, // vectors with wrong dim
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            idx, err := faiss.NewIndexFlatL2(tt.dimension)
            require.NoError(t, err)
            defer idx.Close()

            vectors := generateVectors(tt.numVectors, tt.dimension)
            err = idx.Add(vectors)

            if tt.wantErr {
                require.Error(t, err)
            } else {
                require.NoError(t, err)
                assert.Equal(t, int64(tt.numVectors), idx.Ntotal())
            }
        })
    }
}
```

---

## Pull Request Process

### Before You Start

1. **Open an issue first** for large changes
2. **Check existing PRs** to avoid duplicate work
3. **Fork the repository** and create a branch

### PR Workflow

1. **Create a branch**
   ```bash
   git checkout -b fix/issue-123-index-crash
   # or
   git checkout -b feature/add-ivfpqfs-support
   ```

2. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

3. **Test locally**
   ```bash
   # Build
   go build -tags=nogpu ./...

   # Test
   go test -tags=nogpu -v ./...

   # Lint
   golangci-lint run
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "fix: resolve crash in IndexIVFFlat.Search with empty index

   - Add nil check before search
   - Add test case for empty index search
   - Fixes #123"
   ```

5. **Push and create PR**
   ```bash
   git push origin fix/issue-123-index-crash
   ```
   Then create PR on GitHub.

### PR Requirements

Your PR will be reviewed for:

**Code Quality:**
- ✅ Tests pass in CI (all 11 jobs)
- ✅ Linter passes (golangci-lint)
- ✅ Code coverage maintained or improved
- ✅ No breaking changes (unless discussed)

**Documentation:**
- ✅ Public APIs have godoc comments
- ✅ Complex logic has inline comments
- ✅ README/docs updated if needed
- ✅ CHANGELOG.md updated for user-facing changes

**Testing:**
- ✅ New code has tests
- ✅ Tests cover success and error cases
- ✅ Benchmarks for performance-critical code

### Commit Message Format

We follow conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding or updating tests
- `perf`: Performance improvement
- `refactor`: Code refactoring
- `ci`: CI/CD changes

**Examples:**
```
feat(index): add support for IndexPQFastScan

- Implement NewIndexPQFastScan constructor
- Add SetBBS method for block size configuration
- Include tests and benchmarks

Closes #45

---

fix: prevent crash when searching empty IVF index

Adds nil check before dereferencing quantizer.

Fixes #123

---

docs: improve quickstart guide clarity

- Add more detailed setup instructions
- Include troubleshooting section
- Fix typos in code examples
```

---

## Coding Standards

### Go Style

- Follow [Effective Go](https://go.dev/doc/effective_go)
- Use `gofmt` (enforced by CI)
- Use `golangci-lint` (enforced by CI)
- Read our [Programming Guide](docs/programming-guide.md) - **Highly recommended!**
  - CGO best practices
  - Memory management patterns
  - API design principles
  - Testing patterns

### Specific Guidelines

**Naming:**
```go
// Good
func NewIndexFlatL2(d int) (*IndexFlat, error)
func (idx *IndexFlat) Search(query []float32, k int) ([]float32, []int64, error)

// Bad
func new_index_flat_l2(dimension int) (*index_flat, error)
func (i *IndexFlat) search(q []float32, K int) (dist []float32, ids []int64, e error)
```

**Error Handling:**
```go
// Good
if err := index.Add(vectors); err != nil {
    return fmt.Errorf("failed to add vectors: %w", err)
}

// Bad
index.Add(vectors)  // Ignoring error
```

**Resource Management:**
```go
// Good
index, err := faiss.NewIndexFlatL2(128)
if err != nil {
    return err
}
defer index.Close()  // Always close!

// Bad
index, _ := faiss.NewIndexFlatL2(128)
// Never closed - memory leak!
```

**Comments:**
```go
// Good - explains WHY
// Normalize vectors before adding to IndexFlatIP because
// inner product assumes unit vectors for cosine similarity
normalizeVectors(vectors)

// Bad - explains WHAT (obvious from code)
// Add vectors to index
index.Add(vectors)
```

### CGO Guidelines

When writing CGO code:

1. **Minimize CGO calls** - They have overhead
2. **Handle C memory carefully** - Use `C.free()`
3. **Check return codes** - C APIs return error codes
4. **Use unsafe.Pointer correctly** - Easy to get wrong

**Example:**
```go
func faissIndexAdd(ptr uintptr, vectors []float32, n int) error {
    idx := C.FaissIndex(unsafe.Pointer(ptr))
    vecPtr := (*C.float)(unsafe.Pointer(&vectors[0]))

    ret := C.faiss_Index_add(idx, C.int64_t(n), vecPtr)
    if ret != 0 {
        return fmt.Errorf("FAISS error code: %d", ret)
    }
    return nil
}
```

---

## Project Structure

```
faiss-go/
├── README.md              # Main landing page
├── CONTRIBUTING.md        # This file
├── LICENSE                # MIT license
├── go.mod                 # Go module definition
├── go.sum                 # Dependency checksums
│
├── docs/                  # All documentation
│   ├── README.md          # Documentation hub
│   ├── installation.md    # Detailed install guide
│   ├── quickstart.md      # 5-minute tutorial
│   ├── build-modes.md     # Static libs vs system FAISS
│   ├── api-reference.md   # Complete API docs
│   ├── examples.md        # Code examples
│   ├── testing.md         # Testing strategy
│   ├── benchmarks.md      # Performance data
│   ├── workflows.md       # CI/CD documentation
│   ├── troubleshooting.md # Common issues
│   ├── faq.md             # Frequently asked questions
│   └── changelog.md       # Version history
│
├── examples/              # Example code
│   ├── basic/             # Simple examples
│   ├── semantic-search/   # Semantic search demo
│   └── image-similarity/  # Image search demo
│
├── libs/                  # Pre-built static libraries
│   ├── linux_amd64/       # Linux AMD64 binaries
│   ├── linux_arm64/       # Linux ARM64 binaries
│   ├── darwin_amd64/      # macOS Intel binaries
│   ├── darwin_arm64/      # macOS Apple Silicon binaries
│   └── windows_amd64/     # Windows AMD64 binaries
│
├── test/                  # Integration tests
│   ├── datasets/          # Test datasets
│   ├── helpers/           # Test utilities
│   └── recall/            # Recall validation tests
│
├── .github/
│   └── workflows/         # GitHub Actions CI/CD
│       ├── ci.yml         # Main CI pipeline
│       ├── benchmark.yml  # Performance benchmarks
│       └── ...
│
├── *_test.go              # Unit tests (alongside code)
├── benchmark_test.go      # Benchmarks
│
├── faiss_lib.go           # Static library build mode (default)
├── faiss_system.go        # System FAISS build mode (fallback)
├── faiss_c_impl.cpp       # C++ bridge (for system mode)
│
├── index.go               # Core index interface
├── index_flat.go          # Flat indexes
├── index_ivf.go           # IVF indexes
├── index_hnsw.go          # HNSW indexes
├── index_pq.go            # PQ indexes
├── index_gpu.go           # GPU indexes
├── clustering.go          # K-means clustering
├── transforms.go          # PCA, OPQ, normalization
├── factory.go             # Index factory
└── ...                    # Other index types
```

### Key Files

- **`faiss_lib.go`**: Default build mode using pre-built static libraries
- **`faiss_system.go`**: Fallback mode using system-installed FAISS
- **`faiss_c_impl.cpp`**: C++ bridge between Go and FAISS (system mode only)
- **Build tags**: Control which files are compiled
  - `//go:build !faiss_use_system` → Default (static libs)
  - `//go:build faiss_use_system` → System FAISS

---

## CI/CD Pipeline

Our CI runs **11 parallel jobs** on every push:

### Test Matrix

- **Go versions**: 1.21, 1.22, 1.23, 1.24, 1.25
- **Operating systems**: Ubuntu, macOS
- **Architectures**: AMD64, ARM64 (via runners)
- **Build mode**: Pre-built static libraries (default)

### CI Jobs

1. **Build** (10 jobs)
   - Matrix: 5 Go versions × 2 OSes
   - Build time: ~30 seconds (thanks to pre-built binaries!)
   - Runs: `go build -tags=nogpu`

2. **Test** (10 jobs)
   - Unit tests + integration tests
   - Coverage tracked (Ubuntu + Go 1.25)
   - Runs: `go test -tags=nogpu -coverprofile=coverage.out`

3. **Benchmark** (10 jobs)
   - Quick smoke test benchmarks
   - Runs: `go test -tags=nogpu -bench=. -benchtime=100ms`

4. **Lint** (1 job)
   - golangci-lint with 5-minute timeout
   - Go 1.25 on Ubuntu

**Total runtime**: ~5-10 minutes

### Manual Workflows

Some workflows run manually (via GitHub Actions UI or `gh` CLI):

- **`benchmark.yml`**: Comprehensive benchmarks (quick/full modes)
- **`build-static-libs.yml`**: Build new static libraries for all platforms
- **`gpu-ci.yml`**: GPU-specific tests (requires GPU runner)

See [docs/workflows.md](docs/workflows.md) for details.

---

## Questions?

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/NerdMeNot/faiss-go/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NerdMeNot/faiss-go/discussions)

Thank you for contributing to faiss-go! 🎉
