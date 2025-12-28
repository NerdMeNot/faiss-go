# 🔍 faiss-go

[![Go Reference](https://pkg.go.dev/badge/github.com/NerdMeNot/faiss-go.svg)](https://pkg.go.dev/github.com/NerdMeNot/faiss-go)
[![Go Report Card](https://goreportcard.com/badge/github.com/NerdMeNot/faiss-go)](https://goreportcard.com/report/github.com/NerdMeNot/faiss-go)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI Status](https://github.com/NerdMeNot/faiss-go/workflows/CI/badge.svg)](https://github.com/NerdMeNot/faiss-go/actions)
[![Go Version](https://img.shields.io/github/go-mod/go-version/NerdMeNot/faiss-go)](https://github.com/NerdMeNot/faiss-go)
[![Release](https://img.shields.io/github/v/release/NerdMeNot/faiss-go)](https://github.com/NerdMeNot/faiss-go/releases)

**Go bindings for FAISS** — Bring Facebook's battle-tested billion-scale vector similarity search to the Go ecosystem.

faiss-go provides idiomatic Go bindings for [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search). Build semantic search, recommendation systems, and image similarity at scale with Go's simplicity and FAISS's performance.

---

## 🎯 Why faiss-go?

### The Problem

Python dominates the ML/AI ecosystem, but deploying Python-based similarity search services at scale presents challenges:
- **Operational Complexity**: Managing Python dependencies and virtual environments
- **Concurrency Limitations**: Python's GIL limits concurrent query processing
- **Deployment Overhead**: Complex containerization and dependency management

### The Solution

faiss-go brings FAISS to Go's production-ready ecosystem:

**For Go Developers**:
- ⚡ **Native Performance**: Direct CGO bindings to FAISS C++ library
- 🔒 **Type Safety**: Compile-time type checking and error handling
- 🏗️ **Cloud-Native Ready**: Fits naturally in Go microservices architecture
- 🔄 **Better Concurrency**: Leverage Go's goroutines for parallel queries

**For ML Engineers**:
- 🎯 **Comprehensive Coverage**: Support for all major FAISS index types
- 📊 **Production-Proven**: Same FAISS library powering Meta's billion-scale search
- 🧪 **Well-Tested**: Recall validation and stress tests
- 📚 **Familiar Concepts**: Similar API patterns to FAISS

---

## ✨ Supported Features

### 🏗️ Index Types

**20+ Index Types** covering the full FAISS feature set:

| Category | Indexes | Use Case |
|----------|---------|----------|
| **Exact Search** | `FlatL2`, `FlatIP` | Perfect recall, baseline performance |
| **Fast Approximate** | `IVFFlat`, `HNSW`, `LSH` | High recall with 10-100x speedup |
| **Compressed** | `PQ`, `ScalarQuantizer` | 8-32x memory reduction |
| **Hybrid** | `IVFPQ`, `IVFScalarQuantizer` | Best speed/memory/recall balance |
| **SIMD Optimized** | `PQFastScan`, `IVFPQFastScan` | 2-4x faster queries |
| **GPU Accelerated** | `GpuIndexFlat`, `GpuIndexIVFFlat` | 10-100x faster with CUDA |
| **Billion-Scale** | `OnDisk` variants | Datasets larger than RAM |
| **Special Purpose** | `IDMap`, `Shards`, `PreTransform`, `Refine` | Custom IDs, sharding, transformations |
| **Binary Indexes** | `BinaryFlat`, `BinaryIVF`, `BinaryHash` | Hamming distance search |

### 🚀 Core Operations

- ✅ **Training API** - Index optimization for your data distribution
- ✅ **Serialization** - Save/load indexes to disk
- ✅ **Range Search** - Find all vectors within a distance threshold
- ✅ **Batch Operations** - Efficient bulk add/search
- ✅ **Vector Reconstruction** - Retrieve vectors from compressed indexes
- ✅ **Clustering (Kmeans)** - Vector clustering
- ✅ **Preprocessing** - PCA, OPQ, normalization transforms
- ✅ **Index Factory** - String-based index construction
- ✅ **Custom IDs** - Map external IDs to vectors

---

## 📦 Installation

### Prerequisites

faiss-go requires FAISS to be installed on your system.

> **🚀 GPU Acceleration**: For 10-100x faster search with CUDA, see [GPU Installation Guide](docs/getting-started/gpu-installation.md). The instructions below are for CPU-only builds.

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get update
sudo apt-get install -y libfaiss-dev libopenblas-dev
```

**Linux (Fedora/RHEL)**:
```bash
sudo dnf install -y faiss-devel openblas-devel
```

**macOS**:
```bash
brew install faiss openblas
```

**From Source (CPU)**:
See [FAISS Installation Guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)

**From Source (GPU)**:
See [GPU Installation Guide](docs/getting-started/gpu-installation.md) for CUDA-enabled builds

### Install faiss-go

```bash
go get github.com/NerdMeNot/faiss-go
```

---

## 🚀 Quick Start

### Your First Similarity Search

```go
package main

import (
    "fmt"
    "log"

    "github.com/NerdMeNot/faiss-go"
)

func main() {
    // Create an index for 128-dimensional vectors
    index, err := faiss.NewIndexFlatL2(128)
    if err != nil {
        log.Fatal(err)
    }
    defer index.Close()

    // Add vectors (flattened: [v1_d1, v1_d2, ..., v2_d1, v2_d2, ...])
    vectors := make([]float32, 1000 * 128) // 1000 vectors
    // ... populate with your data ...
    err = index.Add(vectors)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Index contains %d vectors\n", index.Ntotal())

    // Search for 10 nearest neighbors
    query := make([]float32, 128) // Single query vector
    // ... populate query ...
    distances, indices, err := index.Search(query, 10)
    if err != nil {
        log.Fatal(err)
    }

    // Print results
    fmt.Println("\nTop 10 nearest neighbors:")
    for i := 0; i < 10; i++ {
        fmt.Printf("%d. Vector #%d (distance: %.4f)\n",
            i+1, indices[i], distances[i])
    }
}
```

**Build and run**:

```bash
go build
./your-app
```

### Next Steps

- 📖 **[Quick Start Guide](docs/getting-started/quickstart.md)** - Detailed tutorial
- 🎓 **[Index Selection Guide](docs/getting-started/choosing-an-index.md)** - Pick the right index
- 📚 **[Complete Documentation](docs/)** - Comprehensive guides
- 💡 **[Examples](examples/)** - Real-world patterns

---

## 📊 Real-World Use Cases

### 1. Semantic Search

```go
// Create index for document embeddings
index, _ := faiss.NewIndexHNSWFlat(768, 32, faiss.MetricL2)
index.Add(documentEmbeddings) // 768-dim BERT/OpenAI embeddings

// Search
queryEmb := embedText("machine learning tutorial")
distances, indices, _ := index.Search(queryEmb, 10)
```

### 2. Image Similarity

```go
// Compressed index for millions of images
quantizer, _ := faiss.NewIndexFlatL2(2048)
index, _ := faiss.NewIndexIVFPQ(quantizer, 2048, 1000, 16, 8)
index.Train(imageFeatures)  // 2048-dim CNN features
index.Add(imageFeatures)

// Find similar images
_, similar, _ := index.Search(queryFeatures, 20)
```

### 3. Recommendations

```go
// Fast lookup for recommendations
quantizer, _ := faiss.NewIndexFlatL2(128)
index, _ := faiss.NewIndexIVFFlat(quantizer, 128, 4096, faiss.MetricL2)
index.Train(itemEmbeddings)
index.Add(itemEmbeddings)

// Get recommendations
_, recommendedItems, _ := index.Search(userEmbedding, 50)
```

---

## 📚 Documentation

Comprehensive documentation available:

### 🎓 Getting Started
- **[Installation Guide](docs/getting-started/installation.md)** - CPU installation for all platforms
- **[GPU Installation](docs/getting-started/gpu-installation.md)** - 🚀 CUDA setup for 10-100x speedup
- **[Quick Start](docs/getting-started/quickstart.md)** - Your first search in 5 minutes
- **[Choosing an Index](docs/getting-started/choosing-an-index.md)** - Index selection guide
- **[First Index Tutorial](docs/getting-started/first-index.md)** - Hands-on walkthrough

### 📖 Guides
- **[Index Types](docs/guides/index-types.md)** - Complete index reference (coming soon)
- **[Performance Tuning](docs/guides/performance-tuning.md)** - Optimization guide (coming soon)
- **[Benchmarking](docs/guides/benchmarking.md)** - Performance baselines

### 💻 API Reference
- **[pkg.go.dev Documentation](https://pkg.go.dev/github.com/NerdMeNot/faiss-go)** - Complete API reference

### 🧪 Testing & Development
- **[Testing Strategy](docs/testing/strategy.md)** - Comprehensive testing approach
- **[Contributing Guide](docs/development/contributing.md)** - Development guidelines

---

## 🧪 Testing & Quality

faiss-go includes comprehensive testing:

- ✅ **Recall Validation**: Tests verify >95% recall for approximate indexes
- ✅ **Stress Tests**: Scale tests up to 10M+ vectors
- ✅ **Integration Tests**: Full lifecycle for each index type
- ✅ **CI/CD**: Multi-platform, multi-Go-version testing

### Run Tests

```bash
# Quick validation
go test -v ./...

# With coverage
go test -v -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

---

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for:
- Development setup
- Testing guidelines
- PR process

### Quick Links

- 🐛 [Report a Bug](https://github.com/NerdMeNot/faiss-go/issues/new?template=bug_report.md)
- ✨ [Request a Feature](https://github.com/NerdMeNot/faiss-go/issues/new?template=feature_request.md)
- 💬 [Discussions](https://github.com/NerdMeNot/faiss-go/discussions)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

FAISS is licensed under the MIT License - Copyright (c) Meta Platforms, Inc. and affiliates.

---

## 🙏 Acknowledgments

- **[FAISS](https://github.com/facebookresearch/faiss)** by Meta AI Research
- The Go community for CGO best practices

---

## 🔗 Resources

### Official FAISS Resources
- **[FAISS GitHub](https://github.com/facebookresearch/faiss)** - Official repository
- **[FAISS Documentation](https://faiss.ai/)** - FAISS wiki and guides
- **[FAISS Paper](https://arxiv.org/abs/1702.08734)** - Research paper

### Project Links
- **[pkg.go.dev Documentation](https://pkg.go.dev/github.com/NerdMeNot/faiss-go)** - API reference
- **[GitHub Releases](https://github.com/NerdMeNot/faiss-go/releases)** - Version history

---

## 📞 Support

- 📖 **Documentation**: [Complete Guides](docs/)
- ❓ **FAQ**: [Frequently Asked Questions](docs/faq.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/NerdMeNot/faiss-go/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/NerdMeNot/faiss-go/discussions)

---

<div align="center">

**Built for the Go and ML communities**

⭐ **Star us on GitHub** if faiss-go helps you!

[Get Started](docs/getting-started/quickstart.md) • [Documentation](docs/) • [API Reference](https://pkg.go.dev/github.com/NerdMeNot/faiss-go)

</div>
