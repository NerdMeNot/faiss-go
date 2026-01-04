# faiss-go

[![Go Reference](https://pkg.go.dev/badge/github.com/NerdMeNot/faiss-go.svg)](https://pkg.go.dev/github.com/NerdMeNot/faiss-go)
[![Go Report Card](https://goreportcard.com/badge/github.com/NerdMeNot/faiss-go)](https://goreportcard.com/report/github.com/NerdMeNot/faiss-go)
[![codecov](https://codecov.io/gh/NerdMeNot/faiss-go/graph/badge.svg)](https://codecov.io/gh/NerdMeNot/faiss-go)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI Status](https://github.com/NerdMeNot/faiss-go/workflows/CI/badge.svg)](https://github.com/NerdMeNot/faiss-go/actions/workflows/ci.yml)

Go bindings for [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search), enabling efficient similarity search and clustering of dense vectors.

## Features

- **Fast builds** - Pre-built static libraries for major platforms (30-second builds)
- **Multiple index types** - Flat, IVF, HNSW, PQ, LSH, and more
- **Flexible API** - IndexFactory for easy index creation
- **Cross-platform** - Linux and macOS (AMD64 and ARM64)
- **Production ready** - Comprehensive testing across Go 1.21-1.25

## Installation

```bash
go get github.com/NerdMeNot/faiss-go
```

Pre-built binaries are included for:
- Linux (AMD64, ARM64)
- macOS (Intel, Apple Silicon)

No additional dependencies required on supported platforms.

## Quick Start

```go
package main

import (
    "fmt"
    faiss "github.com/NerdMeNot/faiss-go"
)

func main() {
    // Create an index for 128-dimensional vectors
    index, err := faiss.IndexFactory(128, "Flat", faiss.MetricL2)
    if err != nil {
        panic(err)
    }
    defer index.Close()

    // Add vectors
    vectors := make([]float32, 1000*128)
    for i := range vectors {
        vectors[i] = float32(i % 100)
    }
    if err := index.Add(vectors); err != nil {
        panic(err)
    }

    // Search for nearest neighbors
    query := vectors[:128]
    distances, labels, err := index.Search(query, 5)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Nearest neighbors: %v\n", labels)
    fmt.Printf("Distances: %v\n", distances)
}
```

## Index Types

faiss-go supports various index types through the `IndexFactory` function:

| Type | Factory String | Use Case |
|------|---------------|----------|
| Flat | `"Flat"` | Exact search, small datasets |
| IVF | `"IVF100,Flat"` | Large datasets, fast approximate search |
| HNSW | `"HNSW32"` | High recall, graph-based search |
| PQ | `"PQ8"` | Memory-efficient, compressed vectors |
| IVF+PQ | `"IVF100,PQ8"` | Large scale, memory-efficient |
| LSH | `"LSH"` | Binary hashing |
| Scalar Quantizer | `"SQ8"` | Compressed with scalar quantization |
| With PCA | `"PCA64,IVF100,Flat"` | Dimensionality reduction |

### Example: IVF Index

```go
// Create IVF index with 100 clusters
index, err := faiss.IndexFactory(128, "IVF100,Flat", faiss.MetricL2)
if err != nil {
    panic(err)
}
defer index.Close()

// Train on sample data (required for IVF)
if err := index.Train(trainingVectors); err != nil {
    panic(err)
}

// Add vectors
if err := index.Add(vectors); err != nil {
    panic(err)
}

// Search
distances, labels, err := index.Search(query, 10)
```

### Example: HNSW Index

```go
// Create HNSW index with M=32
index, err := faiss.IndexFactory(128, "HNSW32", faiss.MetricL2)
if err != nil {
    panic(err)
}
defer index.Close()

// HNSW doesn't require training
if err := index.Add(vectors); err != nil {
    panic(err)
}

distances, labels, err := index.Search(query, 10)
```

## Metric Types

- `faiss.MetricL2` - Euclidean distance (L2)
- `faiss.MetricInnerProduct` - Inner product (for cosine similarity with normalized vectors)

## Serialization

```go
// Save index to file
if err := faiss.WriteIndex(index, "index.faiss"); err != nil {
    panic(err)
}

// Load index from file
loaded, err := faiss.ReadIndex("index.faiss")
if err != nil {
    panic(err)
}
defer loaded.Close()
```

## Custom IDs with IndexIDMap

```go
// Wrap any index with custom ID mapping
baseIndex, _ := faiss.IndexFactory(128, "Flat", faiss.MetricL2)
idMap, err := faiss.NewIndexIDMap(baseIndex)
if err != nil {
    panic(err)
}
defer idMap.Close()

// Add vectors with custom IDs
ids := []int64{100, 200, 300, 400, 500}
if err := idMap.AddWithIDs(vectors, ids); err != nil {
    panic(err)
}

// Search returns your custom IDs
distances, labels, _ := idMap.Search(query, 5)
// labels contains: [100, 200, ...] (your IDs)
```

## Documentation

- [Getting Started](docs/getting-started/) - Installation, quickstart, index selection
- [Guides](docs/guides/) - API reference, index types, performance tuning
- [Development](docs/development/) - Architecture, contributing, building libraries
- [Reference](docs/reference/) - FAQ, glossary, resources

## Requirements

- Go 1.21 or later
- Supported platform (or system FAISS installation)

For platforms without pre-built binaries:

```bash
go build -tags=faiss_use_system ./...
```

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md) for current limitations and workarounds.

**Key recommendation**: Use `IndexFactory()` for all index creation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI Research
