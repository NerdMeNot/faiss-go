# faiss-go Documentation

Welcome to the faiss-go documentation! This guide will help you get started with billion-scale vector similarity search in Go.

---

## 🚀 Quick Navigation

### New to faiss-go?

Start here to get up and running in minutes:

1. **[Installation Guide](installation.md)** - Set up faiss-go (30 seconds with pre-built binaries!)
2. **[Quickstart Tutorial](quickstart.md)** - Build your first search in 5 minutes
3. **[Examples](examples.md)** - See real-world code examples

### Understanding faiss-go

Learn how faiss-go works and how to use it effectively:

- **[Build Modes](build-modes.md)** - Pre-built binaries vs system FAISS (recommended reading!)
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Index Types Guide](guides/choosing-indexes.md)** - Which index to use for your use case

### Performance & Quality

See how faiss-go performs and how we ensure quality:

- **[Benchmarks](benchmarks.md)** - Performance data and comparisons
- **[Testing Strategy](testing.md)** - How we test faiss-go
- **[CI/CD Workflows](workflows.md)** - Our comprehensive testing pipeline

### Troubleshooting

Having issues? Check these resources:

- **[Troubleshooting Guide](troubleshooting.md)** - Common issues and solutions
- **[FAQ](faq.md)** - Frequently asked questions
- **[GitHub Issues](https://github.com/NerdMeNot/faiss-go/issues)** - Report bugs or request features

### Contributing

Want to contribute? We'd love your help!

- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute
- **[Code of Conduct](../CODE_OF_CONDUCT.md)** - Community guidelines
- **[Development Setup](../CONTRIBUTING.md#development-setup)** - Get set up for development

---

## 📚 Documentation Index

### Getting Started

| Document | Description | Time to Read |
|----------|-------------|--------------|
| [Installation](installation.md) | Detailed setup for all platforms | 5 min |
| [Quickstart](quickstart.md) | Build your first search | 10 min |
| [Build Modes](build-modes.md) | Static libs vs system FAISS | 5 min |

### Using faiss-go

| Document | Description | Time to Read |
|----------|-------------|--------------|
| [API Reference](api-reference.md) | Complete API documentation | Reference |
| [Examples](examples.md) | Real-world code examples | 15 min |
| [Index Types](guides/choosing-indexes.md) | Choosing the right index | 10 min |

### Performance & Testing

| Document | Description | Time to Read |
|----------|-------------|--------------|
| [Benchmarks](benchmarks.md) | Performance data | 10 min |
| [Testing](testing.md) | Testing strategy | 5 min |
| [Workflows](workflows.md) | CI/CD documentation | 10 min |

### Help & Support

| Document | Description | Time to Read |
|----------|-------------|--------------|
| [Troubleshooting](troubleshooting.md) | Common issues | Reference |
| [FAQ](faq.md) | Frequently asked questions | 10 min |
| [Glossary](glossary.md) | FAISS terminology | Reference |

### Additional Resources

| Document | Description | Time to Read |
|----------|-------------|--------------|
| [Changelog](changelog.md) | Version history | Reference |
| [Resources](resources.md) | External links and papers | 5 min |

---

## 🎯 Common Use Cases

Quick links to documentation for specific use cases:

### Semantic Search
- [Quickstart: Semantic Search](quickstart.md#semantic-search)
- [Example: Document Search](examples.md#semantic-document-search)
- [Index Choice: IndexFlatIP or IndexHNSW](guides/choosing-indexes.md#semantic-search)

### Image Similarity
- [Example: Image Search](examples.md#image-similarity)
- [Index Choice: IndexPQ or IndexIVFPQ](guides/choosing-indexes.md#image-search)

### Recommendation Systems
- [Example: Product Recommendations](examples.md#product-recommendations)
- [Index Choice: IndexIVFFlat](guides/choosing-indexes.md#recommendations)

### Billion-Scale Search
- [Guide: Billion-Scale Indexes](guides/billion-scale.md)
- [Index Choice: IndexIVFPQ with OnDisk](guides/choosing-indexes.md#billion-scale)

### GPU Acceleration
- [Installation: GPU Setup](installation.md#gpu-support)
- [Guide: GPU Indexes](guides/gpu-acceleration.md)

---

## 💡 Key Concepts

### What is FAISS?

FAISS (Facebook AI Similarity Search) is a library developed by Meta AI Research for efficient similarity search and clustering of dense vectors. It's used in production at Meta to search billions of vectors.

**Key features:**
- Exact and approximate nearest neighbor search
- Scales to billions of vectors
- GPU acceleration
- Multiple index types optimized for speed/memory/accuracy tradeoffs

### What is faiss-go?

faiss-go provides production-ready Go bindings to FAISS, bringing billion-scale vector search to the Go ecosystem.

**Key advantages:**
- ⚡ **Pre-built binaries** - 30-second builds vs 15-30 minutes
- 🔒 **Type-safe API** - Compile-time guarantees
- 🚀 **True concurrency** - Go goroutines > Python's GIL
- ✅ **Battle-tested** - Comprehensive CI testing Go 1.21-1.25

### Core Operations

**1. Creating an Index**
```go
index, err := faiss.NewIndexFlatL2(dimension)
```

**2. Adding Vectors**
```go
err = index.Add(vectors) // vectors: []float32
```

**3. Searching**
```go
distances, labels, err := index.Search(query, k)
```

**4. Saving/Loading**
```go
err = index.Write("index.faiss")
index, err = faiss.ReadIndex("index.faiss")
```

---

## 🔍 Finding What You Need

### By Topic

- **Installation issues?** → [Installation Guide](installation.md) + [Troubleshooting](troubleshooting.md)
- **How do I...?** → [Examples](examples.md) + [API Reference](api-reference.md)
- **Which index type?** → [Choosing Indexes Guide](guides/choosing-indexes.md)
- **Performance questions?** → [Benchmarks](benchmarks.md)
- **Build failing?** → [Build Modes](build-modes.md) + [Troubleshooting](troubleshooting.md)
- **Contributing?** → [Contributing Guide](../CONTRIBUTING.md)

### By Experience Level

**Beginners:**
1. Read [Installation](installation.md)
2. Follow [Quickstart](quickstart.md)
3. Try [Examples](examples.md)
4. Check [FAQ](faq.md)

**Intermediate:**
1. Read [Build Modes](build-modes.md)
2. Study [Index Types Guide](guides/choosing-indexes.md)
3. Review [API Reference](api-reference.md)
4. Check [Benchmarks](benchmarks.md)

**Advanced:**
1. Read [Testing Strategy](testing.md)
2. Study [CI/CD Workflows](workflows.md)
3. Review source code
4. Contribute! [Contributing Guide](../CONTRIBUTING.md)

---

## 📖 External Resources

- **FAISS Documentation**: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- **FAISS Wiki**: [https://github.com/facebookresearch/faiss/wiki](https://github.com/facebookresearch/faiss/wiki)
- **Research Paper**: [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)
- **Go Package Docs**: [https://pkg.go.dev/github.com/NerdMeNot/faiss-go](https://pkg.go.dev/github.com/NerdMeNot/faiss-go)

---

## 🤝 Community

- **GitHub**: [https://github.com/NerdMeNot/faiss-go](https://github.com/NerdMeNot/faiss-go)
- **Issues**: [Report bugs or request features](https://github.com/NerdMeNot/faiss-go/issues)
- **Discussions**: [Ask questions](https://github.com/NerdMeNot/faiss-go/discussions)

---

## 📝 Documentation Status

Last updated: December 2025
Version: Current with faiss-go v0.x.x
FAISS Version: v1.13.2

**Contributing to docs?** See [Contributing Guide](../CONTRIBUTING.md#documentation).

---

**Ready to get started? Begin with the [Installation Guide](installation.md)!** 🚀
