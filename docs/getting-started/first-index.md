# Your First Index: Step-by-Step Tutorial

This hands-on tutorial will guide you through creating, populating, and querying your first FAISS index in Go.

---

## What You'll Learn

By the end of this tutorial, you'll be able to:
- ✅ Create a FAISS index in Go
- ✅ Add vectors to the index
- ✅ Search for nearest neighbors
- ✅ Understand distance metrics
- ✅ Handle basic error cases

**Time required**: 15-20 minutes

---

## Prerequisites

- Go 1.21 or later installed
- faiss-go package installed (`go get github.com/NerdMeNot/faiss-go`)
- Basic understanding of vectors and similarity search

---

## Step 1: Project Setup

Create a new directory and initialize a Go module:

```bash
mkdir faiss-tutorial
cd faiss-tutorial
go mod init faiss-tutorial
go get github.com/NerdMeNot/faiss-go
```

Create `main.go`:

```bash
touch main.go
```

---

## Step 2: Import and Setup

Open `main.go` and add the imports:

```go
package main

import (
    "fmt"
    "log"
    "math/rand"

    "github.com/NerdMeNot/faiss-go"
)

func main() {
    fmt.Println("Welcome to FAISS-go!")
}
```

Build and run to verify setup:

```bash
go build -tags=faiss_use_lib  # Fast build
./faiss-tutorial
```

Expected output:
```
Welcome to FAISS-go!
```

---

## Step 3: Create Your First Index

An **index** is the core data structure in FAISS. It stores vectors and enables fast similarity search.

Let's create a flat index (exact search) for 128-dimensional vectors:

```go
func main() {
    // Define vector dimension
    dimension := 128

    // Create a flat index using L2 (Euclidean) distance
    index, err := faiss.NewIndexFlatL2(dimension)
    if err != nil {
        log.Fatalf("Failed to create index: %v", err)
    }
    defer index.Close() // Always close to free memory!

    fmt.Printf("Created index for %d-dimensional vectors\n", index.D())
    fmt.Printf("Current index size: %d vectors\n", index.Ntotal())
}
```

**Key Concepts**:
- `dimension`: All vectors must have the same dimension
- `NewIndexFlatL2`: Creates exact search index using L2 distance
- `defer index.Close()`: Essential to prevent memory leaks

Run it:
```bash
go run main.go
```

Output:
```
Created index for 128-dimensional vectors
Current index size: 0 vectors
```

---

## Step 4: Generate Some Vectors

FAISS expects vectors as a flattened `[]float32` slice:
- For N vectors of dimension D, you need a slice of length `N × D`
- Vectors are stored sequentially: `[v1_d1, v1_d2, ..., v1_dD, v2_d1, v2_d2, ..., v2_dD, ...]`

Let's generate random vectors for demonstration:

```go
// Helper function to generate random vectors
func generateRandomVectors(n, d int) []float32 {
    vectors := make([]float32, n*d)
    for i := range vectors {
        vectors[i] = rand.Float32()
    }
    return vectors
}

func main() {
    dimension := 128
    index, err := faiss.NewIndexFlatL2(dimension)
    if err != nil {
        log.Fatalf("Failed to create index: %v", err)
    }
    defer index.Close()

    // Generate 1000 random vectors
    numVectors := 1000
    vectors := generateRandomVectors(numVectors, dimension)

    fmt.Printf("Generated %d vectors\n", numVectors)
    fmt.Printf("Vectors slice length: %d (should be %d × %d = %d)\n",
        len(vectors), numVectors, dimension, numVectors*dimension)
}
```

Output:
```
Generated 1000 vectors
Vectors slice length: 128000 (should be 1000 × 128 = 128000)
```

---

## Step 5: Add Vectors to the Index

Now add the vectors to the index:

```go
func main() {
    dimension := 128
    index, err := faiss.NewIndexFlatL2(dimension)
    if err != nil {
        log.Fatalf("Failed to create index: %v", err)
    }
    defer index.Close()

    // Generate and add vectors
    numVectors := 1000
    vectors := generateRandomVectors(numVectors, dimension)

    err = index.Add(vectors)
    if err != nil {
        log.Fatalf("Failed to add vectors: %v", err)
    }

    fmt.Printf("Successfully added %d vectors to index\n", index.Ntotal())
}
```

Output:
```
Successfully added 1000 vectors to index
```

**Important Notes**:
- `Add()` accepts a flat slice of `[]float32`
- The slice length must be a multiple of the dimension
- `Ntotal()` returns the number of vectors in the index

---

## Step 6: Search for Nearest Neighbors

Now for the fun part — searching!

```go
func main() {
    dimension := 128
    index, err := faiss.NewIndexFlatL2(dimension)
    if err != nil {
        log.Fatalf("Failed to create index: %v", err)
    }
    defer index.Close()

    // Add vectors
    numVectors := 1000
    vectors := generateRandomVectors(numVectors, dimension)
    err = index.Add(vectors)
    if err != nil {
        log.Fatalf("Failed to add vectors: %v", err)
    }

    // Create a query vector
    query := generateRandomVectors(1, dimension)

    // Search for 5 nearest neighbors
    k := 5
    distances, indices, err := index.Search(query, k)
    if err != nil {
        log.Fatalf("Search failed: %v", err)
    }

    // Print results
    fmt.Println("\nTop 5 nearest neighbors:")
    for i := 0; i < k; i++ {
        fmt.Printf("%d. Vector #%d, Distance: %.4f\n",
            i+1, indices[i], distances[i])
    }
}
```

Output:
```
Top 5 nearest neighbors:
1. Vector #742, Distance: 12.3456
2. Vector #123, Distance: 13.7890
3. Vector #456, Distance: 14.2341
4. Vector #789, Distance: 15.6789
5. Vector #234, Distance: 16.1234
```

**Understanding the Results**:
- `indices[]`: IDs of the nearest neighbors (0 to 999)
- `distances[]`: L2 distances (lower = more similar)
- Results are sorted by distance (closest first)

---

## Step 7: Understand Distance Metrics

FAISS supports two main distance metrics:

### L2 Distance (Euclidean)

```go
index, _ := faiss.NewIndexFlatL2(dimension)
```

- Measures geometric distance
- Lower distance = more similar
- Formula: `sqrt(sum((a[i] - b[i])^2))`
- Use when: Vector magnitude matters

### Inner Product

```go
index, _ := faiss.NewIndexFlatIP(dimension)
```

- Measures dot product similarity
- Higher score = more similar
- Formula: `sum(a[i] * b[i])`
- Use when: For cosine similarity with normalized vectors

### Cosine Similarity Example

For cosine similarity, normalize vectors first:

```go
func normalize(v []float32, dimension int) []float32 {
    numVectors := len(v) / dimension
    normalized := make([]float32, len(v))
    copy(normalized, v)

    for i := 0; i < numVectors; i++ {
        start := i * dimension
        end := start + dimension
        vec := normalized[start:end]

        // Compute L2 norm
        var norm float32
        for _, val := range vec {
            norm += val * val
        }
        norm = float32(math.Sqrt(float64(norm)))

        // Normalize
        if norm > 0 {
            for j := range vec {
                vec[j] /= norm
            }
        }
    }

    return normalized
}

// Use it
vectors := generateRandomVectors(1000, 128)
normalized := normalize(vectors, 128)

index, _ := faiss.NewIndexFlatIP(128)
index.Add(normalized)

// Now inner product == cosine similarity!
```

---

## Step 8: Handle Errors Properly

Production-ready error handling:

```go
func main() {
    dimension := 128

    // 1. Check index creation
    index, err := faiss.NewIndexFlatL2(dimension)
    if err != nil {
        log.Fatalf("Failed to create index: %v", err)
    }
    defer index.Close()

    // 2. Validate vector dimensions
    vectors := generateRandomVectors(1000, dimension)
    if len(vectors)%dimension != 0 {
        log.Fatalf("Invalid vector data: length %d not divisible by dimension %d",
            len(vectors), dimension)
    }

    // 3. Check Add operation
    if err := index.Add(vectors); err != nil {
        log.Fatalf("Failed to add vectors: %v", err)
    }

    // 4. Validate query
    query := generateRandomVectors(1, dimension)
    if len(query) != dimension {
        log.Fatalf("Query vector has wrong dimension: got %d, want %d",
            len(query), dimension)
    }

    // 5. Check search
    distances, indices, err := index.Search(query, 5)
    if err != nil {
        log.Fatalf("Search failed: %v", err)
    }

    fmt.Printf("Found %d neighbors\n", len(indices))
}
```

---

## Complete Example

Here's the complete, production-ready example:

```go
package main

import (
    "fmt"
    "log"
    "math"
    "math/rand"

    "github.com/NerdMeNot/faiss-go"
)

func main() {
    // Configuration
    dimension := 128
    numVectors := 1000
    numNeighbors := 5

    // Create index
    fmt.Println("Creating index...")
    index, err := faiss.NewIndexFlatL2(dimension)
    if err != nil {
        log.Fatalf("Failed to create index: %v", err)
    }
    defer index.Close()

    // Generate and add vectors
    fmt.Printf("Generating %d random vectors...\n", numVectors)
    vectors := generateRandomVectors(numVectors, dimension)

    fmt.Println("Adding vectors to index...")
    if err := index.Add(vectors); err != nil {
        log.Fatalf("Failed to add vectors: %v", err)
    }
    fmt.Printf("Index now contains %d vectors\n", index.Ntotal())

    // Create query
    fmt.Println("\nSearching...")
    query := generateRandomVectors(1, dimension)

    // Search
    distances, indices, err := index.Search(query, numNeighbors)
    if err != nil {
        log.Fatalf("Search failed: %v", err)
    }

    // Print results
    fmt.Printf("\nTop %d nearest neighbors:\n", numNeighbors)
    for i := 0; i < numNeighbors; i++ {
        fmt.Printf("  %d. Vector #%d, Distance: %.4f\n",
            i+1, indices[i], distances[i])
    }

    fmt.Println("\n✅ Success! You've completed your first FAISS search.")
}

func generateRandomVectors(n, d int) []float32 {
    vectors := make([]float32, n*d)
    for i := range vectors {
        vectors[i] = rand.Float32()
    }
    return vectors
}

func normalize(v []float32, dimension int) []float32 {
    numVectors := len(v) / dimension
    normalized := make([]float32, len(v))
    copy(normalized, v)

    for i := 0; i < numVectors; i++ {
        start := i * dimension
        end := start + dimension
        vec := normalized[start:end]

        var norm float32
        for _, val := range vec {
            norm += val * val
        }
        norm = float32(math.Sqrt(float64(norm)))

        if norm > 0 {
            for j := range vec {
                vec[j] /= norm
            }
        }
    }

    return normalized
}
```

---

## Next Steps

Congratulations! You've successfully:
- ✅ Created a FAISS index
- ✅ Added vectors
- ✅ Performed similarity search
- ✅ Understood distance metrics

### Continue Learning

1. **[Choosing an Index](choosing-an-index.md)** - Learn about different index types
2. **[API Reference](../api/index-operations.md)** - Explore all operations
3. **[Examples](../examples/)** - See real-world use cases
4. **[Performance Tuning](../guides/performance-tuning.md)** - Optimize for production

---

## Common Questions

**Q: Can I use different vector dimensions in the same index?**
A: No, all vectors in an index must have the same dimension.

**Q: What's the maximum number of vectors?**
A: Depends on available RAM. Flat indexes store all vectors in memory. For billions of vectors, use compressed or on-disk indexes.

**Q: Is the search exact or approximate?**
A: `IndexFlatL2` and `IndexFlatIP` provide exact search (100% recall). Other index types offer faster approximate search.

**Q: Can I update vectors after adding them?**
A: FAISS indexes are generally immutable. To update, remove and re-add vectors (or rebuild the index).

---

## Troubleshooting

**Build fails**: Use pre-built libraries:
```bash
go build -tags=faiss_use_lib
```

**Dimension mismatch**: Ensure `len(vectors) % dimension == 0`

**Memory issues**: Reduce `numVectors` or use a compressed index type

---

**Need help?** Check the [FAQ](../faq.md) or [open an issue](https://github.com/NerdMeNot/faiss-go/issues).
