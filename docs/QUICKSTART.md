# Quick Start Guide

Get up and running with faiss-go in 5 minutes!

## 1. Install Dependencies

### Option A: Use Pre-built Libraries (No Dependencies!)

Skip to step 2 if using pre-built libraries.

### Option B: Build from Source

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential libopenblas-dev
```

**macOS:**
```bash
brew install openblas
```

**Windows:**
Install [MSYS2](https://www.msys2.org/), then:
```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-openblas
```

## 2. Install faiss-go

```bash
go get github.com/NerdMeNot/faiss-go
```

## 3. Write Your First Program

Create `main.go`:

```go
package main

import (
    "fmt"
    "log"
    "math/rand"

    "github.com/NerdMeNot/faiss-go"
)

func main() {
    // Create an index for 128-dimensional vectors
    dim := 128
    index, err := faiss.NewIndexFlatL2(dim)
    if err != nil {
        log.Fatal(err)
    }
    defer index.Close()

    // Generate 1000 random vectors
    numVectors := 1000
    vectors := make([]float32, numVectors*dim)
    for i := range vectors {
        vectors[i] = rand.Float32()
    }

    // Add vectors to the index
    if err := index.Add(vectors); err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Added %d vectors to index\n", index.Ntotal())

    // Create a query vector
    query := make([]float32, dim)
    for i := range query {
        query[i] = rand.Float32()
    }

    // Search for 5 nearest neighbors
    distances, indices, err := index.Search(query, 5)
    if err != nil {
        log.Fatal(err)
    }

    // Print results
    fmt.Println("\nTop 5 nearest neighbors:")
    for i := 0; i < 5; i++ {
        fmt.Printf("%d. Index=%d, Distance=%.4f\n",
            i+1, indices[i], distances[i])
    }
}
```

## 4. Build and Run

### Option A: Pre-built Libraries (Fast!)

```bash
go build -tags=faiss_use_lib
./your-program
```

Build time: ~30 seconds

### Option B: From Source

```bash
go build
./your-program
```

Build time: ~5-10 minutes (first time only, subsequent builds are fast!)

## 5. Expected Output

```
Added 1000 vectors to index

Top 5 nearest neighbors:
1. Index=742, Distance=12.3456
2. Index=123, Distance=13.7890
3. Index=456, Distance=14.2341
4. Index=789, Distance=15.6789
5. Index=234, Distance=16.1234
```

## What's Next?

### Learn More

- 📖 Read the [full documentation](https://pkg.go.dev/github.com/NerdMeNot/faiss-go)
- 📝 Check out [examples](../examples/)
- ❓ Read the [FAQ](FAQ.md)

### Try Different Index Types

**Inner Product (Similarity) Search:**
```go
// Create index with inner product metric
index, _ := faiss.NewIndexFlatIP(dim)

// Normalize vectors for cosine similarity
normalized := normalize(vectors)
index.Add(normalized)
```

### Common Use Cases

#### Semantic Search

```go
// Embed your documents using your favorite embedding model
embeddings := embedDocuments(documents)  // Your embedding function

// Create and populate index
index, _ := faiss.NewIndexFlatIP(dim)
index.Add(embeddings)

// Search for similar documents
queryEmbedding := embedText("your search query")
distances, indices, _ := index.Search(queryEmbedding, 10)

// Get top 10 most similar documents
for i, idx := range indices {
    fmt.Printf("%d. %s (score: %.4f)\n",
        i+1, documents[idx], distances[i])
}
```

#### Image Similarity

```go
// Extract image features using a CNN
features := extractFeatures(images)  // Your feature extraction

// Build index
index, _ := faiss.NewIndexFlatL2(featureDim)
index.Add(features)

// Find similar images
queryFeatures := extractFeatures(queryImage)
_, indices, _ := index.Search(queryFeatures, 5)

// Display similar images
for _, idx := range indices {
    displayImage(images[idx])
}
```

#### Recommendation System

```go
// User/item embeddings
userEmbeddings := trainEmbeddings(users)
itemEmbeddings := trainEmbeddings(items)

// Index items
itemIndex, _ := faiss.NewIndexFlatIP(dim)
itemIndex.Add(itemEmbeddings)

// Recommend items for a user
userEmb := userEmbeddings[userID]
_, recommendedItems, _ := itemIndex.Search(userEmb, 10)

// Show recommendations
for _, itemID := range recommendedItems {
    fmt.Printf("Recommended: %s\n", items[itemID].Name)
}
```

## Tips for Best Performance

1. **Use the right metric:**
   - L2 distance: When vector magnitude matters
   - Inner product: For normalized vectors (cosine similarity)

2. **Normalize for cosine similarity:**
   ```go
   normalized := normalize(vectors)  // Divide by L2 norm
   index, _ := faiss.NewIndexFlatIP(dim)
   index.Add(normalized)
   ```

3. **Batch operations:**
   ```go
   // Add vectors in batches for better performance
   batchSize := 10000
   for i := 0; i < len(vectors); i += batchSize*dim {
       end := min(i+batchSize*dim, len(vectors))
       index.Add(vectors[i:end])
   }
   ```

4. **Choose build mode:**
   - Development: Use `-tags=faiss_use_lib` for fast builds
   - Production: Build from source for optimal performance

## Troubleshooting

### Build Errors

**Error: "BLAS not found"**
```bash
# Install OpenBLAS (see step 1)
# Or use pre-built libraries
go build -tags=faiss_use_lib
```

**Error: "C++ compiler not found"**
```bash
# Install build tools (see step 1)
# Or use pre-built libraries
go build -tags=faiss_use_lib
```

### Runtime Errors

**Error: "not implemented"**

The stub implementation is active. You need to generate the FAISS amalgamation:
```bash
cd scripts
./generate_amalgamation.sh
```

## Need Help?

- 🐛 [Report an issue](https://github.com/NerdMeNot/faiss-go/issues)
- 💬 [Ask a question](https://github.com/NerdMeNot/faiss-go/discussions)
- 📖 [Read the FAQ](FAQ.md)

Happy searching! 🚀
