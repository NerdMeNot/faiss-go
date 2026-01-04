# Quickstart

Build your first vector similarity search in 5 minutes.

## Prerequisites

- Go 1.21 or later
- faiss-go installed: `go get github.com/NerdMeNot/faiss-go`

## Basic Example

Create `main.go`:

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

    // Add vectors (1000 vectors of 128 dimensions each)
    vectors := make([]float32, 1000*128)
    for i := range vectors {
        vectors[i] = float32(i % 100)
    }
    if err := index.Add(vectors); err != nil {
        panic(err)
    }

    fmt.Printf("Index contains %d vectors\n", index.Ntotal())

    // Search for nearest neighbors
    query := vectors[:128] // Use first vector as query
    distances, labels, err := index.Search(query, 5)
    if err != nil {
        panic(err)
    }

    fmt.Println("Top 5 nearest neighbors:")
    for i := range labels {
        fmt.Printf("  %d. Vector #%d (distance: %.2f)\n", i+1, labels[i], distances[i])
    }
}
```

Run it:

```bash
go run main.go
```

Output:
```
Index contains 1000 vectors
Top 5 nearest neighbors:
  1. Vector #0 (distance: 0.00)
  2. Vector #1 (distance: 128.00)
  ...
```

## Using Different Index Types

### IVF (Fast Approximate Search)

```go
// Create IVF index - faster search with slight accuracy tradeoff
index, err := faiss.IndexFactory(128, "IVF100,Flat", faiss.MetricL2)
if err != nil {
    panic(err)
}
defer index.Close()

// IVF requires training before adding vectors
trainingData := generateTrainingData(10000, 128)
if err := index.Train(trainingData); err != nil {
    panic(err)
}

// Now add vectors
if err := index.Add(vectors); err != nil {
    panic(err)
}

// Search
distances, labels, _ := index.Search(query, 10)
```

### HNSW (Graph-Based Search)

```go
// Create HNSW index - high recall, very fast
index, err := faiss.IndexFactory(128, "HNSW32", faiss.MetricL2)
if err != nil {
    panic(err)
}
defer index.Close()

// HNSW doesn't require training
if err := index.Add(vectors); err != nil {
    panic(err)
}

distances, labels, _ := index.Search(query, 10)
```

### PQ (Memory-Efficient)

```go
// Create PQ index - compressed vectors, uses less memory
index, err := faiss.IndexFactory(128, "PQ8", faiss.MetricL2)
if err != nil {
    panic(err)
}
defer index.Close()

// PQ requires training
if err := index.Train(trainingData); err != nil {
    panic(err)
}

if err := index.Add(vectors); err != nil {
    panic(err)
}

distances, labels, _ := index.Search(query, 10)
```

## Using Inner Product (Cosine Similarity)

For normalized vectors, inner product gives cosine similarity:

```go
index, err := faiss.IndexFactory(128, "Flat", faiss.MetricInnerProduct)
if err != nil {
    panic(err)
}
defer index.Close()

// Normalize your vectors before adding
normalizedVectors := normalize(vectors)
index.Add(normalizedVectors)

// Normalize query too
normalizedQuery := normalize(query)
distances, labels, _ := index.Search(normalizedQuery, 10)
// distances are now cosine similarities (higher = more similar)
```

## Saving and Loading Indexes

```go
// Save index to file
if err := faiss.WriteIndex(index, "my_index.faiss"); err != nil {
    panic(err)
}

// Load index from file
loaded, err := faiss.ReadIndex("my_index.faiss")
if err != nil {
    panic(err)
}
defer loaded.Close()

// Use loaded index
distances, labels, _ := loaded.Search(query, 10)
```

## Using Custom IDs

```go
// Create base index
baseIndex, _ := faiss.IndexFactory(128, "Flat", faiss.MetricL2)

// Wrap with ID map
idMap, err := faiss.NewIndexIDMap(baseIndex)
if err != nil {
    panic(err)
}
defer idMap.Close()

// Add vectors with custom IDs
ids := []int64{100, 200, 300, 400, 500}
vectors := make([]float32, 5*128)
// ... fill vectors
if err := idMap.AddWithIDs(vectors, ids); err != nil {
    panic(err)
}

// Search returns your custom IDs
distances, labels, _ := idMap.Search(query, 5)
// labels contains: [100, 200, ...] (your IDs)
```

## Key Concepts

### Dimensions

All vectors in an index must have the same number of dimensions. Common embedding dimensions:
- OpenAI embeddings: 1536
- BERT: 768
- Sentence transformers: 384

### Metrics

- **L2 (Euclidean)**: Lower distance = more similar
- **InnerProduct**: Higher value = more similar (use with normalized vectors for cosine similarity)

### Training

Some indexes need training to learn the data distribution:
- **Flat, HNSW**: No training needed
- **IVF, PQ**: Training required before adding vectors

## Next Steps

- [Choosing an Index](choosing-an-index.md) - Pick the right index for your use case
- [API Reference](../guides/api-reference.md) - Complete API documentation
