// Basic Search Example
//
// This example demonstrates the fundamentals of FAISS vector search:
// - Creating a flat index (exact, brute-force search)
// - Adding vectors
// - Searching for nearest neighbors
// - Using different distance metrics (L2 vs Inner Product)
//
// Run: go run main.go

package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	faiss "github.com/NerdMeNot/faiss-go"
)

func main() {
	fmt.Println("FAISS Go - Basic Search Example")
	fmt.Println("================================")
	fmt.Println()

	// Example 1: L2 (Euclidean) Distance Search
	l2Search()

	fmt.Println()
	fmt.Println("----------------------------------------")
	fmt.Println()

	// Example 2: Inner Product (Cosine Similarity) Search
	innerProductSearch()

	fmt.Println()
	fmt.Println("----------------------------------------")
	fmt.Println()

	// Example 3: Using the Factory Pattern
	factoryPatternSearch()
}

// l2Search demonstrates nearest neighbor search using L2 (Euclidean) distance
func l2Search() {
	fmt.Println("Example 1: L2 (Euclidean) Distance Search")
	fmt.Println("------------------------------------------")

	// Create an index for 128-dimensional vectors
	// IndexFlatL2 performs exact (brute-force) search - accurate but O(n) complexity
	dimension := 128
	index, err := faiss.NewIndexFlatL2(dimension)
	if err != nil {
		log.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	fmt.Printf("Created IndexFlatL2 (dimension=%d)\n", index.D())
	fmt.Printf("  Metric: %s (lower distance = more similar)\n", index.MetricType())
	fmt.Printf("  Is trained: %t (flat indexes don't need training)\n", index.IsTrained())

	// Generate random database vectors
	numVectors := 1000
	vectors := generateRandomVectors(numVectors, dimension)

	// Add vectors to the index
	if err := index.Add(vectors); err != nil {
		log.Fatalf("Failed to add vectors: %v", err)
	}
	fmt.Printf("  Added %d vectors (total: %d)\n", numVectors, index.Ntotal())

	// Search for nearest neighbors of a query vector
	query := generateRandomVectors(1, dimension)
	k := 5 // find 5 nearest neighbors

	distances, labels, err := index.Search(query, k)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("\nTop %d nearest neighbors:\n", k)
	for i := 0; i < k; i++ {
		fmt.Printf("  %d. ID=%d, L2 Distance=%.4f (Euclidean=%.4f)\n",
			i+1, labels[i], distances[i], math.Sqrt(float64(distances[i])))
	}
}

// innerProductSearch demonstrates search using inner product (cosine similarity when normalized)
func innerProductSearch() {
	fmt.Println("Example 2: Inner Product Search")
	fmt.Println("--------------------------------")

	// Create an inner product index
	// When vectors are L2-normalized, inner product equals cosine similarity
	dimension := 64
	index, err := faiss.NewIndexFlatIP(dimension)
	if err != nil {
		log.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	fmt.Printf("Created IndexFlatIP (dimension=%d)\n", index.D())
	fmt.Printf("  Metric: %s (higher score = more similar)\n", index.MetricType())

	// Generate and normalize vectors (required for cosine similarity)
	numVectors := 500
	vectors := generateNormalizedVectors(numVectors, dimension)

	if err := index.Add(vectors); err != nil {
		log.Fatalf("Failed to add vectors: %v", err)
	}
	fmt.Printf("  Added %d normalized vectors\n", numVectors)

	// Search with a normalized query
	query := generateNormalizedVectors(1, dimension)
	k := 5

	scores, labels, err := index.Search(query, k)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("\nTop %d most similar vectors (cosine similarity):\n", k)
	for i := 0; i < k; i++ {
		// Clamp to [-1, 1] to handle floating point precision
		similarity := math.Max(-1, math.Min(1, float64(scores[i])))
		fmt.Printf("  %d. ID=%d, Cosine Similarity=%.4f\n",
			i+1, labels[i], similarity)
	}
}

// factoryPatternSearch demonstrates using IndexFactory for flexibility
func factoryPatternSearch() {
	fmt.Println("Example 3: Using IndexFactory")
	fmt.Println("------------------------------")

	dimension := 128

	// The factory pattern lets you create any index type with a string description
	// This is the same API as Python's faiss.index_factory()
	descriptions := []struct {
		desc   string
		metric faiss.MetricType
		note   string
	}{
		{"Flat", faiss.MetricL2, "Exact L2 search"},
		{"Flat", faiss.MetricInnerProduct, "Exact inner product search"},
	}

	for _, d := range descriptions {
		fmt.Printf("\nCreating index: %q (%s)\n", d.desc, d.note)

		index, err := faiss.IndexFactory(dimension, d.desc, d.metric)
		if err != nil {
			log.Printf("  Failed: %v", err)
			continue
		}
		defer index.Close()

		fmt.Printf("  Metric: %s, Trained: %t\n", index.MetricType(), index.IsTrained())

		// Add some vectors and search
		vectors := generateRandomVectors(100, dimension)
		if err := index.Add(vectors); err != nil {
			log.Printf("  Add failed: %v", err)
			continue
		}

		query := generateRandomVectors(1, dimension)
		distances, labels, err := index.Search(query, 3)
		if err != nil {
			log.Printf("  Search failed: %v", err)
			continue
		}

		fmt.Printf("  Search results: IDs=%v, Distances=%.4f\n", labels, distances)
	}

	fmt.Println("\n  (See other examples for HNSW, IVF, PQ index types)")
}

// generateRandomVectors creates random float32 vectors
func generateRandomVectors(n, d int) []float32 {
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = rand.Float32()
	}
	return vectors
}

// generateNormalizedVectors creates L2-normalized random vectors
func generateNormalizedVectors(n, d int) []float32 {
	vectors := make([]float32, n*d)
	for i := 0; i < n; i++ {
		// Generate random vector
		var norm float64
		offset := i * d
		for j := 0; j < d; j++ {
			val := rand.Float32()*2 - 1 // [-1, 1]
			vectors[offset+j] = val
			norm += float64(val * val)
		}
		// Normalize to unit length
		norm = math.Sqrt(norm)
		for j := 0; j < d; j++ {
			vectors[offset+j] /= float32(norm)
		}
	}
	return vectors
}
