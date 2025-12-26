package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/NerdMeNot/faiss-go"
)

func main() {
	fmt.Println("FAISS Go Inner Product Search Example")
	fmt.Println("======================================")
	fmt.Println()

	// Create an index using inner product (similarity) metric
	dimension := 64
	fmt.Printf("Creating IndexFlatIP (Inner Product) with dimension %d...\n", dimension)

	index, err := faiss.NewIndexFlatIP(dimension)
	if err != nil {
		log.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	fmt.Printf("Index created!\n")
	fmt.Printf("  Metric: %s (higher values = more similar)\n", index.MetricType())
	fmt.Println()

	// Generate normalized vectors (common for inner product similarity)
	numVectors := 500
	fmt.Printf("Generating %d normalized vectors...\n", numVectors)

	vectors := make([]float32, numVectors*dimension)
	for i := 0; i < numVectors; i++ {
		// Generate and normalize each vector
		var norm float32
		offset := i * dimension
		for j := 0; j < dimension; j++ {
			val := rand.Float32()*2 - 1 // Random value in [-1, 1]
			vectors[offset+j] = val
			norm += val * val
		}
		norm = float32(math.Sqrt(float64(norm)))
		// Normalize
		for j := 0; j < dimension; j++ {
			vectors[offset+j] /= norm
		}
	}

	// Add vectors to index
	fmt.Printf("Adding vectors to index...\n")
	if err := index.Add(vectors); err != nil {
		log.Fatalf("Failed to add vectors: %v", err)
	}
	fmt.Printf("Added %d vectors\n", index.Ntotal())
	fmt.Println()

	// Create a normalized query vector
	query := make([]float32, dimension)
	var norm float32
	for i := range query {
		query[i] = rand.Float32()*2 - 1
		norm += query[i] * query[i]
	}
	norm = float32(math.Sqrt(float64(norm)))
	for i := range query {
		query[i] /= norm
	}

	// Search for most similar vectors
	k := 5
	fmt.Printf("Searching for %d most similar vectors...\n", k)

	similarities, indices, err := index.Search(query, k)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("\nTop %d most similar vectors:\n", k)
	fmt.Printf("(Inner product values range from -1 to 1, higher is more similar)\n\n")
	for i := 0; i < k; i++ {
		fmt.Printf("  %2d. Index: %6d, Similarity: %.6f\n", i+1, indices[i], similarities[i])
	}
	fmt.Println()
	fmt.Println("Example completed!")
}
