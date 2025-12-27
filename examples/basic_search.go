// +build ignore

package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/NerdMeNot/faiss-go"
)

func main() {
	fmt.Println("FAISS Go Basic Search Example")
	fmt.Println("==============================")
	fmt.Println()

	// Print build information
	buildInfo := faiss.GetBuildInfo()
	fmt.Println(buildInfo)
	fmt.Println()

	// Create an index for 128-dimensional vectors
	dimension := 128
	fmt.Printf("Creating IndexFlatL2 with dimension %d...\n", dimension)

	index, err := faiss.NewIndexFlatL2(dimension)
	if err != nil {
		log.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	fmt.Printf("Index created successfully!\n")
	fmt.Printf("  Dimension: %d\n", index.D())
	fmt.Printf("  Metric: %s\n", index.MetricType())
	fmt.Printf("  Is trained: %t\n", index.IsTrained())
	fmt.Println()

	// Generate some random vectors
	numVectors := 1000
	fmt.Printf("Generating %d random vectors...\n", numVectors)

	vectors := make([]float32, numVectors*dimension)
	for i := range vectors {
		vectors[i] = rand.Float32()
	}

	// Add vectors to the index
	fmt.Printf("Adding vectors to index...\n")
	if err := index.Add(vectors); err != nil {
		log.Fatalf("Failed to add vectors: %v", err)
	}

	fmt.Printf("Successfully added %d vectors\n", index.Ntotal())
	fmt.Println()

	// Create a query vector
	fmt.Printf("Creating query vector...\n")
	query := make([]float32, dimension)
	for i := range query {
		query[i] = rand.Float32()
	}

	// Search for the 10 nearest neighbors
	k := 10
	fmt.Printf("Searching for %d nearest neighbors...\n", k)

	distances, indices, err := index.Search(query, k)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("\nTop %d nearest neighbors:\n", k)
	for i := 0; i < k; i++ {
		fmt.Printf("  %2d. Index: %6d, Distance: %.6f\n", i+1, indices[i], distances[i])
	}
	fmt.Println()

	// Reset the index
	fmt.Printf("Resetting index...\n")
	if err := index.Reset(); err != nil {
		log.Fatalf("Failed to reset index: %v", err)
	}

	fmt.Printf("Index reset. Total vectors: %d\n", index.Ntotal())
	fmt.Println()
	fmt.Println("Example completed successfully!")
}
