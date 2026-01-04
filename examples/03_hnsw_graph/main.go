// HNSW Graph Example
//
// This example demonstrates the Hierarchical Navigable Small World (HNSW) index:
// - How HNSW uses a multi-layer graph for fast search
// - No training required (unlike IVF)
// - Excellent recall with fast query times
// - Trade-offs between M parameter and performance
//
// HNSW is one of the best-performing ANN algorithms, especially for
// datasets that require high recall with fast queries.
//
// Run: go run main.go

package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	faiss "github.com/NerdMeNot/faiss-go"
)

func main() {
	fmt.Println("FAISS Go - HNSW Graph Example")
	fmt.Println("==============================")
	fmt.Println()

	// Example 1: Basic HNSW Index
	basicHNSW()

	fmt.Println()
	fmt.Println("----------------------------------------")
	fmt.Println()

	// Example 2: Comparing Different M Values
	compareMValues()

	fmt.Println()
	fmt.Println("----------------------------------------")
	fmt.Println()

	// Example 3: HNSW vs Flat Performance
	compareWithFlat()
}

// basicHNSW demonstrates the core HNSW workflow
func basicHNSW() {
	fmt.Println("Example 1: Basic HNSW Index")
	fmt.Println("----------------------------")

	dimension := 128
	numVectors := 10000

	// Create HNSW index using factory
	// "HNSW32" = HNSW with M=32 (each node connected to ~32 neighbors)
	// M is the key parameter controlling the graph connectivity
	M := 32
	index, err := faiss.IndexFactory(dimension, fmt.Sprintf("HNSW%d", M), faiss.MetricL2)
	if err != nil {
		log.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	fmt.Printf("Created HNSW index with M=%d\n", M)
	fmt.Printf("  Is trained: %t (HNSW doesn't need training!)\n", index.IsTrained())

	// Generate database vectors
	vectors := generateRandomVectors(numVectors, dimension)

	// HNSW doesn't require training - vectors can be added immediately
	// Adding vectors builds the graph incrementally
	fmt.Printf("\nAdding %d vectors...\n", numVectors)
	start := time.Now()
	if err := index.Add(vectors); err != nil {
		log.Fatalf("Failed to add vectors: %v", err)
	}
	fmt.Printf("  Added in: %v (%.0f vectors/sec)\n",
		time.Since(start),
		float64(numVectors)/time.Since(start).Seconds())

	// Search
	query := generateRandomVectors(1, dimension)
	k := 10

	start = time.Now()
	distances, labels, err := index.Search(query, k)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}
	fmt.Printf("  Search took: %v\n", time.Since(start))

	fmt.Printf("\nTop %d results:\n", k)
	for i := 0; i < min(5, k); i++ {
		fmt.Printf("  %d. ID=%d, Distance=%.4f (Euclidean=%.4f)\n",
			i+1, labels[i], distances[i], math.Sqrt(float64(distances[i])))
	}
	if k > 5 {
		fmt.Printf("  ... (%d more results)\n", k-5)
	}
}

// compareMValues demonstrates how different M values affect performance
func compareMValues() {
	fmt.Println("Example 2: Comparing M Values")
	fmt.Println("------------------------------")
	fmt.Println("M controls graph connectivity: higher M = better recall, more memory, slower indexing")
	fmt.Println()

	dimension := 128
	numVectors := 20000
	k := 10

	vectors := generateRandomVectors(numVectors, dimension)
	queries := generateRandomVectors(100, dimension)

	// Ground truth from exact search
	flatIndex, _ := faiss.NewIndexFlatL2(dimension)
	defer flatIndex.Close()
	flatIndex.Add(vectors)

	// Calculate ground truth for all queries
	groundTruth := make([]map[int64]bool, 100)
	for q := 0; q < 100; q++ {
		query := queries[q*dimension : (q+1)*dimension]
		_, gtLabels, _ := flatIndex.Search(query, k)
		groundTruth[q] = make(map[int64]bool)
		for _, id := range gtLabels {
			groundTruth[q][id] = true
		}
	}

	// Test different M values
	mValues := []int{8, 16, 32, 64}

	fmt.Printf("%-6s %-15s %-15s %-12s\n", "M", "Index Time", "Search Time", "Recall@10")
	fmt.Printf("%-6s %-15s %-15s %-12s\n", "---", "----------", "-----------", "---------")

	for _, M := range mValues {
		index, err := faiss.IndexFactory(dimension, fmt.Sprintf("HNSW%d", M), faiss.MetricL2)
		if err != nil {
			log.Printf("Failed to create HNSW%d: %v", M, err)
			continue
		}

		// Time indexing
		start := time.Now()
		if err := index.Add(vectors); err != nil {
			log.Printf("Failed to add vectors: %v", err)
			index.Close()
			continue
		}
		indexTime := time.Since(start)

		// Time search and calculate recall
		start = time.Now()
		totalRecall := 0.0
		for q := 0; q < 100; q++ {
			query := queries[q*dimension : (q+1)*dimension]
			_, labels, _ := index.Search(query, k)

			hits := 0
			for _, id := range labels {
				if groundTruth[q][id] {
					hits++
				}
			}
			totalRecall += float64(hits) / float64(k)
		}
		searchTime := time.Since(start) / 100 // average per query
		avgRecall := totalRecall / 100

		fmt.Printf("%-6d %-15v %-15v %.1f%%\n",
			M, indexTime, searchTime, avgRecall*100)

		index.Close()
	}

	fmt.Println("\nRecommendations:")
	fmt.Println("  M=16: Good for memory-constrained scenarios")
	fmt.Println("  M=32: Best balance of speed and accuracy (default)")
	fmt.Println("  M=64: Use when highest recall is critical")
}

// compareWithFlat shows HNSW performance vs brute-force
func compareWithFlat() {
	fmt.Println("Example 3: HNSW vs Flat Performance")
	fmt.Println("-------------------------------------")

	dimension := 128
	numVectors := 50000
	numQueries := 100
	k := 10

	vectors := generateRandomVectors(numVectors, dimension)
	queries := generateRandomVectors(numQueries, dimension)

	// Create both index types
	flatIndex, _ := faiss.NewIndexFlatL2(dimension)
	defer flatIndex.Close()

	hnswIndex, _ := faiss.IndexFactory(dimension, "HNSW32", faiss.MetricL2)
	defer hnswIndex.Close()

	// Index both
	fmt.Printf("Dataset: %d vectors, %d dimensions\n\n", numVectors, dimension)

	fmt.Println("Indexing:")
	start := time.Now()
	flatIndex.Add(vectors)
	fmt.Printf("  Flat:  %v\n", time.Since(start))

	start = time.Now()
	hnswIndex.Add(vectors)
	fmt.Printf("  HNSW:  %v\n", time.Since(start))

	// Search performance
	fmt.Println("\nSearch (average over 100 queries):")

	// Flat search
	start = time.Now()
	for q := 0; q < numQueries; q++ {
		query := queries[q*dimension : (q+1)*dimension]
		flatIndex.Search(query, k)
	}
	flatTime := time.Since(start) / time.Duration(numQueries)
	fmt.Printf("  Flat:  %v per query\n", flatTime)

	// HNSW search
	start = time.Now()
	for q := 0; q < numQueries; q++ {
		query := queries[q*dimension : (q+1)*dimension]
		hnswIndex.Search(query, k)
	}
	hnswTime := time.Since(start) / time.Duration(numQueries)
	fmt.Printf("  HNSW:  %v per query (%.1fx faster)\n",
		hnswTime, float64(flatTime)/float64(hnswTime))

	// Accuracy comparison
	fmt.Println("\nRecall comparison (HNSW vs exact):")
	totalRecall := 0.0
	for q := 0; q < numQueries; q++ {
		query := queries[q*dimension : (q+1)*dimension]

		_, gtLabels, _ := flatIndex.Search(query, k)
		_, hnswLabels, _ := hnswIndex.Search(query, k)

		gtSet := make(map[int64]bool)
		for _, id := range gtLabels {
			gtSet[id] = true
		}

		hits := 0
		for _, id := range hnswLabels {
			if gtSet[id] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(k)
	}
	fmt.Printf("  HNSW Recall@%d: %.1f%%\n", k, totalRecall/float64(numQueries)*100)

	fmt.Println("\nConclusion:")
	fmt.Printf("  HNSW provides %.1fx speedup with >%.0f%% recall\n",
		float64(flatTime)/float64(hnswTime),
		totalRecall/float64(numQueries)*100)
	fmt.Println("  Use Flat for small datasets (<10K) or when exact results are required")
	fmt.Println("  Use HNSW for larger datasets when approximate search is acceptable")
}

// generateRandomVectors creates random float32 vectors
func generateRandomVectors(n, d int) []float32 {
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = rand.Float32()
	}
	return vectors
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
