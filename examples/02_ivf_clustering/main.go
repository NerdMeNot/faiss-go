// IVF Clustering Example
//
// This example demonstrates the Inverted File Index (IVF):
// - How IVF uses clustering to speed up search
// - Training the index on representative data
// - Tuning nprobe for speed/accuracy tradeoff
// - When to use IVF vs Flat indexes
//
// IVF is one of the most important FAISS index types for large-scale search.
// It divides vectors into clusters and only searches relevant clusters at query time.
//
// Run: go run main.go

package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	faiss "github.com/NerdMeNot/faiss-go"
)

func main() {
	fmt.Println("FAISS Go - IVF Clustering Example")
	fmt.Println("==================================")
	fmt.Println()

	// Example 1: Basic IVF Index
	basicIVF()

	fmt.Println()
	fmt.Println("----------------------------------------")
	fmt.Println()

	// Example 2: Tuning nprobe for Speed/Accuracy
	tuneNprobe()

	fmt.Println()
	fmt.Println("----------------------------------------")
	fmt.Println()

	// Example 3: Different IVF Variants
	ivfVariants()
}

// basicIVF demonstrates the core IVF workflow: train, add, search
func basicIVF() {
	fmt.Println("Example 1: Basic IVF Index")
	fmt.Println("---------------------------")

	dimension := 128
	numVectors := 10000
	nlist := 100 // number of clusters (Voronoi cells)

	// Create IVF index using factory
	// "IVF100,Flat" = 100 clusters, flat (uncompressed) storage within clusters
	index, err := faiss.IndexFactory(dimension, fmt.Sprintf("IVF%d,Flat", nlist), faiss.MetricL2)
	if err != nil {
		log.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	fmt.Printf("Created IVF index with %d clusters\n", nlist)
	fmt.Printf("  Is trained: %t (IVF indexes require training!)\n", index.IsTrained())

	// Generate database vectors
	vectors := generateRandomVectors(numVectors, dimension)

	// IMPORTANT: IVF indexes must be trained before adding vectors
	// Training learns the cluster centroids from representative data
	// Rule of thumb: use at least 30 * nlist training vectors
	trainingVectors := vectors[:min(nlist*50, numVectors)*dimension]
	fmt.Printf("\nTraining on %d vectors (recommend >= %d)...\n",
		len(trainingVectors)/dimension, nlist*30)

	start := time.Now()
	if err := index.Train(trainingVectors); err != nil {
		log.Fatalf("Training failed: %v", err)
	}
	fmt.Printf("  Training took: %v\n", time.Since(start))
	fmt.Printf("  Is trained: %t\n", index.IsTrained())

	// Add all vectors to the index
	start = time.Now()
	if err := index.Add(vectors); err != nil {
		log.Fatalf("Failed to add vectors: %v", err)
	}
	fmt.Printf("  Added %d vectors in %v\n", index.Ntotal(), time.Since(start))

	// Search
	query := generateRandomVectors(1, dimension)
	k := 10

	start = time.Now()
	distances, labels, err := index.Search(query, k)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}
	fmt.Printf("\nSearch took: %v\n", time.Since(start))
	fmt.Printf("Top %d results:\n", k)
	for i := 0; i < min(5, k); i++ {
		fmt.Printf("  %d. ID=%d, Distance=%.4f\n", i+1, labels[i], distances[i])
	}
	if k > 5 {
		fmt.Printf("  ... (%d more results)\n", k-5)
	}
}

// tuneNprobe demonstrates the speed/accuracy tradeoff controlled by nprobe
func tuneNprobe() {
	fmt.Println("Example 2: Tuning nprobe")
	fmt.Println("-------------------------")
	fmt.Println("nprobe controls how many clusters to search at query time.")
	fmt.Println("Higher nprobe = better recall but slower search.")
	fmt.Println()

	dimension := 128
	numVectors := 50000
	nlist := 256

	// Create and train IVF index
	index, err := faiss.IndexFactory(dimension, fmt.Sprintf("IVF%d,Flat", nlist), faiss.MetricL2)
	if err != nil {
		log.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	vectors := generateRandomVectors(numVectors, dimension)

	// Train
	trainingData := vectors[:nlist*50*dimension]
	if err := index.Train(trainingData); err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	// Add
	if err := index.Add(vectors); err != nil {
		log.Fatalf("Failed to add vectors: %v", err)
	}

	fmt.Printf("Index: %d vectors, %d clusters\n\n", index.Ntotal(), nlist)

	// Get ground truth with exhaustive flat search
	flatIndex, _ := faiss.NewIndexFlatL2(dimension)
	defer flatIndex.Close()
	flatIndex.Add(vectors)

	query := generateRandomVectors(1, dimension)
	k := 10

	gtDist, gtLabels, _ := flatIndex.Search(query, k)
	gtSet := make(map[int64]bool)
	for _, id := range gtLabels {
		gtSet[id] = true
	}

	// Cast to access IVF-specific methods
	genericIndex := index.(*faiss.GenericIndex)

	// Test different nprobe values
	nprobeValues := []int{1, 4, 16, 64, 128, 256}

	fmt.Printf("%-10s %-15s %-10s\n", "nprobe", "Search Time", "Recall@10")
	fmt.Printf("%-10s %-15s %-10s\n", "------", "-----------", "---------")

	for _, nprobe := range nprobeValues {
		if err := genericIndex.SetNprobe(nprobe); err != nil {
			// May fail if nprobe > nlist
			continue
		}

		// Average over multiple searches
		numQueries := 100
		queries := generateRandomVectors(numQueries, dimension)

		start := time.Now()
		for i := 0; i < numQueries; i++ {
			q := queries[i*dimension : (i+1)*dimension]
			_, _, _ = index.Search(q, k)
		}
		avgTime := time.Since(start) / time.Duration(numQueries)

		// Calculate recall for original query
		distances, labels, _ := index.Search(query, k)
		_ = distances
		hits := 0
		for _, id := range labels {
			if gtSet[id] {
				hits++
			}
		}
		recall := float64(hits) / float64(k)

		fmt.Printf("%-10d %-15v %.1f%%\n", nprobe, avgTime, recall*100)
	}

	fmt.Printf("\nGround truth (exact search) - Top 3: IDs=%v, Distances=%.4f\n",
		gtLabels[:3], gtDist[:3])
}

// ivfVariants shows different IVF configurations
func ivfVariants() {
	fmt.Println("Example 3: IVF Variants")
	fmt.Println("------------------------")
	fmt.Println("IVF can be combined with different storage types:")
	fmt.Println()

	dimension := 64
	numVectors := 5000
	nlist := 50

	vectors := generateRandomVectors(numVectors, dimension)
	query := generateRandomVectors(1, dimension)
	k := 5

	variants := []struct {
		description string
		explain     string
	}{
		{
			fmt.Sprintf("IVF%d,Flat", nlist),
			"IVF with flat storage - highest accuracy, most memory",
		},
		{
			fmt.Sprintf("IVF%d,SQ8", nlist),
			"IVF with 8-bit scalar quantization - 4x less memory",
		},
		{
			fmt.Sprintf("IVF%d,PQ8", nlist),
			"IVF with product quantization - even more compression",
		},
	}

	for _, v := range variants {
		fmt.Printf("\n%s\n  %s\n", v.description, v.explain)

		index, err := faiss.IndexFactory(dimension, v.description, faiss.MetricL2)
		if err != nil {
			fmt.Printf("  Failed to create: %v\n", err)
			continue
		}
		defer index.Close()

		// Train
		if err := index.Train(vectors); err != nil {
			fmt.Printf("  Training failed: %v\n", err)
			continue
		}

		// Add
		if err := index.Add(vectors); err != nil {
			fmt.Printf("  Add failed: %v\n", err)
			continue
		}

		// Search
		distances, labels, err := index.Search(query, k)
		if err != nil {
			fmt.Printf("  Search failed: %v\n", err)
			continue
		}

		fmt.Printf("  Results: IDs=%v\n", labels)
		fmt.Printf("  Distances: %.4f\n", distances)
	}

	fmt.Println("\n\nWhen to use each:")
	fmt.Println("  - IVF,Flat: Best accuracy, dataset fits in memory")
	fmt.Println("  - IVF,SQ8: Good balance of accuracy and memory")
	fmt.Println("  - IVF,PQ8: Maximum compression, billion-scale datasets")
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
