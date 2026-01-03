// PQ Compression Example
//
// This example demonstrates Product Quantization (PQ) for vector compression:
// - How PQ compresses vectors to reduce memory usage
// - Training PQ codebooks
// - Trade-offs between compression and accuracy
// - When to use PQ, SQ, and IVFPQ
//
// PQ is essential for billion-scale vector search where memory is the bottleneck.
// A 128-dimensional float32 vector (512 bytes) can be compressed to just 8-64 bytes.
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
	fmt.Println("FAISS Go - Product Quantization (PQ) Example")
	fmt.Println("=============================================")
	fmt.Println()

	// Example 1: Basic PQ Index
	basicPQ()

	fmt.Println()
	fmt.Println("----------------------------------------")
	fmt.Println()

	// Example 2: Comparing Compression Levels
	compareCompression()

	fmt.Println()
	fmt.Println("----------------------------------------")
	fmt.Println()

	// Example 3: Scalar Quantization (SQ)
	scalarQuantization()

	fmt.Println()
	fmt.Println("----------------------------------------")
	fmt.Println()

	// Example 4: IVF+PQ for Billion-Scale Search
	ivfPQ()
}

// basicPQ demonstrates a simple PQ index
func basicPQ() {
	fmt.Println("Example 1: Basic PQ Index")
	fmt.Println("--------------------------")

	dimension := 128 // Must be divisible by M (number of subquantizers)
	numVectors := 10000

	// Create PQ index using factory
	// "PQ16" = Product Quantization with 16 subquantizers
	// Each subquantizer encodes dimension/16 = 8 dimensions to 1 byte
	// Total compressed size: 16 bytes per vector (vs 512 bytes for float32)
	M := 16 // number of subquantizers (also compression ratio)
	index, err := faiss.IndexFactory(dimension, fmt.Sprintf("PQ%d", M), faiss.MetricL2)
	if err != nil {
		log.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	fmt.Printf("Created PQ index with %d subquantizers\n", M)
	fmt.Printf("  Original vector size: %d bytes (128 x 4 bytes)\n", dimension*4)
	fmt.Printf("  Compressed size: %d bytes (%.1fx compression)\n", M, float64(dimension*4)/float64(M))
	fmt.Printf("  Is trained: %t (PQ requires training to learn codebooks)\n", index.IsTrained())

	// Generate training data
	vectors := generateRandomVectors(numVectors, dimension)

	// Train PQ codebooks
	// PQ learns a codebook of 256 centroids for each subquantizer
	fmt.Printf("\nTraining PQ codebooks on %d vectors...\n", numVectors)
	start := time.Now()
	if err := index.Train(vectors); err != nil {
		log.Fatalf("Training failed: %v", err)
	}
	fmt.Printf("  Training took: %v\n", time.Since(start))
	fmt.Printf("  Is trained: %t\n", index.IsTrained())

	// Add vectors
	start = time.Now()
	if err := index.Add(vectors); err != nil {
		log.Fatalf("Failed to add vectors: %v", err)
	}
	fmt.Printf("  Added %d vectors in %v\n", index.Ntotal(), time.Since(start))

	// Search
	query := generateRandomVectors(1, dimension)
	k := 10

	distances, labels, err := index.Search(query, k)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("\nTop %d results:\n", k)
	for i := 0; i < min(5, k); i++ {
		fmt.Printf("  %d. ID=%d, Distance=%.4f\n", i+1, labels[i], distances[i])
	}

	fmt.Println("\nNote: PQ returns approximate distances due to quantization")
}

// compareCompression compares different compression levels
func compareCompression() {
	fmt.Println("Example 2: Compression vs Accuracy Trade-off")
	fmt.Println("----------------------------------------------")

	dimension := 64 // Must be divisible by all M values we test
	numVectors := 10000
	k := 10

	vectors := generateRandomVectors(numVectors, dimension)
	query := generateRandomVectors(1, dimension)

	// Ground truth from exact search
	flatIndex, _ := faiss.NewIndexFlatL2(dimension)
	defer flatIndex.Close()
	flatIndex.Add(vectors)
	_, gtLabels, _ := flatIndex.Search(query, k)

	gtSet := make(map[int64]bool)
	for _, id := range gtLabels {
		gtSet[id] = true
	}

	// Test different M values (compression levels)
	// M must divide dimension evenly
	mValues := []int{8, 16, 32, 64}

	fmt.Printf("Dimension: %d, Vectors: %d\n\n", dimension, numVectors)
	fmt.Printf("%-5s %-15s %-15s %-12s\n", "M", "Bytes/Vector", "Compression", "Recall@10")
	fmt.Printf("%-5s %-15s %-15s %-12s\n", "---", "------------", "-----------", "---------")

	for _, M := range mValues {
		if dimension%M != 0 {
			continue // M must divide dimension
		}

		index, err := faiss.IndexFactory(dimension, fmt.Sprintf("PQ%d", M), faiss.MetricL2)
		if err != nil {
			log.Printf("Failed to create PQ%d: %v", M, err)
			continue
		}

		// Train and add
		if err := index.Train(vectors); err != nil {
			log.Printf("Training failed: %v", err)
			index.Close()
			continue
		}
		if err := index.Add(vectors); err != nil {
			log.Printf("Add failed: %v", err)
			index.Close()
			continue
		}

		// Search
		_, labels, err := index.Search(query, k)
		if err != nil {
			log.Printf("Search failed: %v", err)
			index.Close()
			continue
		}

		// Calculate recall
		hits := 0
		for _, id := range labels {
			if gtSet[id] {
				hits++
			}
		}
		recall := float64(hits) / float64(k)

		bytesPerVector := M
		compression := float64(dimension*4) / float64(bytesPerVector)

		fmt.Printf("%-5d %-15d %-15.1fx %.1f%%\n",
			M, bytesPerVector, compression, recall*100)

		index.Close()
	}

	fmt.Println("\nHigher M = More bytes = Better accuracy")
	fmt.Println("Choose M based on your memory budget and accuracy requirements")
}

// scalarQuantization demonstrates SQ as a simpler alternative to PQ
func scalarQuantization() {
	fmt.Println("Example 3: Scalar Quantization (SQ)")
	fmt.Println("------------------------------------")
	fmt.Println("SQ is simpler than PQ: just quantize each dimension to fewer bits")
	fmt.Println()

	dimension := 128
	numVectors := 10000
	k := 10

	vectors := generateRandomVectors(numVectors, dimension)
	query := generateRandomVectors(1, dimension)

	// Ground truth
	flatIndex, _ := faiss.NewIndexFlatL2(dimension)
	defer flatIndex.Close()
	flatIndex.Add(vectors)
	_, gtLabels, _ := flatIndex.Search(query, k)

	gtSet := make(map[int64]bool)
	for _, id := range gtLabels {
		gtSet[id] = true
	}

	// Compare SQ variants
	variants := []struct {
		desc           string
		bytesPerVector int
		explain        string
	}{
		{"SQ8", dimension * 1, "8-bit quantization (1 byte per dimension)"},
		{"SQfp16", dimension * 2, "16-bit float (2 bytes per dimension)"},
	}

	fmt.Printf("%-10s %-15s %-15s %-12s\n", "Type", "Bytes/Vector", "Compression", "Recall@10")
	fmt.Printf("%-10s %-15s %-15s %-12s\n", "----", "------------", "-----------", "---------")

	// First show Flat baseline
	fmt.Printf("%-10s %-15d %-15s %-12s\n", "Flat", dimension*4, "1.0x", "100.0%")

	for _, v := range variants {
		index, err := faiss.IndexFactory(dimension, v.desc, faiss.MetricL2)
		if err != nil {
			// SQ types might not all be available
			continue
		}

		// Train and add
		if err := index.Train(vectors); err != nil {
			index.Close()
			continue
		}
		if err := index.Add(vectors); err != nil {
			index.Close()
			continue
		}

		// Search
		_, labels, err := index.Search(query, k)
		if err != nil {
			index.Close()
			continue
		}

		// Calculate recall
		hits := 0
		for _, id := range labels {
			if gtSet[id] {
				hits++
			}
		}
		recall := float64(hits) / float64(k)

		compression := float64(dimension*4) / float64(v.bytesPerVector)

		fmt.Printf("%-10s %-15d %-15.1fx %.1f%%\n",
			v.desc, v.bytesPerVector, compression, recall*100)

		index.Close()
	}

	fmt.Println("\nSQ is best when you need:")
	fmt.Println("  - Simple compression (4x with SQ8)")
	fmt.Println("  - Fast encoding/decoding")
	fmt.Println("  - High accuracy (SQ8 usually has >90% recall)")
}

// ivfPQ demonstrates the combination for billion-scale search
func ivfPQ() {
	fmt.Println("Example 4: IVF+PQ for Large-Scale Search")
	fmt.Println("-----------------------------------------")
	fmt.Println("Combine IVF (fast search via clustering) with PQ (compression)")
	fmt.Println("This is the go-to approach for billion-scale vector search")
	fmt.Println()

	dimension := 128
	numVectors := 50000
	nlist := 100  // number of clusters
	M := 16       // PQ subquantizers
	k := 10

	vectors := generateRandomVectors(numVectors, dimension)
	query := generateRandomVectors(1, dimension)

	// Create IVFPQ index
	// Format: "IVF{nlist},PQ{M}"
	description := fmt.Sprintf("IVF%d,PQ%d", nlist, M)
	index, err := faiss.IndexFactory(dimension, description, faiss.MetricL2)
	if err != nil {
		log.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	fmt.Printf("Created %s index\n", description)
	fmt.Printf("  Clusters: %d\n", nlist)
	fmt.Printf("  PQ bytes per vector: %d\n", M)
	fmt.Printf("  Compression: %.1fx\n", float64(dimension*4)/float64(M))

	// Train
	fmt.Printf("\nTraining on %d vectors...\n", numVectors)
	start := time.Now()
	if err := index.Train(vectors); err != nil {
		log.Fatalf("Training failed: %v", err)
	}
	fmt.Printf("  Training took: %v\n", time.Since(start))

	// Add
	start = time.Now()
	if err := index.Add(vectors); err != nil {
		log.Fatalf("Add failed: %v", err)
	}
	fmt.Printf("  Added %d vectors in %v\n", index.Ntotal(), time.Since(start))

	// Ground truth for recall calculation
	flatIndex, _ := faiss.NewIndexFlatL2(dimension)
	defer flatIndex.Close()
	flatIndex.Add(vectors)
	_, gtLabels, _ := flatIndex.Search(query, k)

	gtSet := make(map[int64]bool)
	for _, id := range gtLabels {
		gtSet[id] = true
	}

	// Set nprobe for the IVF component
	genericIndex := index.(*faiss.GenericIndex)

	fmt.Printf("\nSearch performance with different nprobe:\n")
	fmt.Printf("%-10s %-15s %-12s\n", "nprobe", "Search Time", "Recall@10")
	fmt.Printf("%-10s %-15s %-12s\n", "------", "-----------", "---------")

	for _, nprobe := range []int{1, 4, 16, 32} {
		genericIndex.SetNprobe(nprobe)

		start = time.Now()
		_, labels, _ := index.Search(query, k)
		searchTime := time.Since(start)

		hits := 0
		for _, id := range labels {
			if gtSet[id] {
				hits++
			}
		}
		recall := float64(hits) / float64(k)

		fmt.Printf("%-10d %-15v %.1f%%\n", nprobe, searchTime, recall*100)
	}

	fmt.Println("\nMemory usage comparison (for 1 billion vectors):")
	fmt.Printf("  Flat (512 bytes/vec):     %.1f TB\n", float64(1e9)*512/1e12)
	fmt.Printf("  IVFPQ (%d bytes/vec):     %.1f GB\n", M, float64(1e9)*float64(M)/1e9)
	fmt.Printf("  Memory savings:           %.0fx\n", 512/float64(M))
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
