// Pre-Transform Example
//
// This example demonstrates vector transformations before indexing:
// - PCA for dimensionality reduction
// - OPQ for optimized product quantization
// - Using transforms with the factory pattern
// - When and why to use pre-transforms
//
// Pre-transforms can significantly improve search quality and speed by:
// - Reducing dimensionality (PCA)
// - Decorrelating dimensions for better quantization (OPQ)
// - Enabling efficient compression of high-dimensional vectors
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
	fmt.Println("FAISS Go - Pre-Transform Example")
	fmt.Println("=================================")
	fmt.Println()

	// Example 1: PCA for Dimensionality Reduction
	pcaExample()

	fmt.Println()
	fmt.Println("----------------------------------------")
	fmt.Println()

	// Example 2: PCA + IVF for Large-Scale Search
	pcaWithIVF()

	fmt.Println()
	fmt.Println("----------------------------------------")
	fmt.Println()

	// Example 3: Using Factory with Pre-transforms
	factoryTransforms()
}

// pcaExample demonstrates standalone PCA usage
func pcaExample() {
	fmt.Println("Example 1: PCA for Dimensionality Reduction")
	fmt.Println("--------------------------------------------")

	dIn := 256  // original dimension
	dOut := 64  // reduced dimension
	numVectors := 5000

	fmt.Printf("Reducing from %d to %d dimensions (%.1fx compression)\n", dIn, dOut, float64(dIn)/float64(dOut))

	// Create PCA transform
	pca, err := faiss.NewPCAMatrix(dIn, dOut)
	if err != nil {
		log.Fatalf("Failed to create PCA: %v", err)
	}
	defer pca.Close()

	fmt.Printf("\nPCA created:\n")
	fmt.Printf("  Input dimension:  %d\n", pca.DIn())
	fmt.Printf("  Output dimension: %d\n", pca.DOut())
	fmt.Printf("  Is trained: %t\n", pca.IsTrained())

	// Generate training data
	vectors := generateRandomVectors(numVectors, dIn)

	// Train PCA - learns the principal components
	fmt.Printf("\nTraining on %d vectors...\n", numVectors)
	start := time.Now()
	if err := pca.Train(vectors); err != nil {
		log.Fatalf("PCA training failed: %v", err)
	}
	fmt.Printf("  Training took: %v\n", time.Since(start))
	fmt.Printf("  Is trained: %t\n", pca.IsTrained())

	// Apply PCA to reduce dimensionality
	reduced, err := pca.Apply(vectors)
	if err != nil {
		log.Fatalf("PCA apply failed: %v", err)
	}
	fmt.Printf("  Original size: %d floats\n", len(vectors))
	fmt.Printf("  Reduced size:  %d floats\n", len(reduced))

	// Now use reduced vectors with a flat index
	fmt.Println("\nSearching with reduced vectors:")
	index, _ := faiss.NewIndexFlatL2(dOut)
	defer index.Close()

	index.Add(reduced)

	// Query with PCA-reduced query
	query := generateRandomVectors(1, dIn)
	reducedQuery, _ := pca.Apply(query)

	distances, labels, _ := index.Search(reducedQuery, 5)
	fmt.Printf("  Top 5 results: IDs=%v\n", labels)
	fmt.Printf("  Distances: %.4f\n", distances)

	// Demonstrate reverse transform (approximate reconstruction)
	fmt.Println("\nReverse transform (reconstruction):")
	reconstructed, err := pca.ReverseTransform(reduced[:dOut]) // first vector
	if err != nil {
		fmt.Printf("  Reverse transform error: %v\n", err)
	} else {
		// Calculate reconstruction error
		var mse float64
		for i := 0; i < dIn; i++ {
			diff := float64(vectors[i] - reconstructed[i])
			mse += diff * diff
		}
		mse /= float64(dIn)
		fmt.Printf("  MSE of first vector: %.6f\n", mse)
		fmt.Printf("  (Lower MSE = better reconstruction)\n")
	}
}

// pcaWithIVF shows PCA + IVF for large-scale search
func pcaWithIVF() {
	fmt.Println("Example 2: PCA + IVF for Large-Scale Search")
	fmt.Println("---------------------------------------------")
	fmt.Println("Reduce dimension first, then use IVF for fast search")
	fmt.Println()

	dIn := 512   // high-dimensional input
	dOut := 128  // reduced dimension
	numVectors := 20000
	nlist := 100

	vectors := generateRandomVectors(numVectors, dIn)
	query := generateRandomVectors(1, dIn)

	// Method 1: Direct search on high-dimensional vectors
	fmt.Println("Method 1: Direct flat search (baseline)")
	flatIndex, _ := faiss.NewIndexFlatL2(dIn)
	defer flatIndex.Close()

	start := time.Now()
	flatIndex.Add(vectors)
	addTime := time.Since(start)

	start = time.Now()
	gtDistances, gtLabels, _ := flatIndex.Search(query, 10)
	searchTime := time.Since(start)

	fmt.Printf("  Add time:    %v\n", addTime)
	fmt.Printf("  Search time: %v\n", searchTime)
	fmt.Printf("  Top result:  ID=%d, Distance=%.4f\n", gtLabels[0], gtDistances[0])

	// Method 2: PCA reduction + IVF search
	fmt.Println("\nMethod 2: PCA reduction + IVF search")

	// Train PCA
	pca, _ := faiss.NewPCAMatrix(dIn, dOut)
	defer pca.Close()

	start = time.Now()
	if err := pca.Train(vectors); err != nil {
		log.Fatalf("PCA training failed: %v", err)
	}
	pcaTrainTime := time.Since(start)
	fmt.Printf("  PCA training: %v\n", pcaTrainTime)

	// Reduce all vectors
	start = time.Now()
	reducedVectors, _ := pca.Apply(vectors)
	pcaApplyTime := time.Since(start)
	fmt.Printf("  PCA apply:    %v\n", pcaApplyTime)

	// Create and train IVF index on reduced vectors
	ivfIndex, _ := faiss.IndexFactory(dOut, fmt.Sprintf("IVF%d,Flat", nlist), faiss.MetricL2)
	defer ivfIndex.Close()

	start = time.Now()
	ivfIndex.Train(reducedVectors)
	ivfTrainTime := time.Since(start)
	fmt.Printf("  IVF training: %v\n", ivfTrainTime)

	start = time.Now()
	ivfIndex.Add(reducedVectors)
	ivfAddTime := time.Since(start)
	fmt.Printf("  IVF add:      %v\n", ivfAddTime)

	// Search with reduced query
	reducedQuery, _ := pca.Apply(query)

	// Set higher nprobe for better recall
	genericIdx := ivfIndex.(*faiss.GenericIndex)
	genericIdx.SetNprobe(10)

	start = time.Now()
	distances, labels, _ := ivfIndex.Search(reducedQuery, 10)
	ivfSearchTime := time.Since(start)
	fmt.Printf("  Search time:  %v (%.1fx faster)\n",
		ivfSearchTime, float64(searchTime)/float64(ivfSearchTime))
	fmt.Printf("  Top result:   ID=%d, Distance=%.4f\n", labels[0], distances[0])

	// Calculate recall
	gtSet := make(map[int64]bool)
	for _, id := range gtLabels {
		gtSet[id] = true
	}
	hits := 0
	for _, id := range labels {
		if gtSet[id] {
			hits++
		}
	}
	fmt.Printf("  Recall@10:    %.0f%%\n", float64(hits)/10*100)

	fmt.Println("\nSummary:")
	fmt.Printf("  Memory: %d -> %d bytes per vector (%.1fx reduction)\n",
		dIn*4, dOut*4, float64(dIn)/float64(dOut))
}

// factoryTransforms shows using factory strings with transforms
func factoryTransforms() {
	fmt.Println("Example 3: Factory Strings with Pre-transforms")
	fmt.Println("------------------------------------------------")
	fmt.Println("The factory can automatically create indexes with pre-transforms")
	fmt.Println()

	dimension := 128
	numVectors := 10000
	k := 5

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

	// Test factory strings with transforms
	configs := []struct {
		description string
		explain     string
	}{
		{
			"Flat",
			"Baseline: exact search, no transform",
		},
		{
			"PCA64,Flat",
			"PCA to 64 dims, then exact search",
		},
		{
			"PCA64,IVF50,Flat",
			"PCA to 64 dims, then IVF clustering",
		},
	}

	fmt.Printf("%-20s %-12s %-12s\n", "Factory String", "Time", "Recall@5")
	fmt.Printf("%-20s %-12s %-12s\n", "--------------", "----", "--------")

	for _, cfg := range configs {
		index, err := faiss.IndexFactory(dimension, cfg.description, faiss.MetricL2)
		if err != nil {
			fmt.Printf("%-20s %-12s (failed: %v)\n", cfg.description, "-", err)
			continue
		}

		// Train if needed
		if !index.IsTrained() {
			if err := index.Train(vectors); err != nil {
				fmt.Printf("%-20s %-12s (train failed)\n", cfg.description, "-")
				index.Close()
				continue
			}
		}

		// Add vectors
		if err := index.Add(vectors); err != nil {
			fmt.Printf("%-20s %-12s (add failed)\n", cfg.description, "-")
			index.Close()
			continue
		}

		// Set nprobe for IVF
		if gen, ok := index.(*faiss.GenericIndex); ok {
			gen.SetNprobe(10) // May fail for non-IVF, that's OK
		}

		// Search
		start := time.Now()
		_, labels, err := index.Search(query, k)
		elapsed := time.Since(start)
		if err != nil {
			fmt.Printf("%-20s %-12s (search failed)\n", cfg.description, "-")
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
		recall := float64(hits) / float64(k) * 100

		fmt.Printf("%-20s %-12v %.0f%%\n", cfg.description, elapsed, recall)

		index.Close()
	}

	fmt.Println("\nFactory transform strings:")
	fmt.Println("  PCA{d}    - PCA reduction to d dimensions")
	fmt.Println("  OPQ{M}    - Optimized rotation for M subspaces")
	fmt.Println("  RR{d}     - Random rotation (preserves distances)")
	fmt.Println()
	fmt.Println("Example combinations:")
	fmt.Println("  'PCA64,Flat'           - Reduce to 64 dims, exact search")
	fmt.Println("  'PCA64,IVF100,Flat'    - Reduce, then IVF clustering")
	fmt.Println("  'OPQ8,IVF100,PQ8'      - OPQ + IVF + PQ (best compression)")
}

// generateRandomVectors creates random float32 vectors
func generateRandomVectors(n, d int) []float32 {
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = rand.Float32()
	}
	return vectors
}

// generateCorrelatedVectors creates vectors where some dimensions are correlated
// This shows PCA's value - it can identify and compress correlations
func generateCorrelatedVectors(n, d int) []float32 {
	vectors := make([]float32, n*d)
	for i := 0; i < n; i++ {
		offset := i * d
		// First half: random
		for j := 0; j < d/2; j++ {
			vectors[offset+j] = rand.Float32()
		}
		// Second half: correlated with first half (with noise)
		for j := d / 2; j < d; j++ {
			srcIdx := j - d/2
			vectors[offset+j] = vectors[offset+srcIdx]*0.8 + rand.Float32()*0.2
		}
	}
	return vectors
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func calculateMSE(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.MaxFloat64
	}
	var mse float64
	for i := range a {
		diff := float64(a[i] - b[i])
		mse += diff * diff
	}
	return mse / float64(len(a))
}
