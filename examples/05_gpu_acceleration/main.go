// GPU Acceleration Example
//
// This example demonstrates GPU-accelerated vector search with FAISS:
// - Creating GPU indexes for massive speedup
// - Moving indexes between CPU and GPU
// - Multi-GPU support
// - Performance comparison CPU vs GPU
//
// REQUIREMENTS:
// - NVIDIA GPU with CUDA support
// - FAISS compiled with GPU support
// - Build with: go build -tags=gpu
//
// Without GPU hardware, this example will not compile.
// See the CPU examples (01-04) for non-GPU usage.
//
// Run: go build -tags=gpu && ./main

//go:build gpu
// +build gpu

package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	faiss "github.com/NerdMeNot/faiss-go"
)

func main() {
	fmt.Println("FAISS Go - GPU Acceleration Example")
	fmt.Println("====================================")
	fmt.Println()

	// Check GPU availability
	numGPUs, err := faiss.GetNumGPUs()
	if err != nil {
		log.Fatalf("Failed to get GPU count: %v", err)
	}
	fmt.Printf("Detected %d GPU(s)\n\n", numGPUs)

	if numGPUs == 0 {
		log.Fatal("No GPUs detected. This example requires CUDA-capable GPU.")
	}

	// Example 1: Basic GPU Index
	basicGPU()

	fmt.Println()
	fmt.Println("----------------------------------------")
	fmt.Println()

	// Example 2: GPU vs CPU Performance
	cpuVsGPU()

	fmt.Println()
	fmt.Println("----------------------------------------")
	fmt.Println()

	// Example 3: GPU IVF Index
	gpuIVF()
}

// basicGPU demonstrates creating and using a GPU flat index
func basicGPU() {
	fmt.Println("Example 1: Basic GPU Index")
	fmt.Println("---------------------------")

	dimension := 128
	numVectors := 100000

	// Create GPU resources
	// This manages GPU memory and CUDA streams
	res, err := faiss.NewStandardGpuResources()
	if err != nil {
		log.Fatalf("Failed to create GPU resources: %v", err)
	}
	defer res.Free()

	// Optional: Set temporary memory size (default is usually fine)
	// res.SetTempMemory(64 * 1024 * 1024) // 64 MB

	// Create GPU flat index on device 0
	deviceID := 0
	index, err := faiss.NewGpuIndexFlatL2(res, dimension, deviceID)
	if err != nil {
		log.Fatalf("Failed to create GPU index: %v", err)
	}
	defer index.Close()

	fmt.Printf("Created GPU IndexFlatL2 on device %d\n", deviceID)
	fmt.Printf("  Dimension: %d\n", index.D())

	// Generate and add vectors
	vectors := generateRandomVectors(numVectors, dimension)

	start := time.Now()
	if err := index.Add(vectors); err != nil {
		log.Fatalf("Failed to add vectors: %v", err)
	}
	fmt.Printf("  Added %d vectors in %v\n", index.Ntotal(), time.Since(start))

	// Search
	numQueries := 1000
	queries := generateRandomVectors(numQueries, dimension)
	k := 10

	start = time.Now()
	distances, labels, err := index.Search(queries, k)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}
	searchTime := time.Since(start)

	fmt.Printf("\nSearched %d queries in %v (%.0f queries/sec)\n",
		numQueries, searchTime,
		float64(numQueries)/searchTime.Seconds())

	// Show first result
	fmt.Printf("First query - Top result: ID=%d, Distance=%.4f\n",
		labels[0], distances[0])
}

// cpuVsGPU compares CPU and GPU search performance
func cpuVsGPU() {
	fmt.Println("Example 2: CPU vs GPU Performance")
	fmt.Println("-----------------------------------")

	dimension := 128
	numVectors := 500000
	numQueries := 1000
	k := 10

	vectors := generateRandomVectors(numVectors, dimension)
	queries := generateRandomVectors(numQueries, dimension)

	fmt.Printf("Dataset: %d vectors, %d queries, dimension=%d\n\n", numVectors, numQueries, dimension)

	// CPU Flat Index
	fmt.Println("CPU IndexFlatL2:")
	cpuIndex, _ := faiss.NewIndexFlatL2(dimension)
	defer cpuIndex.Close()

	start := time.Now()
	cpuIndex.Add(vectors)
	cpuAddTime := time.Since(start)
	fmt.Printf("  Add time: %v\n", cpuAddTime)

	start = time.Now()
	for i := 0; i < numQueries; i++ {
		q := queries[i*dimension : (i+1)*dimension]
		cpuIndex.Search(q, k)
	}
	cpuSearchTime := time.Since(start)
	fmt.Printf("  Search time: %v (%.0f queries/sec)\n",
		cpuSearchTime, float64(numQueries)/cpuSearchTime.Seconds())

	// GPU Flat Index
	fmt.Println("\nGPU IndexFlatL2:")
	res, _ := faiss.NewStandardGpuResources()
	defer res.Free()

	gpuIndex, err := faiss.NewGpuIndexFlatL2(res, dimension, 0)
	if err != nil {
		log.Printf("Failed to create GPU index: %v", err)
		return
	}
	defer gpuIndex.Close()

	start = time.Now()
	gpuIndex.Add(vectors)
	gpuAddTime := time.Since(start)
	fmt.Printf("  Add time: %v\n", gpuAddTime)

	// GPU can handle batched queries efficiently
	start = time.Now()
	gpuIndex.Search(queries, k)
	gpuSearchTime := time.Since(start)
	fmt.Printf("  Search time (batched): %v (%.0f queries/sec)\n",
		gpuSearchTime, float64(numQueries)/gpuSearchTime.Seconds())

	// Comparison
	fmt.Printf("\nSpeedup (GPU vs CPU):\n")
	fmt.Printf("  Add:    %.1fx %s\n",
		speedup(cpuAddTime, gpuAddTime),
		faster(cpuAddTime, gpuAddTime))
	fmt.Printf("  Search: %.1fx %s\n",
		speedup(cpuSearchTime, gpuSearchTime),
		faster(cpuSearchTime, gpuSearchTime))
}

// gpuIVF demonstrates GPU IVF index with training
func gpuIVF() {
	fmt.Println("Example 3: GPU IVF Index")
	fmt.Println("--------------------------")

	dimension := 128
	numVectors := 200000
	nlist := 256

	// Create GPU resources
	res, err := faiss.NewStandardGpuResources()
	if err != nil {
		log.Fatalf("Failed to create GPU resources: %v", err)
	}
	defer res.Free()

	// Create quantizer (flat index for cluster centers)
	quantizer, err := faiss.NewGpuIndexFlatL2(res, dimension, 0)
	if err != nil {
		log.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	// Create IVF index
	index, err := faiss.NewGpuIndexIVFFlat(res, quantizer, dimension, nlist, 0, faiss.MetricL2)
	if err != nil {
		log.Fatalf("Failed to create GPU IVF index: %v", err)
	}
	defer index.Close()

	fmt.Printf("Created GPU IVFFlat with %d clusters\n", nlist)

	// Generate data
	vectors := generateRandomVectors(numVectors, dimension)

	// Train - this runs K-means clustering on GPU (very fast!)
	trainSize := nlist * 100
	fmt.Printf("\nTraining on %d vectors...\n", trainSize)
	start := time.Now()
	if err := index.Train(vectors[:trainSize*dimension]); err != nil {
		log.Fatalf("Training failed: %v", err)
	}
	fmt.Printf("  Training took: %v\n", time.Since(start))

	// Add vectors
	start = time.Now()
	if err := index.Add(vectors); err != nil {
		log.Fatalf("Add failed: %v", err)
	}
	fmt.Printf("  Added %d vectors in %v\n", index.Ntotal(), time.Since(start))

	// Search with different nprobe values
	numQueries := 1000
	queries := generateRandomVectors(numQueries, dimension)
	k := 10

	fmt.Printf("\nSearch performance with different nprobe:\n")
	fmt.Printf("%-10s %-15s %-15s\n", "nprobe", "Time", "Queries/sec")
	fmt.Printf("%-10s %-15s %-15s\n", "------", "----", "-----------")

	for _, nprobe := range []int{1, 8, 32, 64} {
		index.SetNprobe(nprobe)

		start = time.Now()
		index.Search(queries, k)
		elapsed := time.Since(start)

		fmt.Printf("%-10d %-15v %-15.0f\n",
			nprobe, elapsed, float64(numQueries)/elapsed.Seconds())
	}

	fmt.Println("\nGPU IVF provides the best of both worlds:")
	fmt.Println("  - IVF clustering for sub-linear search complexity")
	fmt.Println("  - GPU parallelism for massive throughput")
}

// generateRandomVectors creates random float32 vectors
func generateRandomVectors(n, d int) []float32 {
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = rand.Float32()
	}
	return vectors
}

func speedup(cpu, gpu time.Duration) float64 {
	if gpu == 0 {
		return 0
	}
	return float64(cpu) / float64(gpu)
}

func faster(cpu, gpu time.Duration) string {
	if gpu < cpu {
		return "(GPU faster)"
	}
	return "(CPU faster)"
}
