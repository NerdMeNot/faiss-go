package faiss

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// ========================================
// K-Selection Utilities
// ========================================

// KMin finds the k smallest values and their indices
//
// Python equivalent: faiss.kmin
//
// Example:
//   vals := []float32{3.0, 1.0, 4.0, 1.0, 5.0}
//   minVals, minIdx := faiss.KMin(vals, 3)
//   // minVals = [1.0, 1.0, 3.0]
//   // minIdx = [1, 3, 0]
func KMin(vals []float32, k int) ([]float32, []int64) {
	if k <= 0 || len(vals) == 0 {
		return []float32{}, []int64{}
	}
	if k > len(vals) {
		k = len(vals)
	}

	// Create index array
	type pair struct {
		val float32
		idx int64
	}
	pairs := make([]pair, len(vals))
	for i, v := range vals {
		pairs[i] = pair{v, int64(i)}
	}

	// Sort by value (ascending)
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].val < pairs[j].val
	})

	// Extract top k
	minVals := make([]float32, k)
	minIdx := make([]int64, k)
	for i := 0; i < k; i++ {
		minVals[i] = pairs[i].val
		minIdx[i] = pairs[i].idx
	}

	return minVals, minIdx
}

// KMax finds the k largest values and their indices
//
// Python equivalent: faiss.kmax
//
// Example:
//   vals := []float32{3.0, 1.0, 4.0, 1.0, 5.0}
//   maxVals, maxIdx := faiss.KMax(vals, 3)
//   // maxVals = [5.0, 4.0, 3.0]
//   // maxIdx = [4, 2, 0]
func KMax(vals []float32, k int) ([]float32, []int64) {
	if k <= 0 || len(vals) == 0 {
		return []float32{}, []int64{}
	}
	if k > len(vals) {
		k = len(vals)
	}

	// Create index array
	type pair struct {
		val float32
		idx int64
	}
	pairs := make([]pair, len(vals))
	for i, v := range vals {
		pairs[i] = pair{v, int64(i)}
	}

	// Sort by value (descending)
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].val > pairs[j].val
	})

	// Extract top k
	maxVals := make([]float32, k)
	maxIdx := make([]int64, k)
	for i := 0; i < k; i++ {
		maxVals[i] = pairs[i].val
		maxIdx[i] = pairs[i].idx
	}

	return maxVals, maxIdx
}

// ========================================
// Random Number Generation
// ========================================

// RandUniform generates n random floats uniformly in [0, 1)
//
// Python equivalent: faiss.rand
//
// Example:
//   vals := faiss.RandUniform(1000)
func RandUniform(n int) []float32 {
	if n <= 0 {
		return []float32{}
	}

	vals := make([]float32, n)
	for i := 0; i < n; i++ {
		vals[i] = rand.Float32()
	}
	return vals
}

// RandNormal generates n random floats from standard normal distribution N(0,1)
//
// Python equivalent: faiss.randn
//
// Example:
//   vals := faiss.RandNormal(1000)
func RandNormal(n int) []float32 {
	if n <= 0 {
		return []float32{}
	}

	vals := make([]float32, n)
	for i := 0; i < n; i++ {
		vals[i] = float32(rand.NormFloat64())
	}
	return vals
}

// RandSeed sets the random seed for reproducibility
//
// Example:
//   faiss.RandSeed(42)  // Reproducible results
func RandSeed(seed int64) {
	rand.Seed(seed)
}

// ========================================
// Vector Utilities
// ========================================

// Fvec2Bvec converts float vectors to binary vectors by thresholding at 0
//
// Python equivalent: faiss.fvec2bvec
//
// Example:
//   fvec := []float32{-1.0, 0.5, -0.3, 1.2}  // 4 values
//   bvec := faiss.Fvec2Bvec(fvec)             // [false, true, false, true] -> 0b1010 = 10
func Fvec2Bvec(fvec []float32) []uint8 {
	if len(fvec) == 0 {
		return []uint8{}
	}

	// Number of bytes needed (8 bits per byte)
	nbytes := (len(fvec) + 7) / 8
	bvec := make([]uint8, nbytes)

	for i, val := range fvec {
		if val > 0 {
			byteIdx := i / 8
			bitIdx := uint(i % 8)
			bvec[byteIdx] |= (1 << bitIdx)
		}
	}

	return bvec
}

// BitstringHammingDistance computes Hamming distance between two binary strings
//
// Python equivalent: faiss.hamming
//
// Example:
//   a := []uint8{0b10101010}
//   b := []uint8{0b11001100}
//   dist := faiss.BitstringHammingDistance(a, b)  // 4
func BitstringHammingDistance(a, b []uint8) int {
	if len(a) != len(b) {
		return -1
	}

	dist := 0
	for i := range a {
		// XOR to find differing bits, then count them
		xor := a[i] ^ b[i]
		dist += popcount(xor)
	}
	return dist
}

// popcount counts the number of 1 bits in a byte
func popcount(x uint8) int {
	count := 0
	for x != 0 {
		count++
		x &= x - 1 // Clear least significant 1 bit
	}
	return count
}

// ========================================
// Distance Computations
// ========================================

// L2Distance computes L2 (Euclidean) distance between two vectors
//
// Example:
//   a := []float32{1.0, 2.0, 3.0}
//   b := []float32{4.0, 5.0, 6.0}
//   dist := faiss.L2Distance(a, b)  // sqrt(27) ≈ 5.196
func L2Distance(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vectors must have same length")
	}

	sum := float32(0)
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum))), nil
}

// InnerProduct computes inner product between two vectors
//
// Example:
//   a := []float32{1.0, 2.0, 3.0}
//   b := []float32{4.0, 5.0, 6.0}
//   ip := faiss.InnerProduct(a, b)  // 32.0
func InnerProduct(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vectors must have same length")
	}

	sum := float32(0)
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum, nil
}

// CosineSimilarity computes cosine similarity between two vectors
// Returns value in [-1, 1] where 1 = identical direction, -1 = opposite
//
// Example:
//   a := []float32{1.0, 0.0}
//   b := []float32{0.0, 1.0}
//   sim := faiss.CosineSimilarity(a, b)  // 0.0 (perpendicular)
func CosineSimilarity(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vectors must have same length")
	}

	dotProduct := float32(0)
	normA := float32(0)
	normB := float32(0)

	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0, fmt.Errorf("cannot compute cosine similarity with zero vector")
	}

	return dotProduct / float32(math.Sqrt(float64(normA)*float64(normB))), nil
}

// ========================================
// Batch Operations
// ========================================

// BatchL2Distance computes L2 distances for batches of vectors
// queries and database should be flat arrays (n*d and m*d)
//
// Returns n×m matrix of distances
func BatchL2Distance(queries, database []float32, d int) ([]float32, error) {
	if len(queries)%d != 0 || len(database)%d != 0 {
		return nil, fmt.Errorf("vector lengths must be multiple of dimension")
	}

	nq := len(queries) / d
	nb := len(database) / d
	distances := make([]float32, nq*nb)

	for i := 0; i < nq; i++ {
		for j := 0; j < nb; j++ {
			sum := float32(0)
			for k := 0; k < d; k++ {
				diff := queries[i*d+k] - database[j*d+k]
				sum += diff * diff
			}
			distances[i*nb+j] = float32(math.Sqrt(float64(sum)))
		}
	}

	return distances, nil
}

// BatchInnerProduct computes inner products for batches of vectors
// queries and database should be flat arrays (n*d and m*d)
//
// Returns n×m matrix of inner products
func BatchInnerProduct(queries, database []float32, d int) ([]float32, error) {
	if len(queries)%d != 0 || len(database)%d != 0 {
		return nil, fmt.Errorf("vector lengths must be multiple of dimension")
	}

	nq := len(queries) / d
	nb := len(database) / d
	products := make([]float32, nq*nb)

	for i := 0; i < nq; i++ {
		for j := 0; j < nb; j++ {
			sum := float32(0)
			for k := 0; k < d; k++ {
				sum += queries[i*d+k] * database[j*d+k]
			}
			products[i*nb+j] = sum
		}
	}

	return products, nil
}

// ========================================
// Index Utilities
// ========================================

// GetIndexDescription returns a human-readable description of an index
//
// Example:
//   desc := faiss.GetIndexDescription(index)
//   // "IndexFlatL2(d=128, ntotal=10000)"
func GetIndexDescription(index Index) string {
	return fmt.Sprintf("%T(d=%d, ntotal=%d, metric=%s)",
		index, index.D(), index.Ntotal(), index.MetricType())
}

// IsIndexTrained checks if an index is trained
func IsIndexTrained(index Index) bool {
	return index.IsTrained()
}

// GetIndexSize returns the memory footprint estimate in bytes
func GetIndexSize(index Index) int64 {
	d := int64(index.D())
	n := index.Ntotal()

	switch idx := index.(type) {
	case *IndexFlat:
		// d * ntotal * 4 bytes per float32
		return d * n * 4

	case *IndexPQ:
		// Compressed: M * nbits per vector + codebook
		nbytes := int64(idx.M * idx.nbits / 8)
		return n*nbytes + d*256*4 // rough estimate

	case *IndexHNSW:
		// Vectors + HNSW graph
		return d*n*4 + n*int64(idx.M)*8 // rough estimate

	default:
		// Generic fallback
		return d * n * 4
	}
}
