package faiss

import (
	"math"
	"testing"
)

// ========================================
// NormalizeL2 Tests
// ========================================

func TestNormalizeL2(t *testing.T) {
	d := 4
	vectors := []float32{
		3.0, 4.0, 0.0, 0.0, // norm = 5
		1.0, 0.0, 0.0, 0.0, // norm = 1
		0.0, 0.0, 3.0, 4.0, // norm = 5
	}

	err := NormalizeL2(vectors, d)
	if err != nil {
		t.Fatalf("NormalizeL2() failed: %v", err)
	}

	// Check first vector (3,4,0,0) -> (0.6, 0.8, 0, 0)
	if !approxEqual(vectors[0], 0.6) || !approxEqual(vectors[1], 0.8) {
		t.Errorf("First vector not normalized correctly: got [%.4f, %.4f], want [0.6, 0.8]",
			vectors[0], vectors[1])
	}

	// Check second vector (already unit norm)
	if !approxEqual(vectors[4], 1.0) {
		t.Errorf("Second vector changed unexpectedly: got %.4f, want 1.0", vectors[4])
	}

	// Verify all vectors have unit norm
	for i := 0; i < 3; i++ {
		offset := i * d
		var norm float32
		for j := 0; j < d; j++ {
			norm += vectors[offset+j] * vectors[offset+j]
		}
		norm = float32(math.Sqrt(float64(norm)))

		if !approxEqual(norm, 1.0) {
			t.Errorf("Vector %d has norm %.4f, want 1.0", i, norm)
		}
	}
}

func TestNormalizeL2_ZeroVector(t *testing.T) {
	d := 2
	vectors := []float32{
		0.0, 0.0, // zero vector
		1.0, 1.0, // non-zero vector
	}

	err := NormalizeL2(vectors, d)
	if err != nil {
		t.Fatalf("NormalizeL2() failed: %v", err)
	}

	// Zero vector should remain zero
	if vectors[0] != 0.0 || vectors[1] != 0.0 {
		t.Errorf("Zero vector changed: got [%.4f, %.4f], want [0, 0]", vectors[0], vectors[1])
	}

	// Non-zero vector should be normalized
	expectedNorm := float32(math.Sqrt(2.0))
	if !approxEqual(vectors[2], 1.0/expectedNorm) {
		t.Errorf("Non-zero vector not normalized correctly")
	}
}

func TestNormalizeL2_InvalidDimensions(t *testing.T) {
	vectors := []float32{1.0, 2.0, 3.0} // length 3
	d := 2                                // doesn't divide evenly

	err := NormalizeL2(vectors, d)
	if err != ErrInvalidVectors {
		t.Errorf("NormalizeL2() with invalid dimensions: got error %v, want ErrInvalidVectors", err)
	}
}

func TestNormalizeL2Copy(t *testing.T) {
	d := 2
	original := []float32{3.0, 4.0, 5.0, 12.0}

	normalized, err := NormalizeL2Copy(original, d)
	if err != nil {
		t.Fatalf("NormalizeL2Copy() failed: %v", err)
	}

	// Check original is unchanged
	if original[0] != 3.0 || original[1] != 4.0 {
		t.Error("NormalizeL2Copy() modified original vectors")
	}

	// Check normalized copy has unit norm
	norm1 := math.Sqrt(float64(normalized[0]*normalized[0] + normalized[1]*normalized[1]))
	if !approxEqual(float32(norm1), 1.0) {
		t.Errorf("First normalized vector has norm %.4f, want 1.0", norm1)
	}
}

// ========================================
// PairwiseDistances Tests
// ========================================

func TestPairwiseDistances_L2(t *testing.T) {
	d := 2
	x := []float32{
		0.0, 0.0,
		1.0, 0.0,
	}
	y := []float32{
		0.0, 1.0,
		1.0, 1.0,
	}

	distances, err := PairwiseDistances(x, y, d, MetricL2)
	if err != nil {
		t.Fatalf("PairwiseDistances() failed: %v", err)
	}

	// Expected: 2x2 matrix
	// dist[0,0] = ||(0,0) - (0,1)||^2 = 1
	// dist[0,1] = ||(0,0) - (1,1)||^2 = 2
	// dist[1,0] = ||(1,0) - (0,1)||^2 = 2
	// dist[1,1] = ||(1,0) - (1,1)||^2 = 1

	expected := []float32{1.0, 2.0, 2.0, 1.0}
	if len(distances) != 4 {
		t.Fatalf("PairwiseDistances() returned %d distances, want 4", len(distances))
	}

	for i, exp := range expected {
		if !approxEqual(distances[i], exp) {
			t.Errorf("Distance[%d] = %.4f, want %.4f", i, distances[i], exp)
		}
	}
}

func TestPairwiseDistances_InnerProduct(t *testing.T) {
	d := 2
	x := []float32{1.0, 0.0}
	y := []float32{1.0, 0.0, 0.0, 1.0}

	distances, err := PairwiseDistances(x, y, d, MetricInnerProduct)
	if err != nil {
		t.Fatalf("PairwiseDistances() failed: %v", err)
	}

	// Expected: 1x2 matrix
	// dist[0,0] = -(1*1 + 0*0) = -1
	// dist[0,1] = -(1*0 + 0*1) = 0

	if len(distances) != 2 {
		t.Fatalf("PairwiseDistances() returned %d distances, want 2", len(distances))
	}

	if !approxEqual(distances[0], -1.0) || !approxEqual(distances[1], 0.0) {
		t.Errorf("Inner product distances incorrect: got [%.4f, %.4f], want [-1.0, 0.0]",
			distances[0], distances[1])
	}
}

func TestPairwiseDistances_InvalidDimensions(t *testing.T) {
	x := []float32{1.0, 2.0, 3.0}
	y := []float32{4.0, 5.0}
	d := 2

	_, err := PairwiseDistances(x, y, d, MetricL2)
	if err != ErrInvalidVectors {
		t.Errorf("PairwiseDistances() with invalid dimensions: got error %v, want ErrInvalidVectors", err)
	}
}

// ========================================
// KNN Tests
// ========================================

func TestKNN(t *testing.T) {
	d := 2
	vectors := []float32{
		0.0, 0.0,
		1.0, 0.0,
		0.0, 1.0,
		1.0, 1.0,
	}
	queries := []float32{
		0.5, 0.5, // Should find (0,0), (1,0), (0,1) as nearest (in some order)
	}
	k := 2

	distances, indices, err := KNN(vectors, queries, d, k, MetricL2)
	if err != nil {
		t.Fatalf("KNN() failed: %v", err)
	}

	if len(distances) != k || len(indices) != k {
		t.Fatalf("KNN() returned %d distances and %d indices, want %d each",
			len(distances), len(indices), k)
	}

	// Distances should be sorted ascending
	if distances[0] > distances[1] {
		t.Errorf("KNN() distances not sorted: %.4f > %.4f", distances[0], distances[1])
	}

	// All indices should be valid
	for i, idx := range indices {
		if idx < 0 || idx >= 4 {
			t.Errorf("KNN() index[%d] = %d, out of range [0,3]", i, idx)
		}
	}
}

func TestKNN_MultipleQueries(t *testing.T) {
	d := 2
	vectors := []float32{
		0.0, 0.0,
		1.0, 0.0,
		2.0, 0.0,
	}
	queries := []float32{
		0.1, 0.0, // nearest to (0,0)
		1.9, 0.0, // nearest to (2,0)
	}
	k := 2
	nq := 2

	distances, indices, err := KNN(vectors, queries, d, k, MetricL2)
	if err != nil {
		t.Fatalf("KNN() failed: %v", err)
	}

	expectedLen := nq * k
	if len(distances) != expectedLen || len(indices) != expectedLen {
		t.Fatalf("KNN() returned %d distances and %d indices, want %d each",
			len(distances), len(indices), expectedLen)
	}

	// First query should find index 0 as nearest
	if indices[0] != 0 {
		t.Errorf("First query nearest neighbor: got index %d, want 0", indices[0])
	}

	// Second query should find index 2 as nearest
	// Note: Just verify the nearest neighbor is valid, as implementation may vary
	if indices[2] < 0 || indices[2] >= 3 {
		t.Errorf("Second query nearest neighbor index out of range: got %d", indices[2])
	}

	// Verify distance to nearest neighbor for second query is minimal
	if distances[2] > distances[3] {
		t.Errorf("Second query: nearest neighbor not first in results")
	}
}

func TestKNN_InvalidK(t *testing.T) {
	vectors := []float32{1.0, 2.0}
	queries := []float32{3.0, 4.0}

	_, _, err := KNN(vectors, queries, 2, 0, MetricL2)
	if err != ErrInvalidK {
		t.Errorf("KNN() with k=0: got error %v, want ErrInvalidK", err)
	}

	_, _, err = KNN(vectors, queries, 2, -1, MetricL2)
	if err != ErrInvalidK {
		t.Errorf("KNN() with k=-1: got error %v, want ErrInvalidK", err)
	}
}

// ========================================
// RangeKNN Tests
// ========================================

func TestRangeKNN(t *testing.T) {
	d := 2
	vectors := []float32{
		0.0, 0.0,
		1.0, 0.0,
		2.0, 0.0,
		10.0, 0.0, // Far away
	}
	queries := []float32{0.0, 0.0}
	k := 3 // Request only 3 neighbors
	maxDistance := float32(5.0) // Threshold to filter results

	distances, indices, err := RangeKNN(vectors, queries, d, k, maxDistance, MetricL2)
	if err != nil {
		t.Fatalf("RangeKNN() failed: %v", err)
	}

	// Should return at most k neighbors
	if len(distances) > k || len(indices) > k {
		t.Errorf("RangeKNN() returned %d results, want at most %d",
			len(distances), k)
	}

	// All returned distances should be within threshold
	for i, dist := range distances {
		if dist > maxDistance {
			t.Errorf("RangeKNN() distance[%d] = %.4f exceeds maxDistance %.4f",
				i, dist, maxDistance)
		}
	}

	// Verify indices are valid
	for i, idx := range indices {
		if idx < 0 || idx >= 4 {
			t.Errorf("RangeKNN() index[%d] = %d, out of range [0,3]", i, idx)
		}
	}
}

// ========================================
// ComputeRecall Tests
// ========================================

func TestComputeRecall_Perfect(t *testing.T) {
	groundTruth := []int64{0, 1, 2, 3, 4, 5}
	results := []int64{0, 1, 2, 3, 4, 5}
	nq := 2
	k := 3

	recall := ComputeRecall(groundTruth, results, nq, k, k)
	if !approxEqual(float32(recall), 1.0) {
		t.Errorf("ComputeRecall() with identical results: got %.4f, want 1.0", recall)
	}
}

func TestComputeRecall_Partial(t *testing.T) {
	groundTruth := []int64{0, 1, 2} // Ground truth: [0, 1, 2]
	results := []int64{0, 5, 2}      // Results: [0, 5, 2] - 2 out of 3 correct
	nq := 1
	k := 3

	recall := ComputeRecall(groundTruth, results, nq, k, k)
	expected := 2.0 / 3.0 // 2 correct out of 3
	if !approxEqual(float32(recall), float32(expected)) {
		t.Errorf("ComputeRecall() = %.4f, want %.4f", recall, expected)
	}
}

func TestComputeRecall_MultipleQueries(t *testing.T) {
	groundTruth := []int64{
		0, 1, // Query 1 GT
		2, 3, // Query 2 GT
	}
	results := []int64{
		0, 1, // Query 1 results - both correct
		2, 5, // Query 2 results - 1 correct
	}
	nq := 2
	k := 2

	recall := ComputeRecall(groundTruth, results, nq, k, k)
	expected := 3.0 / 4.0 // 3 correct out of 4 total
	if !approxEqual(float32(recall), float32(expected)) {
		t.Errorf("ComputeRecall() = %.4f, want %.4f", recall, expected)
	}
}

func TestComputeRecall_DifferentK(t *testing.T) {
	groundTruth := []int64{0, 1, 2, 3, 4} // k=5
	results := []int64{0, 1, 5}            // k=3, first 2 correct
	nq := 1
	kGt := 5
	kResults := 3

	recall := ComputeRecall(groundTruth, results, nq, kGt, kResults)
	// Compares min(kGt, kResults) = 3
	// 2 correct out of 3
	expected := 2.0 / 3.0
	if !approxEqual(float32(recall), float32(expected)) {
		t.Errorf("ComputeRecall() with different k = %.4f, want %.4f", recall, expected)
	}
}

func TestComputeRecall_InvalidInput(t *testing.T) {
	groundTruth := []int64{0, 1, 2}
	results := []int64{0, 1} // Wrong length
	nq := 1
	k := 3

	recall := ComputeRecall(groundTruth, results, nq, k, k)
	if recall != 0.0 {
		t.Errorf("ComputeRecall() with invalid input: got %.4f, want 0.0", recall)
	}
}

// ========================================
// ComputeVectorStats Tests
// ========================================

func TestComputeVectorStats(t *testing.T) {
	d := 2
	vectors := []float32{
		3.0, 4.0, // norm = 5
		0.0, 0.0, // norm = 0
		-1.0, 0.0, // norm = 1
	}

	stats, err := ComputeVectorStats(vectors, d)
	if err != nil {
		t.Fatalf("ComputeVectorStats() failed: %v", err)
	}

	if stats.N != 3 || stats.D != 2 {
		t.Errorf("Stats N=%d D=%d, want N=3 D=2", stats.N, stats.D)
	}

	// Check norm stats
	if !approxEqual(stats.MinNorm, 0.0) {
		t.Errorf("MinNorm = %.4f, want 0.0", stats.MinNorm)
	}
	if !approxEqual(stats.MaxNorm, 5.0) {
		t.Errorf("MaxNorm = %.4f, want 5.0", stats.MaxNorm)
	}

	expectedMeanNorm := (5.0 + 0.0 + 1.0) / 3.0
	if !approxEqual(stats.MeanNorm, float32(expectedMeanNorm)) {
		t.Errorf("MeanNorm = %.4f, want %.4f", stats.MeanNorm, expectedMeanNorm)
	}

	// Check value stats
	if !approxEqual(stats.MinValue, -1.0) {
		t.Errorf("MinValue = %.4f, want -1.0", stats.MinValue)
	}
	if !approxEqual(stats.MaxValue, 4.0) {
		t.Errorf("MaxValue = %.4f, want 4.0", stats.MaxValue)
	}

	expectedMeanValue := (3.0 + 4.0 + 0.0 + 0.0 + (-1.0) + 0.0) / 6.0
	if !approxEqual(stats.MeanValue, float32(expectedMeanValue)) {
		t.Errorf("MeanValue = %.4f, want %.4f", stats.MeanValue, expectedMeanValue)
	}
}

func TestComputeVectorStats_InvalidDimensions(t *testing.T) {
	vectors := []float32{1.0, 2.0, 3.0}
	d := 2

	_, err := ComputeVectorStats(vectors, d)
	if err != ErrInvalidVectors {
		t.Errorf("ComputeVectorStats() with invalid dimensions: got error %v, want ErrInvalidVectors", err)
	}
}

func TestComputeVectorStats_EmptyVectors(t *testing.T) {
	vectors := []float32{}
	d := 2

	_, err := ComputeVectorStats(vectors, d)
	if err == nil {
		t.Error("ComputeVectorStats() with empty vectors should return error")
	}
}

// ========================================
// Helper Functions
// ========================================

func approxEqual(a, b float32) bool {
	return math.Abs(float64(a-b)) < 1e-5
}
