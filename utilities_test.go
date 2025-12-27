package faiss

import (
	"math"
	"testing"
)

// ========================================
// K-Selection Tests
// ========================================

func TestKMin(t *testing.T) {
	vals := []float32{5.0, 2.0, 8.0, 1.0, 9.0, 3.0}
	k := 3

	minVals, minIdx := KMin(vals, k)

	if len(minVals) != k || len(minIdx) != k {
		t.Errorf("Expected %d results, got %d values and %d indices", k, len(minVals), len(minIdx))
	}

	// Check values are sorted ascending
	if minVals[0] != 1.0 || minVals[1] != 2.0 || minVals[2] != 3.0 {
		t.Errorf("Expected [1.0, 2.0, 3.0], got %v", minVals)
	}

	// Check indices
	if minIdx[0] != 3 || minIdx[1] != 1 || minIdx[2] != 5 {
		t.Errorf("Expected indices [3, 1, 5], got %v", minIdx)
	}
}

func TestKMax(t *testing.T) {
	vals := []float32{5.0, 2.0, 8.0, 1.0, 9.0, 3.0}
	k := 3

	maxVals, maxIdx := KMax(vals, k)

	if len(maxVals) != k || len(maxIdx) != k {
		t.Errorf("Expected %d results, got %d values and %d indices", k, len(maxVals), len(maxIdx))
	}

	// Check values are sorted descending
	if maxVals[0] != 9.0 || maxVals[1] != 8.0 || maxVals[2] != 5.0 {
		t.Errorf("Expected [9.0, 8.0, 5.0], got %v", maxVals)
	}

	// Check indices
	if maxIdx[0] != 4 || maxIdx[1] != 2 || maxIdx[2] != 0 {
		t.Errorf("Expected indices [4, 2, 0], got %v", maxIdx)
	}
}

func TestKMinKMaxEdgeCases(t *testing.T) {
	// Empty array
	vals := []float32{}
	minVals, minIdx := KMin(vals, 5)
	if len(minVals) != 0 || len(minIdx) != 0 {
		t.Error("Expected empty results for empty input")
	}

	// k larger than array
	vals = []float32{1.0, 2.0, 3.0}
	minVals, minIdx = KMin(vals, 10)
	if len(minVals) != 3 || len(minIdx) != 3 {
		t.Error("Should return all elements when k > len")
	}

	// k = 0
	minVals, minIdx = KMin(vals, 0)
	if len(minVals) != 0 || len(minIdx) != 0 {
		t.Error("Expected empty results for k=0")
	}
}

// ========================================
// Random Number Generation Tests
// ========================================

func TestRandUniform(t *testing.T) {
	n := 1000
	vals := RandUniform(n)

	if len(vals) != n {
		t.Errorf("Expected %d values, got %d", n, len(vals))
	}

	// Check values are in [0, 1)
	for i, v := range vals {
		if v < 0 || v >= 1 {
			t.Errorf("Value %d out of range [0,1): %f", i, v)
		}
	}

	// Check some randomness (not all same)
	allSame := true
	first := vals[0]
	for _, v := range vals {
		if v != first {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("All random values are the same")
	}
}

func TestRandNormal(t *testing.T) {
	n := 10000
	vals := RandNormal(n)

	if len(vals) != n {
		t.Errorf("Expected %d values, got %d", n, len(vals))
	}

	// Compute mean (should be close to 0)
	sum := float32(0)
	for _, v := range vals {
		sum += v
	}
	mean := sum / float32(n)

	if math.Abs(float64(mean)) > 0.1 {
		t.Errorf("Mean should be close to 0, got %f", mean)
	}

	// Compute variance (should be close to 1)
	varSum := float32(0)
	for _, v := range vals {
		diff := v - mean
		varSum += diff * diff
	}
	variance := varSum / float32(n)

	if math.Abs(float64(variance)-1.0) > 0.1 {
		t.Errorf("Variance should be close to 1, got %f", variance)
	}
}

func TestRandSeed(t *testing.T) {
	RandSeed(42)
	vals1 := RandUniform(100)

	RandSeed(42)
	vals2 := RandUniform(100)

	// With same seed, should get same results
	for i := range vals1 {
		if vals1[i] != vals2[i] {
			t.Error("Same seed should produce same random sequence")
			break
		}
	}
}

// ========================================
// Vector Utilities Tests
// ========================================

func TestFvec2Bvec(t *testing.T) {
	fvec := []float32{-1.0, 0.5, -0.3, 1.2, 0.0, -0.1, 0.8, 1.0}
	bvec := Fvec2Bvec(fvec)

	// 8 floats = 1 byte
	if len(bvec) != 1 {
		t.Errorf("Expected 1 byte, got %d", len(bvec))
	}

	// Expected bits: 0b01001011 = 75
	// Position 0: -1.0 < 0 -> 0
	// Position 1: 0.5 > 0 -> 1
	// Position 2: -0.3 < 0 -> 0
	// Position 3: 1.2 > 0 -> 1
	// Position 4: 0.0 = 0 -> 0
	// Position 5: -0.1 < 0 -> 0
	// Position 6: 0.8 > 0 -> 1
	// Position 7: 1.0 > 0 -> 1
	expected := uint8(0b11001010) // Reading right to left
	if bvec[0] != expected {
		t.Errorf("Expected %08b, got %08b", expected, bvec[0])
	}
}

func TestBitstringHammingDistance(t *testing.T) {
	a := []uint8{0b10101010}
	b := []uint8{0b11001100}

	dist := BitstringHammingDistance(a, b)

	// XOR: 0b01100110 has 4 bits set
	expected := 4
	if dist != expected {
		t.Errorf("Expected Hamming distance %d, got %d", expected, dist)
	}
}

func TestBitstringHammingDistanceMultipleBytes(t *testing.T) {
	a := []uint8{0xFF, 0x00, 0xAA}
	b := []uint8{0x00, 0xFF, 0xAA}

	dist := BitstringHammingDistance(a, b)

	// First byte: 8 different bits
	// Second byte: 8 different bits
	// Third byte: 0 different bits
	expected := 16
	if dist != expected {
		t.Errorf("Expected Hamming distance %d, got %d", expected, dist)
	}
}

func TestBitstringHammingDistanceLengthMismatch(t *testing.T) {
	a := []uint8{0xFF}
	b := []uint8{0xFF, 0x00}

	dist := BitstringHammingDistance(a, b)
	if dist != -1 {
		t.Errorf("Expected -1 for length mismatch, got %d", dist)
	}
}

// ========================================
// Distance Computation Tests
// ========================================

func TestL2Distance(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0}
	b := []float32{4.0, 5.0, 6.0}

	dist, err := L2Distance(a, b)
	if err != nil {
		t.Fatalf("L2Distance failed: %v", err)
	}

	// sqrt((3^2 + 3^2 + 3^2)) = sqrt(27) ≈ 5.196
	expected := float32(math.Sqrt(27))
	if !almostEqual(dist, expected, 0.001) {
		t.Errorf("Expected distance %f, got %f", expected, dist)
	}
}

func TestInnerProduct(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0}
	b := []float32{4.0, 5.0, 6.0}

	ip, err := InnerProduct(a, b)
	if err != nil {
		t.Fatalf("InnerProduct failed: %v", err)
	}

	// 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
	expected := float32(32.0)
	if ip != expected {
		t.Errorf("Expected inner product %f, got %f", expected, ip)
	}
}

func TestCosineSimilarity(t *testing.T) {
	// Perpendicular vectors
	a := []float32{1.0, 0.0}
	b := []float32{0.0, 1.0}

	sim, err := CosineSimilarity(a, b)
	if err != nil {
		t.Fatalf("CosineSimilarity failed: %v", err)
	}

	if !almostEqual(sim, 0.0, 0.001) {
		t.Errorf("Expected similarity 0.0 (perpendicular), got %f", sim)
	}

	// Identical vectors
	a = []float32{1.0, 1.0}
	b = []float32{1.0, 1.0}
	sim, err = CosineSimilarity(a, b)
	if err != nil {
		t.Fatalf("CosineSimilarity failed: %v", err)
	}

	if !almostEqual(sim, 1.0, 0.001) {
		t.Errorf("Expected similarity 1.0 (identical), got %f", sim)
	}

	// Opposite vectors
	a = []float32{1.0, 0.0}
	b = []float32{-1.0, 0.0}
	sim, err = CosineSimilarity(a, b)
	if err != nil {
		t.Fatalf("CosineSimilarity failed: %v", err)
	}

	if !almostEqual(sim, -1.0, 0.001) {
		t.Errorf("Expected similarity -1.0 (opposite), got %f", sim)
	}
}

func TestDistanceMismatchedLengths(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0}
	b := []float32{4.0, 5.0}

	_, err := L2Distance(a, b)
	if err == nil {
		t.Error("Expected error for mismatched lengths")
	}

	_, err = InnerProduct(a, b)
	if err == nil {
		t.Error("Expected error for mismatched lengths")
	}

	_, err = CosineSimilarity(a, b)
	if err == nil {
		t.Error("Expected error for mismatched lengths")
	}
}

// ========================================
// Batch Operations Tests
// ========================================

func TestBatchL2Distance(t *testing.T) {
	d := 4
	queries := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0} // 2 queries
	database := []float32{1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0} // 3 database vectors

	distances, err := BatchL2Distance(queries, database, d)
	if err != nil {
		t.Fatalf("BatchL2Distance failed: %v", err)
	}

	// Should return 2x3 matrix = 6 distances
	if len(distances) != 6 {
		t.Errorf("Expected 6 distances, got %d", len(distances))
	}
}

func TestBatchInnerProduct(t *testing.T) {
	d := 3
	queries := []float32{1.0, 0.0, 0.0, 0.0, 1.0, 0.0} // 2 queries
	database := []float32{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0} // 3 database vectors

	products, err := BatchInnerProduct(queries, database, d)
	if err != nil {
		t.Fatalf("BatchInnerProduct failed: %v", err)
	}

	// Should return 2x3 matrix = 6 products
	if len(products) != 6 {
		t.Errorf("Expected 6 products, got %d", len(products))
	}

	// First query [1,0,0] · first database [1,0,0] = 1
	if products[0] != 1.0 {
		t.Errorf("Expected product 1.0, got %f", products[0])
	}

	// First query [1,0,0] · second database [0,1,0] = 0
	if products[1] != 0.0 {
		t.Errorf("Expected product 0.0, got %f", products[1])
	}
}

// ========================================
// Index Utilities Tests
// ========================================

func TestGetIndexDescription(t *testing.T) {
	index, _ := NewIndexFlatL2(64)
	defer index.Close()

	desc := GetIndexDescription(index)
	if desc == "" {
		t.Error("Expected non-empty description")
	}
}

func TestIsIndexTrained(t *testing.T) {
	flatIndex, _ := NewIndexFlatL2(64)
	defer flatIndex.Close()

	if !IsIndexTrained(flatIndex) {
		t.Error("Flat index should be trained")
	}

	quantizer, _ := NewIndexFlatL2(64)
	defer quantizer.Close()

	ivfIndex, _ := NewIndexIVFFlat(quantizer, 64, 10, MetricL2)
	defer ivfIndex.Close()

	if IsIndexTrained(ivfIndex) {
		t.Error("IVF index should not be trained initially")
	}
}

func TestGetIndexSize(t *testing.T) {
	d := 64
	nb := 1000

	index, _ := NewIndexFlatL2(d)
	defer index.Close()

	vectors := generateVectors(nb, d)
	index.Add(vectors)

	size := GetIndexSize(index)

	// Flat index: d * ntotal * 4 bytes
	expectedSize := int64(d) * int64(nb) * 4
	if size != expectedSize {
		t.Errorf("Expected size %d, got %d", expectedSize, size)
	}
}
