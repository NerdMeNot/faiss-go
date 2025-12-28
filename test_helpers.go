package faiss

import (
	"math"
	"math/rand"
)

// Test helper functions shared across internal test files
// (inlined to avoid import cycles with test/helpers package)

// generateVectors creates random test vectors
func generateVectors(n, d int) []float32 {
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = rand.Float32()
	}
	return vectors
}

// generateBinaryVectors creates random binary test vectors
func generateBinaryVectors(n, d int) []uint8 {
	bytesPerVector := d / 8
	vectors := make([]uint8, n*bytesPerVector)
	for i := range vectors {
		vectors[i] = uint8(rand.Intn(256))
	}
	return vectors
}

// almostEqual checks if two float32 values are approximately equal
func almostEqual(a, b, tolerance float32) bool {
	return math.Abs(float64(a-b)) < float64(tolerance)
}
