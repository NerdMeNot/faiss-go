package faiss

import (
	"math"
	"math/rand"
)

// Test helper functions shared across internal test files in the faiss package.
// These are inlined here to avoid import cycles with test/helpers package.

// generateVectors creates random test vectors
func generateVectors(n, d int) []float32 {
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = rand.Float32()
	}
	return vectors
}

// almostEqual checks if two float32 values are approximately equal
func almostEqual(a, b, tolerance float32) bool {
	return math.Abs(float64(a-b)) < float64(tolerance)
}
