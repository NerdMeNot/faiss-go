package helpers

import (
	"math"
	"math/rand"
)

// GenerateVectors creates random float32 vectors for testing
func GenerateVectors(n, d int) []float32 {
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = rand.Float32()
	}
	return vectors
}

// GenerateBinaryVectors creates random binary vectors for testing
func GenerateBinaryVectors(n, d int) []uint8 {
	bytesPerVec := d / 8
	vectors := make([]uint8, n*bytesPerVec)
	for i := range vectors {
		vectors[i] = uint8(rand.Intn(256))
	}
	return vectors
}

// AlmostEqual checks if two float32 values are approximately equal within a tolerance
func AlmostEqual(a, b float32, tolerance float32) bool {
	return math.Abs(float64(a-b)) < float64(tolerance)
}
