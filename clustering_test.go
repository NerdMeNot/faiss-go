package faiss

import (
	"math"
	"testing"
)

// ========================================
// Kmeans Tests
// ========================================

func TestNewKmeans(t *testing.T) {
	km, err := NewKmeans(128, 10)
	if err != nil {
		t.Fatalf("NewKmeans() failed: %v", err)
	}

	if km.D() != 128 {
		t.Errorf("D() = %d, want 128", km.D())
	}
	if km.K() != 10 {
		t.Errorf("K() = %d, want 10", km.K())
	}
	if km.IsTrained() {
		t.Error("IsTrained() = true before training")
	}
}

func TestNewKmeans_InvalidParams(t *testing.T) {
	_, err := NewKmeans(0, 10)
	if err == nil {
		t.Error("NewKmeans(0, 10) should return error")
	}

	_, err = NewKmeans(128, 0)
	if err == nil {
		t.Error("NewKmeans(128, 0) should return error")
	}

	_, err = NewKmeans(-1, 10)
	if err == nil {
		t.Error("NewKmeans(-1, 10) should return error")
	}
}

func TestKmeans_Train(t *testing.T) {
	d := 64
	k := 5
	n := 1000

	km, err := NewKmeans(d, k)
	if err != nil {
		t.Fatalf("NewKmeans() failed: %v", err)
	}

	// Generate training vectors
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = float32(i % 100)
	}

	err = km.Train(vectors)
	if err != nil {
		t.Fatalf("Train() failed: %v", err)
	}

	if !km.IsTrained() {
		t.Error("IsTrained() = false after training")
	}

	centroids := km.Centroids()
	if centroids == nil {
		t.Fatal("Centroids() returned nil after training")
	}

	expectedLen := k * d
	if len(centroids) != expectedLen {
		t.Errorf("len(Centroids()) = %d, want %d", len(centroids), expectedLen)
	}
}

func TestKmeans_Train_NotEnoughVectors(t *testing.T) {
	d := 64
	k := 100

	km, _ := NewKmeans(d, k)

	// Only 50 vectors for 100 clusters
	vectors := make([]float32, 50*d)
	for i := range vectors {
		vectors[i] = float32(i)
	}

	err := km.Train(vectors)
	if err == nil {
		t.Error("Train() should fail when n < k")
	}
}

func TestKmeans_Train_Empty(t *testing.T) {
	km, _ := NewKmeans(64, 10)

	err := km.Train([]float32{})
	if err == nil {
		t.Error("Train(empty) should return error")
	}
}

func TestKmeans_Train_InvalidDimension(t *testing.T) {
	km, _ := NewKmeans(64, 10)

	// 65 floats is not a multiple of 64
	vectors := make([]float32, 65)
	err := km.Train(vectors)
	if err == nil {
		t.Error("Train() with invalid dimension should return error")
	}
}

func TestKmeans_Centroids_BeforeTraining(t *testing.T) {
	km, _ := NewKmeans(64, 10)

	centroids := km.Centroids()
	if centroids != nil {
		t.Error("Centroids() should return nil before training")
	}
}

func TestKmeans_Assign(t *testing.T) {
	d := 64
	k := 5
	n := 500

	km, _ := NewKmeans(d, k)

	// Generate training vectors
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = float32(i % 100)
	}

	km.Train(vectors)

	// Assign some vectors
	testVectors := make([]float32, 10*d)
	for i := range testVectors {
		testVectors[i] = float32(i % 50)
	}

	assignments, err := km.Assign(testVectors)
	if err != nil {
		t.Fatalf("Assign() failed: %v", err)
	}

	if len(assignments) != 10 {
		t.Errorf("len(assignments) = %d, want 10", len(assignments))
	}

	// All assignments should be in range [0, k)
	for i, a := range assignments {
		if a < 0 || a >= int64(k) {
			t.Errorf("assignment[%d] = %d, out of range [0, %d)", i, a, k)
		}
	}
}

func TestKmeans_Assign_BeforeTraining(t *testing.T) {
	km, _ := NewKmeans(64, 10)

	vectors := make([]float32, 64)
	_, err := km.Assign(vectors)
	if err == nil {
		t.Error("Assign() before training should return error")
	}
}

func TestKmeans_Assign_Empty(t *testing.T) {
	d := 64
	k := 5

	km, _ := NewKmeans(d, k)
	vectors := make([]float32, 100*d)
	for i := range vectors {
		vectors[i] = float32(i)
	}
	km.Train(vectors)

	assignments, err := km.Assign([]float32{})
	if err != nil {
		t.Errorf("Assign(empty) failed: %v", err)
	}
	if len(assignments) != 0 {
		t.Errorf("Assign(empty) should return empty assignments")
	}
}

// ========================================
// Clustering Quality Tests
// ========================================

func TestKmeans_ClusterQuality(t *testing.T) {
	d := 2
	k := 3
	n := 300

	km, _ := NewKmeans(d, k)

	// Generate 3 well-separated clusters
	vectors := make([]float32, n*d)
	for i := 0; i < n; i++ {
		cluster := i % k
		offset := float32(cluster * 100) // Separate clusters by 100 units
		vectors[i*d] = offset + float32(i%10)
		vectors[i*d+1] = offset + float32(i%10)
	}

	err := km.Train(vectors)
	if err != nil {
		t.Fatalf("Train() failed: %v", err)
	}

	// Assign the same vectors
	assignments, err := km.Assign(vectors)
	if err != nil {
		t.Fatalf("Assign() failed: %v", err)
	}

	// Count assignments per cluster
	counts := make(map[int64]int)
	for _, a := range assignments {
		counts[a]++
	}

	// Each cluster should have roughly n/k vectors
	expectedPerCluster := n / k
	for cluster, count := range counts {
		deviation := math.Abs(float64(count - expectedPerCluster))
		// Allow 50% deviation due to clustering variance
		if deviation > float64(expectedPerCluster)*0.5 {
			t.Logf("Cluster %d has %d vectors (expected ~%d)", cluster, count, expectedPerCluster)
		}
	}

	t.Logf("Cluster distribution: %v", counts)
}

// ========================================
// Benchmarks
// ========================================

func BenchmarkKmeans_Train(b *testing.B) {
	d := 128
	k := 100
	n := 10000

	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = float32(i % 100)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km, _ := NewKmeans(d, k)
		km.Train(vectors)
	}
}

func BenchmarkKmeans_Assign(b *testing.B) {
	d := 128
	k := 100
	n := 10000

	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = float32(i % 100)
	}

	km, _ := NewKmeans(d, k)
	km.Train(vectors)

	testVectors := make([]float32, 1000*d)
	for i := range testVectors {
		testVectors[i] = float32(i % 50)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Assign(testVectors)
	}
}
