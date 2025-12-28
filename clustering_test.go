package faiss

import (
	"math"
	"testing"
)

func TestNewKmeans(t *testing.T) {
	d := 64
	k := 10
	niter := 25

	kmeans, err := NewKmeans(d, k, niter)
	if err != nil {
		t.Fatalf("Failed to create Kmeans: %v", err)
	}
	defer kmeans.Close()

	if kmeans.D() != d {
		t.Errorf("Expected dimension %d, got %d", d, kmeans.D())
	}

	if kmeans.K() != k {
		t.Errorf("Expected k=%d, got %d", k, kmeans.K())
	}

	if kmeans.Niter() != niter {
		t.Errorf("Expected niter=%d, got %d", niter, kmeans.Niter())
	}
}

func TestNewKmeans_InvalidParameters(t *testing.T) {
	tests := []struct {
		name  string
		d     int
		k     int
		niter int
	}{
		{"zero dimension", 0, 10, 25},
		{"negative dimension", -1, 10, 25},
		{"zero k", 64, 0, 25},
		{"negative k", 64, -1, 25},
		{"zero niter", 64, 10, 0},
		{"negative niter", 64, 10, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewKmeans(tt.d, tt.k, tt.niter)
			if err == nil {
				t.Error("Expected error for invalid parameters")
			}
		})
	}
}

func TestKmeans_SetNiter(t *testing.T) {
	kmeans, err := NewKmeans(32, 5, 10)
	if err != nil {
		t.Fatalf("Failed to create Kmeans: %v", err)
	}
	defer kmeans.Close()

	newNiter := 50
	if err := kmeans.SetNiter(newNiter); err != nil {
		t.Fatalf("SetNiter failed: %v", err)
	}

	if kmeans.Niter() != newNiter {
		t.Errorf("Expected niter=%d, got %d", newNiter, kmeans.Niter())
	}

	// Test invalid value
	if err := kmeans.SetNiter(0); err == nil {
		t.Error("Expected error for zero niter")
	}

	if err := kmeans.SetNiter(-1); err == nil {
		t.Error("Expected error for negative niter")
	}
}

func TestKmeans_SetVerbose(t *testing.T) {
	kmeans, err := NewKmeans(32, 5, 10)
	if err != nil {
		t.Fatalf("Failed to create Kmeans: %v", err)
	}
	defer kmeans.Close()

	// Should not error
	if err := kmeans.SetVerbose(true); err != nil {
		t.Errorf("SetVerbose(true) failed: %v", err)
	}

	if err := kmeans.SetVerbose(false); err != nil {
		t.Errorf("SetVerbose(false) failed: %v", err)
	}
}

func TestKmeans_SetSeed(t *testing.T) {
	kmeans, err := NewKmeans(32, 5, 10)
	if err != nil {
		t.Fatalf("Failed to create Kmeans: %v", err)
	}
	defer kmeans.Close()

	// Should not error
	if err := kmeans.SetSeed(42); err != nil {
		t.Errorf("SetSeed failed: %v", err)
	}
}

func TestKmeans_Train(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping training test in short mode")
	}

	d := 32
	k := 5
	niter := 10
	n := 1000

	kmeans, err := NewKmeans(d, k, niter)
	if err != nil {
		t.Fatalf("Failed to create Kmeans: %v", err)
	}
	defer kmeans.Close()

	// Generate random vectors with some clustering structure
	vectors := make([]float32, n*d)
	for i := 0; i < n; i++ {
		clusterID := i % k
		baseValue := float32(clusterID) * 10.0
		for j := 0; j < d; j++ {
			// Add some noise around cluster centers
			vectors[i*d+j] = baseValue + float32(j)*0.1
		}
	}

	// Train
	if err := kmeans.Train(vectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Get centroids
	centroids := kmeans.Centroids()
	if centroids == nil {
		t.Fatal("Centroids should not be nil after training")
	}

	expectedLen := k * d
	if len(centroids) != expectedLen {
		t.Errorf("Expected %d centroids, got %d", expectedLen, len(centroids))
	}

	// Verify centroids are reasonable (not all zeros)
	allZero := true
	for _, c := range centroids {
		if c != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("Centroids should not all be zero")
	}
}

func TestKmeans_Train_EmptyVectors(t *testing.T) {
	kmeans, err := NewKmeans(32, 5, 10)
	if err != nil {
		t.Fatalf("Failed to create Kmeans: %v", err)
	}
	defer kmeans.Close()

	err = kmeans.Train([]float32{})
	if err == nil {
		t.Error("Expected error for empty vectors")
	}
}

func TestKmeans_Train_InvalidVectorSize(t *testing.T) {
	d := 32
	kmeans, err := NewKmeans(d, 5, 10)
	if err != nil {
		t.Fatalf("Failed to create Kmeans: %v", err)
	}
	defer kmeans.Close()

	// Not a multiple of dimension
	invalidVectors := make([]float32, d+1)
	err = kmeans.Train(invalidVectors)
	if err != ErrInvalidVectors {
		t.Errorf("Expected ErrInvalidVectors, got %v", err)
	}
}

func TestKmeans_Centroids_BeforeTraining(t *testing.T) {
	kmeans, err := NewKmeans(32, 5, 10)
	if err != nil {
		t.Fatalf("Failed to create Kmeans: %v", err)
	}
	defer kmeans.Close()

	centroids := kmeans.Centroids()
	if centroids != nil {
		t.Error("Centroids should be nil before training")
	}
}

func TestNewClustering(t *testing.T) {
	d := 64
	k := 10

	clustering, err := NewClustering(d, k)
	if err != nil {
		t.Fatalf("Failed to create Clustering: %v", err)
	}
	defer clustering.Close()

	if clustering.D() != d {
		t.Errorf("Expected dimension %d, got %d", d, clustering.D())
	}

	if clustering.K() != k {
		t.Errorf("Expected k=%d, got %d", k, clustering.K())
	}
}

func TestNewClustering_InvalidParameters(t *testing.T) {
	tests := []struct {
		name string
		d    int
		k    int
	}{
		{"zero dimension", 0, 10},
		{"negative dimension", -1, 10},
		{"zero k", 64, 0},
		{"negative k", 64, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewClustering(tt.d, tt.k)
			if err == nil {
				t.Error("Expected error for invalid parameters")
			}
		})
	}
}

func TestClustering_SetNiter(t *testing.T) {
	clustering, err := NewClustering(32, 5)
	if err != nil {
		t.Fatalf("Failed to create Clustering: %v", err)
	}
	defer clustering.Close()

	newNiter := 50
	clustering.SetNiter(newNiter)

	if clustering.Niter() != newNiter {
		t.Errorf("Expected niter=%d, got %d", newNiter, clustering.Niter())
	}
}

func TestClustering_Train(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping training test in short mode")
	}

	d := 32
	k := 5
	n := 500

	clustering, err := NewClustering(d, k)
	if err != nil {
		t.Fatalf("Failed to create Clustering: %v", err)
	}
	defer clustering.Close()

	clustering.SetNiter(20)

	// Generate clustered data
	vectors := make([]float32, n*d)
	for i := 0; i < n; i++ {
		clusterID := i % k
		for j := 0; j < d; j++ {
			// Create distinct clusters
			vectors[i*d+j] = float32(clusterID*100 + j)
		}
	}

	// Train
	if err := clustering.Train(vectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Get centroids
	centroids := clustering.Centroids()
	if len(centroids) != k*d {
		t.Errorf("Expected %d centroid values, got %d", k*d, len(centroids))
	}

	// Verify centroids are different from each other
	for i := 0; i < k-1; i++ {
		same := true
		for j := 0; j < d; j++ {
			if math.Abs(float64(centroids[i*d+j]-centroids[(i+1)*d+j])) > 0.01 {
				same = false
				break
			}
		}
		if same {
			t.Error("Adjacent centroids should be different")
		}
	}
}

func TestClustering_Train_EmptyVectors(t *testing.T) {
	clustering, err := NewClustering(32, 5)
	if err != nil {
		t.Fatalf("Failed to create Clustering: %v", err)
	}
	defer clustering.Close()

	err = clustering.Train([]float32{})
	if err == nil {
		t.Error("Expected error for empty vectors")
	}
}

func TestClustering_Train_InvalidVectorSize(t *testing.T) {
	d := 32
	clustering, err := NewClustering(d, 5)
	if err != nil {
		t.Fatalf("Failed to create Clustering: %v", err)
	}
	defer clustering.Close()

	// Not a multiple of dimension
	invalidVectors := make([]float32, d+1)
	err = clustering.Train(invalidVectors)
	if err != ErrInvalidVectors {
		t.Errorf("Expected ErrInvalidVectors, got %v", err)
	}
}
