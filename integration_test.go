package faiss

import (
	"os"
	"testing"
)

// ========================================
// End-to-End Integration Tests
// ========================================

// TestCompleteIndexLifecycle tests the full lifecycle of an index
func TestCompleteIndexLifecycle(t *testing.T) {
	d := 128
	nb := 1000
	nq := 10
	k := 5

	// Create index
	index, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Generate and add vectors
	vectors := generateVectors(nb, d)
	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Verify vectors were added
	if index.Ntotal() != int64(nb) {
		t.Errorf("Expected %d vectors, got %d", nb, index.Ntotal())
	}

	// Perform search
	queries := generateVectors(nq, d)
	distances, indices, err := index.Search(queries, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Validate results
	if len(distances) != nq*k || len(indices) != nq*k {
		t.Errorf("Unexpected result size")
	}

	// Save to file
	filename := "/tmp/test_index_lifecycle.faiss"
	defer os.Remove(filename)

	if err := WriteIndex(index, filename); err != nil {
		t.Fatalf("WriteIndex failed: %v", err)
	}

	// Load from file
	loadedIndex, err := ReadIndex(filename)
	if err != nil {
		t.Fatalf("ReadIndex failed: %v", err)
	}
	defer loadedIndex.Close()

	// Verify loaded index
	if loadedIndex.D() != d {
		t.Errorf("Loaded index dimension mismatch")
	}
	if loadedIndex.Ntotal() != int64(nb) {
		t.Errorf("Loaded index vector count mismatch")
	}

	// Search on loaded index
	distances2, indices2, err := loadedIndex.Search(queries, k)
	if err != nil {
		t.Fatalf("Search on loaded index failed: %v", err)
	}

	// Results should match
	for i := range distances {
		if !almostEqual(distances[i], distances2[i], 0.001) {
			t.Errorf("Distance mismatch at %d: %f vs %f", i, distances[i], distances2[i])
		}
		if indices[i] != indices2[i] {
			t.Errorf("Index mismatch at %d: %d vs %d", i, indices[i], indices2[i])
		}
	}

	// Reset
	if err := loadedIndex.Reset(); err != nil {
		t.Fatalf("Reset failed: %v", err)
	}
	if loadedIndex.Ntotal() != 0 {
		t.Error("Index should be empty after reset")
	}
}

// TestIVFPipelineTrainAddSearch tests complete IVF workflow
func TestIVFPipelineTrainAddSearch(t *testing.T) {
	d := 64
	nlist := 20
	nTrain := 1000
	nAdd := 5000
	nq := 10
	k := 10

	// Create quantizer
	quantizer, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	// Create IVF index
	index, err := NewIndexIVFFlat(quantizer, d, nlist, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IVF index: %v", err)
	}
	defer index.Close()

	// Generate training data
	trainingVectors := generateVectors(nTrain, d)

	// Train
	if err := index.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	if !index.IsTrained() {
		t.Error("Index should be trained")
	}

	// Add vectors
	addVectors := generateVectors(nAdd, d)
	if err := index.Add(addVectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	if index.Ntotal() != int64(nAdd) {
		t.Errorf("Expected %d vectors, got %d", nAdd, index.Ntotal())
	}

	// Adjust search parameters
	if err := index.SetNprobe(5); err != nil {
		t.Fatalf("SetNprobe failed: %v", err)
	}

	// Search
	queries := generateVectors(nq, d)
	distances, indices, err := index.Search(queries, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Validate
	if len(distances) != nq*k || len(indices) != nq*k {
		t.Error("Search returned wrong number of results")
	}

	// All indices should be valid
	for _, idx := range indices {
		if idx < 0 || idx >= int64(nAdd) {
			t.Errorf("Invalid index: %d", idx)
		}
	}
}

// TestPQCompressionPipeline tests complete PQ workflow
func TestPQCompressionPipeline(t *testing.T) {
	d := 128
	M := 8
	nbits := 8
	nb := 10000 // PQ with nbits=8 requires 39 * 256 = 9984 training points

	// Create index
	index, err := NewIndexPQ(d, M, nbits, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create PQ index: %v", err)
	}
	defer index.Close()

	// Train
	trainingVectors := generateVectors(nb, d)
	if err := index.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Add
	if err := index.Add(trainingVectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Verify compression ratio
	ratio := index.CompressionRatio()
	if ratio <= 1.0 {
		t.Errorf("Expected compression ratio > 1, got %f", ratio)
	}

	// Search
	queries := generateVectors(10, d)
	distances, indices, err := index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(distances) != 100 || len(indices) != 100 {
		t.Error("Search returned wrong number of results")
	}

	// TODO: IndexPQ serialization not yet fully supported
	// filename := "/tmp/test_pq_pipeline.faiss"
	// defer os.Remove(filename)
	//
	// if err := WriteIndex(index, filename); err != nil {
	// 	t.Fatalf("WriteIndex failed: %v", err)
	// }
	//
	// loadedIndex, err := ReadIndex(filename)
	// if err != nil {
	// 	t.Fatalf("ReadIndex failed: %v", err)
	// }
	// defer loadedIndex.Close()
	//
	// // Search on loaded index
	// _, _, err = loadedIndex.Search(queries, 10)
	// if err != nil {
	// 	t.Fatalf("Search on loaded index failed: %v", err)
	// }
}

// TestMultiMetricSearch tests different metrics
func TestMultiMetricSearch(t *testing.T) {
	d := 64
	nb := 1000
	k := 5

	vectors := generateVectors(nb, d)
	queries := generateVectors(5, d)

	// Test L2
	indexL2, _ := NewIndexFlatL2(d)
	defer indexL2.Close()
	indexL2.Add(vectors)
	distL2, _, err := indexL2.Search(queries, k)
	if err != nil {
		t.Fatalf("L2 search failed: %v", err)
	}

	// Test Inner Product
	indexIP, _ := NewIndexFlatIP(d)
	defer indexIP.Close()
	indexIP.Add(vectors)
	distIP, _, err := indexIP.Search(queries, k)
	if err != nil {
		t.Fatalf("IP search failed: %v", err)
	}

	// Distances should be different for different metrics
	allSame := true
	for i := range distL2 {
		if !almostEqual(distL2[i], distIP[i], 0.001) {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("L2 and IP distances should be different")
	}
}

// TestBinarySearchPipeline tests binary vector workflow
func TestBinarySearchPipeline(t *testing.T) {
	d := 256
	nb := 1000
	nq := 10
	k := 5

	// Create index
	index, err := NewIndexBinaryFlat(d)
	if err != nil {
		t.Fatalf("Failed to create binary index: %v", err)
	}
	defer index.Close()

	// Add binary vectors
	vectors := generateBinaryVectors(nb, d)
	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search
	queries := generateBinaryVectors(nq, d)
	distances, indices, err := index.Search(queries, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Validate Hamming distances
	for _, dist := range distances {
		if dist < 0 || dist > int32(d) {
			t.Errorf("Invalid Hamming distance: %d", dist)
		}
	}

	// All indices should be valid
	for _, idx := range indices {
		if idx < 0 || idx >= int64(nb) {
			t.Errorf("Invalid index: %d", idx)
		}
	}

	// Save and load - TODO: Implement binary index serialization
	// filename := "/tmp/test_binary_pipeline.faiss"
	// defer os.Remove(filename)
	//
	// if err := WriteBinaryIndex(index, filename); err != nil {
	// 	t.Fatalf("WriteBinaryIndex failed: %v", err)
	// }
	//
	// loadedIndex, err := ReadBinaryIndex(filename)
	// if err != nil {
	// 	t.Fatalf("ReadBinaryIndex failed: %v", err)
	// }
	// defer loadedIndex.Close()
	//
	// if loadedIndex.Ntotal() != int64(nb) {
	// 	t.Error("Loaded index has wrong vector count")
	// }
}

// TestClusteringIntegration tests clustering functionality
func TestClusteringIntegration(t *testing.T) {
	d := 64
	nb := 1000
	nCentroids := 10

	vectors := generateVectors(nb, d)

	// Create clustering
	clustering, err := NewClustering(d, nCentroids)
	if err != nil {
		t.Fatalf("Failed to create clustering: %v", err)
	}
	defer clustering.Close()

	// Set parameters
	clustering.SetNiter(20)
	// clustering.SetNredo(5)  // TODO: Implement SetNredo if needed

	// Train clustering (no separate index needed)
	if err := clustering.Train(vectors); err != nil {
		t.Fatalf("Clustering failed: %v", err)
	}

	// Get centroids
	centroids := clustering.Centroids()
	expectedLen := nCentroids * d
	if len(centroids) != expectedLen {
		t.Errorf("Expected %d centroids, got %d", expectedLen, len(centroids))
	}
}

// TestTransformPipeline tests dimensionality reduction pipeline
func TestTransformPipeline(t *testing.T) {
	dIn := 128
	dOut := 64
	nb := 500

	// Create PCA
	pca, err := NewPCAMatrix(dIn, dOut)
	if err != nil {
		t.Fatalf("Failed to create PCA: %v", err)
	}
	defer pca.Close()

	// Train PCA
	trainingVectors := generateVectors(nb, dIn)
	if err := pca.Train(trainingVectors); err != nil {
		t.Fatalf("PCA training failed: %v", err)
	}

	// Transform vectors
	testVectors := generateVectors(100, dIn)
	transformed, err := pca.Apply(testVectors)
	if err != nil {
		t.Fatalf("PCA apply failed: %v", err)
	}

	if len(transformed) != 100*dOut {
		t.Errorf("Wrong transformed size: expected %d, got %d", 100*dOut, len(transformed))
	}

	// Create index on transformed space
	index, _ := NewIndexFlatL2(dOut)
	defer index.Close()

	// Add transformed vectors
	if err := index.Add(transformed); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Transform and search
	queries := generateVectors(5, dIn)
	transformedQueries, _ := pca.Apply(queries)
	distances, indices, err := index.Search(transformedQueries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(distances) != 50 || len(indices) != 50 {
		t.Error("Search returned wrong number of results")
	}
}

// TestFactoryString tests index creation via factory
func TestFactoryString(t *testing.T) {
	d := 64
	nb := 10000 // PQ8 requires 39 * 256 = 9984 training points

	tests := []struct {
		name    string
		factory string
	}{
		{"Flat", "Flat"},
		{"IVF100_Flat", "IVF100,Flat"},
		{"PQ8", "PQ8"},
		{"IVF100_PQ8", "IVF100,PQ8"},
		{"HNSW32", "HNSW32"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			index, err := IndexFactory(d, tt.factory, MetricL2)
			if err != nil {
				t.Fatalf("IndexFactory failed for %s: %v", tt.factory, err)
			}
			defer index.Close()

			// Generate training vectors
			trainingVectors := generateVectors(nb, d)

			// Train if needed
			if !index.IsTrained() {
				if err := index.Train(trainingVectors); err != nil {
					t.Fatalf("Training failed: %v", err)
				}
			}

			// Add vectors
			if err := index.Add(trainingVectors); err != nil {
				t.Fatalf("Add failed: %v", err)
			}

			// Search
			queries := generateVectors(5, d)
			_, _, err = index.Search(queries, 10)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}
		})
	}
}
