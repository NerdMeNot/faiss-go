package faiss

import (
	"testing"
)

// ========================================
// IndexRefine Tests
// ========================================

func TestIndexRefine(t *testing.T) {
	d := 64
	nlist := 10
	nb := 500

	// Create base index (IVF for speed)
	quantizer, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	baseIndex, err := NewIndexIVFFlat(quantizer, d, nlist, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create base index: %v", err)
	}
	defer baseIndex.Close()

	// Create refine index (Flat for accuracy)
	refineIndex, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create refine index: %v", err)
	}
	defer refineIndex.Close()

	// Create IndexRefine
	index, err := NewIndexRefine(baseIndex, refineIndex)
	if err != nil {
		t.Fatalf("Failed to create IndexRefine: %v", err)
	}
	defer index.Close()

	// Check properties
	if index.D() != d {
		t.Errorf("Expected dimension %d, got %d", d, index.D())
	}
	if index.MetricType() != MetricL2 {
		t.Errorf("Expected MetricL2, got %v", index.MetricType())
	}

	// Train
	trainingVectors := generateVectors(nb, d)
	if err := index.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Set k_factor
	if err := index.SetK_factor(2.0); err != nil {
		t.Fatalf("SetK_factor failed: %v", err)
	}

	// Add vectors
	if err := index.Add(trainingVectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search
	queries := generateVectors(5, d)
	distances, indices, err := index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(distances) != 50 || len(indices) != 50 {
		t.Error("Search returned wrong number of results")
	}

	// Reset
	if err := index.Reset(); err != nil {
		t.Fatalf("Reset failed: %v", err)
	}
	if index.Ntotal() != 0 {
		t.Errorf("Expected ntotal 0 after reset, got %d", index.Ntotal())
	}
}

func TestIndexRefineMismatchedDimensions(t *testing.T) {
	base, _ := NewIndexFlatL2(64)
	defer base.Close()

	refine, _ := NewIndexFlatL2(32)
	defer refine.Close()

	_, err := NewIndexRefine(base, refine)
	if err == nil {
		t.Error("Expected error for mismatched dimensions")
	}
}

func TestIndexRefineMismatchedMetrics(t *testing.T) {
	base, _ := NewIndexFlatL2(64)
	defer base.Close()

	refine, _ := NewIndexFlatIP(64)
	defer refine.Close()

	_, err := NewIndexRefine(base, refine)
	if err == nil {
		t.Error("Expected error for mismatched metrics")
	}
}

// ========================================
// IndexPreTransform Tests
// ========================================

func TestIndexPreTransform(t *testing.T) {
	dIn := 128
	dOut := 64
	nb := 500

	// Create PCA transform
	pca, err := NewPCAMatrix(dIn, dOut)
	if err != nil {
		t.Fatalf("Failed to create PCA: %v", err)
	}
	defer pca.Close()

	// Create index
	baseIndex, err := NewIndexFlatL2(dOut)
	if err != nil {
		t.Fatalf("Failed to create base index: %v", err)
	}
	defer baseIndex.Close()

	// Create IndexPreTransform
	index, err := NewIndexPreTransform(pca, baseIndex)
	if err != nil {
		t.Fatalf("Failed to create IndexPreTransform: %v", err)
	}
	defer index.Close()

	// Check properties
	if index.D() != dIn {
		t.Errorf("Expected input dimension %d, got %d", dIn, index.D())
	}

	// Train
	trainingVectors := generateVectors(nb, dIn)
	if err := index.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	if !index.IsTrained() {
		t.Error("Index should be trained after training")
	}

	// Add vectors (will be transformed automatically)
	if err := index.Add(trainingVectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search (queries will be transformed automatically)
	queries := generateVectors(5, dIn)
	distances, indices, err := index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(distances) != 50 || len(indices) != 50 {
		t.Error("Search returned wrong number of results")
	}
}

func TestIndexPreTransformDimensionMismatch(t *testing.T) {
	pca, _ := NewPCAMatrix(128, 64)
	defer pca.Close()

	index, _ := NewIndexFlatL2(32) // Wrong dimension
	defer index.Close()

	_, err := NewIndexPreTransform(pca, index)
	if err == nil {
		t.Error("Expected error for dimension mismatch")
	}
}

// ========================================
// IndexShards Tests
// ========================================

func TestIndexShards(t *testing.T) {
	d := 64
	nb := 500

	// Create sharded index
	shards, err := NewIndexShards(d, MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IndexShards: %v", err)
	}
	defer shards.Close()

	// Create and add shards
	shard1, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create shard1: %v", err)
	}
	defer shard1.Close()

	shard2, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create shard2: %v", err)
	}
	defer shard2.Close()

	if err := shards.AddShard(shard1); err != nil {
		t.Fatalf("Failed to add shard1: %v", err)
	}
	if err := shards.AddShard(shard2); err != nil {
		t.Fatalf("Failed to add shard2: %v", err)
	}

	// Add vectors (distributed across shards)
	vectors := generateVectors(nb, d)
	if err := shards.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search across all shards
	queries := generateVectors(5, d)
	distances, indices, err := shards.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(distances) != 50 || len(indices) != 50 {
		t.Error("Search returned wrong number of results")
	}

	// Check total vectors
	total := shards.Ntotal()
	if total != int64(nb) {
		t.Errorf("Expected ntotal %d, got %d", nb, total)
	}

	// Reset
	if err := shards.Reset(); err != nil {
		t.Fatalf("Reset failed: %v", err)
	}
	if shards.Ntotal() != 0 {
		t.Errorf("Expected ntotal 0 after reset, got %d", shards.Ntotal())
	}
}

func TestIndexShardsDimensionMismatch(t *testing.T) {
	shards, _ := NewIndexShards(64, MetricL2)
	defer shards.Close()

	shard, _ := NewIndexFlatL2(32) // Wrong dimension
	defer shard.Close()

	err := shards.AddShard(shard)
	if err == nil {
		t.Error("Expected error for dimension mismatch")
	}
}

func TestIndexShardsNoShards(t *testing.T) {
	shards, _ := NewIndexShards(64, MetricL2)
	defer shards.Close()

	vectors := generateVectors(100, 64)
	err := shards.Add(vectors)
	if err == nil {
		t.Error("Expected error when adding to empty shards")
	}
}
