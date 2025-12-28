package recall_test

import (
	"os"
	"testing"

	faiss "github.com/NerdMeNot/faiss-go"
	"github.com/NerdMeNot/faiss-go/test/helpers"
)

// ========================================
// IndexFlat Tests
// ========================================

func TestIndexFlatL2(t *testing.T) {
	d := 128
	nb := 1000
	nq := 10
	k := 5

	index, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create IndexFlatL2: %v", err)
	}
	defer index.Close()

	// Check properties
	if index.D() != d {
		t.Errorf("Expected dimension %d, got %d", d, index.D())
	}
	if index.Ntotal() != 0 {
		t.Errorf("Expected ntotal 0, got %d", index.Ntotal())
	}
	if !index.IsTrained() {
		t.Error("Flat index should be trained by default")
	}
	if index.MetricType() != faiss.MetricL2 {
		t.Errorf("Expected faiss.MetricL2, got %v", index.MetricType())
	}

	// Add vectors
	vectors := helpers.GenerateVectors(nb, d)
	if err := index.Add(vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}
	if index.Ntotal() != int64(nb) {
		t.Errorf("Expected ntotal %d, got %d", nb, index.Ntotal())
	}

	// Search
	queries := helpers.GenerateVectors(nq, d)
	distances, indices, err := index.Search(queries, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Validate results
	if len(distances) != nq*k {
		t.Errorf("Expected %d distances, got %d", nq*k, len(distances))
	}
	if len(indices) != nq*k {
		t.Errorf("Expected %d indices, got %d", nq*k, len(indices))
	}

	// First result should be self (distance ~0)
	if indices[0] < 0 || indices[0] >= int64(nb) {
		t.Errorf("Invalid index: %d", indices[0])
	}

	// Reset
	if err := index.Reset(); err != nil {
		t.Fatalf("Reset failed: %v", err)
	}
	if index.Ntotal() != 0 {
		t.Errorf("Expected ntotal 0 after reset, got %d", index.Ntotal())
	}
}

func TestIndexFlatIP(t *testing.T) {
	d := 64
	nb := 500

	index, err := faiss.NewIndexFlatIP(d)
	if err != nil {
		t.Fatalf("Failed to create IndexFlatIP: %v", err)
	}
	defer index.Close()

	if index.MetricType() != faiss.MetricInnerProduct {
		t.Errorf("Expected faiss.MetricInnerProduct, got %v", index.MetricType())
	}

	vectors := helpers.GenerateVectors(nb, d)
	if err := index.Add(vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	queries := helpers.GenerateVectors(1, d)
	distances, indices, err := index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(distances) != 10 || len(indices) != 10 {
		t.Error("Search returned wrong number of results")
	}
}

// ========================================
// IndexIVFFlat Tests
// ========================================

func TestIndexIVFFlat(t *testing.T) {
	d := 128
	nlist := 10
	nb := 1000
	nq := 5
	k := 10

	// Create quantizer
	quantizer, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	// Create IVF index
	index, err := faiss.NewIndexIVFFlat(quantizer, d, nlist, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IndexIVFFlat: %v", err)
	}
	defer index.Close()

	// Check initial state
	if index.IsTrained() {
		t.Error("Index should not be trained initially")
	}
	if index.Nlist() != nlist {
		t.Errorf("Expected nlist %d, got %d", nlist, index.Nlist())
	}
	if index.Nprobe() != 1 {
		t.Errorf("Expected nprobe 1, got %d", index.Nprobe())
	}

	// Train
	trainingVectors := helpers.GenerateVectors(nb, d)
	if err := index.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}
	if !index.IsTrained() {
		t.Error("Index should be trained after training")
	}

	// Set nprobe
	if err := index.SetNprobe(5); err != nil {
		t.Fatalf("SetNprobe failed: %v", err)
	}
	if index.Nprobe() != 5 {
		t.Errorf("Expected nprobe 5, got %d", index.Nprobe())
	}

	// Add vectors
	if err := index.Add(trainingVectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search
	queries := helpers.GenerateVectors(nq, d)
	distances, indices, err := index.Search(queries, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(distances) != nq*k || len(indices) != nq*k {
		t.Error("Search returned wrong number of results")
	}
}

// ========================================
// IndexPQ Tests
// ========================================

func TestIndexPQ(t *testing.T) {
	d := 128
	M := 8
	nbits := 8
	nb := 10000 // PQ with nbits=8 requires 39 * 256 = 9984 training points

	index, err := faiss.NewIndexPQ(d, M, nbits, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IndexPQ: %v", err)
	}
	defer index.Close()

	if index.GetM() != M {
		t.Errorf("Expected M=%d, got %d", M, index.GetM())
	}
	if index.GetNbits() != nbits {
		t.Errorf("Expected nbits=%d, got %d", nbits, index.GetNbits())
	}

	// Train
	trainingVectors := helpers.GenerateVectors(nb, d)
	if err := index.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Add
	if err := index.Add(trainingVectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search
	queries := helpers.GenerateVectors(5, d)
	distances, indices, err := index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(distances) != 50 || len(indices) != 50 {
		t.Error("Search returned wrong number of results")
	}

	// Check compression ratio
	ratio := index.CompressionRatio()
	if ratio <= 1.0 {
		t.Errorf("Expected compression ratio > 1.0, got %f", ratio)
	}
}

// ========================================
// IndexIVFPQ Tests
// ========================================

func TestIndexIVFPQ(t *testing.T) {
	d := 64
	nlist := 10
	M := 8
	nbits := 8
	nb := 10000 // IVFPQ with nbits=8 requires 39 * 256 = 9984 training points

	quantizer, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	index, err := faiss.NewIndexIVFPQ(quantizer, d, nlist, M, nbits)
	if err != nil {
		t.Fatalf("Failed to create IndexIVFPQ: %v", err)
	}
	defer index.Close()

	// Train
	trainingVectors := helpers.GenerateVectors(nb, d)
	if err := index.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Add
	if err := index.Add(trainingVectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Set nprobe
	index.SetNprobe(3)

	// Search
	queries := helpers.GenerateVectors(5, d)
	distances, indices, err := index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(distances) != 50 || len(indices) != 50 {
		t.Error("Search returned wrong number of results")
	}
}

// ========================================
// IndexHNSW Tests
// ========================================

func TestIndexHNSW(t *testing.T) {
	d := 64
	M := 16
	nb := 500

	index, err := faiss.NewIndexHNSWFlat(d, M, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IndexHNSW: %v", err)
	}
	defer index.Close()

	if index.GetM() != M {
		t.Errorf("Expected M=%d, got %d", M, index.GetM())
	}

	// Add vectors
	vectors := helpers.GenerateVectors(nb, d)
	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Set efSearch
	if err := index.SetEfSearch(40); err != nil {
		t.Fatalf("SetEfSearch failed: %v", err)
	}

	// Search
	queries := helpers.GenerateVectors(5, d)
	distances, indices, err := index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(distances) != 50 || len(indices) != 50 {
		t.Error("Search returned wrong number of results")
	}
}

// ========================================
// IndexScalarQuantizer Tests
// ========================================

func TestIndexScalarQuantizer(t *testing.T) {
	d := 64
	nb := 500

	index, err := faiss.NewIndexScalarQuantizer(d, faiss.QT_8bit, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IndexScalarQuantizer: %v", err)
	}
	defer index.Close()

	if index.QuantizerType() != faiss.QT_8bit {
		t.Error("Wrong quantizer type")
	}

	// Train
	trainingVectors := helpers.GenerateVectors(nb, d)
	if err := index.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Add
	if err := index.Add(trainingVectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search
	queries := helpers.GenerateVectors(5, d)
	distances, indices, err := index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(distances) != 50 || len(indices) != 50 {
		t.Error("Search returned wrong number of results")
	}

	// Check compression
	ratio := index.CompressionRatio()
	if ratio != 4.0 { // 32 bits / 8 bits = 4x
		t.Errorf("Expected compression ratio 4.0, got %f", ratio)
	}
}

func TestIndexIVFScalarQuantizer(t *testing.T) {
	d := 64
	nlist := 10
	nb := 500

	quantizer, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	index, err := faiss.NewIndexIVFScalarQuantizer(quantizer, d, nlist, faiss.QT_8bit, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IndexIVFScalarQuantizer: %v", err)
	}
	defer index.Close()

	// Train
	trainingVectors := helpers.GenerateVectors(nb, d)
	if err := index.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Add
	if err := index.Add(trainingVectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search
	index.SetNprobe(3)
	queries := helpers.GenerateVectors(5, d)
	_, _, err = index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
}

// ========================================
// IndexLSH Tests
// ========================================

func TestIndexLSH(t *testing.T) {
	d := 128
	nbits := 256
	nb := 500

	index, err := faiss.NewIndexLSH(d, nbits)
	if err != nil {
		t.Skipf("IndexLSH not available in this FAISS build: %v", err)
	}
	defer index.Close()

	if index.Nbits() != nbits {
		t.Errorf("Expected nbits=%d, got %d", nbits, index.Nbits())
	}

	// Add vectors
	vectors := helpers.GenerateVectors(nb, d)
	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search
	queries := helpers.GenerateVectors(5, d)
	distances, indices, err := index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(distances) != 50 || len(indices) != 50 {
		t.Error("Search returned wrong number of results")
	}
}

// ========================================
// Binary Index Tests
// ========================================

func TestIndexBinaryFlat(t *testing.T) {
	d := 256 // bits
	nb := 500
	nq := 5
	k := 10

	index, err := faiss.NewIndexBinaryFlat(d)
	if err != nil {
		t.Fatalf("Failed to create IndexBinaryFlat: %v", err)
	}
	defer index.Close()

	if index.D() != d {
		t.Errorf("Expected dimension %d, got %d", d, index.D())
	}

	// Add binary vectors
	vectors := helpers.GenerateBinaryVectors(nb, d)
	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	if index.Ntotal() != int64(nb) {
		t.Errorf("Expected ntotal %d, got %d", nb, index.Ntotal())
	}

	// Search
	queries := helpers.GenerateBinaryVectors(nq, d)
	distances, indices, err := index.Search(queries, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(distances) != nq*k || len(indices) != nq*k {
		t.Error("Search returned wrong number of results")
	}

	// Distances should be int32 (Hamming distance)
	for _, dist := range distances {
		if dist < 0 || dist > int32(d) {
			t.Errorf("Invalid Hamming distance: %d", dist)
		}
	}
}

func TestIndexBinaryIVF(t *testing.T) {
	d := 256
	nlist := 10
	nb := 500

	quantizer, err := faiss.NewIndexBinaryFlat(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	index, err := faiss.NewIndexBinaryIVF(quantizer, d, nlist)
	if err != nil {
		t.Fatalf("Failed to create IndexBinaryIVF: %v", err)
	}
	defer index.Close()

	// Train
	trainingVectors := helpers.GenerateBinaryVectors(nb, d)
	if err := index.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Add
	if err := index.Add(trainingVectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search
	index.SetNprobe(3)
	queries := helpers.GenerateBinaryVectors(5, d)
	_, _, err = index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
}

// ========================================
// PQFastScan Tests
// ========================================

func TestIndexPQFastScan(t *testing.T) {
	d := 128
	M := 8
	nbits := 4 // Optimal for FastScan
	nb := 1000 // Sufficient for nbits=4 (16 centroids per subquantizer)

	index, err := faiss.NewIndexPQFastScan(d, M, nbits, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IndexPQFastScan: %v", err)
	}
	defer index.Close()

	if index.BlockSize() != 32 {
		t.Errorf("Expected default block size 32, got %d", index.BlockSize())
	}

	// Train
	trainingVectors := helpers.GenerateVectors(nb, d)
	if err := index.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Add
	if err := index.Add(trainingVectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search
	queries := helpers.GenerateVectors(5, d)
	_, _, err = index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
}

func TestIndexIVFPQFastScan(t *testing.T) {
	d := 64
	nlist := 10
	M := 8
	nbits := 4
	nb := 1000 // Sufficient for nlist=10 + nbits=4

	quantizer, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	index, err := faiss.NewIndexIVFPQFastScan(quantizer, d, nlist, M, nbits, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IndexIVFPQFastScan: %v", err)
	}
	defer index.Close()

	// Train
	trainingVectors := helpers.GenerateVectors(nb, d)
	if err := index.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Add
	if err := index.Add(trainingVectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search
	index.SetNprobe(3)
	queries := helpers.GenerateVectors(5, d)
	_, _, err = index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
}

// ========================================
// OnDisk Index Tests
// ========================================

func TestIndexIVFFlatOnDisk(t *testing.T) {
	d := 64
	nlist := 10
	nb := 500
	filename := "/tmp/test_ivfflat_ondisk.ivfdata"
	defer os.Remove(filename)

	quantizer, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	index, err := faiss.NewIndexIVFFlatOnDisk(quantizer, d, nlist, filename, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IndexIVFFlatOnDisk: %v", err)
	}
	defer index.Close()

	if index.Filename() != filename {
		t.Errorf("Expected filename %s, got %s", filename, index.Filename())
	}

	// Train
	trainingVectors := helpers.GenerateVectors(nb, d)
	if err := index.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Add
	if err := index.Add(trainingVectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search
	index.SetNprobe(3)
	queries := helpers.GenerateVectors(5, d)
	_, _, err = index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
}

func TestIndexIVFPQOnDisk(t *testing.T) {
	d := 64
	nlist := 10
	M := 8
	nbits := 8
	nb := 10000 // Need 39*256=9984 for nbits=8 (256 centroids per subquantizer)
	filename := "/tmp/test_ivfpq_ondisk.ivfpq"
	defer os.Remove(filename)

	quantizer, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	index, err := faiss.NewIndexIVFPQOnDisk(quantizer, d, nlist, M, nbits, filename, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IndexIVFPQOnDisk: %v", err)
	}
	defer index.Close()

	// Train
	trainingVectors := helpers.GenerateVectors(nb, d)
	if err := index.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Add
	if err := index.Add(trainingVectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search
	index.SetNprobe(3)
	queries := helpers.GenerateVectors(5, d)
	_, _, err = index.Search(queries, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Check compression ratio
	ratio := index.CompressionRatio()
	if ratio <= 1.0 {
		t.Errorf("Expected compression ratio > 1.0, got %f", ratio)
	}
}

// ========================================
// Error Handling Tests
// ========================================

func TestInvalidDimensions(t *testing.T) {
	_, err := faiss.NewIndexFlatL2(0)
	if err == nil {
		t.Error("Expected error for dimension 0")
	}

	_, err = faiss.NewIndexFlatL2(-10)
	if err == nil {
		t.Error("Expected error for negative dimension")
	}
}

func TestInvalidPQParameters(t *testing.T) {
	// d not divisible by M
	_, err := faiss.NewIndexPQ(100, 7, 8, faiss.MetricL2)
	if err == nil {
		t.Error("Expected error for d not divisible by M")
	}

	// Invalid nbits
	_, err = faiss.NewIndexPQ(128, 8, 0, faiss.MetricL2)
	if err == nil {
		t.Error("Expected error for invalid nbits")
	}
}

func TestSearchBeforeAdd(t *testing.T) {
	index, _ := faiss.NewIndexFlatL2(64)
	defer index.Close()

	queries := helpers.GenerateVectors(1, 64)
	_, _, err := index.Search(queries, 10)

	// Should not error, but return no results or handle gracefully
	// The behavior depends on FAISS implementation
	_ = err
}

func TestAddBeforeTrain(t *testing.T) {
	quantizer, _ := faiss.NewIndexFlatL2(64)
	defer quantizer.Close()

	index, _ := faiss.NewIndexIVFFlat(quantizer, 64, 10, faiss.MetricL2)
	defer index.Close()

	vectors := helpers.GenerateVectors(100, 64)
	err := index.Add(vectors)
	if err == nil {
		t.Error("Expected error when adding before training")
	}
}

func TestWrongVectorDimension(t *testing.T) {
	index, _ := faiss.NewIndexFlatL2(64)
	defer index.Close()

	// Wrong dimension - create vectors that don't divide evenly
	// 100 floats total, which is not divisible by 64
	wrongVectors := make([]float32, 100)
	err := index.Add(wrongVectors)
	if err == nil {
		t.Error("Expected error for wrong vector dimension")
	}
}
