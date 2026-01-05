package faiss

import (
	"testing"
)

// ========================================
// IndexScalarQuantizer Creation Tests
// ========================================

func TestNewIndexScalarQuantizer(t *testing.T) {
	tests := []struct {
		name          string
		d             int
		qtype         QuantizerType
		metric        MetricType
		needsTraining bool
	}{
		{"8bit_L2", 64, QT_8bit, MetricL2, true},
		{"8bit_IP", 64, QT_8bit, MetricInnerProduct, true},
		{"4bit_L2", 64, QT_4bit, MetricL2, true},
		{"fp16_L2", 64, QT_fp16, MetricL2, false}, // fp16 may not need training
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx, err := NewIndexScalarQuantizer(tt.d, tt.qtype, tt.metric)
			if err != nil {
				t.Fatalf("NewIndexScalarQuantizer() failed: %v", err)
			}
			defer idx.Close()

			if idx.D() != tt.d {
				t.Errorf("D() = %d, want %d", idx.D(), tt.d)
			}
			if idx.QuantizerType() != tt.qtype {
				t.Errorf("QuantizerType() = %v, want %v", idx.QuantizerType(), tt.qtype)
			}
			if idx.MetricType() != tt.metric {
				t.Errorf("MetricType() = %v, want %v", idx.MetricType(), tt.metric)
			}
			if idx.Ntotal() != 0 {
				t.Errorf("Ntotal() = %d, want 0", idx.Ntotal())
			}
			// Training requirements depend on quantizer type
			// fp16 doesn't need training, others do
			if tt.needsTraining && idx.IsTrained() {
				t.Error("IsTrained() = true, want false (SQ needs training)")
			}
		})
	}
}

// ========================================
// IndexScalarQuantizer Train Tests
// ========================================

func TestIndexScalarQuantizer_Train(t *testing.T) {
	idx, _ := NewIndexScalarQuantizer(4, QT_8bit, MetricL2)
	defer idx.Close()

	// Generate training vectors with variance
	vectors := make([]float32, 4*100)
	for i := range vectors {
		vectors[i] = float32(i % 50)
	}

	err := idx.Train(vectors)
	if err != nil {
		t.Fatalf("Train() failed: %v", err)
	}

	if !idx.IsTrained() {
		t.Error("IsTrained() = false after Train()")
	}
}

func TestIndexScalarQuantizer_Train_Empty(t *testing.T) {
	idx, _ := NewIndexScalarQuantizer(4, QT_8bit, MetricL2)
	defer idx.Close()

	err := idx.Train([]float32{})
	if err == nil {
		t.Error("Train(empty) should return error")
	}
}

func TestIndexScalarQuantizer_Train_InvalidDimension(t *testing.T) {
	idx, _ := NewIndexScalarQuantizer(4, QT_8bit, MetricL2)
	defer idx.Close()

	err := idx.Train([]float32{1, 2, 3})
	if err == nil {
		t.Error("Train() with invalid dimension should error")
	}
}

// ========================================
// IndexScalarQuantizer Add Tests
// ========================================

func TestIndexScalarQuantizer_Add(t *testing.T) {
	idx, _ := NewIndexScalarQuantizer(4, QT_8bit, MetricL2)
	defer idx.Close()

	// Train first
	trainVectors := make([]float32, 4*100)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 50)
	}
	idx.Train(trainVectors)

	// Add vectors
	vectors := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}

	err := idx.Add(vectors)
	if err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	if idx.Ntotal() != 2 {
		t.Errorf("Ntotal() = %d, want 2", idx.Ntotal())
	}
}

func TestIndexScalarQuantizer_Add_BeforeTrain(t *testing.T) {
	idx, _ := NewIndexScalarQuantizer(4, QT_8bit, MetricL2)
	defer idx.Close()

	err := idx.Add([]float32{1, 2, 3, 4})
	if err == nil {
		t.Error("Add() before Train() should return error")
	}
}

func TestIndexScalarQuantizer_Add_Empty(t *testing.T) {
	idx, _ := NewIndexScalarQuantizer(4, QT_8bit, MetricL2)
	defer idx.Close()

	// Train first
	trainVectors := make([]float32, 4*100)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 50)
	}
	idx.Train(trainVectors)

	err := idx.Add([]float32{})
	if err != nil {
		t.Errorf("Add(empty) should not error: %v", err)
	}
}

func TestIndexScalarQuantizer_Add_InvalidDimension(t *testing.T) {
	idx, _ := NewIndexScalarQuantizer(4, QT_8bit, MetricL2)
	defer idx.Close()

	// Train first
	trainVectors := make([]float32, 4*100)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 50)
	}
	idx.Train(trainVectors)

	err := idx.Add([]float32{1, 2, 3})
	if err == nil {
		t.Error("Add() with invalid dimension should error")
	}
}

// ========================================
// IndexScalarQuantizer Search Tests
// ========================================

func TestIndexScalarQuantizer_Search(t *testing.T) {
	idx, _ := NewIndexScalarQuantizer(4, QT_8bit, MetricL2)
	defer idx.Close()

	// Train
	trainVectors := make([]float32, 4*200)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 50)
	}
	idx.Train(trainVectors)

	// Add
	vectors := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	idx.Add(vectors)

	// Search
	query := []float32{1, 0, 0, 0}
	distances, indices, err := idx.Search(query, 2)
	if err != nil {
		t.Fatalf("Search() failed: %v", err)
	}

	if len(distances) != 2 || len(indices) != 2 {
		t.Errorf("Search() returned %d distances, %d indices, want 2 each",
			len(distances), len(indices))
	}

	// First result should be index 0 (closest to query)
	if indices[0] != 0 {
		t.Errorf("First result index = %d, want 0", indices[0])
	}
}

func TestIndexScalarQuantizer_Search_Empty(t *testing.T) {
	idx, _ := NewIndexScalarQuantizer(4, QT_8bit, MetricL2)
	defer idx.Close()

	_, _, err := idx.Search([]float32{}, 5)
	if err == nil {
		t.Error("Search(empty) should return error")
	}
}

func TestIndexScalarQuantizer_Search_InvalidDimension(t *testing.T) {
	idx, _ := NewIndexScalarQuantizer(4, QT_8bit, MetricL2)
	defer idx.Close()

	// Train and add
	trainVectors := make([]float32, 4*100)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 50)
	}
	idx.Train(trainVectors)
	idx.Add([]float32{1, 2, 3, 4})

	_, _, err := idx.Search([]float32{1, 2, 3}, 5)
	if err == nil {
		t.Error("Search() with invalid dimension should error")
	}
}

// ========================================
// IndexScalarQuantizer Reset Tests
// ========================================

func TestIndexScalarQuantizer_Reset(t *testing.T) {
	idx, _ := NewIndexScalarQuantizer(4, QT_8bit, MetricL2)
	defer idx.Close()

	// Train and add
	trainVectors := make([]float32, 4*100)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 50)
	}
	idx.Train(trainVectors)
	idx.Add([]float32{1, 2, 3, 4})

	if idx.Ntotal() != 1 {
		t.Errorf("Ntotal() before reset = %d, want 1", idx.Ntotal())
	}

	err := idx.Reset()
	if err != nil {
		t.Fatalf("Reset() failed: %v", err)
	}

	if idx.Ntotal() != 0 {
		t.Errorf("Ntotal() after reset = %d, want 0", idx.Ntotal())
	}
}

// ========================================
// IndexScalarQuantizer Close Tests
// ========================================

func TestIndexScalarQuantizer_Close(t *testing.T) {
	idx, _ := NewIndexScalarQuantizer(4, QT_8bit, MetricL2)

	err := idx.Close()
	if err != nil {
		t.Errorf("First Close() failed: %v", err)
	}

	err = idx.Close()
	if err != nil {
		t.Errorf("Second Close() failed: %v", err)
	}
}

// ========================================
// IndexScalarQuantizer CompressionRatio Tests
// ========================================

func TestIndexScalarQuantizer_CompressionRatio(t *testing.T) {
	tests := []struct {
		qtype    QuantizerType
		expected float64
	}{
		{QT_8bit, 4.0},          // 32/8 = 4x
		{QT_4bit, 8.0},          // 32/4 = 8x
		{QT_fp16, 2.0},          // 32/16 = 2x
		{QT_6bit, 32.0 / 6.0},   // ~5.33x
		{QT_8bit_uniform, 4.0},  // 32/8 = 4x
		{QT_4bit_uniform, 8.0},  // 32/4 = 8x
		{QT_8bit_direct, 4.0},   // 32/8 = 4x
	}

	for _, tt := range tests {
		idx, err := NewIndexScalarQuantizer(64, tt.qtype, MetricL2)
		if err != nil {
			t.Fatalf("Failed to create SQ with qtype %v: %v", tt.qtype, err)
		}

		ratio := idx.CompressionRatio()
		if ratio != tt.expected {
			t.Errorf("CompressionRatio() for qtype %v = %f, want %f",
				tt.qtype, ratio, tt.expected)
		}

		idx.Close()
	}
}

// ========================================
// IndexScalarQuantizer Unsupported Operations
// ========================================

func TestIndexScalarQuantizer_SetNprobe(t *testing.T) {
	idx, _ := NewIndexScalarQuantizer(4, QT_8bit, MetricL2)
	defer idx.Close()

	err := idx.SetNprobe(10)
	if err == nil {
		t.Error("SetNprobe() should return error (not supported)")
	}
}

func TestIndexScalarQuantizer_SetEfSearch(t *testing.T) {
	idx, _ := NewIndexScalarQuantizer(4, QT_8bit, MetricL2)
	defer idx.Close()

	err := idx.SetEfSearch(10)
	if err == nil {
		t.Error("SetEfSearch() should return error (not supported)")
	}
}

// ========================================
// IndexIVFScalarQuantizer Tests
// ========================================

func TestNewIndexIVFScalarQuantizer(t *testing.T) {
	quantizer, err := NewIndexFlatL2(64)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}
	defer quantizer.Close()

	idx, err := NewIndexIVFScalarQuantizer(quantizer, 64, 10, QT_8bit, MetricL2)
	if err != nil {
		t.Fatalf("NewIndexIVFScalarQuantizer() failed: %v", err)
	}
	defer idx.Close()

	if idx.D() != 64 {
		t.Errorf("D() = %d, want 64", idx.D())
	}
	if idx.Nlist() != 10 {
		t.Errorf("Nlist() = %d, want 10", idx.Nlist())
	}
	if idx.Nprobe() != 1 {
		t.Errorf("Nprobe() = %d, want 1 (default)", idx.Nprobe())
	}
	if idx.QuantizerType() != QT_8bit {
		t.Errorf("QuantizerType() = %v, want QT_8bit", idx.QuantizerType())
	}
	if idx.MetricType() != MetricL2 {
		t.Errorf("MetricType() = %v, want MetricL2", idx.MetricType())
	}
}

func TestNewIndexIVFScalarQuantizer_NilQuantizer(t *testing.T) {
	_, err := NewIndexIVFScalarQuantizer(nil, 64, 10, QT_8bit, MetricL2)
	if err == nil {
		t.Error("NewIndexIVFScalarQuantizer(nil, ...) should return error")
	}
}

func TestIndexIVFScalarQuantizer_TrainAndAdd(t *testing.T) {
	quantizer, _ := NewIndexFlatL2(4)
	defer quantizer.Close()

	idx, _ := NewIndexIVFScalarQuantizer(quantizer, 4, 5, QT_8bit, MetricL2)
	defer idx.Close()

	// Train with enough vectors (need at least nlist * 39 = 195 vectors)
	trainVectors := make([]float32, 4*200)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 50)
	}

	err := idx.Train(trainVectors)
	if err != nil {
		t.Fatalf("Train() failed: %v", err)
	}

	if !idx.IsTrained() {
		t.Error("IsTrained() = false after Train()")
	}

	// Add vectors
	err = idx.Add([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	if err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	if idx.Ntotal() != 2 {
		t.Errorf("Ntotal() = %d, want 2", idx.Ntotal())
	}
}

func TestIndexIVFScalarQuantizer_SetNprobe(t *testing.T) {
	quantizer, _ := NewIndexFlatL2(4)
	defer quantizer.Close()

	idx, _ := NewIndexIVFScalarQuantizer(quantizer, 4, 10, QT_8bit, MetricL2)
	defer idx.Close()

	err := idx.SetNprobe(5)
	if err != nil {
		t.Fatalf("SetNprobe(5) failed: %v", err)
	}

	if idx.Nprobe() != 5 {
		t.Errorf("Nprobe() = %d, want 5", idx.Nprobe())
	}
}

func TestIndexIVFScalarQuantizer_SetNprobe_Invalid(t *testing.T) {
	quantizer, _ := NewIndexFlatL2(4)
	defer quantizer.Close()

	idx, _ := NewIndexIVFScalarQuantizer(quantizer, 4, 10, QT_8bit, MetricL2)
	defer idx.Close()

	err := idx.SetNprobe(0)
	if err == nil {
		t.Error("SetNprobe(0) should error")
	}

	err = idx.SetNprobe(100)
	if err == nil {
		t.Error("SetNprobe(100) should error (nlist=10)")
	}
}

func TestIndexIVFScalarQuantizer_Search(t *testing.T) {
	quantizer, _ := NewIndexFlatL2(4)
	defer quantizer.Close()

	idx, _ := NewIndexIVFScalarQuantizer(quantizer, 4, 5, QT_8bit, MetricL2)
	defer idx.Close()

	// Train (need at least nlist * 39 = 195 vectors)
	trainVectors := make([]float32, 4*200)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 50)
	}
	idx.Train(trainVectors)
	idx.SetNprobe(5) // Search all clusters

	// Add
	vectors := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	idx.Add(vectors)

	// Search
	query := []float32{1, 0, 0, 0}
	distances, indices, err := idx.Search(query, 2)
	if err != nil {
		t.Fatalf("Search() failed: %v", err)
	}

	if len(distances) != 2 || len(indices) != 2 {
		t.Errorf("Search() returned %d distances, %d indices, want 2 each",
			len(distances), len(indices))
	}
}

func TestIndexIVFScalarQuantizer_Train_NotEnoughVectors(t *testing.T) {
	quantizer, _ := NewIndexFlatL2(4)
	defer quantizer.Close()

	idx, _ := NewIndexIVFScalarQuantizer(quantizer, 4, 10, QT_8bit, MetricL2)
	defer idx.Close()

	// Only 5 vectors but 10 clusters
	trainVectors := make([]float32, 4*5)
	err := idx.Train(trainVectors)
	if err == nil {
		t.Error("Train() with fewer vectors than clusters should error")
	}
}

func TestIndexIVFScalarQuantizer_CompressionRatio(t *testing.T) {
	quantizer, _ := NewIndexFlatL2(64)
	defer quantizer.Close()

	idx, _ := NewIndexIVFScalarQuantizer(quantizer, 64, 10, QT_8bit, MetricL2)
	defer idx.Close()

	ratio := idx.CompressionRatio()
	if ratio != 4.0 {
		t.Errorf("CompressionRatio() = %f, want 4.0", ratio)
	}
}

func TestIndexIVFScalarQuantizer_SetEfSearch(t *testing.T) {
	quantizer, _ := NewIndexFlatL2(4)
	defer quantizer.Close()

	idx, _ := NewIndexIVFScalarQuantizer(quantizer, 4, 10, QT_8bit, MetricL2)
	defer idx.Close()

	err := idx.SetEfSearch(10)
	if err == nil {
		t.Error("SetEfSearch() should return error (not HNSW)")
	}
}

// ========================================
// Interface Compliance Tests
// ========================================

func TestIndexScalarQuantizer_ImplementsIndex(t *testing.T) {
	var _ Index = (*IndexScalarQuantizer)(nil)
}

func TestIndexIVFScalarQuantizer_ImplementsIndex(t *testing.T) {
	var _ Index = (*IndexIVFScalarQuantizer)(nil)
}
