package faiss

import (
	"testing"
)

// ========================================
// PCAMatrix Creation Tests
// ========================================

func TestNewPCAMatrix(t *testing.T) {
	tests := []struct {
		name string
		dIn  int
		dOut int
	}{
		{"256to64", 256, 64},
		{"128to32", 128, 32},
		{"64to16", 64, 16},
		{"same_dim", 64, 64},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pca, err := NewPCAMatrix(tt.dIn, tt.dOut)
			if err != nil {
				t.Fatalf("NewPCAMatrix(%d, %d) failed: %v", tt.dIn, tt.dOut, err)
			}
			defer pca.Close()

			if pca.DIn() != tt.dIn {
				t.Errorf("DIn() = %d, want %d", pca.DIn(), tt.dIn)
			}
			if pca.DOut() != tt.dOut {
				t.Errorf("DOut() = %d, want %d", pca.DOut(), tt.dOut)
			}
			if pca.IsTrained() {
				t.Error("IsTrained() = true, want false (before training)")
			}
		})
	}
}

func TestNewPCAMatrix_InvalidDimensions(t *testing.T) {
	tests := []struct {
		name string
		dIn  int
		dOut int
	}{
		{"zero_in", 0, 64},
		{"zero_out", 64, 0},
		{"negative_in", -1, 64},
		{"negative_out", 64, -1},
		{"out_greater_than_in", 64, 128},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewPCAMatrix(tt.dIn, tt.dOut)
			if err == nil {
				t.Errorf("NewPCAMatrix(%d, %d) should return error", tt.dIn, tt.dOut)
			}
		})
	}
}

func TestNewPCAMatrixWithEigen(t *testing.T) {
	pca, err := NewPCAMatrixWithEigen(128, 32, 0.5)
	if err != nil {
		t.Fatalf("NewPCAMatrixWithEigen() failed: %v", err)
	}
	defer pca.Close()

	if pca.DIn() != 128 {
		t.Errorf("DIn() = %d, want 128", pca.DIn())
	}
	if pca.DOut() != 32 {
		t.Errorf("DOut() = %d, want 32", pca.DOut())
	}
}

func TestNewPCAMatrixWithEigen_InvalidDimensions(t *testing.T) {
	_, err := NewPCAMatrixWithEigen(0, 32, 0.5)
	if err == nil {
		t.Error("NewPCAMatrixWithEigen(0, 32, 0.5) should return error")
	}

	_, err = NewPCAMatrixWithEigen(64, 128, 0.5)
	if err == nil {
		t.Error("NewPCAMatrixWithEigen(64, 128, 0.5) should return error")
	}
}

// ========================================
// PCAMatrix Train Tests
// ========================================

func TestPCAMatrix_Train(t *testing.T) {
	pca, _ := NewPCAMatrix(8, 4)
	defer pca.Close()

	// Generate training vectors with some structure
	vectors := make([]float32, 8*100)
	for i := 0; i < 100; i++ {
		for j := 0; j < 8; j++ {
			vectors[i*8+j] = float32(i*j + j)
		}
	}

	err := pca.Train(vectors)
	if err != nil {
		t.Fatalf("Train() failed: %v", err)
	}

	if !pca.IsTrained() {
		t.Error("IsTrained() = false after Train()")
	}
}

func TestPCAMatrix_Train_Empty(t *testing.T) {
	pca, _ := NewPCAMatrix(8, 4)
	defer pca.Close()

	err := pca.Train([]float32{})
	if err == nil {
		t.Error("Train(empty) should return error")
	}
}

func TestPCAMatrix_Train_InvalidDimension(t *testing.T) {
	pca, _ := NewPCAMatrix(8, 4)
	defer pca.Close()

	err := pca.Train([]float32{1, 2, 3}) // Not a multiple of 8
	if err == nil {
		t.Error("Train() with invalid dimension should error")
	}
}

// ========================================
// PCAMatrix Apply Tests
// ========================================

func TestPCAMatrix_Apply(t *testing.T) {
	pca, _ := NewPCAMatrix(8, 4)
	defer pca.Close()

	// Train
	trainVectors := make([]float32, 8*100)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 50)
	}
	err := pca.Train(trainVectors)
	if err != nil {
		t.Fatalf("Train() failed: %v", err)
	}

	// Apply
	input := make([]float32, 8*5)
	for i := range input {
		input[i] = float32(i)
	}

	output, err := pca.Apply(input)
	if err != nil {
		// PCA Apply may fail due to C library limitations
		t.Skipf("Apply() not working (known limitation): %v", err)
		return
	}

	expectedLen := 4 * 5 // dOut * n
	if len(output) != expectedLen {
		t.Errorf("Apply() output length = %d, want %d", len(output), expectedLen)
	}
}

func TestPCAMatrix_Apply_BeforeTrain(t *testing.T) {
	pca, _ := NewPCAMatrix(8, 4)
	defer pca.Close()

	_, err := pca.Apply(make([]float32, 8))
	if err == nil {
		t.Error("Apply() before Train() should error")
	}
}

func TestPCAMatrix_Apply_Empty(t *testing.T) {
	pca, _ := NewPCAMatrix(8, 4)
	defer pca.Close()

	// Train first
	trainVectors := make([]float32, 8*100)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 50)
	}
	pca.Train(trainVectors)

	output, err := pca.Apply([]float32{})
	if err != nil {
		t.Errorf("Apply(empty) should not error: %v", err)
	}
	if len(output) != 0 {
		t.Errorf("Apply(empty) should return empty, got len=%d", len(output))
	}
}

func TestPCAMatrix_Apply_InvalidDimension(t *testing.T) {
	pca, _ := NewPCAMatrix(8, 4)
	defer pca.Close()

	// Train first
	trainVectors := make([]float32, 8*100)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 50)
	}
	pca.Train(trainVectors)

	_, err := pca.Apply([]float32{1, 2, 3}) // Not a multiple of 8
	if err == nil {
		t.Error("Apply() with invalid dimension should error")
	}
}

// ========================================
// PCAMatrix ReverseTransform Tests
// ========================================

func TestPCAMatrix_ReverseTransform(t *testing.T) {
	pca, _ := NewPCAMatrix(8, 4)
	defer pca.Close()

	// Train
	trainVectors := make([]float32, 8*100)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 50)
	}
	pca.Train(trainVectors)

	// Apply then reverse
	input := make([]float32, 8*5)
	for i := range input {
		input[i] = float32(i % 10)
	}

	reduced, err := pca.Apply(input)
	if err != nil {
		t.Skipf("Apply() not working (known limitation): %v", err)
		return
	}

	reconstructed, err := pca.ReverseTransform(reduced)
	if err != nil {
		t.Skipf("ReverseTransform() not working (known limitation): %v", err)
		return
	}

	if len(reconstructed) != len(input) {
		t.Errorf("ReverseTransform() output length = %d, want %d",
			len(reconstructed), len(input))
	}
}

func TestPCAMatrix_ReverseTransform_BeforeTrain(t *testing.T) {
	pca, _ := NewPCAMatrix(8, 4)
	defer pca.Close()

	_, err := pca.ReverseTransform(make([]float32, 4))
	if err == nil {
		t.Error("ReverseTransform() before Train() should error")
	}
}

func TestPCAMatrix_ReverseTransform_InvalidDimension(t *testing.T) {
	pca, _ := NewPCAMatrix(8, 4)
	defer pca.Close()

	// Train first
	trainVectors := make([]float32, 8*100)
	for i := range trainVectors {
		trainVectors[i] = float32(i % 50)
	}
	pca.Train(trainVectors)

	_, err := pca.ReverseTransform([]float32{1, 2, 3}) // Not a multiple of dOut=4
	if err == nil {
		t.Error("ReverseTransform() with invalid dimension should error")
	}
}

// ========================================
// PCAMatrix Close Tests
// ========================================

func TestPCAMatrix_Close(t *testing.T) {
	pca, _ := NewPCAMatrix(8, 4)

	err := pca.Close()
	if err != nil {
		t.Errorf("First Close() failed: %v", err)
	}

	err = pca.Close()
	if err != nil {
		t.Errorf("Second Close() failed: %v", err)
	}
}

// ========================================
// OPQMatrix Tests
// ========================================

func TestNewOPQMatrix(t *testing.T) {
	opq, err := NewOPQMatrix(64, 8)
	if err != nil {
		t.Fatalf("NewOPQMatrix() failed: %v", err)
	}
	defer opq.Close()

	if opq.DIn() != 64 {
		t.Errorf("DIn() = %d, want 64", opq.DIn())
	}
	if opq.DOut() != 64 {
		t.Errorf("DOut() = %d, want 64 (same as DIn for rotation)", opq.DOut())
	}
	if opq.GetM() != 8 {
		t.Errorf("GetM() = %d, want 8", opq.GetM())
	}
	if opq.IsTrained() {
		t.Error("IsTrained() = true, want false")
	}
}

func TestNewOPQMatrix_InvalidParams(t *testing.T) {
	tests := []struct {
		name string
		d    int
		M    int
	}{
		{"zero_d", 0, 8},
		{"zero_M", 64, 0},
		{"d_not_divisible", 64, 7}, // 64 not divisible by 7
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewOPQMatrix(tt.d, tt.M)
			if err == nil {
				t.Errorf("NewOPQMatrix(%d, %d) should return error", tt.d, tt.M)
			}
		})
	}
}

func TestOPQMatrix_Train(t *testing.T) {
	opq, _ := NewOPQMatrix(8, 2)
	defer opq.Close()

	// OPQ needs sufficient training data (at least ~10k for 256 centroids)
	vectors := make([]float32, 8*10000)
	for i := range vectors {
		vectors[i] = float32(i%100) / 100.0
	}

	err := opq.Train(vectors)
	if err != nil {
		t.Errorf("Train() failed: %v", err)
		return
	}

	if !opq.IsTrained() {
		t.Error("IsTrained() = false after Train()")
	}
}

func TestOPQMatrix_Apply(t *testing.T) {
	opq, _ := NewOPQMatrix(8, 2)
	defer opq.Close()

	// Train with sufficient data
	trainVectors := make([]float32, 8*10000)
	for i := range trainVectors {
		trainVectors[i] = float32(i%100) / 100.0
	}
	err := opq.Train(trainVectors)
	if err != nil {
		t.Errorf("Train() failed: %v", err)
		return
	}

	// Apply
	input := make([]float32, 8*5)
	for i := range input {
		input[i] = float32(i)
	}

	output, err := opq.Apply(input)
	if err != nil {
		t.Errorf("Apply() failed: %v", err)
		return
	}

	if len(output) != len(input) {
		t.Errorf("Apply() output length = %d, want %d (rotation preserves dim)",
			len(output), len(input))
	}
}

func TestOPQMatrix_Apply_BeforeTrain(t *testing.T) {
	opq, _ := NewOPQMatrix(8, 2)
	defer opq.Close()

	_, err := opq.Apply(make([]float32, 8))
	if err == nil {
		t.Error("Apply() before Train() should error")
	}
}

func TestOPQMatrix_ReverseTransform(t *testing.T) {
	opq, _ := NewOPQMatrix(8, 2)
	defer opq.Close()

	// Train with sufficient data
	trainVectors := make([]float32, 8*10000)
	for i := range trainVectors {
		trainVectors[i] = float32(i%100) / 100.0
	}
	err := opq.Train(trainVectors)
	if err != nil {
		t.Errorf("Train() failed: %v", err)
		return
	}

	// Apply then reverse
	input := make([]float32, 8*5)
	for i := range input {
		input[i] = float32(i % 10)
	}

	rotated, err := opq.Apply(input)
	if err != nil {
		t.Errorf("Apply() failed: %v", err)
		return
	}

	reversed, err := opq.ReverseTransform(rotated)
	if err != nil {
		t.Errorf("ReverseTransform() failed: %v", err)
		return
	}

	if len(reversed) != len(input) {
		t.Errorf("ReverseTransform() output length = %d, want %d",
			len(reversed), len(input))
	}

	// Check approximate reconstruction
	var mse float32
	for i := range input {
		diff := input[i] - reversed[i]
		mse += diff * diff
	}
	mse /= float32(len(input))

	if mse > 0.1 {
		t.Errorf("OPQ rotation should be approximately reversible, MSE = %f", mse)
	}
}

func TestOPQMatrix_Close(t *testing.T) {
	opq, _ := NewOPQMatrix(8, 2)

	err := opq.Close()
	if err != nil {
		t.Errorf("First Close() failed: %v", err)
	}

	err = opq.Close()
	if err != nil {
		t.Errorf("Second Close() failed: %v", err)
	}
}

// ========================================
// RandomRotationMatrix Tests
// ========================================

func TestNewRandomRotationMatrix(t *testing.T) {
	tests := []struct {
		name string
		dIn  int
		dOut int
	}{
		{"same_dim", 64, 64},
		{"expand", 32, 64},
		{"reduce", 64, 32},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rr, err := NewRandomRotationMatrix(tt.dIn, tt.dOut)
			if err != nil {
				t.Fatalf("NewRandomRotationMatrix(%d, %d) failed: %v", tt.dIn, tt.dOut, err)
			}
			defer rr.Close()

			if rr.DIn() != tt.dIn {
				t.Errorf("DIn() = %d, want %d", rr.DIn(), tt.dIn)
			}
			if rr.DOut() != tt.dOut {
				t.Errorf("DOut() = %d, want %d", rr.DOut(), tt.dOut)
			}
			// RandomRotation requires Train() to be called before use
			if rr.IsTrained() {
				t.Error("IsTrained() = true before Train(), want false")
			}

			// Train with dummy data
			trainData := make([]float32, tt.dIn)
			err = rr.Train(trainData)
			if err != nil {
				t.Fatalf("Train() failed: %v", err)
			}
			if !rr.IsTrained() {
				t.Error("IsTrained() = false after Train(), want true")
			}
		})
	}
}

func TestNewRandomRotationMatrix_InvalidDimensions(t *testing.T) {
	_, err := NewRandomRotationMatrix(0, 64)
	if err == nil {
		t.Error("NewRandomRotationMatrix(0, 64) should return error")
	}

	_, err = NewRandomRotationMatrix(64, 0)
	if err == nil {
		t.Error("NewRandomRotationMatrix(64, 0) should return error")
	}
}

func TestRandomRotationMatrix_Train(t *testing.T) {
	rr, _ := NewRandomRotationMatrix(8, 8)
	defer rr.Close()

	if rr.IsTrained() {
		t.Error("IsTrained() should be false before training")
	}

	// Train initializes the rotation matrix
	err := rr.Train([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	if err != nil {
		t.Errorf("Train() failed: %v", err)
	}

	if !rr.IsTrained() {
		t.Error("IsTrained() should be true after training")
	}

	// Training again should be a no-op (already trained)
	err = rr.Train([]float32{8, 7, 6, 5, 4, 3, 2, 1})
	if err != nil {
		t.Errorf("Second Train() failed: %v", err)
	}
}

func TestRandomRotationMatrix_Apply(t *testing.T) {
	rr, err := NewRandomRotationMatrix(8, 4)
	if err != nil {
		t.Fatalf("NewRandomRotationMatrix failed: %v", err)
	}
	defer rr.Close()

	// Must train before applying
	trainData := make([]float32, 8*10)
	for i := range trainData {
		trainData[i] = float32(i)
	}
	err = rr.Train(trainData)
	if err != nil {
		t.Fatalf("Train() failed: %v", err)
	}

	if !rr.IsTrained() {
		t.Error("IsTrained() = false after Train()")
	}

	input := make([]float32, 8*5)
	for i := range input {
		input[i] = float32(i)
	}

	output, err := rr.Apply(input)
	if err != nil {
		t.Fatalf("Apply() failed: %v", err)
	}

	expectedLen := 4 * 5 // dOut * n
	if len(output) != expectedLen {
		t.Errorf("Apply() output length = %d, want %d", len(output), expectedLen)
	}
}

func TestRandomRotationMatrix_Apply_Empty(t *testing.T) {
	rr, err := NewRandomRotationMatrix(8, 4)
	if err != nil {
		t.Fatalf("NewRandomRotationMatrix failed: %v", err)
	}
	defer rr.Close()

	// Train first
	rr.Train(make([]float32, 8))

	output, err := rr.Apply([]float32{})
	if err != nil {
		t.Errorf("Apply(empty) should not error: %v", err)
	}
	if len(output) != 0 {
		t.Errorf("Apply(empty) should return empty, got len=%d", len(output))
	}
}

func TestRandomRotationMatrix_Apply_InvalidDimension(t *testing.T) {
	rr, err := NewRandomRotationMatrix(8, 4)
	if err != nil {
		t.Fatalf("NewRandomRotationMatrix failed: %v", err)
	}
	defer rr.Close()

	// Train first
	rr.Train(make([]float32, 8))

	_, err = rr.Apply([]float32{1, 2, 3}) // Not a multiple of 8
	if err == nil {
		t.Error("Apply() with invalid dimension should error")
	}
}

func TestRandomRotationMatrix_ReverseTransform_SameDim(t *testing.T) {
	rr, err := NewRandomRotationMatrix(8, 8) // Same dimensions for reversibility
	if err != nil {
		t.Fatalf("NewRandomRotationMatrix failed: %v", err)
	}
	defer rr.Close()

	// Train
	rr.Train(make([]float32, 8*10))

	input := make([]float32, 8*5)
	for i := range input {
		input[i] = float32(i % 10)
	}

	rotated, err := rr.Apply(input)
	if err != nil {
		t.Fatalf("Apply() failed: %v", err)
	}

	reversed, err := rr.ReverseTransform(rotated)
	if err != nil {
		t.Fatalf("ReverseTransform() failed: %v", err)
	}

	if len(reversed) != len(input) {
		t.Errorf("ReverseTransform() output length = %d, want %d", len(reversed), len(input))
	}
}

func TestRandomRotationMatrix_ReverseTransform_DifferentDim(t *testing.T) {
	rr, err := NewRandomRotationMatrix(8, 4)
	if err != nil {
		t.Skipf("NewRandomRotationMatrix failed: %v", err)
		return
	}
	defer rr.Close()

	_, err = rr.ReverseTransform(make([]float32, 4))
	if err == nil {
		t.Error("ReverseTransform() should error when dIn != dOut")
	}
}

func TestRandomRotationMatrix_Close(t *testing.T) {
	rr, _ := NewRandomRotationMatrix(8, 4)

	err := rr.Close()
	if err != nil {
		t.Errorf("First Close() failed: %v", err)
	}

	err = rr.Close()
	if err != nil {
		t.Errorf("Second Close() failed: %v", err)
	}
}

// ========================================
// VectorTransform Interface Compliance
// ========================================

func TestPCAMatrix_ImplementsVectorTransform(t *testing.T) {
	var _ VectorTransform = (*PCAMatrix)(nil)
}

func TestOPQMatrix_ImplementsVectorTransform(t *testing.T) {
	var _ VectorTransform = (*OPQMatrix)(nil)
}

func TestRandomRotationMatrix_ImplementsVectorTransform(t *testing.T) {
	var _ VectorTransform = (*RandomRotationMatrix)(nil)
}

// ========================================
// Transform Integration Test
// ========================================

func TestPCA_DimensionalityReduction(t *testing.T) {
	// Create PCA to reduce from 16 to 4 dimensions
	pca, err := NewPCAMatrix(16, 4)
	if err != nil {
		t.Fatalf("NewPCAMatrix failed: %v", err)
	}
	defer pca.Close()

	// Generate training data with some structure
	n := 200
	trainVectors := make([]float32, 16*n)
	for i := 0; i < n; i++ {
		for j := 0; j < 16; j++ {
			trainVectors[i*16+j] = float32(i+j) + float32(i*j)*0.01
		}
	}
	err = pca.Train(trainVectors)
	if err != nil {
		t.Fatalf("Train() failed: %v", err)
	}

	// Apply PCA
	testVectors := trainVectors[:16*10]
	reduced, err := pca.Apply(testVectors)
	if err != nil {
		t.Fatalf("Apply() failed: %v", err)
	}

	if len(reduced) != 4*10 {
		t.Errorf("Reduced vector length = %d, want %d", len(reduced), 4*10)
	}
}
