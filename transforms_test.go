package faiss

import (
	"testing"
)

// ========================================
// PCAMatrix Tests
// ========================================

func TestPCAMatrix(t *testing.T) {
	dIn := 128
	dOut := 64
	nb := 500

	pca, err := NewPCAMatrix(dIn, dOut)
	if err != nil {
		t.Fatalf("Failed to create PCAMatrix: %v", err)
	}
	defer pca.Close()

	// Check properties
	if pca.DIn() != dIn {
		t.Errorf("Expected dIn=%d, got %d", dIn, pca.DIn())
	}
	if pca.DOut() != dOut {
		t.Errorf("Expected dOut=%d, got %d", dOut, pca.DOut())
	}
	if pca.IsTrained() {
		t.Error("PCA should not be trained initially")
	}

	// Train
	trainingVectors := generateVectors(nb, dIn)
	if err := pca.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}
	if !pca.IsTrained() {
		t.Error("PCA should be trained after training")
	}

	// Apply transformation
	testVectors := generateVectors(10, dIn)
	transformed, err := pca.Apply(testVectors)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}

	expectedLen := 10 * dOut
	if len(transformed) != expectedLen {
		t.Errorf("Expected %d output elements, got %d", expectedLen, len(transformed))
	}

	// Reverse transformation
	reversed, err := pca.ReverseTransform(transformed)
	if err != nil {
		t.Fatalf("ReverseTransform failed: %v", err)
	}

	if len(reversed) != 10*dIn {
		t.Errorf("Expected %d reversed elements, got %d", 10*dIn, len(reversed))
	}
}

func TestPCAMatrixWithEigenPower(t *testing.T) {
	dIn := 64
	dOut := 32
	eigenPower := 0.5
	nb := 500

	pca, err := NewPCAMatrixWithEigenPower(dIn, dOut, eigenPower)
	if err != nil {
		t.Fatalf("Failed to create PCA with eigen power: %v", err)
	}
	defer pca.Close()

	trainingVectors := generateVectors(nb, dIn)
	if err := pca.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	testVectors := generateVectors(5, dIn)
	_, err = pca.Apply(testVectors)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
}

func TestPCAInvalidDimensions(t *testing.T) {
	_, err := NewPCAMatrix(64, 128)
	if err == nil {
		t.Error("Expected error for dOut > dIn")
	}

	_, err = NewPCAMatrix(0, 32)
	if err == nil {
		t.Error("Expected error for invalid dIn")
	}
}

func TestPCAApplyBeforeTrain(t *testing.T) {
	pca, _ := NewPCAMatrix(64, 32)
	defer pca.Close()

	vectors := generateVectors(10, 64)
	_, err := pca.Apply(vectors)
	if err == nil {
		t.Error("Expected error when applying before training")
	}
}

// ========================================
// OPQMatrix Tests
// ========================================

func TestOPQMatrix(t *testing.T) {
	d := 64
	M := 8
	nb := 500

	opq, err := NewOPQMatrix(d, M)
	if err != nil {
		t.Fatalf("Failed to create OPQMatrix: %v", err)
	}
	defer opq.Close()

	if opq.DIn() != d || opq.DOut() != d {
		t.Error("OPQ should preserve dimension")
	}
	if opq.GetM() != M {
		t.Errorf("Expected M=%d, got %d", M, opq.GetM())
	}

	// Train
	trainingVectors := generateVectors(nb, d)
	if err := opq.Train(trainingVectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Apply
	testVectors := generateVectors(10, d)
	transformed, err := opq.Apply(testVectors)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}

	if len(transformed) != len(testVectors) {
		t.Error("OPQ should preserve vector count and dimension")
	}
}

func TestOPQInvalidParameters(t *testing.T) {
	_, err := NewOPQMatrix(63, 8)
	if err == nil {
		t.Error("Expected error for d not divisible by M")
	}

	_, err = NewOPQMatrix(64, 0)
	if err == nil {
		t.Error("Expected error for invalid M")
	}
}

// ========================================
// RandomRotationMatrix Tests
// ========================================

func TestRandomRotationMatrix(t *testing.T) {
	d := 64
	nb := 100

	rotation, err := NewRandomRotationMatrix(d, d)
	if err != nil {
		t.Fatalf("Failed to create RandomRotationMatrix: %v", err)
	}
	defer rotation.Close()

	if rotation.DIn() != d || rotation.DOut() != d {
		t.Error("Random rotation should preserve dimension")
	}

	// Random rotation doesn't need training
	if !rotation.IsTrained() {
		t.Error("Random rotation should be trained by default")
	}

	// Apply
	testVectors := generateVectors(nb, d)
	rotated, err := rotation.Apply(testVectors)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}

	if len(rotated) != len(testVectors) {
		t.Error("Rotation should preserve vector count and dimension")
	}

	// Verify rotation preserves L2 norm (approximately)
	// For a single vector
	vec := testVectors[:d]
	rotVec := rotated[:d]

	origNorm := float32(0)
	rotNorm := float32(0)
	for i := 0; i < d; i++ {
		origNorm += vec[i] * vec[i]
		rotNorm += rotVec[i] * rotVec[i]
	}

	if !almostEqual(origNorm, rotNorm, 0.01) {
		t.Errorf("Rotation should preserve L2 norm: %f vs %f", origNorm, rotNorm)
	}
}

func TestRandomRotationInvalidDimensions(t *testing.T) {
	_, err := NewRandomRotationMatrix(0, 64)
	if err == nil {
		t.Error("Expected error for invalid dimensions")
	}
}
