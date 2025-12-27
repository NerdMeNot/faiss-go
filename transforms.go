package faiss

import (
	"fmt"
	"runtime"
)

// VectorTransform is an interface for vector transformations (PCA, OPQ, etc.)
type VectorTransform interface {
	DIn() int     // input dimension
	DOut() int    // output dimension
	IsTrained() bool
	Train(vectors []float32) error
	Apply(vectors []float32) ([]float32, error)
	ReverseTransform(vectors []float32) ([]float32, error)
	Close() error
}

// ========================================
// PCAMatrix - Principal Component Analysis
// ========================================

// PCAMatrix performs dimensionality reduction using PCA
//
// Python equivalent: faiss.PCAMatrix
//
// Example:
//   // Reduce from 256 to 64 dimensions
//   pca, _ := faiss.NewPCAMatrix(256, 64)
//   pca.Train(trainingVectors)
//
//   // Apply to vectors
//   reduced, _ := pca.Apply(vectors)
//
//   // Reconstruct (approximate) original
//   reconstructed, _ := pca.ReverseTransform(reduced)
type PCAMatrix struct {
	ptr       uintptr // C pointer
	dIn       int     // input dimension
	dOut      int     // output dimension
	isTrained bool    // training status
	_eigenVec []float32 // eigenvectors (reserved for future introspection)
	_eigenVal []float32 // eigenvalues (reserved for future introspection)
	_mean     []float32 // mean vector (reserved for future introspection)
}

// NewPCAMatrix creates a new PCA transformation matrix
func NewPCAMatrix(dIn, dOut int) (*PCAMatrix, error) {
	if dIn <= 0 || dOut <= 0 || dOut > dIn {
		return nil, fmt.Errorf("invalid dimensions: dIn=%d, dOut=%d (need 0 < dOut <= dIn)", dIn, dOut)
	}

	var ptr uintptr
	ret := faiss_PCAMatrix_new(&ptr, int64(dIn), int64(dOut), 0, 0)
	if ret != 0 {
		return nil, fmt.Errorf("failed to create PCAMatrix")
	}

	pca := &PCAMatrix{
		ptr:       ptr,
		dIn:       dIn,
		dOut:      dOut,
		isTrained: false,
	}

	runtime.SetFinalizer(pca, func(p *PCAMatrix) {
		p.Close()
	})

	return pca, nil
}

// NewPCAMatrixWithEigen creates a PCA matrix that also computes eigenvalues
func NewPCAMatrixWithEigen(dIn, dOut int, eigenPower float32) (*PCAMatrix, error) {
	if dIn <= 0 || dOut <= 0 || dOut > dIn {
		return nil, fmt.Errorf("invalid dimensions: dIn=%d, dOut=%d (need 0 < dOut <= dIn)", dIn, dOut)
	}

	var ptr uintptr
	ret := faiss_PCAMatrix_new(&ptr, int64(dIn), int64(dOut), eigenPower, 1)
	if ret != 0 {
		return nil, fmt.Errorf("failed to create PCAMatrix")
	}

	pca := &PCAMatrix{
		ptr:       ptr,
		dIn:       dIn,
		dOut:      dOut,
		isTrained: false,
	}

	runtime.SetFinalizer(pca, func(p *PCAMatrix) {
		p.Close()
	})

	return pca, nil
}

// DIn returns the input dimension
func (pca *PCAMatrix) DIn() int {
	return pca.dIn
}

// DOut returns the output dimension
func (pca *PCAMatrix) DOut() int {
	return pca.dOut
}

// IsTrained returns whether the PCA has been trained
func (pca *PCAMatrix) IsTrained() bool {
	return pca.isTrained
}

// Train trains the PCA on the given vectors
func (pca *PCAMatrix) Train(vectors []float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("empty training vectors")
	}
	if len(vectors)%pca.dIn != 0 {
		return fmt.Errorf("vectors length must be multiple of input dimension %d", pca.dIn)
	}

	n := int64(len(vectors) / pca.dIn)
	ret := faiss_VectorTransform_train(pca.ptr, n, &vectors[0])
	if ret != 0 {
		return fmt.Errorf("PCA training failed")
	}

	pca.isTrained = true
	return nil
}

// Apply applies the PCA transformation to vectors
func (pca *PCAMatrix) Apply(vectors []float32) ([]float32, error) {
	if !pca.isTrained {
		return nil, fmt.Errorf("PCA must be trained before applying")
	}
	if len(vectors) == 0 {
		return []float32{}, nil
	}
	if len(vectors)%pca.dIn != 0 {
		return nil, fmt.Errorf("vectors length must be multiple of input dimension %d", pca.dIn)
	}

	n := len(vectors) / pca.dIn
	output := make([]float32, n*pca.dOut)

	ret := faiss_VectorTransform_apply(pca.ptr, int64(n), &vectors[0], &output[0])
	if ret != 0 {
		return nil, fmt.Errorf("PCA apply failed")
	}

	return output, nil
}

// ReverseTransform attempts to reverse the PCA transformation (approximate reconstruction)
func (pca *PCAMatrix) ReverseTransform(vectors []float32) ([]float32, error) {
	if !pca.isTrained {
		return nil, fmt.Errorf("PCA must be trained before reverse transform")
	}
	if len(vectors) == 0 {
		return []float32{}, nil
	}
	if len(vectors)%pca.dOut != 0 {
		return nil, fmt.Errorf("vectors length must be multiple of output dimension %d", pca.dOut)
	}

	n := len(vectors) / pca.dOut
	output := make([]float32, n*pca.dIn)

	ret := faiss_VectorTransform_reverse_transform(pca.ptr, int64(n), &vectors[0], &output[0])
	if ret != 0 {
		return nil, fmt.Errorf("PCA reverse transform failed")
	}

	return output, nil
}

// Close frees the PCA matrix
func (pca *PCAMatrix) Close() error {
	if pca.ptr != 0 {
		faiss_VectorTransform_free(pca.ptr)
		pca.ptr = 0
	}
	return nil
}

// ========================================
// OPQMatrix - Optimized Product Quantization
// ========================================

// OPQMatrix performs rotation for optimal product quantization
//
// Python equivalent: faiss.OPQMatrix
//
// Example:
//   // 128-dim vectors, 8 subspaces
//   opq, _ := faiss.NewOPQMatrix(128, 8)
//   opq.Train(trainingVectors)
//   rotated, _ := opq.Apply(vectors)
type OPQMatrix struct {
	ptr       uintptr // C pointer
	d         int     // dimension
	M         int     // number of subspaces
	isTrained bool    // training status
}

// NewOPQMatrix creates a new OPQ transformation matrix
func NewOPQMatrix(d, M int) (*OPQMatrix, error) {
	if d <= 0 || M <= 0 || d%M != 0 {
		return nil, fmt.Errorf("invalid parameters: d=%d, M=%d (d must be divisible by M)", d, M)
	}

	var ptr uintptr
	ret := faiss_OPQMatrix_new(&ptr, int64(d), int64(M))
	if ret != 0 {
		return nil, fmt.Errorf("failed to create OPQMatrix")
	}

	opq := &OPQMatrix{
		ptr:       ptr,
		d:         d,
		M:         M,
		isTrained: false,
	}

	runtime.SetFinalizer(opq, func(o *OPQMatrix) {
		o.Close()
	})

	return opq, nil
}

// DIn returns the input dimension
func (opq *OPQMatrix) DIn() int {
	return opq.d
}

// DOut returns the output dimension (same as input for rotation)
func (opq *OPQMatrix) DOut() int {
	return opq.d
}

// IsTrained returns whether the OPQ has been trained
func (opq *OPQMatrix) IsTrained() bool {
	return opq.isTrained
}

// GetM returns the number of subspaces
func (opq *OPQMatrix) GetM() int {
	return opq.M
}

// Train trains the OPQ on the given vectors
func (opq *OPQMatrix) Train(vectors []float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("empty training vectors")
	}
	if len(vectors)%opq.d != 0 {
		return fmt.Errorf("vectors length must be multiple of dimension %d", opq.d)
	}

	n := int64(len(vectors) / opq.d)
	ret := faiss_VectorTransform_train(opq.ptr, n, &vectors[0])
	if ret != 0 {
		return fmt.Errorf("OPQ training failed")
	}

	opq.isTrained = true
	return nil
}

// Apply applies the OPQ rotation to vectors
func (opq *OPQMatrix) Apply(vectors []float32) ([]float32, error) {
	if !opq.isTrained {
		return nil, fmt.Errorf("OPQ must be trained before applying")
	}
	if len(vectors) == 0 {
		return []float32{}, nil
	}
	if len(vectors)%opq.d != 0 {
		return nil, fmt.Errorf("vectors length must be multiple of dimension %d", opq.d)
	}

	n := len(vectors) / opq.d
	output := make([]float32, len(vectors))

	ret := faiss_VectorTransform_apply(opq.ptr, int64(n), &vectors[0], &output[0])
	if ret != 0 {
		return nil, fmt.Errorf("OPQ apply failed")
	}

	return output, nil
}

// ReverseTransform reverses the OPQ rotation
func (opq *OPQMatrix) ReverseTransform(vectors []float32) ([]float32, error) {
	if !opq.isTrained {
		return nil, fmt.Errorf("OPQ must be trained before reverse transform")
	}
	if len(vectors) == 0 {
		return []float32{}, nil
	}
	if len(vectors)%opq.d != 0 {
		return nil, fmt.Errorf("vectors length must be multiple of dimension %d", opq.d)
	}

	n := len(vectors) / opq.d
	output := make([]float32, len(vectors))

	ret := faiss_VectorTransform_reverse_transform(opq.ptr, int64(n), &vectors[0], &output[0])
	if ret != 0 {
		return nil, fmt.Errorf("OPQ reverse transform failed")
	}

	return output, nil
}

// Close frees the OPQ matrix
func (opq *OPQMatrix) Close() error {
	if opq.ptr != 0 {
		faiss_VectorTransform_free(opq.ptr)
		opq.ptr = 0
	}
	return nil
}

// ========================================
// RandomRotationMatrix
// ========================================

// RandomRotationMatrix performs a random rotation of vectors
//
// Python equivalent: faiss.RandomRotationMatrix
//
// Example:
//   rr, _ := faiss.NewRandomRotationMatrix(128, 128)
//   rotated, _ := rr.Apply(vectors)
type RandomRotationMatrix struct {
	ptr       uintptr // C pointer
	dIn       int     // input dimension
	dOut      int     // output dimension
	isTrained bool    // always true (no training needed)
}

// NewRandomRotationMatrix creates a new random rotation matrix
func NewRandomRotationMatrix(dIn, dOut int) (*RandomRotationMatrix, error) {
	if dIn <= 0 || dOut <= 0 {
		return nil, fmt.Errorf("invalid dimensions: dIn=%d, dOut=%d", dIn, dOut)
	}

	var ptr uintptr
	ret := faiss_RandomRotationMatrix_new(&ptr, int64(dIn), int64(dOut))
	if ret != 0 {
		return nil, fmt.Errorf("failed to create RandomRotationMatrix")
	}

	rr := &RandomRotationMatrix{
		ptr:       ptr,
		dIn:       dIn,
		dOut:      dOut,
		isTrained: true, // no training needed
	}

	runtime.SetFinalizer(rr, func(r *RandomRotationMatrix) {
		r.Close()
	})

	return rr, nil
}

// DIn returns the input dimension
func (rr *RandomRotationMatrix) DIn() int {
	return rr.dIn
}

// DOut returns the output dimension
func (rr *RandomRotationMatrix) DOut() int {
	return rr.dOut
}

// IsTrained returns whether the transform is ready (always true)
func (rr *RandomRotationMatrix) IsTrained() bool {
	return rr.isTrained
}

// Train is a no-op (random rotation doesn't need training)
func (rr *RandomRotationMatrix) Train(vectors []float32) error {
	return nil
}

// Apply applies the random rotation to vectors
func (rr *RandomRotationMatrix) Apply(vectors []float32) ([]float32, error) {
	if len(vectors) == 0 {
		return []float32{}, nil
	}
	if len(vectors)%rr.dIn != 0 {
		return nil, fmt.Errorf("vectors length must be multiple of input dimension %d", rr.dIn)
	}

	n := len(vectors) / rr.dIn
	output := make([]float32, n*rr.dOut)

	ret := faiss_VectorTransform_apply(rr.ptr, int64(n), &vectors[0], &output[0])
	if ret != 0 {
		return nil, fmt.Errorf("random rotation apply failed")
	}

	return output, nil
}

// ReverseTransform reverses the random rotation (if dIn == dOut)
func (rr *RandomRotationMatrix) ReverseTransform(vectors []float32) ([]float32, error) {
	if rr.dIn != rr.dOut {
		return nil, fmt.Errorf("reverse transform only available when dIn == dOut")
	}
	if len(vectors) == 0 {
		return []float32{}, nil
	}
	if len(vectors)%rr.dOut != 0 {
		return nil, fmt.Errorf("vectors length must be multiple of output dimension %d", rr.dOut)
	}

	n := len(vectors) / rr.dOut
	output := make([]float32, n*rr.dIn)

	ret := faiss_VectorTransform_reverse_transform(rr.ptr, int64(n), &vectors[0], &output[0])
	if ret != 0 {
		return nil, fmt.Errorf("random rotation reverse transform failed")
	}

	return output, nil
}

// Close frees the random rotation matrix
func (rr *RandomRotationMatrix) Close() error {
	if rr.ptr != 0 {
		faiss_VectorTransform_free(rr.ptr)
		rr.ptr = 0
	}
	return nil
}
