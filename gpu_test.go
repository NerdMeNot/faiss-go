//go:build gpu
// +build gpu

package faiss

import (
	"testing"
)

// ========================================
// StandardGpuResources Tests
// ========================================

func TestNewStandardGpuResources(t *testing.T) {
	res, err := NewStandardGpuResources()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer res.Close()

	if res.ptr == 0 {
		t.Error("Expected non-zero pointer")
	}
}

func TestStandardGpuResources_SetTempMemory(t *testing.T) {
	res, err := NewStandardGpuResources()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer res.Close()

	// Set temp memory to 256MB
	err = res.SetTempMemory(256 * 1024 * 1024)
	if err != nil {
		t.Errorf("SetTempMemory() failed: %v", err)
	}

	if res.GetTempMemory() != 256*1024*1024 {
		t.Errorf("GetTempMemory() = %d, want %d", res.GetTempMemory(), 256*1024*1024)
	}
}

func TestStandardGpuResources_SetTempMemory_Invalid(t *testing.T) {
	res, err := NewStandardGpuResources()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer res.Close()

	err = res.SetTempMemory(0)
	if err == nil {
		t.Error("SetTempMemory(0) should return error")
	}

	err = res.SetTempMemory(-1)
	if err == nil {
		t.Error("SetTempMemory(-1) should return error")
	}
}

func TestStandardGpuResources_Close(t *testing.T) {
	res, err := NewStandardGpuResources()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}

	err = res.Close()
	if err != nil {
		t.Errorf("Close() failed: %v", err)
	}

	// Second close should be safe
	err = res.Close()
	if err != nil {
		t.Errorf("Second Close() failed: %v", err)
	}
}

// ========================================
// GpuClonerOptions Tests
// ========================================

func TestNewGpuClonerOptions(t *testing.T) {
	opts := NewGpuClonerOptions()
	if opts == nil {
		t.Fatal("NewGpuClonerOptions returned nil")
	}

	// Test setters don't panic
	opts.SetUseFloat16(true)
	opts.SetUseFloat16CoarseQuantizer(true)
	opts.SetUsePrecomputed(true)
	opts.SetVerbose(true)
}

// ========================================
// GetNumGpus Tests
// ========================================

func TestGetNumGpus(t *testing.T) {
	numGpus := GetNumGpus()
	// Just verify it doesn't panic and returns >= 0
	if numGpus < 0 {
		t.Errorf("GetNumGpus() = %d, expected >= 0", numGpus)
	}
	t.Logf("Number of GPUs: %d", numGpus)
}

// ========================================
// GpuIndexFlat Tests
// ========================================

func TestNewGpuIndexFlatL2(t *testing.T) {
	res, err := NewStandardGpuResources()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer res.Close()

	idx, err := NewGpuIndexFlatL2(res, 128, 0)
	if err != nil {
		t.Fatalf("NewGpuIndexFlatL2() failed: %v", err)
	}
	defer idx.Close()

	if idx.D() != 128 {
		t.Errorf("D() = %d, want 128", idx.D())
	}
	if idx.Ntotal() != 0 {
		t.Errorf("Ntotal() = %d, want 0", idx.Ntotal())
	}
	if !idx.IsTrained() {
		t.Error("IsTrained() = false, want true")
	}
	if idx.MetricType() != MetricL2 {
		t.Errorf("MetricType() = %v, want MetricL2", idx.MetricType())
	}
}

func TestNewGpuIndexFlatIP(t *testing.T) {
	res, err := NewStandardGpuResources()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer res.Close()

	idx, err := NewGpuIndexFlatIP(res, 64, 0)
	if err != nil {
		t.Fatalf("NewGpuIndexFlatIP() failed: %v", err)
	}
	defer idx.Close()

	if idx.MetricType() != MetricInnerProduct {
		t.Errorf("MetricType() = %v, want MetricInnerProduct", idx.MetricType())
	}
}

func TestGpuIndexFlat_AddSearch(t *testing.T) {
	res, err := NewStandardGpuResources()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer res.Close()

	d := 64
	idx, err := NewGpuIndexFlatL2(res, d, 0)
	if err != nil {
		t.Fatalf("NewGpuIndexFlatL2() failed: %v", err)
	}
	defer idx.Close()

	// Add vectors
	n := 100
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = float32(i % 100)
	}

	err = idx.Add(vectors)
	if err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	if idx.Ntotal() != int64(n) {
		t.Errorf("Ntotal() = %d, want %d", idx.Ntotal(), n)
	}

	// Search
	query := vectors[:d]
	distances, indices, err := idx.Search(query, 5)
	if err != nil {
		t.Fatalf("Search() failed: %v", err)
	}

	if len(distances) != 5 || len(indices) != 5 {
		t.Errorf("Search() returned %d distances, %d indices, want 5 each",
			len(distances), len(indices))
	}

	// First result should be index 0 (exact match)
	if indices[0] != 0 {
		t.Errorf("First result index = %d, want 0", indices[0])
	}
	if distances[0] != 0 {
		t.Errorf("First result distance = %f, want 0", distances[0])
	}
}

func TestGpuIndexFlat_InvalidDimension(t *testing.T) {
	res, err := NewStandardGpuResources()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer res.Close()

	_, err = NewGpuIndexFlatL2(res, 0, 0)
	if err == nil {
		t.Error("NewGpuIndexFlatL2(0) should return error")
	}

	_, err = NewGpuIndexFlatL2(res, -1, 0)
	if err == nil {
		t.Error("NewGpuIndexFlatL2(-1) should return error")
	}
}

func TestGpuIndexFlat_NilResources(t *testing.T) {
	_, err := NewGpuIndexFlatL2(nil, 64, 0)
	if err == nil {
		t.Error("NewGpuIndexFlatL2(nil) should return error")
	}
}

func TestGpuIndexFlat_SetNprobe(t *testing.T) {
	res, err := NewStandardGpuResources()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer res.Close()

	idx, err := NewGpuIndexFlatL2(res, 64, 0)
	if err != nil {
		t.Fatalf("NewGpuIndexFlatL2() failed: %v", err)
	}
	defer idx.Close()

	// SetNprobe should fail for flat index
	err = idx.SetNprobe(10)
	if err == nil {
		t.Error("SetNprobe() should return error for flat index")
	}
}

func TestGpuIndexFlat_SetEfSearch(t *testing.T) {
	res, err := NewStandardGpuResources()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer res.Close()

	idx, err := NewGpuIndexFlatL2(res, 64, 0)
	if err != nil {
		t.Fatalf("NewGpuIndexFlatL2() failed: %v", err)
	}
	defer idx.Close()

	// SetEfSearch should fail for flat index
	err = idx.SetEfSearch(10)
	if err == nil {
		t.Error("SetEfSearch() should return error for flat index")
	}
}

func TestGpuIndexFlat_Reset(t *testing.T) {
	res, err := NewStandardGpuResources()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer res.Close()

	d := 64
	idx, err := NewGpuIndexFlatL2(res, d, 0)
	if err != nil {
		t.Fatalf("NewGpuIndexFlatL2() failed: %v", err)
	}
	defer idx.Close()

	// Add some vectors
	vectors := make([]float32, 10*d)
	for i := range vectors {
		vectors[i] = float32(i)
	}
	idx.Add(vectors)

	if idx.Ntotal() != 10 {
		t.Errorf("Ntotal() = %d, want 10", idx.Ntotal())
	}

	// Reset
	err = idx.Reset()
	if err != nil {
		t.Fatalf("Reset() failed: %v", err)
	}

	if idx.Ntotal() != 0 {
		t.Errorf("Ntotal() after reset = %d, want 0", idx.Ntotal())
	}
}

// ========================================
// IndexCpuToGpu Tests
// ========================================

func TestIndexCpuToGpu(t *testing.T) {
	res, err := NewStandardGpuResources()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer res.Close()

	// Create CPU index and add vectors
	d := 64
	cpuIdx, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("NewIndexFlatL2() failed: %v", err)
	}
	defer cpuIdx.Close()

	vectors := make([]float32, 100*d)
	for i := range vectors {
		vectors[i] = float32(i % 100)
	}
	cpuIdx.Add(vectors)

	// Transfer to GPU
	gpuIdx, err := IndexCpuToGpu(res, 0, cpuIdx)
	if err != nil {
		t.Fatalf("IndexCpuToGpu() failed: %v", err)
	}
	defer gpuIdx.Close()

	if gpuIdx.D() != d {
		t.Errorf("D() = %d, want %d", gpuIdx.D(), d)
	}
	if gpuIdx.Ntotal() != 100 {
		t.Errorf("Ntotal() = %d, want 100", gpuIdx.Ntotal())
	}

	// Search on GPU
	query := vectors[:d]
	distances, indices, err := gpuIdx.Search(query, 5)
	if err != nil {
		t.Fatalf("Search() failed: %v", err)
	}

	if len(distances) != 5 || len(indices) != 5 {
		t.Errorf("Search() returned %d distances, %d indices, want 5 each",
			len(distances), len(indices))
	}
}

func TestIndexCpuToGpu_NilResources(t *testing.T) {
	cpuIdx, _ := NewIndexFlatL2(64)
	defer cpuIdx.Close()

	_, err := IndexCpuToGpu(nil, 0, cpuIdx)
	if err == nil {
		t.Error("IndexCpuToGpu(nil, ...) should return error")
	}
}

func TestIndexCpuToGpu_NilIndex(t *testing.T) {
	res, err := NewStandardGpuResources()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer res.Close()

	_, err = IndexCpuToGpu(res, 0, nil)
	if err == nil {
		t.Error("IndexCpuToGpu(..., nil) should return error")
	}
}

// ========================================
// GpuIndexIVFFlat Tests
// ========================================

func TestNewGpuIndexIVFFlat(t *testing.T) {
	res, err := NewStandardGpuResources()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer res.Close()

	d := 64
	nlist := 10
	quantizer, err := NewGpuIndexFlatL2(res, d, 0)
	if err != nil {
		t.Fatalf("NewGpuIndexFlatL2() failed: %v", err)
	}
	defer quantizer.Close()

	idx, err := NewGpuIndexIVFFlat(res, quantizer, d, nlist, 0, MetricL2)
	if err != nil {
		t.Fatalf("NewGpuIndexIVFFlat() failed: %v", err)
	}
	defer idx.Close()

	if idx.D() != d {
		t.Errorf("D() = %d, want %d", idx.D(), d)
	}
	if idx.Nlist() != nlist {
		t.Errorf("Nlist() = %d, want %d", idx.Nlist(), nlist)
	}
	if idx.IsTrained() {
		t.Error("IsTrained() = true, want false before training")
	}
}

func TestGpuIndexIVFFlat_TrainAddSearch(t *testing.T) {
	res, err := NewStandardGpuResources()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer res.Close()

	d := 64
	nlist := 10
	quantizer, err := NewGpuIndexFlatL2(res, d, 0)
	if err != nil {
		t.Fatalf("NewGpuIndexFlatL2() failed: %v", err)
	}
	defer quantizer.Close()

	idx, err := NewGpuIndexIVFFlat(res, quantizer, d, nlist, 0, MetricL2)
	if err != nil {
		t.Fatalf("NewGpuIndexIVFFlat() failed: %v", err)
	}
	defer idx.Close()

	// Generate training/test vectors
	n := 1000
	vectors := make([]float32, n*d)
	for i := range vectors {
		vectors[i] = float32(i % 100)
	}

	// Train
	err = idx.Train(vectors)
	if err != nil {
		t.Fatalf("Train() failed: %v", err)
	}

	if !idx.IsTrained() {
		t.Error("IsTrained() = false after training")
	}

	// Set nprobe
	err = idx.SetNprobe(5)
	if err != nil {
		t.Fatalf("SetNprobe() failed: %v", err)
	}

	if idx.Nprobe() != 5 {
		t.Errorf("Nprobe() = %d, want 5", idx.Nprobe())
	}

	// Add vectors
	err = idx.Add(vectors)
	if err != nil {
		t.Fatalf("Add() failed: %v", err)
	}

	if idx.Ntotal() != int64(n) {
		t.Errorf("Ntotal() = %d, want %d", idx.Ntotal(), n)
	}

	// Search
	query := vectors[:d]
	distances, indices, err := idx.Search(query, 10)
	if err != nil {
		t.Fatalf("Search() failed: %v", err)
	}

	if len(distances) != 10 || len(indices) != 10 {
		t.Errorf("Search() returned %d distances, %d indices, want 10 each",
			len(distances), len(indices))
	}
}

func TestGpuIndexIVFFlat_SetNprobe_Invalid(t *testing.T) {
	res, err := NewStandardGpuResources()
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer res.Close()

	d := 64
	nlist := 10
	quantizer, _ := NewGpuIndexFlatL2(res, d, 0)
	defer quantizer.Close()

	idx, _ := NewGpuIndexIVFFlat(res, quantizer, d, nlist, 0, MetricL2)
	defer idx.Close()

	err = idx.SetNprobe(0)
	if err == nil {
		t.Error("SetNprobe(0) should return error")
	}

	err = idx.SetNprobe(nlist + 1)
	if err == nil {
		t.Error("SetNprobe(nlist+1) should return error")
	}
}

// ========================================
// GpuIndex Interface Compliance Tests
// ========================================

func TestGpuIndex_ImplementsIndex(t *testing.T) {
	var _ Index = (*GpuIndex)(nil)
	var _ Index = (*GpuIndexFlat)(nil)
	var _ Index = (*GpuIndexIVFFlat)(nil)
}
