// Package faiss provides Go bindings for FAISS (Facebook AI Similarity Search).
//
// FAISS is a library for efficient similarity search and clustering of dense vectors.
// This package embeds FAISS, so no separate installation is required.
//
// # Build Modes
//
// The package supports two build modes:
//
//  1. Source Build (default): Compiles FAISS from amalgamated source
//     go build
//
//  2. Pre-built Libraries: Uses pre-compiled static libraries
//     go build -tags=faiss_use_lib
//
// # Quick Start
//
//	// Create an index for 128-dimensional vectors
//	index, err := faiss.NewIndexFlatL2(128)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer index.Close()
//
//	// Add vectors ([]float32 with length = dimension * numVectors)
//	vectors := []float32{ /* ... */ }
//	if err := index.Add(vectors); err != nil {
//	    log.Fatal(err)
//	}
//
//	// Search for k nearest neighbors
//	query := []float32{ /* ... dimension floats ... */ }
//	distances, indices, err := index.Search(query, 10)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
// # Index Types
//
// Supported index types:
//   - IndexFlatL2: Exact L2 distance search
//   - IndexFlatIP: Exact inner product search
//
// More index types (IVF, PQ, HNSW) coming soon.
//
// # Thread Safety
//
// Index operations are not thread-safe by default. Use external synchronization
// or create separate indexes per goroutine.
package faiss

import (
	"errors"
	"fmt"
	"runtime"
)

// Version information
const (
	// Version is the faiss-go binding version
	Version = "0.1.0-alpha"
	// FAISSVersion is the embedded FAISS library version
	FAISSVersion = "1.8.0" // Will be updated when amalgamation is generated
)

var (
	// ErrInvalidDimension is returned when dimension is invalid
	ErrInvalidDimension = errors.New("faiss: invalid dimension (must be > 0)")
	// ErrInvalidVectors is returned when vector data is invalid
	ErrInvalidVectors = errors.New("faiss: invalid vectors (length must be multiple of dimension)")
	// ErrIndexNotTrained is returned when operation requires trained index
	ErrIndexNotTrained = errors.New("faiss: index not trained")
	// ErrNullPointer is returned when C pointer is null
	ErrNullPointer = errors.New("faiss: null pointer")
)

// MetricType defines the distance metric used by an index
type MetricType int

const (
	// MetricInnerProduct uses inner product (higher is more similar)
	MetricInnerProduct MetricType = 0
	// MetricL2 uses L2 (Euclidean) distance (lower is more similar)
	MetricL2 MetricType = 1
)

// String returns the string representation of the metric type
func (m MetricType) String() string {
	switch m {
	case MetricInnerProduct:
		return "InnerProduct"
	case MetricL2:
		return "L2"
	default:
		return fmt.Sprintf("MetricType(%d)", m)
	}
}

// Index interface is defined in index.go to avoid duplication

// IndexFlat represents a flat (brute-force) index
type IndexFlat struct {
	ptr       uintptr     // C pointer to FaissIndexFlat
	d         int         // dimension
	metric    MetricType  // metric type
	ntotal    int64       // number of vectors
	isTrained bool        // always true for flat indexes
}

// Ensure IndexFlat implements Index
var _ Index = (*IndexFlat)(nil)

// NewIndexFlatL2 creates a new flat index using L2 distance
func NewIndexFlatL2(d int) (*IndexFlat, error) {
	if d <= 0 {
		return nil, ErrInvalidDimension
	}

	ptr, err := faissIndexFlatL2New(d)
	if err != nil {
		return nil, fmt.Errorf("faiss: failed to create IndexFlatL2: %w", err)
	}

	idx := &IndexFlat{
		ptr:       ptr,
		d:         d,
		metric:    MetricL2,
		ntotal:    0,
		isTrained: true,
	}

	// Set finalizer to ensure cleanup
	runtime.SetFinalizer(idx, func(i *IndexFlat) {
		if i.ptr != 0 {
			_ = i.Close()
		}
	})

	return idx, nil
}

// NewIndexFlatIP creates a new flat index using inner product
func NewIndexFlatIP(d int) (*IndexFlat, error) {
	if d <= 0 {
		return nil, ErrInvalidDimension
	}

	ptr, err := faissIndexFlatIPNew(d)
	if err != nil {
		return nil, fmt.Errorf("faiss: failed to create IndexFlatIP: %w", err)
	}

	idx := &IndexFlat{
		ptr:       ptr,
		d:         d,
		metric:    MetricInnerProduct,
		ntotal:    0,
		isTrained: true,
	}

	runtime.SetFinalizer(idx, func(i *IndexFlat) {
		if i.ptr != 0 {
			_ = i.Close()
		}
	})

	return idx, nil
}

// D returns the dimension of the vectors
func (idx *IndexFlat) D() int {
	return idx.d
}

// Ntotal returns the total number of vectors in the index
func (idx *IndexFlat) Ntotal() int64 {
	if idx.ptr == 0 {
		return 0
	}
	// Will be updated from C when we add vectors
	return idx.ntotal
}

// IsTrained returns whether the index has been trained (always true for flat indexes)
func (idx *IndexFlat) IsTrained() bool {
	return idx.isTrained
}

// MetricType returns the metric type used by the index
func (idx *IndexFlat) MetricType() MetricType {
	return idx.metric
}

// Train is a no-op for flat indexes (they don't require training)
func (idx *IndexFlat) Train(vectors []float32) error {
	// Flat indexes don't need training
	return nil
}

// Add adds vectors to the index
func (idx *IndexFlat) Add(vectors []float32) error {
	if idx.ptr == 0 {
		return ErrNullPointer
	}

	if len(vectors) == 0 {
		return nil // nothing to add
	}

	if len(vectors)%idx.d != 0 {
		return ErrInvalidVectors
	}

	n := len(vectors) / idx.d

	if err := faissIndexAdd(idx.ptr, vectors, n); err != nil {
		return fmt.Errorf("faiss: failed to add vectors: %w", err)
	}

	idx.ntotal += int64(n)
	return nil
}

// Search searches for the k nearest neighbors of the query vectors
func (idx *IndexFlat) Search(queries []float32, k int) (distances []float32, indices []int64, err error) {
	if idx.ptr == 0 {
		return nil, nil, ErrNullPointer
	}

	if len(queries) == 0 {
		return []float32{}, []int64{}, nil
	}

	if len(queries)%idx.d != 0 {
		return nil, nil, ErrInvalidVectors
	}

	nq := len(queries) / idx.d

	if k <= 0 {
		return nil, nil, errors.New("faiss: k must be positive")
	}

	distances = make([]float32, nq*k)
	indices = make([]int64, nq*k)

	if err := faissIndexSearch(idx.ptr, queries, nq, k, distances, indices); err != nil {
		return nil, nil, fmt.Errorf("faiss: search failed: %w", err)
	}

	return distances, indices, nil
}

// Reset removes all vectors from the index
func (idx *IndexFlat) Reset() error {
	if idx.ptr == 0 {
		return ErrNullPointer
	}

	if err := faissIndexReset(idx.ptr); err != nil {
		return fmt.Errorf("faiss: reset failed: %w", err)
	}

	idx.ntotal = 0
	return nil
}

// Close releases resources associated with the index
func (idx *IndexFlat) Close() error {
	if idx.ptr == 0 {
		return nil // already closed
	}

	err := faissIndexFree(idx.ptr)
	idx.ptr = 0
	idx.ntotal = 0

	if err != nil {
		return fmt.Errorf("faiss: failed to free index: %w", err)
	}

	return nil
}

// GetBuildInfo returns information about how faiss-go was built
func GetBuildInfo() BuildInfo {
	return getBuildInfo()
}

// BuildInfo contains information about the build configuration
type BuildInfo struct {
	// Version is the faiss-go version
	Version string
	// FAISSVersion is the FAISS library version
	FAISSVersion string
	// BuildMode is either "source" or "prebuilt"
	BuildMode string
	// Compiler is the C++ compiler used
	Compiler string
	// Platform is the OS/architecture
	Platform string
	// BLASBackend is the BLAS library used
	BLASBackend string
}

// String returns a formatted string of build information
func (bi BuildInfo) String() string {
	return fmt.Sprintf("faiss-go %s (FAISS %s)\nBuild: %s\nCompiler: %s\nPlatform: %s\nBLAS: %s",
		bi.Version, bi.FAISSVersion, bi.BuildMode, bi.Compiler, bi.Platform, bi.BLASBackend)
}
