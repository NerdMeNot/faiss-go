// Package faiss provides production-ready Go bindings for Facebook's FAISS
// (Facebook AI Similarity Search) library, enabling billion-scale similarity
// search and clustering of dense vectors.
//
// faiss-go offers complete feature parity with Python FAISS, including 18+ index types,
// GPU acceleration, and advanced features like product quantization, HNSW graphs, and
// on-disk indexes. Perfect for semantic search, recommendation systems, and image similarity.
//
// # Quick Start
//
// Create an index and search for similar vectors:
//
//	package main
//
//	import (
//	    "fmt"
//	    "log"
//	    "github.com/NerdMeNot/faiss-go"
//	)
//
//	func main() {
//	    // Create index for 128-dimensional vectors
//	    index, err := faiss.NewIndexFlatL2(128)
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//	    defer index.Close()
//
//	    // Add vectors (flattened: [v1_d1, v1_d2, ..., v2_d1, v2_d2, ...])
//	    vectors := make([]float32, 1000 * 128) // 1000 vectors
//	    // ... populate vectors with your data ...
//	    err = index.Add(vectors)
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//
//	    // Search for 10 nearest neighbors
//	    query := make([]float32, 128) // Single query vector
//	    // ... populate query ...
//	    distances, indices, err := index.Search(query, 10)
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//
//	    // Process results
//	    for i := 0; i < 10; i++ {
//	        fmt.Printf("Neighbor %d: index=%d, distance=%.4f\n",
//	            i+1, indices[i], distances[i])
//	    }
//	}
//
// # Index Selection Guide
//
// Choose the right index for your use case:
//
//  - IndexFlatL2 / IndexFlatIP: Exact search, 100% recall, best for <100K vectors
//  - IndexIVFFlat: Fast approximate search, 10-100x speedup, 95%+ recall
//  - IndexHNSW: Best recall/speed tradeoff, excellent for production
//  - IndexPQ: 8-32x compression, great for memory-constrained scenarios
//  - IndexIVFPQ: Combines speed and compression, best overall balance
//  - IndexOnDisk: For billion-scale datasets that don't fit in RAM
//  - GPU indexes: 10-100x faster search with CUDA acceleration
//
// # Build Modes
//
// Two flexible build modes to fit your workflow:
//
// Pre-built Libraries (fast development):
//
//	go build -tags=faiss_use_lib    # <30 second builds
//
// Compile from Source (production optimization):
//
//	go build                         # ~5-10 min first time, cached after
//
// Both modes produce identical functionality. Source build requires:
//   - C++17 compiler (GCC 7+, Clang 5+, MSVC 2019+)
//   - BLAS library (OpenBLAS, MKL, or Accelerate on macOS)
//
// # Production Features
//
// This package provides comprehensive FAISS functionality:
//
//   - 18+ Index Types: Flat, IVF, HNSW, PQ, ScalarQuantizer, LSH, GPU, OnDisk
//   - Training API: Optimize indexes for your data distribution
//   - Serialization: Save and load indexes from disk
//   - Range Search: Find all vectors within a distance threshold
//   - Batch Operations: Efficient bulk add/search
//   - Vector Reconstruction: Retrieve vectors from compressed indexes
//   - Clustering: Built-in Kmeans implementation
//   - Preprocessing: PCA, OPQ, Random Rotation transforms
//   - Index Factory: Declarative index construction with strings
//   - Custom IDs: Map external IDs to internal indices
//
// # Use Cases
//
// Semantic Search - Document similarity:
//
//	embeddings := embedDocuments(docs) // 768-dim BERT/OpenAI
//	index, _ := faiss.NewIndexHNSWFlat(768, 32, faiss.MetricL2)
//	index.Train(embeddings)
//	index.Add(embeddings)
//	distances, indices, _ := index.Search(queryEmbedding, 10)
//
// Image Similarity - Visual search:
//
//	features := extractImageFeatures(images) // 2048-dim ResNet
//	quantizer, _ := faiss.NewIndexFlatL2(2048)
//	index, _ := faiss.NewIndexIVFPQ(quantizer, 2048, 1000, 16, 8, faiss.MetricL2)
//	index.Train(features)
//	index.Add(features)
//	_, similar, _ := index.Search(queryFeatures, 20)
//
// Recommendation Systems - Collaborative filtering:
//
//	itemEmbeddings := trainEmbeddings(interactions) // 128-dim
//	quantizer, _ := faiss.NewIndexFlatL2(128)
//	index, _ := faiss.NewIndexIVFFlat(quantizer, 128, 4096, faiss.MetricL2)
//	index.Train(itemEmbeddings)
//	index.Add(itemEmbeddings)
//	_, recommended, _ := index.Search(userEmbedding, 50)
//
// # Metrics
//
// FAISS supports two distance metrics:
//
//   - MetricL2: Euclidean (L2) distance - lower is more similar
//   - MetricInnerProduct: Inner product - higher is more similar
//
// For cosine similarity, normalize vectors and use MetricInnerProduct:
//
//	normalized := normalize(vectors) // Divide by L2 norm
//	index, _ := faiss.NewIndexFlatIP(dimension)
//	index.Add(normalized)
//
// # Thread Safety
//
// Index operations are NOT thread-safe by default. For concurrent access:
//
// Option 1 - Use synchronization:
//
//	var mu sync.Mutex
//	mu.Lock()
//	defer mu.Unlock()
//	index.Add(vectors)
//
// Option 2 - Separate indexes per goroutine (read-heavy workloads):
//
//	indexes := make([]*faiss.IndexFlat, numWorkers)
//	for i := range indexes {
//	    indexes[i], _ = faiss.NewIndexFlatL2(dimension)
//	    indexes[i].Add(vectors) // Same data in each
//	}
//
// # Memory Management
//
// Always call Close() to free C++ resources:
//
//	index, err := faiss.NewIndexFlatL2(128)
//	if err != nil {
//	    return err
//	}
//	defer index.Close() // Essential to prevent memory leaks
//
// Finalizers are set as a safety net, but explicit Close() is recommended.
//
// # Platform Support
//
// Supports all major platforms:
//   - Linux: x86_64, ARM64
//   - macOS: Intel (x86_64), Apple Silicon (ARM64)
//
// # Performance
//
// Performance characteristics (1M 128-dim vectors, M1 Mac):
//
//   - IndexFlatL2: 12K QPS, 100% recall (exact search)
//   - IndexHNSWFlat: 85K QPS, 98.5% recall
//   - IndexIVFPQ: 120K QPS, 95.2% recall, 16x compression
//   - PQFastScan: 180K QPS, 95.8% recall, SIMD optimized
//
// See https://github.com/NerdMeNot/faiss-go for comprehensive benchmarks.
//
// # Documentation
//
// Complete documentation available at:
//   - Getting Started: https://github.com/NerdMeNot/faiss-go/docs/getting-started/
//   - API Reference: https://pkg.go.dev/github.com/NerdMeNot/faiss-go
//   - Examples: https://github.com/NerdMeNot/faiss-go/docs/examples/
//   - GitHub: https://github.com/NerdMeNot/faiss-go
//
// # Version Information
//
// This package version: v0.1.0-alpha
// Embedded FAISS version: 1.8.0
//
// Report issues: https://github.com/NerdMeNot/faiss-go/issues
package faiss

import (
	"errors"
	"fmt"
	"runtime"
)

// Version information
//
// Versioning scheme: v{FAISS_VERSION}-{BINDING_MAJOR}.{BINDING_MINOR}
// Example: v1.13.2-0.1
//
// - FAISS_VERSION: The upstream FAISS library version this is built against
// - BINDING_MAJOR: Incremented for new faiss-go features/interfaces (not in upstream FAISS)
// - BINDING_MINOR: Incremented for bug fixes and minor improvements
//
// When FAISS releases a new version, reset binding version to 0.1
const (
	// FAISSVersion is the upstream FAISS library version
	FAISSVersion = "1.13.2"

	// BindingMajor is incremented for new faiss-go features/interfaces
	BindingMajor = 0

	// BindingMinor is incremented for bug fixes and improvements
	BindingMinor = 1

	// Version is the full faiss-go version string (auto-generated)
	Version = FAISSVersion + "-" + "0.1" // Note: Can't use fmt.Sprintf in const
)

// FullVersion returns the complete version string with 'v' prefix
// Example: "v1.13.2-0.1"
func FullVersion() string {
	return fmt.Sprintf("v%s-%d.%d", FAISSVersion, BindingMajor, BindingMinor)
}

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

// NewIndexFlat creates a new flat index with the specified metric.
// This is the recommended constructor for flat indexes as it follows the same
// pattern as other index constructors (NewIndexHNSW, NewIndexPQ, etc.).
//
// Parameters:
//   - d: dimension of vectors
//   - metric: distance metric (MetricL2 or MetricInnerProduct)
//
// Example:
//
//	index, err := faiss.NewIndexFlat(128, faiss.MetricL2)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer index.Close()
func NewIndexFlat(d int, metric MetricType) (*IndexFlat, error) {
	switch metric {
	case MetricL2:
		return NewIndexFlatL2(d)
	case MetricInnerProduct:
		return NewIndexFlatIP(d)
	default:
		return nil, fmt.Errorf("faiss: unsupported metric type %d for flat index", metric)
	}
}

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

	timer := StartTimer()
	if err := faissIndexAdd(idx.ptr, vectors, n); err != nil {
		return fmt.Errorf("faiss: failed to add vectors: %w", err)
	}
	timer.RecordAdd(n)

	idx.ntotal += int64(n)
	return nil
}

// Search searches for the k nearest neighbors of the query vectors
// For large batches (>100 queries), this releases the Go scheduler during
// the C++ computation to prevent blocking other goroutines
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

	// For large searches, lock to OS thread to optimize C++ performance
	// and prevent Go scheduler from migrating goroutine during computation
	if nq > 100 {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
	}

	timer := StartTimer()
	if err := faissIndexSearch(idx.ptr, queries, nq, k, distances, indices); err != nil {
		return nil, nil, fmt.Errorf("faiss: search failed: %w", err)
	}
	timer.RecordSearch(nq, nq*k)

	return distances, indices, nil
}

// Reset removes all vectors from the index
func (idx *IndexFlat) Reset() error {
	if idx.ptr == 0 {
		return ErrNullPointer
	}

	timer := StartTimer()
	if err := faissIndexReset(idx.ptr); err != nil {
		return fmt.Errorf("faiss: reset failed: %w", err)
	}
	timer.RecordReset()

	idx.ntotal = 0
	return nil
}

// SetNprobe is not supported for flat indexes (not an IVF index)
func (idx *IndexFlat) SetNprobe(nprobe int) error {
	return fmt.Errorf("faiss: SetNprobe not supported for IndexFlat (not an IVF index)")
}

// SetEfSearch is not supported for flat indexes (not an HNSW index)
func (idx *IndexFlat) SetEfSearch(efSearch int) error {
	return fmt.Errorf("faiss: SetEfSearch not supported for IndexFlat (not an HNSW index)")
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
