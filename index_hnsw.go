package faiss

import (
	"fmt"
)

// NewIndexHNSWFlat creates a new HNSW index with flat (uncompressed) storage.
//
// HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest
// neighbor search algorithm that provides excellent recall/speed tradeoffs.
//
// Parameters:
//   - d: dimension of vectors
//   - M: number of connections per layer (typical values: 16, 32, 64)
//        Higher M = better recall but more memory and slower build
//   - metric: distance metric (MetricL2 or MetricInnerProduct)
//
// The index does NOT require training. You can add vectors immediately.
//
// Recommended M values:
//   - M=16: Fast build, moderate memory (good for prototyping)
//   - M=32: Balanced (recommended for production)
//   - M=64: Best recall, higher memory
//
// Python equivalent: faiss.IndexHNSWFlat(d, M)
//
// Example:
//
//	index, err := faiss.NewIndexHNSWFlat(128, 32, faiss.MetricL2)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer index.Close()
//
//	// No training needed for HNSW
//	err = index.Add(vectors)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	distances, indices, err := index.Search(query, 10)
func NewIndexHNSWFlat(d, M int, metric MetricType) (Index, error) {
	if d <= 0 {
		return nil, ErrInvalidDimension
	}
	if M <= 0 {
		return nil, fmt.Errorf("faiss: M must be positive")
	}

	// Use the factory pattern internally
	description := fmt.Sprintf("HNSW%d,Flat", M)
	return IndexFactory(d, description, metric)
}

// NewIndexHNSW creates a new HNSW index (alias for NewIndexHNSWFlat for compatibility).
//
// This is equivalent to NewIndexHNSWFlat and provided for API compatibility
// with Python FAISS where IndexHNSWFlat is the main HNSW implementation.
func NewIndexHNSW(d, M int, metric MetricType) (Index, error) {
	return NewIndexHNSWFlat(d, M, metric)
}
