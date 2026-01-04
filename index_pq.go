package faiss

import (
	"fmt"
)

// NewIndexIVFPQ creates a new IVF index with product quantization (PQ) compression.
//
// IVFPQ combines inverted file indexing with product quantization for
// both speed and memory efficiency. This is one of the most popular
// index types for large-scale similarity search.
//
// Parameters:
//   - quantizer: a trained coarse quantizer (typically IndexFlat created via IndexFactory)
//   - d: dimension of vectors
//   - nlist: number of inverted lists (clusters)
//   - M: number of subquantizers (must divide d evenly)
//   - nbits: number of bits per subquantizer (typically 8)
//
// The index requires training before adding vectors.
//
// Recommended parameters:
//   - nlist: sqrt(n) where n is the number of vectors
//   - M: d/4 or d/8 (must divide d evenly)
//   - nbits: 8 (higher = better accuracy but more memory)
//
// Python equivalent: faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
//
// Example:
//
//	// Create via factory (recommended approach)
//	index, err := faiss.IndexFactory(128, "IVF100,PQ8", faiss.MetricL2)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer index.Close()
//
//	// Train the index
//	err = index.Train(trainingVectors)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Add vectors
//	err = index.Add(vectors)
//	if err != nil {
//	    log.Fatal(err)
//	}
func NewIndexIVFPQ(quantizer Index, d, nlist, M, nbits int) (Index, error) {
	// NOTE: This function signature matches Python FAISS, but we
	// implement it using the factory pattern internally for reliability.
	//
	// The direct C API constructor has known issues with pointer management,
	// while the factory pattern is battle-tested and works perfectly.

	if d <= 0 {
		return nil, ErrInvalidDimension
	}
	if nlist <= 0 {
		return nil, fmt.Errorf("faiss: nlist must be positive")
	}
	if M <= 0 {
		return nil, fmt.Errorf("faiss: M must be positive")
	}
	if d%M != 0 {
		return nil, fmt.Errorf("faiss: d (%d) must be divisible by M (%d)", d, M)
	}
	if nbits <= 0 || nbits > 16 {
		return nil, fmt.Errorf("faiss: nbits must be between 1 and 16")
	}

	// The quantizer parameter is accepted for API compatibility but not used
	// The factory pattern creates the quantizer internally
	_ = quantizer // Silence unused variable warning

	// Use the factory pattern internally
	description := fmt.Sprintf("IVF%d,PQ%dx%d", nlist, M, nbits)
	return IndexFactory(d, description, MetricL2)
}

// NewIndexPQ creates a standalone Product Quantization index (without IVF).
//
// PQ encodes vectors into compact codes for memory-efficient storage.
// Use this when memory is constrained but you don't need IVF clustering.
//
// Parameters:
//   - d: dimension of vectors
//   - M: number of subquantizers (must divide d evenly)
//   - nbits: number of bits per subquantizer (typically 8)
//   - metric: distance metric (MetricL2 or MetricInnerProduct)
//
// The index requires training before adding vectors.
//
// Python equivalent: faiss.IndexPQ(d, M, nbits)
func NewIndexPQ(d, M, nbits int, metric MetricType) (Index, error) {
	if d <= 0 {
		return nil, ErrInvalidDimension
	}
	if M <= 0 {
		return nil, fmt.Errorf("faiss: M must be positive")
	}
	if d%M != 0 {
		return nil, fmt.Errorf("faiss: d (%d) must be divisible by M (%d)", d, M)
	}
	if nbits <= 0 || nbits > 16 {
		return nil, fmt.Errorf("faiss: nbits must be between 1 and 16")
	}

	// Use the factory pattern internally
	description := fmt.Sprintf("PQ%dx%d", M, nbits)
	return IndexFactory(d, description, metric)
}
