package faiss

import (
	"fmt"
)

// RangeSearchResult contains the results of a range search
// For each query, it returns all vectors within the specified radius
type RangeSearchResult struct {
	Nq        int       // Number of queries
	Lims      []int64   // Offsets: lims[i] to lims[i+1] are results for query i
	Labels    []int64   // Labels of results
	Distances []float32 // Distances of results
}

// GetResults returns the results for a specific query
func (rsr *RangeSearchResult) GetResults(queryIdx int) (labels []int64, distances []float32) {
	if queryIdx < 0 || queryIdx >= rsr.Nq {
		return nil, nil
	}

	start := rsr.Lims[queryIdx]
	end := rsr.Lims[queryIdx+1]

	return rsr.Labels[start:end], rsr.Distances[start:end]
}

// NumResults returns the number of results for a specific query
func (rsr *RangeSearchResult) NumResults(queryIdx int) int {
	if queryIdx < 0 || queryIdx >= rsr.Nq {
		return 0
	}
	return int(rsr.Lims[queryIdx+1] - rsr.Lims[queryIdx])
}

// TotalResults returns the total number of results across all queries
func (rsr *RangeSearchResult) TotalResults() int {
	if rsr.Nq == 0 {
		return 0
	}
	return int(rsr.Lims[rsr.Nq])
}

// RangeSearch performs range search on indexes that support it
//
// Returns all vectors within the specified radius for each query.
// The radius interpretation depends on the metric:
//   - L2: radius is the maximum squared L2 distance
//   - Inner Product: radius is the minimum inner product
//
// Python equivalent: lims, D, I = index.range_search(x, radius)
//
// Example:
//   result, err := index.RangeSearch(queries, 0.5)
//   for i := 0; i < result.Nq; i++ {
//       labels, distances := result.GetResults(i)
//       fmt.Printf("Query %d: %d results\n", i, len(labels))
//   }
func (idx *IndexFlat) RangeSearch(queries []float32, radius float32) (*RangeSearchResult, error) {
	if idx.ptr == 0 {
		return nil, ErrNullPointer
	}
	if len(queries) == 0 {
		return &RangeSearchResult{Nq: 0, Lims: []int64{0}, Labels: []int64{}, Distances: []float32{}}, nil
	}
	if len(queries)%idx.d != 0 {
		return nil, ErrInvalidVectors
	}

	nq := len(queries) / idx.d

	resultPtr, lims, labels, distances, err := faissIndexRangeSearch(idx.ptr, queries, nq, radius)
	if err != nil {
		return nil, fmt.Errorf("faiss: range search failed: %w", err)
	}
	defer faissRangeSearchResultFree(resultPtr)

	// Copy results (FAISS owns the memory, we need to copy)
	result := &RangeSearchResult{
		Nq:        nq,
		Lims:      make([]int64, nq+1),
		Labels:    make([]int64, len(labels)),
		Distances: make([]float32, len(distances)),
	}

	copy(result.Lims, lims)
	copy(result.Labels, labels)
	copy(result.Distances, distances)

	return result, nil
}

// RangeSearch for IVF indexes
func (idx *IndexIVFFlat) RangeSearch(queries []float32, radius float32) (*RangeSearchResult, error) {
	if idx.ptr == 0 {
		return nil, ErrNullPointer
	}
	if !idx.isTrained {
		return nil, ErrNotTrained
	}
	if len(queries) == 0 {
		return &RangeSearchResult{Nq: 0, Lims: []int64{0}, Labels: []int64{}, Distances: []float32{}}, nil
	}
	if len(queries)%idx.d != 0 {
		return nil, ErrInvalidVectors
	}

	nq := len(queries) / idx.d

	resultPtr, lims, labels, distances, err := faissIndexRangeSearch(idx.ptr, queries, nq, radius)
	if err != nil {
		return nil, fmt.Errorf("faiss: range search failed: %w", err)
	}
	defer faissRangeSearchResultFree(resultPtr)

	result := &RangeSearchResult{
		Nq:        nq,
		Lims:      make([]int64, nq+1),
		Labels:    make([]int64, len(labels)),
		Distances: make([]float32, len(distances)),
	}

	copy(result.Lims, lims)
	copy(result.Labels, labels)
	copy(result.Distances, distances)

	return result, nil
}

// RangeSearch for HNSW indexes
func (idx *IndexHNSW) RangeSearch(queries []float32, radius float32) (*RangeSearchResult, error) {
	if idx.ptr == 0 {
		return nil, ErrNullPointer
	}
	if len(queries) == 0 {
		return &RangeSearchResult{Nq: 0, Lims: []int64{0}, Labels: []int64{}, Distances: []float32{}}, nil
	}
	if len(queries)%idx.d != 0 {
		return nil, ErrInvalidVectors
	}

	nq := len(queries) / idx.d

	resultPtr, lims, labels, distances, err := faissIndexRangeSearch(idx.ptr, queries, nq, radius)
	if err != nil {
		return nil, fmt.Errorf("faiss: range search failed: %w", err)
	}
	defer faissRangeSearchResultFree(resultPtr)

	result := &RangeSearchResult{
		Nq:        nq,
		Lims:      make([]int64, nq+1),
		Labels:    make([]int64, len(labels)),
		Distances: make([]float32, len(distances)),
	}

	copy(result.Lims, lims)
	copy(result.Labels, labels)
	copy(result.Distances, distances)

	return result, nil
}
