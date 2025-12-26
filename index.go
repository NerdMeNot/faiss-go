package faiss

// This file contains the extended Index interface and common operations
// matching Python FAISS completeness

import (
	"errors"
)

var (
	// ErrNotTrained is returned when an operation requires a trained index
	ErrNotTrained = errors.New("faiss: index not trained")
	// ErrIDNotFound is returned when an ID is not in the index
	ErrIDNotFound = errors.New("faiss: ID not found")
	// ErrInvalidK is returned when k is invalid for search
	ErrInvalidK = errors.New("faiss: k must be positive")
	// ErrInvalidRadius is returned when radius is invalid
	ErrInvalidRadius = errors.New("faiss: invalid radius")
)

// Index is the base interface for all FAISS indexes
// This matches the Python FAISS Index API
type Index interface {
	// Basic properties
	D() int                // Dimension of vectors
	Ntotal() int64         // Total number of indexed vectors
	IsTrained() bool       // Whether the index has been trained
	MetricType() MetricType // Metric type (L2, IP, etc.)

	// Training (required for some index types like IVF, PQ)
	Train(vectors []float32) error

	// Adding vectors
	Add(vectors []float32) error

	// Searching
	Search(queries []float32, k int) (distances []float32, indices []int64, err error)

	// Management
	Reset() error  // Remove all vectors
	Close() error  // Free resources
}

// IndexWithIDs extends Index to support custom IDs
type IndexWithIDs interface {
	Index

	// Add vectors with custom IDs
	AddWithIDs(vectors []float32, ids []int64) error

	// Remove vectors by IDs
	RemoveIDs(ids []int64) error
}

// IndexWithReconstruction extends Index to support vector reconstruction
type IndexWithReconstruction interface {
	Index

	// Reconstruct a single vector by its index
	Reconstruct(id int64) ([]float32, error)

	// Reconstruct multiple vectors
	ReconstructN(start, n int64) ([]float32, error)

	// Reconstruct a batch of vectors by their indices
	ReconstructBatch(ids []int64) ([]float32, error)
}

// IndexWithRangeSearch extends Index to support range-based search
type IndexWithRangeSearch interface {
	Index

	// RangeSearch finds all vectors within a radius
	RangeSearch(queries []float32, radius float32) (*RangeSearchResult, error)
}

// IndexWithAssign extends Index to support assignment (clustering)
type IndexWithAssign interface {
	Index

	// Assign vectors to their nearest cluster
	Assign(vectors []float32) ([]int64, error)
}

// RangeSearchResult contains results from range search
type RangeSearchResult struct {
	Nq      int       // Number of queries
	Lims    []int64   // Limits array (length nq+1)
	Labels  []int64   // Labels for each result
	Distances []float32 // Distances for each result
}

// SearchResult is a structured search result
type SearchResult struct {
	Distances []float32
	Labels    []int64
	Nq        int  // Number of queries
	K         int  // Number of neighbors per query
}

// NewSearchResult creates a SearchResult from raw arrays
func NewSearchResult(distances []float32, labels []int64, nq, k int) *SearchResult {
	return &SearchResult{
		Distances: distances,
		Labels:    labels,
		Nq:        nq,
		K:         k,
	}
}

// Get returns the distance and label for query i, neighbor j
func (sr *SearchResult) Get(i, j int) (distance float32, label int64) {
	idx := i*sr.K + j
	return sr.Distances[idx], sr.Labels[idx]
}

// GetNeighbors returns all neighbors for query i
func (sr *SearchResult) GetNeighbors(i int) (distances []float32, labels []int64) {
	start := i * sr.K
	end := start + sr.K
	return sr.Distances[start:end], sr.Labels[start:end]
}
