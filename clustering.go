package faiss

import (
	"fmt"
)

// Kmeans performs k-means clustering on vectors
//
// Python equivalent: faiss.Kmeans
//
// Example:
//
//	kmeans, _ := faiss.NewKmeans(128, 100) // d=128, k=100
//	kmeans.Train(trainingVectors)
//	centroids := kmeans.Centroids()
type Kmeans struct {
	d          int       // dimension
	k          int       // number of clusters
	centroids  []float32 // cluster centroids (k * d)
	isTrained  bool      // training status
}

// NewKmeans creates a new k-means clustering object
//
// Parameters:
//   - d: dimension of vectors
//   - k: number of clusters
func NewKmeans(d, k int) (*Kmeans, error) {
	if d <= 0 {
		return nil, ErrInvalidDimension
	}
	if k <= 0 {
		return nil, fmt.Errorf("faiss: k must be positive")
	}

	km := &Kmeans{
		d:         d,
		k:         k,
		isTrained: false,
	}

	return km, nil
}

// D returns the dimension
func (km *Kmeans) D() int {
	return km.d
}

// K returns the number of clusters
func (km *Kmeans) K() int {
	return km.k
}

// Train performs k-means clustering on the training vectors
//
// Parameters:
//   - vectors: training vectors (n vectors of dimension d)
//
// The centroids are stored internally and can be accessed via Centroids()
func (km *Kmeans) Train(vectors []float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("faiss: cannot train on empty vectors")
	}
	if len(vectors)%km.d != 0 {
		return ErrInvalidVectors
	}

	n := len(vectors) / km.d
	if n < km.k {
		return fmt.Errorf("faiss: need at least %d training vectors for %d clusters, got %d", km.k, km.k, n)
	}

	// Allocate space for centroids
	km.centroids = make([]float32, km.k*km.d)

	// Call faiss_kmeans_clustering
	err := faiss_kmeans_clustering(km.d, n, km.k, vectors, km.centroids)
	if err != nil {
		return fmt.Errorf("faiss: k-means clustering failed: %w", err)
	}

	km.isTrained = true
	return nil
}

// Centroids returns the cluster centroids (k * d floats)
// Must call Train() first
func (km *Kmeans) Centroids() []float32 {
	if !km.isTrained {
		return nil
	}
	return km.centroids
}

// IsTrained returns whether the clustering has been trained
func (km *Kmeans) IsTrained() bool {
	return km.isTrained
}

// Assign assigns vectors to their nearest cluster using L2 distance
//
// Parameters:
//   - vectors: vectors to assign (n vectors of dimension d)
//
// Returns: cluster assignments (n integers, each in range [0, k))
func (km *Kmeans) Assign(vectors []float32) ([]int64, error) {
	if !km.isTrained {
		return nil, fmt.Errorf("faiss: must train before assigning")
	}
	if len(vectors) == 0 {
		return []int64{}, nil
	}
	if len(vectors)%km.d != 0 {
		return nil, ErrInvalidVectors
	}

	n := len(vectors) / km.d
	assignments := make([]int64, n)

	// Create a flat index with centroids to perform assignment
	idx, err := NewIndexFlatL2(km.d)
	if err != nil {
		return nil, fmt.Errorf("faiss: failed to create index for assignment: %w", err)
	}
	defer idx.Close()

	// Add centroids to index
	if err := idx.Add(km.centroids); err != nil {
		return nil, fmt.Errorf("faiss: failed to add centroids: %w", err)
	}

	// Search for nearest centroid for each vector
	_, labels, err := idx.Search(vectors, 1)
	if err != nil {
		return nil, fmt.Errorf("faiss: assignment search failed: %w", err)
	}

	copy(assignments, labels)
	return assignments, nil
}

// NOTE: Full Clustering API (faiss.Clustering) not exposed due to C binding complexity.
// Use Kmeans which provides the core functionality via faiss_kmeans_clustering.
