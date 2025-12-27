package faiss

import (
	"fmt"
	"runtime"
)

// Kmeans performs k-means clustering on vectors
//
// Python equivalent: faiss.Kmeans
//
// Example:
//   kmeans, _ := faiss.NewKmeans(128, 100, 25) // d=128, k=100, niter=25
//   kmeans.Train(trainingVectors)
//   centroids := kmeans.Centroids()
//   assignments, _ := kmeans.Assign(vectors)
type Kmeans struct {
	ptr        uintptr  // C pointer
	d          int      // dimension
	k          int      // number of clusters
	niter      int      // number of iterations
	centroids  []float32 // cluster centroids (k * d)
	_obj       []float32 // objective function values (reserved for future use)
	isTrained  bool     // training status
}

// NewKmeans creates a new k-means clustering object
//
// Parameters:
//   - d: dimension of vectors
//   - k: number of clusters
//   - niter: number of k-means iterations (default: 25)
//
// Optional configuration can be done via SetXXX methods before training
func NewKmeans(d, k, niter int) (*Kmeans, error) {
	if d <= 0 {
		return nil, ErrInvalidDimension
	}
	if k <= 0 {
		return nil, fmt.Errorf("faiss: k must be positive")
	}
	if niter <= 0 {
		return nil, fmt.Errorf("faiss: niter must be positive")
	}

	ptr, err := faissKmeansNew(d, k)
	if err != nil {
		return nil, fmt.Errorf("faiss: failed to create Kmeans: %w", err)
	}

	km := &Kmeans{
		ptr:       ptr,
		d:         d,
		k:         k,
		niter:     niter,
		isTrained: false,
	}

	runtime.SetFinalizer(km, func(k *Kmeans) {
		if k.ptr != 0 {
			_ = k.Close()
		}
	})

	// Set default parameters
	_ = faissKmeansSetNiter(ptr, niter)

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

// Niter returns the number of iterations
func (km *Kmeans) Niter() int {
	return km.niter
}

// SetNiter sets the number of k-means iterations
func (km *Kmeans) SetNiter(niter int) error {
	if niter <= 0 {
		return fmt.Errorf("faiss: niter must be positive")
	}

	if err := faissKmeansSetNiter(km.ptr, niter); err != nil {
		return err
	}

	km.niter = niter
	return nil
}

// SetVerbose enables/disables verbose output during training
func (km *Kmeans) SetVerbose(verbose bool) error {
	v := 0
	if verbose {
		v = 1
	}
	return faissKmeansSetVerbose(km.ptr, v)
}

// SetSeed sets the random seed for initialization
func (km *Kmeans) SetSeed(seed int64) error {
	return faissKmeansSetSeed(km.ptr, seed)
}

// Train performs k-means clustering on the training vectors
//
// Parameters:
//   - vectors: training vectors (n vectors of dimension d)
//
// The centroids are stored internally and can be accessed via Centroids()
func (km *Kmeans) Train(vectors []float32) error {
	if km.ptr == 0 {
		return ErrNullPointer
	}
	if len(vectors) == 0 {
		return fmt.Errorf("faiss: cannot train on empty vectors")
	}
	if len(vectors)%km.d != 0 {
		return ErrInvalidVectors
	}

	n := len(vectors) / km.d

	// Train
	if err := faissKmeansTrain(km.ptr, vectors, n); err != nil {
		return fmt.Errorf("faiss: k-means training failed: %w", err)
	}

	// Retrieve centroids
	km.centroids = make([]float32, km.k*km.d)
	if err := faissKmeansGetCentroids(km.ptr, km.centroids); err != nil {
		return fmt.Errorf("faiss: failed to get centroids: %w", err)
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

// Assign assigns vectors to their nearest cluster
//
// Parameters:
//   - vectors: vectors to assign (n vectors of dimension d)
//
// Returns: cluster assignments (n integers, each in range [0, k))
func (km *Kmeans) Assign(vectors []float32) ([]int64, error) {
	if km.ptr == 0 {
		return nil, ErrNullPointer
	}
	if !km.isTrained {
		return nil, ErrNotTrained
	}
	if len(vectors) == 0 {
		return []int64{}, nil
	}
	if len(vectors)%km.d != 0 {
		return nil, ErrInvalidVectors
	}

	n := len(vectors) / km.d
	assignments := make([]int64, n)

	if err := faissKmeansAssign(km.ptr, vectors, n, assignments); err != nil {
		return nil, fmt.Errorf("faiss: assignment failed: %w", err)
	}

	return assignments, nil
}

// IsTrained returns whether the clustering has been trained
func (km *Kmeans) IsTrained() bool {
	return km.isTrained
}

// Close releases resources
func (km *Kmeans) Close() error {
	if km.ptr == 0 {
		return nil
	}

	err := faissKmeansFree(km.ptr)
	km.ptr = 0
	km.centroids = nil
	km.isTrained = false

	if err != nil {
		return fmt.Errorf("faiss: failed to free Kmeans: %w", err)
	}

	return nil
}

// Clustering is a more flexible clustering interface
// Python equivalent: faiss.Clustering
type Clustering struct {
	*Kmeans
	// Can add more clustering algorithms here in the future
}

// NewClustering creates a new clustering object (currently wraps Kmeans)
func NewClustering(d, k int) (*Clustering, error) {
	kmeans, err := NewKmeans(d, k, 25) // default 25 iterations
	if err != nil {
		return nil, err
	}

	return &Clustering{Kmeans: kmeans}, nil
}
