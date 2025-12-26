package faiss

import (
	"fmt"
	"math"
)

// NormalizeL2 normalizes vectors to unit L2 norm (in place)
//
// This is commonly used before adding vectors to an IndexFlatIP
// for cosine similarity search.
//
// Python equivalent: faiss.normalize_L2(x)
//
// Example:
//   vectors := []float32{ /* your vectors */ }
//   faiss.NormalizeL2(vectors, dimension)
//   index.Add(vectors)  // Now using cosine similarity
func NormalizeL2(vectors []float32, d int) error {
	if len(vectors)%d != 0 {
		return ErrInvalidVectors
	}

	n := len(vectors) / d

	for i := 0; i < n; i++ {
		offset := i * d

		// Compute L2 norm
		var norm float32
		for j := 0; j < d; j++ {
			val := vectors[offset+j]
			norm += val * val
		}
		norm = float32(math.Sqrt(float64(norm)))

		// Avoid division by zero
		if norm < 1e-10 {
			continue
		}

		// Normalize
		for j := 0; j < d; j++ {
			vectors[offset+j] /= norm
		}
	}

	return nil
}

// NormalizeL2Copy normalizes vectors to unit L2 norm (creates a copy)
//
// Returns a new slice with normalized vectors, leaving the input unchanged.
func NormalizeL2Copy(vectors []float32, d int) ([]float32, error) {
	if len(vectors)%d != 0 {
		return nil, ErrInvalidVectors
	}

	normalized := make([]float32, len(vectors))
	copy(normalized, vectors)

	if err := NormalizeL2(normalized, d); err != nil {
		return nil, err
	}

	return normalized, nil
}

// PairwiseDistances computes pairwise distances between two sets of vectors
//
// Python equivalent: faiss.pairwise_distances(x, y, metric)
//
// Parameters:
//   - x: first set of vectors (n1 vectors of dimension d)
//   - y: second set of vectors (n2 vectors of dimension d)
//   - d: dimension
//   - metric: distance metric
//
// Returns: distance matrix of size n1 x n2
func PairwiseDistances(x, y []float32, d int, metric MetricType) ([]float32, error) {
	if len(x)%d != 0 || len(y)%d != 0 {
		return nil, ErrInvalidVectors
	}

	n1 := len(x) / d
	n2 := len(y) / d
	distances := make([]float32, n1*n2)

	for i := 0; i < n1; i++ {
		for j := 0; j < n2; j++ {
			var dist float32

			offsetX := i * d
			offsetY := j * d

			if metric == MetricL2 {
				// L2 distance
				var sum float32
				for k := 0; k < d; k++ {
					diff := x[offsetX+k] - y[offsetY+k]
					sum += diff * diff
				}
				dist = sum // Squared L2 distance

			} else if metric == MetricInnerProduct {
				// Inner product (negative for similarity)
				var sum float32
				for k := 0; k < d; k++ {
					sum += x[offsetX+k] * y[offsetY+k]
				}
				dist = -sum // Negate for distance
			}

			distances[i*n2+j] = dist
		}
	}

	return distances, nil
}

// KNN performs k-nearest neighbor search on a matrix
//
// This is a standalone function that doesn't require creating an index.
// For repeated searches, it's better to create an index.
//
// Parameters:
//   - vectors: database vectors (n vectors of dimension d)
//   - queries: query vectors (nq vectors of dimension d)
//   - d: dimension
//   - k: number of neighbors
//   - metric: distance metric
//
// Returns: distances and indices for each query
func KNN(vectors, queries []float32, d, k int, metric MetricType) ([]float32, []int64, error) {
	if len(vectors)%d != 0 || len(queries)%d != 0 {
		return nil, nil, ErrInvalidVectors
	}
	if k <= 0 {
		return nil, nil, ErrInvalidK
	}

	// Create a temporary flat index
	var index *IndexFlat
	var err error

	if metric == MetricL2 {
		index, err = NewIndexFlatL2(d)
	} else {
		index, err = NewIndexFlatIP(d)
	}
	if err != nil {
		return nil, nil, err
	}
	defer index.Close()

	// Add vectors and search
	if err := index.Add(vectors); err != nil {
		return nil, nil, err
	}

	return index.Search(queries, k)
}

// RangeKNN finds k nearest neighbors within a maximum distance
//
// This combines k-NN search with a distance threshold.
//
// Parameters:
//   - vectors: database vectors
//   - queries: query vectors
//   - d: dimension
//   - k: maximum number of neighbors
//   - maxDistance: maximum distance threshold
//   - metric: distance metric
//
// Returns: distances and indices (may have fewer than k results per query)
func RangeKNN(vectors, queries []float32, d, k int, maxDistance float32, metric MetricType) ([]float32, []int64, error) {
	// Perform regular k-NN search
	distances, indices, err := KNN(vectors, queries, d, k, metric)
	if err != nil {
		return nil, nil, err
	}

	// Filter results by distance threshold
	nq := len(queries) / d
	filteredDist := make([]float32, 0, len(distances))
	filteredIdx := make([]int64, 0, len(indices))

	for i := 0; i < nq; i++ {
		for j := 0; j < k; j++ {
			idx := i*k + j
			if distances[idx] <= maxDistance {
				filteredDist = append(filteredDist, distances[idx])
				filteredIdx = append(filteredIdx, indices[idx])
			}
		}
	}

	return filteredDist, filteredIdx, nil
}

// ComputeRecall computes recall between ground truth and search results
//
// Recall = fraction of true neighbors found in the results
//
// Parameters:
//   - groundTruth: ground truth neighbor indices (nq x k_gt)
//   - results: search result indices (nq x k_results)
//   - nq: number of queries
//   - kGt: k for ground truth
//   - kResults: k for results
//
// Returns: recall value between 0 and 1
func ComputeRecall(groundTruth, results []int64, nq, kGt, kResults int) float64 {
	if len(groundTruth) != nq*kGt || len(results) != nq*kResults {
		return 0.0
	}

	totalCorrect := 0
	k := kGt
	if kResults < kGt {
		k = kResults
	}

	for i := 0; i < nq; i++ {
		gtSet := make(map[int64]bool)
		for j := 0; j < kGt; j++ {
			gtSet[groundTruth[i*kGt+j]] = true
		}

		correct := 0
		for j := 0; j < k; j++ {
			if gtSet[results[i*kResults+j]] {
				correct++
			}
		}
		totalCorrect += correct
	}

	recall := float64(totalCorrect) / float64(nq*k)
	return recall
}

// VectorStats computes statistics about a set of vectors
type VectorStats struct {
	N          int       // Number of vectors
	D          int       // Dimension
	MinNorm    float32   // Minimum L2 norm
	MaxNorm    float32   // Maximum L2 norm
	MeanNorm   float32   // Mean L2 norm
	MinValue   float32   // Minimum component value
	MaxValue   float32   // Maximum component value
	MeanValue  float32   // Mean component value
}

// ComputeVectorStats computes statistics for a set of vectors
func ComputeVectorStats(vectors []float32, d int) (*VectorStats, error) {
	if len(vectors)%d != 0 {
		return nil, ErrInvalidVectors
	}

	n := len(vectors) / d
	if n == 0 {
		return nil, fmt.Errorf("faiss: no vectors")
	}

	stats := &VectorStats{
		N:        n,
		D:        d,
		MinNorm:  float32(math.MaxFloat32),
		MaxNorm:  0,
		MinValue: float32(math.MaxFloat32),
		MaxValue: float32(-math.MaxFloat32),
	}

	var sumNorm, sumValue float64

	for i := 0; i < n; i++ {
		offset := i * d
		var norm float32

		for j := 0; j < d; j++ {
			val := vectors[offset+j]

			// Update value stats
			if val < stats.MinValue {
				stats.MinValue = val
			}
			if val > stats.MaxValue {
				stats.MaxValue = val
			}
			sumValue += float64(val)

			// Compute norm
			norm += val * val
		}

		norm = float32(math.Sqrt(float64(norm)))

		// Update norm stats
		if norm < stats.MinNorm {
			stats.MinNorm = norm
		}
		if norm > stats.MaxNorm {
			stats.MaxNorm = norm
		}
		sumNorm += float64(norm)
	}

	stats.MeanNorm = float32(sumNorm / float64(n))
	stats.MeanValue = float32(sumValue / float64(n*d))

	return stats, nil
}
