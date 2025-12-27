package datasets

import (
	"math"
	"math/rand"
)

// DataDistribution represents different data distribution types
type DataDistribution int

const (
	// UniformRandom generates uniformly distributed random vectors
	UniformRandom DataDistribution = iota
	// GaussianClustered generates vectors in Gaussian clusters
	GaussianClustered
	// PowerLaw generates vectors with power-law distance distribution
	PowerLaw
	// Normalized generates normalized (unit length) vectors
	Normalized
	// Sparse generates sparse vectors (many zeros)
	Sparse
)

// GeneratorConfig configures synthetic data generation
type GeneratorConfig struct {
	N            int              // Number of vectors
	D            int              // Dimension
	Distribution DataDistribution // Distribution type
	NumClusters  int              // For clustered data
	Sparsity     float64          // For sparse data (0.0-1.0)
	Seed         int64            // Random seed for reproducibility
}

// SyntheticDataset contains generated vectors and metadata
type SyntheticDataset struct {
	Vectors    []float32 // Flattened vectors (N*D)
	Queries    []float32 // Query vectors
	Labels     []int     // Cluster labels (for clustered data)
	N          int       // Number of vectors
	D          int       // Dimension
	NumQueries int       // Number of query vectors
}

// GenerateSyntheticData creates synthetic vectors based on configuration
func GenerateSyntheticData(config GeneratorConfig) *SyntheticDataset {
	if config.Seed != 0 {
		rand.Seed(config.Seed)
	}

	dataset := &SyntheticDataset{
		Vectors: make([]float32, config.N*config.D),
		N:       config.N,
		D:       config.D,
	}

	switch config.Distribution {
	case UniformRandom:
		generateUniformRandom(dataset.Vectors, config.N, config.D)
	case GaussianClustered:
		dataset.Labels = generateGaussianClustered(dataset.Vectors, config.N, config.D, config.NumClusters)
	case PowerLaw:
		generatePowerLaw(dataset.Vectors, config.N, config.D)
	case Normalized:
		generateNormalized(dataset.Vectors, config.N, config.D)
	case Sparse:
		generateSparse(dataset.Vectors, config.N, config.D, config.Sparsity)
	}

	return dataset
}

// GenerateQueries creates query vectors from the same distribution
func (d *SyntheticDataset) GenerateQueries(numQueries int, distribution DataDistribution) {
	d.Queries = make([]float32, numQueries*d.D)
	d.NumQueries = numQueries

	switch distribution {
	case UniformRandom:
		generateUniformRandom(d.Queries, numQueries, d.D)
	case GaussianClustered:
		// Use same cluster centers if labels exist
		if len(d.Labels) > 0 {
			numClusters := max(d.Labels...) + 1
			generateGaussianClustered(d.Queries, numQueries, d.D, numClusters)
		} else {
			generateUniformRandom(d.Queries, numQueries, d.D)
		}
	case Normalized:
		generateNormalized(d.Queries, numQueries, d.D)
	case Sparse:
		generateSparse(d.Queries, numQueries, d.D, 0.8)
	default:
		generateUniformRandom(d.Queries, numQueries, d.D)
	}
}

// generateUniformRandom creates uniformly distributed random vectors
func generateUniformRandom(vectors []float32, n, d int) {
	for i := 0; i < n*d; i++ {
		vectors[i] = rand.Float32()
	}
}

// generateGaussianClustered creates vectors in Gaussian clusters
func generateGaussianClustered(vectors []float32, n, d, numClusters int) []int {
	if numClusters <= 0 {
		numClusters = 10
	}

	// Generate cluster centers
	centers := make([][]float32, numClusters)
	for i := 0; i < numClusters; i++ {
		centers[i] = make([]float32, d)
		for j := 0; j < d; j++ {
			centers[i][j] = rand.Float32() * 100.0 // Spread centers
		}
	}

	// Assign vectors to clusters and add Gaussian noise
	labels := make([]int, n)
	for i := 0; i < n; i++ {
		clusterID := rand.Intn(numClusters)
		labels[i] = clusterID

		for j := 0; j < d; j++ {
			// Gaussian noise around cluster center
			noise := rand.NormFloat64() * 5.0 // Stddev = 5
			vectors[i*d+j] = centers[clusterID][j] + float32(noise)
		}
	}

	return labels
}

// generatePowerLaw creates vectors with power-law distance distribution
func generatePowerLaw(vectors []float32, n, d int) {
	// First vector at origin
	for j := 0; j < d; j++ {
		vectors[j] = 0.0
	}

	// Subsequent vectors with power-law distances
	for i := 1; i < n; i++ {
		// Power-law distribution: distance ~ i^(-alpha)
		alpha := 1.5
		distance := math.Pow(float64(i), -alpha) * 100.0

		// Random direction
		direction := make([]float32, d)
		norm := float32(0.0)
		for j := 0; j < d; j++ {
			direction[j] = rand.Float32()*2.0 - 1.0
			norm += direction[j] * direction[j]
		}
		norm = float32(math.Sqrt(float64(norm)))

		// Place vector at scaled distance
		for j := 0; j < d; j++ {
			vectors[i*d+j] = direction[j] / norm * float32(distance)
		}
	}
}

// generateNormalized creates unit-length (L2-normalized) vectors
func generateNormalized(vectors []float32, n, d int) {
	for i := 0; i < n; i++ {
		// Generate random vector
		norm := float32(0.0)
		for j := 0; j < d; j++ {
			vectors[i*d+j] = rand.Float32()*2.0 - 1.0
			norm += vectors[i*d+j] * vectors[i*d+j]
		}
		norm = float32(math.Sqrt(float64(norm)))

		// Normalize to unit length
		if norm > 0 {
			for j := 0; j < d; j++ {
				vectors[i*d+j] /= norm
			}
		}
	}
}

// generateSparse creates sparse vectors with many zeros
func generateSparse(vectors []float32, n, d int, sparsity float64) {
	if sparsity < 0 {
		sparsity = 0
	}
	if sparsity > 1 {
		sparsity = 1
	}

	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			if rand.Float64() > sparsity {
				vectors[i*d+j] = rand.Float32()
			} else {
				vectors[i*d+j] = 0.0
			}
		}
	}
}

// GenerateRealisticEmbeddings creates vectors that simulate real embeddings
// (e.g., BERT, OpenAI, etc.) with realistic properties
func GenerateRealisticEmbeddings(n, d int) *SyntheticDataset {
	dataset := &SyntheticDataset{
		Vectors: make([]float32, n*d),
		N:       n,
		D:       d,
	}

	// Real embeddings typically:
	// 1. Are somewhat normalized
	// 2. Have some correlation between dimensions
	// 3. Form clusters based on semantic similarity

	numClusters := int(math.Sqrt(float64(n))) // ~sqrt(n) clusters
	if numClusters < 10 {
		numClusters = 10
	}
	if numClusters > 1000 {
		numClusters = 1000
	}

	// Generate with clusters
	dataset.Labels = generateGaussianClustered(dataset.Vectors, n, d, numClusters)

	// Normalize (many embedding models output normalized vectors)
	for i := 0; i < n; i++ {
		norm := float32(0.0)
		for j := 0; j < d; j++ {
			norm += dataset.Vectors[i*d+j] * dataset.Vectors[i*d+j]
		}
		norm = float32(math.Sqrt(float64(norm)))

		if norm > 0 {
			for j := 0; j < d; j++ {
				dataset.Vectors[i*d+j] /= norm
			}
		}
	}

	return dataset
}

// GenerateCorrelatedVectors creates vectors with correlated dimensions
// (useful for testing PCA, dimensionality reduction)
func GenerateCorrelatedVectors(n, d, intrinsicDim int) *SyntheticDataset {
	if intrinsicDim > d {
		intrinsicDim = d
	}

	dataset := &SyntheticDataset{
		Vectors: make([]float32, n*d),
		N:       n,
		D:       d,
	}

	// Generate in low-dimensional space
	lowDim := make([]float32, n*intrinsicDim)
	generateUniformRandom(lowDim, n, intrinsicDim)

	// Random projection to high-dimensional space
	projection := make([]float32, intrinsicDim*d)
	generateUniformRandom(projection, intrinsicDim, d)

	// Project: vectors = lowDim @ projection
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			sum := float32(0.0)
			for k := 0; k < intrinsicDim; k++ {
				sum += lowDim[i*intrinsicDim+k] * projection[k*d+j]
			}
			dataset.Vectors[i*d+j] = sum
		}
	}

	return dataset
}

// GenerateStandardSizes returns standard dataset sizes for testing
func GenerateStandardSizes() []struct {
	Name string
	N    int
	D    int
} {
	return []struct {
		Name string
		N    int
		D    int
	}{
		{"tiny", 100, 32},
		{"small", 1_000, 128},
		{"medium", 10_000, 256},
		{"large", 100_000, 512},
		{"xlarge", 1_000_000, 768},
		{"bert-small", 10_000, 768},
		{"bert-medium", 100_000, 768},
		{"openai-small", 10_000, 1536},
		{"resnet-small", 10_000, 2048},
	}
}

// Helper function to find max in slice
func max(nums ...int) int {
	if len(nums) == 0 {
		return 0
	}
	m := nums[0]
	for _, n := range nums[1:] {
		if n > m {
			m = n
		}
	}
	return m
}
