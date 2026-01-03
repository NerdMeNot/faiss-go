package datasets

import (
	"math"
	"math/rand"
	"os"
	"testing"
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

// GeneratePerturbedQueries creates query vectors as noisy perturbations of actual vectors
// This ensures queries have known nearest neighbors (the vectors they were perturbed from)
// noiseLevel controls the amount of noise (0.0 = identical, 0.1 = 10% noise, etc.)
func (d *SyntheticDataset) GeneratePerturbedQueries(numQueries int, noiseLevel float32) {
	if len(d.Vectors) == 0 {
		panic("Cannot generate perturbed queries from empty dataset")
	}

	d.Queries = make([]float32, numQueries*d.D)
	d.NumQueries = numQueries

	// For each query, pick a uniformly distributed random vector and add noise
	// We distribute source vectors evenly across the dataset to ensure good coverage
	for i := 0; i < numQueries; i++ {
		// Use modulo to distribute queries evenly, with some randomness
		// This ensures we sample from across the entire dataset
		baseIdx := (i * d.N) / numQueries  // Evenly distribute
		offset := rand.Intn(max(1, d.N/numQueries))  // Add some randomness
		srcIdx := (baseIdx + offset) % d.N

		// Copy the vector and add Gaussian noise
		for j := 0; j < d.D; j++ {
			originalValue := d.Vectors[srcIdx*d.D+j]
			noise := float32(rand.NormFloat64()) * noiseLevel
			d.Queries[i*d.D+j] = originalValue + noise
		}
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

// DatasetConfig defines the size parameters for a dataset
type DatasetConfig struct {
	N   int // Number of vectors
	D   int // Dimension
	NQ  int // Number of queries
	K   int // Number of neighbors for recall
}

// CI-friendly dataset sizes (small, fast tests)
var CIDatasetConfigs = map[string]DatasetConfig{
	// Recall tests - small datasets for quick validation
	"ivf_recall":        {N: 5000, D: 128, NQ: 50, K: 10},
	"hnsw_recall":       {N: 5000, D: 128, NQ: 50, K: 10},
	"pq_recall":         {N: 5000, D: 128, NQ: 50, K: 10},
	"ivfpq_best":        {N: 10000, D: 256, NQ: 100, K: 10},
	"ivf_optimal":       {N: 10000, D: 256, NQ: 100, K: 10},
	"ivf_training":      {N: 10000, D: 128, NQ: 100, K: 10},

	// Scenario tests - reduced for CI
	"semantic_search":   {N: 10000, D: 384, NQ: 100, K: 10},
	"image_similarity":  {N: 10000, D: 512, NQ: 50, K: 10},
	"recommendations":   {N: 10000, D: 256, NQ: 50, K: 50},

	// Parameter sweeps - minimal datasets
	"param_sweep":       {N: 5000, D: 128, NQ: 50, K: 10},
	"high_dimensional":  {N: 5000, D: 1536, NQ: 50, K: 10},
}

// Local testing dataset sizes (medium, realistic tests)
var LocalDatasetConfigs = map[string]DatasetConfig{
	// Recall tests - medium datasets for quality validation
	"ivf_recall":        {N: 10000, D: 128, NQ: 100, K: 10},
	"hnsw_recall":       {N: 10000, D: 256, NQ: 100, K: 10},
	"pq_recall":         {N: 10000, D: 128, NQ: 100, K: 10},
	"ivfpq_best":        {N: 100000, D: 256, NQ: 100, K: 10},
	"ivf_optimal":       {N: 100000, D: 256, NQ: 100, K: 10},
	"ivf_training":      {N: 50000, D: 128, NQ: 100, K: 10},

	// Scenario tests - realistic sizes
	"semantic_search":   {N: 50000, D: 768, NQ: 500, K: 10},
	"image_similarity":  {N: 50000, D: 2048, NQ: 100, K: 10},
	"recommendations":   {N: 50000, D: 256, NQ: 100, K: 50},

	// Parameter sweeps - full testing
	"param_sweep":       {N: 10000, D: 256, NQ: 100, K: 10},
	"high_dimensional":  {N: 10000, D: 1536, NQ: 100, K: 10},
}

// IsCI detects if running in a CI environment
func IsCI() bool {
	// Check common CI environment variables
	ciEnvVars := []string{
		"CI",
		"CONTINUOUS_INTEGRATION",
		"GITHUB_ACTIONS",
		"GITLAB_CI",
		"CIRCLECI",
		"TRAVIS",
		"JENKINS_URL",
	}

	for _, envVar := range ciEnvVars {
		if os.Getenv(envVar) != "" {
			return true
		}
	}

	return false
}

// GetDatasetConfig returns the appropriate dataset configuration based on environment
// If name is not found, returns a default small configuration
func GetDatasetConfig(name string) DatasetConfig {
	// Check if we're in CI mode or short test mode
	if IsCI() || testing.Short() {
		if config, ok := CIDatasetConfigs[name]; ok {
			return config
		}
		// Default CI config if name not found
		return DatasetConfig{N: 5000, D: 128, NQ: 50, K: 10}
	}

	// Local/full test mode
	if config, ok := LocalDatasetConfigs[name]; ok {
		return config
	}

	// Default local config if name not found
	return DatasetConfig{N: 10000, D: 256, NQ: 100, K: 10}
}

// GenerateClusteredDataWithGroundTruth creates clustered data with known nearest neighbors
// This is superior to random data for testing because:
// - Recall is predictable (vectors in same cluster should be nearest neighbors)
// - Tests are reproducible with fixed seed
// - Can validate that indexes correctly identify cluster membership
func GenerateClusteredDataWithGroundTruth(n, d, numClusters int, seed int64) *SyntheticDataset {
	if seed != 0 {
		rand.Seed(seed)
	}

	dataset := &SyntheticDataset{
		Vectors: make([]float32, n*d),
		N:       n,
		D:       d,
	}

	// Generate well-separated cluster centers for better recall
	centers := make([][]float32, numClusters)
	for i := 0; i < numClusters; i++ {
		centers[i] = make([]float32, d)
		// Place clusters far apart in a grid pattern for better separation
		gridSize := int(math.Ceil(math.Sqrt(float64(numClusters))))
		gridX := i % gridSize
		gridY := i / gridSize

		for j := 0; j < d; j++ {
			// First two dimensions determine grid position, rest are random
			if j == 0 {
				centers[i][j] = float32(gridX) * 100.0
			} else if j == 1 {
				centers[i][j] = float32(gridY) * 100.0
			} else {
				centers[i][j] = rand.Float32() * 10.0 // Small variance in other dims
			}
		}
	}

	// Assign vectors to clusters with tight clustering (low stddev)
	labels := make([]int, n)
	for i := 0; i < n; i++ {
		clusterID := i % numClusters // Evenly distribute across clusters
		labels[i] = clusterID

		for j := 0; j < d; j++ {
			// Tight Gaussian noise around cluster center (stddev = 1.0)
			noise := rand.NormFloat64() * 1.0
			dataset.Vectors[i*d+j] = centers[clusterID][j] + float32(noise)
		}
	}

	dataset.Labels = labels
	return dataset
}

// GenerateQueriesFromClusters creates queries that are close to specific clusters
// This allows for predictable recall testing:
// - Query i will have its K nearest neighbors in cluster i % numClusters
func (d *SyntheticDataset) GenerateQueriesFromClusters(numQueries int, noiseLevel float32) {
	if len(d.Labels) == 0 {
		panic("Dataset must have cluster labels to generate cluster-based queries")
	}

	numClusters := max(d.Labels...) + 1
	d.Queries = make([]float32, numQueries*d.D)
	d.NumQueries = numQueries

	// Compute cluster centers from actual data
	centers := make([][]float32, numClusters)
	counts := make([]int, numClusters)
	for i := 0; i < numClusters; i++ {
		centers[i] = make([]float32, d.D)
	}

	// Average vectors in each cluster to find center
	for i := 0; i < d.N; i++ {
		clusterID := d.Labels[i]
		counts[clusterID]++
		for j := 0; j < d.D; j++ {
			centers[clusterID][j] += d.Vectors[i*d.D+j]
		}
	}
	for i := 0; i < numClusters; i++ {
		if counts[i] > 0 {
			for j := 0; j < d.D; j++ {
				centers[i][j] /= float32(counts[i])
			}
		}
	}

	// Generate queries near cluster centers
	for i := 0; i < numQueries; i++ {
		clusterID := i % numClusters
		for j := 0; j < d.D; j++ {
			noise := float32(rand.NormFloat64()) * noiseLevel
			d.Queries[i*d.D+j] = centers[clusterID][j] + noise
		}
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
