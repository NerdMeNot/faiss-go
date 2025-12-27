package datasets

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// RealDataset represents a real-world dataset for testing
type RealDataset struct {
	Name        string
	Vectors     []float32 // Base vectors
	Queries     []float32 // Query vectors
	GroundTruth [][]int64 // Ground truth nearest neighbors for each query
	N           int       // Number of base vectors
	NQ          int       // Number of query vectors
	D           int       // Dimension
	K           int       // Number of neighbors in ground truth
}

// DatasetInfo contains metadata about available datasets
type DatasetInfo struct {
	Name        string
	Description string
	BaseFile    string // Filename for base vectors
	QueryFile   string // Filename for query vectors
	GTFile      string // Filename for ground truth
	N           int    // Number of vectors
	NQ          int    // Number of queries
	D           int    // Dimension
	Format      string // fvecs, bvecs, etc.
	URL         string // Download URL
}

// AvailableDatasets returns information about standard benchmark datasets
func AvailableDatasets() []DatasetInfo {
	return []DatasetInfo{
		{
			Name:        "SIFT1M",
			Description: "1M SIFT descriptors (128-dim)",
			BaseFile:    "sift1m_base.fvecs",
			QueryFile:   "sift1m_query.fvecs",
			GTFile:      "sift1m_groundtruth.ivecs",
			N:           1000000,
			NQ:          10000,
			D:           128,
			Format:      "fvecs",
			URL:         "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
		},
		{
			Name:        "SIFT10K",
			Description: "10K SIFT descriptors (subset of SIFT1M)",
			BaseFile:    "sift10k_base.fvecs",
			QueryFile:   "sift10k_query.fvecs",
			GTFile:      "sift10k_groundtruth.ivecs",
			N:           10000,
			NQ:          100,
			D:           128,
			Format:      "fvecs",
			URL:         "(generated from SIFT1M)",
		},
		{
			Name:        "GIST1M",
			Description: "1M GIST descriptors (960-dim)",
			BaseFile:    "gist1m_base.fvecs",
			QueryFile:   "gist1m_query.fvecs",
			GTFile:      "gist1m_groundtruth.ivecs",
			N:           1000000,
			NQ:          1000,
			D:           960,
			Format:      "fvecs",
			URL:         "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
		},
	}
}

// LoadDataset loads a dataset from the testdata directory
func LoadDataset(name string, testdataPath string) (*RealDataset, error) {
	// Find dataset info
	var info DatasetInfo
	found := false
	for _, ds := range AvailableDatasets() {
		if ds.Name == name {
			info = ds
			found = true
			break
		}
	}

	if !found {
		return nil, fmt.Errorf("unknown dataset: %s", name)
	}

	// Load vectors
	basePath := filepath.Join(testdataPath, "embeddings", info.BaseFile)
	vectors, n, d, err := loadFVecs(basePath)
	if err != nil {
		return nil, fmt.Errorf("failed to load base vectors: %w", err)
	}

	// Load queries
	queryPath := filepath.Join(testdataPath, "embeddings", info.QueryFile)
	queries, nq, dq, err := loadFVecs(queryPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load queries: %w", err)
	}

	if d != dq {
		return nil, fmt.Errorf("dimension mismatch: base=%d, query=%d", d, dq)
	}

	// Load ground truth
	gtPath := filepath.Join(testdataPath, "embeddings", info.GTFile)
	groundTruth, k, err := loadGroundTruth(gtPath, nq)
	if err != nil {
		return nil, fmt.Errorf("failed to load ground truth: %w", err)
	}

	return &RealDataset{
		Name:        name,
		Vectors:     vectors,
		Queries:     queries,
		GroundTruth: groundTruth,
		N:           n,
		NQ:          nq,
		D:           d,
		K:           k,
	}, nil
}

// loadFVecs loads vectors from .fvecs format
// Format: [d][v1_1][v1_2]...[v1_d][d][v2_1]...
// Each value is a 4-byte float32
func loadFVecs(filename string) ([]float32, int, int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, 0, 0, err
	}
	defer file.Close()

	// Read first dimension
	var d int32
	if err := binary.Read(file, binary.LittleEndian, &d); err != nil {
		return nil, 0, 0, fmt.Errorf("failed to read dimension: %w", err)
	}

	// Get file size to calculate number of vectors
	stat, err := file.Stat()
	if err != nil {
		return nil, 0, 0, err
	}

	dim := int(d)
	bytesPerVector := 4 + dim*4 // 4 bytes for d, then d*4 bytes for vector
	n := int(stat.Size()) / bytesPerVector

	if int64(n*bytesPerVector) != stat.Size() {
		return nil, 0, 0, fmt.Errorf("file size mismatch")
	}

	// Allocate and read vectors
	vectors := make([]float32, n*dim)

	// Seek back to start
	if _, err := file.Seek(0, 0); err != nil {
		return nil, 0, 0, err
	}

	for i := 0; i < n; i++ {
		// Read dimension (should match first one)
		var dimCheck int32
		if err := binary.Read(file, binary.LittleEndian, &dimCheck); err != nil {
			return nil, 0, 0, fmt.Errorf("failed to read dimension at vector %d: %w", i, err)
		}
		if dimCheck != d {
			return nil, 0, 0, fmt.Errorf("dimension mismatch at vector %d: expected %d, got %d", i, d, dimCheck)
		}

		// Read vector
		for j := 0; j < dim; j++ {
			if err := binary.Read(file, binary.LittleEndian, &vectors[i*dim+j]); err != nil {
				return nil, 0, 0, fmt.Errorf("failed to read vector %d component %d: %w", i, j, err)
			}
		}
	}

	return vectors, n, dim, nil
}

// loadGroundTruth loads ground truth nearest neighbors from .ivecs format
// Format: [k][id1][id2]...[idk][k][id1]...
// Each value is a 4-byte int32
func loadGroundTruth(filename string, expectedQueries int) ([][]int64, int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, 0, err
	}
	defer file.Close()

	// Read first k
	var k int32
	if err := binary.Read(file, binary.LittleEndian, &k); err != nil {
		return nil, 0, fmt.Errorf("failed to read k: %w", err)
	}

	// Allocate ground truth
	groundTruth := make([][]int64, expectedQueries)

	// Seek back to start
	if _, err := file.Seek(0, 0); err != nil {
		return nil, 0, err
	}

	for i := 0; i < expectedQueries; i++ {
		// Read k
		var kCheck int32
		if err := binary.Read(file, binary.LittleEndian, &kCheck); err != nil {
			if err == io.EOF {
				return nil, 0, fmt.Errorf("unexpected EOF at query %d", i)
			}
			return nil, 0, fmt.Errorf("failed to read k at query %d: %w", i, err)
		}
		if kCheck != k {
			return nil, 0, fmt.Errorf("k mismatch at query %d: expected %d, got %d", i, k, kCheck)
		}

		// Read IDs
		groundTruth[i] = make([]int64, int(k))
		for j := 0; j < int(k); j++ {
			var id int32
			if err := binary.Read(file, binary.LittleEndian, &id); err != nil {
				return nil, 0, fmt.Errorf("failed to read ID at query %d, neighbor %d: %w", i, j, err)
			}
			groundTruth[i][j] = int64(id)
		}
	}

	return groundTruth, int(k), nil
}

// SaveFVecs saves vectors to .fvecs format (useful for creating test datasets)
func SaveFVecs(filename string, vectors []float32, n, d int) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	dim := int32(d)
	for i := 0; i < n; i++ {
		// Write dimension
		if err := binary.Write(file, binary.LittleEndian, dim); err != nil {
			return fmt.Errorf("failed to write dimension: %w", err)
		}

		// Write vector
		for j := 0; j < d; j++ {
			if err := binary.Write(file, binary.LittleEndian, vectors[i*d+j]); err != nil {
				return fmt.Errorf("failed to write component: %w", err)
			}
		}
	}

	return nil
}

// SaveGroundTruth saves ground truth to .ivecs format
func SaveGroundTruth(filename string, groundTruth [][]int64) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	for i, ids := range groundTruth {
		k := int32(len(ids))

		// Write k
		if err := binary.Write(file, binary.LittleEndian, k); err != nil {
			return fmt.Errorf("failed to write k: %w", err)
		}

		// Write IDs
		for j, id := range ids {
			id32 := int32(id)
			if err := binary.Write(file, binary.LittleEndian, id32); err != nil {
				return fmt.Errorf("failed to write ID at query %d, neighbor %d: %w", i, j, err)
			}
		}
	}

	return nil
}

// CreateSubset creates a smaller subset from a large dataset
// Useful for creating SIFT10K from SIFT1M, etc.
func CreateSubset(source *RealDataset, nBase, nQuery int) *RealDataset {
	if nBase > source.N {
		nBase = source.N
	}
	if nQuery > source.NQ {
		nQuery = source.NQ
	}

	subset := &RealDataset{
		Name:    fmt.Sprintf("%s-subset-%d", source.Name, nBase),
		Vectors: source.Vectors[:nBase*source.D],
		Queries: source.Queries[:nQuery*source.D],
		N:       nBase,
		NQ:      nQuery,
		D:       source.D,
	}

	// Update ground truth to only reference vectors in subset
	subset.GroundTruth = make([][]int64, nQuery)
	for i := 0; i < nQuery; i++ {
		subset.GroundTruth[i] = make([]int64, 0, source.K)
		for _, id := range source.GroundTruth[i] {
			if id < int64(nBase) {
				subset.GroundTruth[i] = append(subset.GroundTruth[i], id)
			}
		}
	}

	if len(subset.GroundTruth[0]) > 0 {
		subset.K = len(subset.GroundTruth[0])
	}

	return subset
}

// IsDatasetAvailable checks if a dataset is downloaded
func IsDatasetAvailable(name string, testdataPath string) bool {
	for _, ds := range AvailableDatasets() {
		if ds.Name == name {
			basePath := filepath.Join(testdataPath, "embeddings", ds.BaseFile)
			_, err := os.Stat(basePath)
			return err == nil
		}
	}
	return false
}
