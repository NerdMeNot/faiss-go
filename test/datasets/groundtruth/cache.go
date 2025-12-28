package groundtruth

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"unsafe"

	faiss "github.com/NerdMeNot/faiss-go"
)

// CachedGroundTruth represents serializable ground truth data
type CachedGroundTruth struct {
	Version  int              `json:"version"`  // Cache format version
	N        int              `json:"n"`        // Number of vectors
	NQ       int              `json:"nq"`       // Number of queries
	D        int              `json:"d"`        // Dimension
	K        int              `json:"k"`        // Number of neighbors
	Metric   string           `json:"metric"`   // "L2" or "IP"
	DataHash string           `json:"dataHash"` // SHA256 of vectors+queries
	Results  []GroundTruthIDs `json:"results"`  // Ground truth IDs for each query
}

// GroundTruthIDs contains just the IDs (distances can be recomputed if needed)
type GroundTruthIDs struct {
	IDs []int64 `json:"ids"`
}

const (
	cacheVersion    = 1
	cacheDir        = "test/datasets/groundtruth"
	cacheFilePrefix = "gt_cache_"
)

// GenerateCacheKey creates a unique cache key for a dataset configuration
func GenerateCacheKey(vectors, queries []float32, d, k int, metric faiss.MetricType) string {
	// Create hash of data
	hasher := sha256.New()

	// Hash vectors (use unsafe for efficient byte conversion)
	vectorBytes := (*[1 << 30]byte)(unsafe.Pointer(&vectors[0]))[:len(vectors)*4:len(vectors)*4]
	hasher.Write(vectorBytes)

	// Hash queries
	queryBytes := (*[1 << 30]byte)(unsafe.Pointer(&queries[0]))[:len(queries)*4:len(queries)*4]
	hasher.Write(queryBytes)

	// Hash parameters
	paramStr := fmt.Sprintf("_d%d_k%d_m%d", d, k, metric)
	hasher.Write([]byte(paramStr))

	hash := fmt.Sprintf("%x", hasher.Sum(nil))

	// Return short hash (first 16 chars is enough for uniqueness in our case)
	return hash[:16]
}

// GetCachePath returns the file path for a cache key
func GetCachePath(cacheKey string) string {
	return filepath.Join(cacheDir, cacheFilePrefix+cacheKey+".json")
}

// LoadFromCache attempts to load ground truth from cache
func LoadFromCache(cacheKey string, n, nq, d, k int, metric faiss.MetricType) ([]GroundTruthIDs, bool) {
	cachePath := GetCachePath(cacheKey)

	// Check if cache file exists
	if _, err := os.Stat(cachePath); os.IsNotExist(err) {
		return nil, false
	}

	// Read cache file
	data, err := os.ReadFile(cachePath)
	if err != nil {
		return nil, false
	}

	// Parse JSON
	var cached CachedGroundTruth
	if err := json.Unmarshal(data, &cached); err != nil {
		return nil, false
	}

	// Validate cache
	metricStr := "L2"
	if metric == faiss.MetricInnerProduct {
		metricStr = "IP"
	}

	if cached.Version != cacheVersion ||
		cached.N != n ||
		cached.NQ != nq ||
		cached.D != d ||
		cached.K != k ||
		cached.Metric != metricStr {
		// Cache is stale or incompatible
		return nil, false
	}

	return cached.Results, true
}

// SaveToCache saves ground truth results to cache
func SaveToCache(cacheKey string, groundTruth []GroundTruthIDs, n, nq, d, k int, metric faiss.MetricType, dataHash string) error {
	// Create cache directory if it doesn't exist
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	metricStr := "L2"
	if metric == faiss.MetricInnerProduct {
		metricStr = "IP"
	}

	cached := CachedGroundTruth{
		Version:  cacheVersion,
		N:        n,
		NQ:       nq,
		D:        d,
		K:        k,
		Metric:   metricStr,
		DataHash: dataHash,
		Results:  groundTruth,
	}

	// Marshal to JSON
	data, err := json.MarshalIndent(cached, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal cache: %w", err)
	}

	// Write to file
	cachePath := GetCachePath(cacheKey)
	if err := os.WriteFile(cachePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write cache file: %w", err)
	}

	return nil
}

// ClearCache removes all cached ground truth files
func ClearCache() error {
	// Check if cache directory exists
	if _, err := os.Stat(cacheDir); os.IsNotExist(err) {
		return nil // Nothing to clear
	}

	// Read cache directory
	entries, err := os.ReadDir(cacheDir)
	if err != nil {
		return fmt.Errorf("failed to read cache directory: %w", err)
	}

	// Remove cache files
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		// Only remove cache files (starting with cacheFilePrefix)
		if len(entry.Name()) > len(cacheFilePrefix) && entry.Name()[:len(cacheFilePrefix)] == cacheFilePrefix {
			cachePath := filepath.Join(cacheDir, entry.Name())
			if err := os.Remove(cachePath); err != nil {
				return fmt.Errorf("failed to remove cache file %s: %w", entry.Name(), err)
			}
		}
	}

	return nil
}
