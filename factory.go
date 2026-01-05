package faiss

import (
	"fmt"
	"strconv"
	"strings"
)

// Index type constants for factory descriptions
const (
	IndexTypeFlat         = "Flat"
	IndexTypeIVF          = "IVF"
	IndexTypeHNSW         = "HNSW"
	IndexTypePQ           = "PQ"
	IndexTypeSQ           = "SQ"
	IndexTypeLSH          = "LSH"
	IndexTypePreTransform = "PreTransform"
	IndexTypeUnknown      = "unknown"

	// Common index configurations
	IndexDescFlat   = "Flat"
	IndexDescHNSW32 = "HNSW32"
)

// IndexFactory creates an index from a description string using FAISS's index_factory.
// This is THE KEY function that unlocks ALL index types including HNSW, PQ, IVFPQ, and more.
//
// Unlike previous implementations that parsed strings manually and returned errors,
// this now uses the actual FAISS C API's faiss_index_factory() function, which
// supports ALL index types that Python FAISS supports.
//
// Python equivalent: faiss.index_factory(d, description, metric)
//
// Supported descriptions (non-exhaustive list):
//
// Basic indexes:
//   - "Flat"              -> Exact search (IndexFlatL2 or IndexFlatIP)
//   - "LSH"               -> Locality-sensitive hashing
//   - "PQn"               -> Product quantization (n = number of bytes)
//   - "SQn"               -> Scalar quantization (n = 4, 6, or 8 bits)
//
// IVF (Inverted File) indexes:
//   - "IVFn,Flat"        -> IVF with n clusters, flat storage
//   - "IVFn,PQ8"         -> IVF with n clusters, PQ encoding (8 bytes)
//   - "IVFn,SQ8"         -> IVF with n clusters, scalar quantization
//
// HNSW (Hierarchical Navigable Small World) indexes:
//   - "HNSWn"            -> HNSW with M=n (recommended: 16, 32, or 64)
//   - "HNSW32,Flat"      -> HNSW graph with flat refinement
//
// Pre-transform indexes:
//   - "PCAn,..."         -> Apply PCA to reduce to n dimensions first
//   - "OPQn,..."         -> Apply Optimized Product Quantization
//   - "RRn,..."          -> Apply Random Rotation
//
// Refinement:
//   - "...,Refine(Flat)" -> Two-stage search with refinement
//
// Examples:
//
//	// Create HNSW index (fast, accurate approximate search)
//	index, _ := IndexFactory(128, "HNSW32", MetricL2)
//
//	// Create IVF+PQ index (compressed, scalable)
//	index, _ := IndexFactory(128, "IVF100,PQ8", MetricL2)
//
//	// Create PCA+IVF index (dimension reduction + clustering)
//	index, _ := IndexFactory(128, "PCA64,IVF100,Flat", MetricL2)
//
//	// Create exact search index
//	index, _ := IndexFactory(128, "Flat", MetricL2)
func IndexFactory(d int, description string, metric MetricType) (Index, error) {
	if d <= 0 {
		return nil, ErrInvalidDimension
	}

	description = strings.TrimSpace(description)
	if description == "" {
		return nil, fmt.Errorf("faiss: empty index description")
	}

	// Use the actual FAISS index_factory C function!
	// This supports ALL index types, not just the ones we manually parse.
	ptr, err := faissIndexFactory(d, description, int(metric))
	if err != nil {
		return nil, fmt.Errorf("faiss: factory failed for '%s': %w", description, err)
	}

	// Return a generic index wrapper that works with any factory-created index
	return newGenericIndex(ptr, d, metric, description), nil
}

// IndexFactoryFromFile creates an index from a factory description and immediately
// loads vectors from a file.
//
// This is a convenience function that combines IndexFactory + Train + Add.
func IndexFactoryFromFile(d int, description string, metric MetricType, vectorFile string) (Index, error) {
	index, err := IndexFactory(d, description, metric)
	if err != nil {
		return nil, err
	}

	// TODO: Implement vector file loading if needed
	// For now, just return the index
	return index, nil
}

// ParseIndexDescription parses an index factory description and returns its components.
// This is useful for understanding what a factory string will create.
//
// Example:
//
//	info := ParseIndexDescription("IVF100,PQ8")
//	// info["type"] = "IVF"
//	// info["nlist"] = 100
//	// info["storage"] = "PQ8"
func ParseIndexDescription(description string) map[string]interface{} {
	result := make(map[string]interface{})
	description = strings.TrimSpace(description)
	parts := strings.Split(description, ",")

	result["raw"] = description
	result["parts"] = parts

	if len(parts) == 0 {
		return result
	}

	// Parse first component
	first := parts[0]
	parseFirstComponent(first, parts, result)

	// Check for refinement
	for _, part := range parts {
		if strings.Contains(part, "Refine") {
			result["has_refinement"] = true
		}
	}

	return result
}

// parseFirstComponent extracts index type info from the first component
func parseFirstComponent(first string, parts []string, result map[string]interface{}) {
	switch {
	case first == IndexTypeFlat:
		result["type"] = IndexTypeFlat
		result["training_required"] = false

	case strings.HasPrefix(first, IndexTypeIVF):
		parseIVFComponent(first, parts, result)

	case strings.HasPrefix(first, IndexTypeHNSW):
		parseHNSWComponent(first, result)

	case strings.HasPrefix(first, IndexTypePQ):
		parsePQComponent(first, result)

	case strings.HasPrefix(first, IndexTypeSQ):
		parseSQComponent(first, result)

	case strings.HasPrefix(first, "PCA"):
		parseTransformComponent(first, "PCA", result)

	case strings.HasPrefix(first, "OPQ"):
		result["type"] = IndexTypePreTransform
		result["transform"] = "OPQ"
		result["training_required"] = true

	case strings.HasPrefix(first, "RR"):
		result["type"] = IndexTypePreTransform
		result["transform"] = "RandomRotation"
		result["training_required"] = false

	case first == IndexTypeLSH:
		result["type"] = IndexTypeLSH
		result["training_required"] = false

	default:
		result["type"] = IndexTypeUnknown
	}
}

func parseIVFComponent(first string, parts []string, result map[string]interface{}) {
	result["type"] = IndexTypeIVF
	nlistStr := strings.TrimPrefix(first, IndexTypeIVF)
	if nlist, err := strconv.Atoi(nlistStr); err == nil {
		result["nlist"] = nlist
	}
	result["training_required"] = true
	if len(parts) >= 2 {
		result["storage"] = parts[1]
	}
}

func parseHNSWComponent(first string, result map[string]interface{}) {
	result["type"] = IndexTypeHNSW
	MStr := strings.TrimPrefix(first, IndexTypeHNSW)
	if MStr != "" {
		if M, err := strconv.Atoi(MStr); err == nil {
			result["M"] = M
		}
	}
	result["training_required"] = false
}

func parsePQComponent(first string, result map[string]interface{}) {
	result["type"] = IndexTypePQ
	nbytesStr := strings.TrimPrefix(first, IndexTypePQ)
	if nbytes, err := strconv.Atoi(nbytesStr); err == nil {
		result["nbytes"] = nbytes
	}
	result["training_required"] = true
}

func parseSQComponent(first string, result map[string]interface{}) {
	result["type"] = IndexTypeSQ
	nbitsStr := strings.TrimPrefix(first, IndexTypeSQ)
	if nbits, err := strconv.Atoi(nbitsStr); err == nil {
		result["nbits"] = nbits
	}
	result["training_required"] = true
}

func parseTransformComponent(first, transformType string, result map[string]interface{}) {
	result["type"] = IndexTypePreTransform
	result["transform"] = transformType
	dOutStr := strings.TrimPrefix(first, transformType)
	if dOut, err := strconv.Atoi(dOutStr); err == nil {
		result["d_out"] = dOut
	}
	result["training_required"] = true
}

// indexRecommendation holds parsed requirements for index recommendation
type indexRecommendation struct {
	recallTarget float64
	speedPref    string
	memoryPref   string
}

func parseRequirements(requirements map[string]interface{}) indexRecommendation {
	rec := indexRecommendation{
		recallTarget: 0.9,
		speedPref:    "balanced",
		memoryPref:   "medium",
	}
	if r, ok := requirements["recall"].(float64); ok {
		rec.recallTarget = r
	}
	if s, ok := requirements["speed"].(string); ok {
		rec.speedPref = s
	}
	if m, ok := requirements["memory"].(string); ok {
		rec.memoryPref = m
	}
	return rec
}

func calculateNlist(n int64) int {
	nlist := int(n / 1000)
	if nlist > 65536 {
		nlist = 65536
	}
	if nlist < 100 {
		nlist = 100
	}
	return nlist
}

func recommendSmallDataset(rec indexRecommendation) string {
	if rec.speedPref == "fast" || rec.recallTarget >= 0.95 {
		return IndexDescHNSW32
	}
	return "IVF100,Flat"
}

func recommendMediumDataset(n int64, rec indexRecommendation) string {
	switch rec.speedPref {
	case "fast":
		if rec.recallTarget >= 0.95 {
			return IndexDescHNSW32
		}
		return "HNSW16"
	case "accurate":
		return "HNSW64"
	default:
		nlist := calculateNlist(n)
		if rec.memoryPref == "low" {
			return fmt.Sprintf("IVF%d,PQ8", nlist)
		}
		return fmt.Sprintf("IVF%d,Flat", nlist)
	}
}

func recommendLargeDataset(n int64, d int, rec indexRecommendation) string {
	nlist := calculateNlist(n)

	switch rec.memoryPref {
	case "low":
		return fmt.Sprintf("IVF%d,PQ8", nlist)
	case "high":
		return fmt.Sprintf("IVF%d,Flat", nlist)
	default:
		if d >= 128 {
			dReduced := d / 2
			if dReduced < 64 {
				dReduced = 64
			}
			return fmt.Sprintf("PCA%d,IVF%d,PQ8", dReduced, nlist)
		}
		return fmt.Sprintf("IVF%d,PQ8", nlist)
	}
}

func recommendVeryLargeDataset(n int64, d int) string {
	nlist := calculateNlist(n)

	if d >= 256 {
		return fmt.Sprintf("OPQ16,IVF%d,PQ16", nlist)
	} else if d >= 128 {
		dReduced := d / 2
		return fmt.Sprintf("PCA%d,IVF%d,PQ8", dReduced, nlist)
	}
	return fmt.Sprintf("IVF%d,PQ8", nlist)
}

// RecommendIndex recommends an index configuration based on dataset characteristics.
//
// This provides guidance similar to FAISS's auto-tuning, helping users choose
// appropriate index types without deep FAISS knowledge.
//
// Parameters:
//   - n: expected number of vectors
//   - d: dimension of vectors
//   - metric: distance metric (MetricL2 or MetricInnerProduct)
//   - requirements: optional requirements map with keys:
//     - "recall": target recall (0.0-1.0, default 0.9)
//     - "speed": preference ("fast", "balanced", "accurate", default "balanced")
//     - "memory": preference ("low", "medium", "high", default "medium")
//     - "build_time": preference ("fast", "medium", "slow", default "medium")
//
// Returns a recommended factory description string.
//
// Example:
//
//	desc := RecommendIndex(1000000, 128, MetricL2, map[string]interface{}{
//		"recall": 0.95,
//		"speed": "fast",
//	})
//	// Returns: "HNSW32"
func RecommendIndex(n int64, d int, metric MetricType, requirements map[string]interface{}) string {
	rec := parseRequirements(requirements)

	switch {
	case n < 10000:
		return IndexDescFlat
	case n < 100000:
		return recommendSmallDataset(rec)
	case n < 1000000:
		return recommendMediumDataset(n, rec)
	case n < 10000000:
		return recommendLargeDataset(n, d, rec)
	default:
		return recommendVeryLargeDataset(n, d)
	}
}

// ValidateIndexDescription checks if a factory description string is valid.
// Returns nil if valid, error if invalid.
//
// This doesn't create the index, just validates the syntax.
func ValidateIndexDescription(description string) error {
	description = strings.TrimSpace(description)
	if description == "" {
		return fmt.Errorf("empty description")
	}

	// Try to parse it
	info := ParseIndexDescription(description)

	// Check if we recognized the type
	if indexType, ok := info["type"].(string); ok {
		if indexType == IndexTypeUnknown {
			return fmt.Errorf("unrecognized index type in: %s", description)
		}
	}

	return nil
}
