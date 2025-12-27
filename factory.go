package faiss

import (
	"fmt"
	"strconv"
	"strings"
)

const (
	indexTypeFlat       = "Flat"
	indexTypeIndexFlatL2 = "IndexFlatL2" // for factory string matching
)

// IndexFactory creates an index from a description string
//
// This provides a convenient way to create indexes with various configurations
// similar to Python FAISS's index_factory function.
//
// Python equivalent: faiss.index_factory(d, description, metric)
//
// Supported descriptions:
//   - "Flat"              -> IndexFlatL2 or IndexFlatIP
//   - "IVFn,Flat"        -> IndexIVFFlat with n clusters
//   - "HNSW32"           -> IndexHNSW with M=32
//   - "PQ8"              -> IndexPQ with M=8
//   - "IVFn,PQ8"         -> IndexIVFPQ with n clusters and M=8
//
// Examples:
//   index, _ := faiss.IndexFactory(128, "Flat", faiss.MetricL2)
//   index, _ := faiss.IndexFactory(128, "IVF100,Flat", faiss.MetricL2)
//   index, _ := faiss.IndexFactory(128, "HNSW32", faiss.MetricL2)
//
//nolint:gocyclo,goconst // Factory pattern naturally has high complexity
func IndexFactory(d int, description string, metric MetricType) (Index, error) {
	if d <= 0 {
		return nil, ErrInvalidDimension
	}

	description = strings.TrimSpace(description)
	parts := strings.Split(description, ",")

	// Parse the description
	switch {
	case description == indexTypeFlat:
		// Simple flat index
		if metric == MetricL2 {
			return NewIndexFlatL2(d)
		}
		return NewIndexFlatIP(d)

	case strings.HasPrefix(parts[0], "IVF") && len(parts) >= 2:
		// IVF-based index
		// Extract nlist from "IVFnnn"
		nlistStr := strings.TrimPrefix(parts[0], "IVF")
		nlist, err := strconv.Atoi(nlistStr)
		if err != nil {
			return nil, fmt.Errorf("faiss: invalid IVF description: %s", parts[0])
		}

		// Create quantizer
		var quantizer Index
		if metric == MetricL2 {
			quantizer, err = NewIndexFlatL2(d)
		} else {
			quantizer, err = NewIndexFlatIP(d)
		}
		if err != nil {
			return nil, err
		}

		// Check second part for storage type
		if parts[1] == "Flat" {
			return NewIndexIVFFlat(quantizer, d, nlist, metric)
		}

		// Support PQ encoding: "PQnn" where nn is M value
		if strings.HasPrefix(parts[1], "PQ") {
			MStr := strings.TrimPrefix(parts[1], "PQ")
			M, err := strconv.Atoi(MStr)
			if err != nil {
				return nil, fmt.Errorf("faiss: invalid PQ description: %s", parts[1])
			}
			nbits := 8 // default nbits for PQ
			return NewIndexIVFPQ(quantizer, d, nlist, M, nbits)
		}

		// Future: support SQ, etc.
		return nil, fmt.Errorf("faiss: unsupported IVF storage type: %s", parts[1])

	case strings.HasPrefix(description, "PQ"):
		// Standalone PQ index: "PQnn" where nn is M value
		MStr := strings.TrimPrefix(description, "PQ")
		M, err := strconv.Atoi(MStr)
		if err != nil {
			return nil, fmt.Errorf("faiss: invalid PQ description: %s", description)
		}
		nbits := 8 // default nbits for PQ
		return NewIndexPQ(d, M, nbits, metric)

	case strings.HasPrefix(description, "HNSW"):
		// HNSW index
		// Extract M from "HNSWnn"
		MStr := strings.TrimPrefix(description, "HNSW")
		M := 32 // default
		if MStr != "" {
			var err error
			M, err = strconv.Atoi(MStr)
			if err != nil {
				return nil, fmt.Errorf("faiss: invalid HNSW description: %s", description)
			}
		}

		return NewIndexHNSWFlat(d, M, metric)

	case strings.HasPrefix(description, "IDMap"):
		// ID mapping wrapper
		// Format: "IDMap,<base_description>"
		if len(parts) < 2 {
			return nil, fmt.Errorf("faiss: IDMap requires base index description")
		}

		baseDesc := strings.Join(parts[1:], ",")
		baseIndex, err := IndexFactory(d, baseDesc, metric)
		if err != nil {
			return nil, err
		}

		return NewIndexIDMap(baseIndex)

	default:
		return nil, fmt.Errorf("faiss: unsupported index description: %s", description)
	}
}

// ParseIndexDescription parses an index factory description and returns its components
func ParseIndexDescription(description string) map[string]interface{} {
	result := make(map[string]interface{})
	description = strings.TrimSpace(description)
	parts := strings.Split(description, ",")

	result["raw"] = description

	switch {
	case description == indexTypeFlat:
		result["type"] = indexTypeFlat

	case strings.HasPrefix(parts[0], "IVF"):
		result["type"] = "IVF"
		nlistStr := strings.TrimPrefix(parts[0], "IVF")
		if nlist, err := strconv.Atoi(nlistStr); err == nil {
			result["nlist"] = nlist
		}
		if len(parts) >= 2 {
			result["storage"] = parts[1]
		}

	case strings.HasPrefix(description, "HNSW"):
		result["type"] = "HNSW"
		MStr := strings.TrimPrefix(description, "HNSW")
		if MStr != "" {
			if M, err := strconv.Atoi(MStr); err == nil {
				result["M"] = M
			}
		}

	case strings.HasPrefix(description, "IDMap"):
		result["type"] = "IDMap"
		if len(parts) >= 2 {
			result["base"] = strings.Join(parts[1:], ",")
		}
	}

	return result
}

// RecommendIndex recommends an index configuration based on dataset size and requirements
//
// Parameters:
//   - n: expected number of vectors
//   - d: dimension
//   - metric: distance metric
//   - requirements: map of requirements (e.g., "recall": 0.95, "speed": "fast")
//
// Returns a recommended index description string for use with IndexFactory
func RecommendIndex(n int64, d int, metric MetricType, requirements map[string]interface{}) string {
	// Extract requirements
	recallTarget := 0.9 // default
	if r, ok := requirements["recall"].(float64); ok {
		recallTarget = r
	}

	speedPref := "balanced" // "fast", "balanced", "accurate"
	if s, ok := requirements["speed"].(string); ok {
		speedPref = s
	}

	// Recommendation logic based on dataset size and requirements
	switch {
	case n < 10000:
		// Small dataset: use exact search
		return indexTypeFlat

	case n < 1000000 && speedPref == "fast":
		// Medium dataset, need speed: use HNSW
		if recallTarget >= 0.95 {
			return "HNSW32"
		}
		return "HNSW16"

	case n < 1000000:
		// Medium dataset, balanced: use IVF
		nlist := int(n / 1000)
		if nlist < 100 {
			nlist = 100
		}
		return fmt.Sprintf("IVF%d,Flat", nlist)

	case n >= 1000000 && speedPref == "accurate":
		// Large dataset, need accuracy: use HNSW with larger M
		return "HNSW48"

	default:
		// Large dataset: use IVF with appropriate nlist
		nlist := int(n / 1000)
		if nlist > 65536 {
			nlist = 65536
		}
		return fmt.Sprintf("IVF%d,Flat", nlist)
	}
}
