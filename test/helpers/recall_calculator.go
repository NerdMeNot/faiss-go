package helpers

import (
	"fmt"
	"math"
	"time"
)

// RecallMetrics contains quality metrics for search results
type RecallMetrics struct {
	Recall1   float64 // Recall@1
	Recall10  float64 // Recall@10
	Recall100 float64 // Recall@100
	RecallK   float64 // Recall@K for specified K
	Precision float64 // Precision (relevant results / total results)
	MRR       float64 // Mean Reciprocal Rank
	NDCG      float64 // Normalized Discounted Cumulative Gain
}

// PerformanceMetrics contains performance measurements
type PerformanceMetrics struct {
	P50Latency    time.Duration // Median latency
	P95Latency    time.Duration // 95th percentile latency
	P99Latency    time.Duration // 99th percentile latency
	QPS           float64       // Queries per second
	TotalTime     time.Duration // Total time for all queries
	AvgLatency    time.Duration // Average latency
	MinLatency    time.Duration // Minimum latency
	MaxLatency    time.Duration // Maximum latency
}

// SearchResult represents search results for a single query
type SearchResult struct {
	IDs       []int64   // Retrieved IDs
	Distances []float32 // Distances to query
}

// GroundTruth represents ground truth results for evaluation
type GroundTruth struct {
	IDs       []int64   // True nearest neighbor IDs
	Distances []float32 // True distances
}

// CalculateRecall computes recall@k between approximate and ground truth results
// Recall@k = (# of ground truth neighbors in top-k results) / k
func CalculateRecall(groundTruth []GroundTruth, results []SearchResult, k int) float64 {
	if len(groundTruth) == 0 || len(results) == 0 {
		return 0.0
	}

	if len(groundTruth) != len(results) {
		panic(fmt.Sprintf("mismatched lengths: groundTruth=%d, results=%d", len(groundTruth), len(results)))
	}

	totalRecall := 0.0
	numQueries := len(groundTruth)

	for i := 0; i < numQueries; i++ {
		gt := groundTruth[i]
		res := results[i]

		// Adjust k if results have fewer than k entries
		actualK := k
		if len(res.IDs) < k {
			actualK = len(res.IDs)
		}
		if len(gt.IDs) < k {
			actualK = len(gt.IDs)
		}

		if actualK == 0 {
			continue
		}

		// Create set of ground truth IDs (top-k)
		gtSet := make(map[int64]bool, actualK)
		for j := 0; j < actualK && j < len(gt.IDs); j++ {
			gtSet[gt.IDs[j]] = true
		}

		// Count how many results are in ground truth
		matches := 0
		for j := 0; j < actualK && j < len(res.IDs); j++ {
			if gtSet[res.IDs[j]] {
				matches++
			}
		}

		totalRecall += float64(matches) / float64(actualK)
	}

	return totalRecall / float64(numQueries)
}

// CalculatePrecision computes precision for search results
// Precision = (# of relevant results) / (# of retrieved results)
func CalculatePrecision(groundTruth []GroundTruth, results []SearchResult, k int) float64 {
	if len(groundTruth) == 0 || len(results) == 0 {
		return 0.0
	}

	totalPrecision := 0.0
	numQueries := len(groundTruth)

	for i := 0; i < numQueries; i++ {
		gt := groundTruth[i]
		res := results[i]

		actualK := k
		if len(res.IDs) < k {
			actualK = len(res.IDs)
		}

		if actualK == 0 {
			continue
		}

		// Create set of all ground truth IDs
		gtSet := make(map[int64]bool)
		for _, id := range gt.IDs {
			gtSet[id] = true
		}

		// Count relevant results in top-k
		relevant := 0
		for j := 0; j < actualK && j < len(res.IDs); j++ {
			if gtSet[res.IDs[j]] {
				relevant++
			}
		}

		totalPrecision += float64(relevant) / float64(actualK)
	}

	return totalPrecision / float64(numQueries)
}

// CalculateMRR computes Mean Reciprocal Rank
// MRR = average of (1 / rank of first relevant result)
func CalculateMRR(groundTruth []GroundTruth, results []SearchResult) float64 {
	if len(groundTruth) == 0 || len(results) == 0 {
		return 0.0
	}

	totalRR := 0.0
	numQueries := len(groundTruth)

	for i := 0; i < numQueries; i++ {
		gt := groundTruth[i]
		res := results[i]

		// Create set of ground truth IDs
		gtSet := make(map[int64]bool)
		for _, id := range gt.IDs {
			gtSet[id] = true
		}

		// Find first relevant result
		for j := 0; j < len(res.IDs); j++ {
			if gtSet[res.IDs[j]] {
				totalRR += 1.0 / float64(j+1)
				break
			}
		}
	}

	return totalRR / float64(numQueries)
}

// CalculateNDCG computes Normalized Discounted Cumulative Gain
// NDCG@k measures ranking quality with position discounting
func CalculateNDCG(groundTruth []GroundTruth, results []SearchResult, k int) float64 {
	if len(groundTruth) == 0 || len(results) == 0 {
		return 0.0
	}

	totalNDCG := 0.0
	numQueries := len(groundTruth)

	for i := 0; i < numQueries; i++ {
		gt := groundTruth[i]
		res := results[i]

		actualK := k
		if len(res.IDs) < k {
			actualK = len(res.IDs)
		}

		if actualK == 0 {
			continue
		}

		// Create relevance map (position in ground truth = relevance)
		relevanceMap := make(map[int64]float64)
		for j := 0; j < len(gt.IDs); j++ {
			// Higher relevance for better ground truth position
			relevanceMap[gt.IDs[j]] = float64(len(gt.IDs) - j)
		}

		// Calculate DCG (Discounted Cumulative Gain)
		dcg := 0.0
		for j := 0; j < actualK && j < len(res.IDs); j++ {
			rel := relevanceMap[res.IDs[j]]
			dcg += rel / math.Log2(float64(j+2)) // +2 because log2(1)=0
		}

		// Calculate ideal DCG (best possible ranking)
		idcg := 0.0
		for j := 0; j < actualK && j < len(gt.IDs); j++ {
			rel := float64(len(gt.IDs) - j)
			idcg += rel / math.Log2(float64(j+2))
		}

		if idcg > 0 {
			totalNDCG += dcg / idcg
		}
	}

	return totalNDCG / float64(numQueries)
}

// CalculateAllMetrics computes all quality metrics
func CalculateAllMetrics(groundTruth []GroundTruth, results []SearchResult, k int) RecallMetrics {
	metrics := RecallMetrics{
		RecallK:   CalculateRecall(groundTruth, results, k),
		Precision: CalculatePrecision(groundTruth, results, k),
		MRR:       CalculateMRR(groundTruth, results),
		NDCG:      CalculateNDCG(groundTruth, results, k),
	}

	// Calculate recall at standard k values
	metrics.Recall1 = CalculateRecall(groundTruth, results, 1)
	metrics.Recall10 = CalculateRecall(groundTruth, results, 10)
	metrics.Recall100 = CalculateRecall(groundTruth, results, 100)

	return metrics
}

// MeasureLatencies measures latency statistics for a set of queries
func MeasureLatencies(latencies []time.Duration) PerformanceMetrics {
	if len(latencies) == 0 {
		return PerformanceMetrics{}
	}

	// Sort latencies for percentile calculation
	sorted := make([]time.Duration, len(latencies))
	copy(sorted, latencies)

	// Simple sort (bubble sort is fine for test code)
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	// Calculate percentiles
	p50Index := len(sorted) * 50 / 100
	p95Index := len(sorted) * 95 / 100
	p99Index := len(sorted) * 99 / 100

	// Calculate total and average
	total := time.Duration(0)
	for _, lat := range latencies {
		total += lat
	}
	avg := total / time.Duration(len(latencies))

	// Calculate QPS
	qps := 0.0
	if avg > 0 {
		qps = float64(time.Second) / float64(avg)
	}

	return PerformanceMetrics{
		P50Latency: sorted[p50Index],
		P95Latency: sorted[p95Index],
		P99Latency: sorted[p99Index],
		QPS:        qps,
		TotalTime:  total,
		AvgLatency: avg,
		MinLatency: sorted[0],
		MaxLatency: sorted[len(sorted)-1],
	}
}

// FormatMetrics returns a human-readable string of recall metrics
func (m RecallMetrics) String() string {
	return fmt.Sprintf(
		"Recall@1=%.4f, Recall@10=%.4f, Recall@100=%.4f, Precision=%.4f, MRR=%.4f, NDCG=%.4f",
		m.Recall1, m.Recall10, m.Recall100, m.Precision, m.MRR, m.NDCG,
	)
}

// FormatPerformance returns a human-readable string of performance metrics
func (p PerformanceMetrics) String() string {
	return fmt.Sprintf(
		"QPS=%.0f, P50=%v, P95=%v, P99=%v, Avg=%v",
		p.QPS,
		p.P50Latency.Round(time.Microsecond),
		p.P95Latency.Round(time.Microsecond),
		p.P99Latency.Round(time.Microsecond),
		p.AvgLatency.Round(time.Microsecond),
	)
}
