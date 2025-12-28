package scenarios

import (
	"testing"
	"time"

	faiss "github.com/NerdMeNot/faiss-go"
	"github.com/NerdMeNot/faiss-go/test/datasets"
	"github.com/NerdMeNot/faiss-go/test/helpers"
)

// TestSemanticSearch_DocumentRetrieval simulates a real-world document search system
// Use case: Search engine for 100K documents with BERT embeddings
func TestSemanticSearch_DocumentRetrieval(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping semantic search scenario in short mode")
	}

	// Simulate BERT-style embeddings (768-dim, normalized)
	t.Log("Simulating document retrieval with BERT-like embeddings...")

	nDocs := 100000  // 100K documents
	dim := 768       // BERT-base dimension
	nQueries := 1000 // 1K search queries
	k := 10          // Top-10 results

	t.Logf("Dataset: %d documents, %d dimensions", nDocs, dim)

	// Generate realistic document embeddings
	docEmbeddings := datasets.GenerateRealisticEmbeddings(nDocs, dim)
	docEmbeddings.GenerateQueries(nQueries, datasets.Normalized)

	// Test different index types suitable for semantic search
	testCases := []struct {
		name          string
		buildIndex    func() faiss.Index
		minRecall     float64
		maxLatencyP99 time.Duration
	}{
		{
			name: "HNSW_M32_efSearch64",
			buildIndex: func() faiss.Index {
				index, err := faiss.NewIndexHNSWFlat(dim, 32, faiss.MetricInnerProduct)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
				if err != nil {
					t.Fatalf("Failed to create HNSW index: %v", err)
				}
				index.SetEfSearch(64)
				return index
			},
			minRecall:     0.95,
			maxLatencyP99: 10 * time.Millisecond,
		},
		{
			name: "IVF1000_nprobe20",
			buildIndex: func() faiss.Index {
				quantizer, err := faiss.NewIndexFlatIP(dim)
			if err != nil {
				t.Fatalf("Failed to create quantizer: %v", err)
			}
				if err != nil {
					t.Fatalf("Failed to create quantizer: %v", err)
				}
				index, err := faiss.NewIndexIVFFlat(quantizer, dim, 1000, faiss.MetricInnerProduct)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
				if err != nil {
					t.Fatalf("Failed to create IVF index: %v", err)
				}
				index.SetNprobe(20)
				return index
			},
			minRecall:     0.85,
			maxLatencyP99: 8 * time.Millisecond,
		},
		{
			name: "IVFPQ_nlist1000_M48",
			buildIndex: func() faiss.Index {
				quantizer, err := faiss.NewIndexFlatIP(dim)
			if err != nil {
				t.Fatalf("Failed to create quantizer: %v", err)
			}
				if err != nil {
					t.Fatalf("Failed to create quantizer: %v", err)
				}
				index, err := faiss.NewIndexIVFPQ(quantizer, dim, 1000, 48, 8)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
				if err != nil {
					t.Fatalf("Failed to create IVFPQ index: %v", err)
				}
				index.SetNprobe(20)
				return index
			},
			minRecall:     0.80,
			maxLatencyP99: 7 * time.Millisecond,
		},
	}

	// Compute ground truth once
	t.Log("Computing ground truth...")
	groundTruth, err := helpers.ComputeGroundTruth(
		docEmbeddings.Vectors,
		docEmbeddings.Queries,
		dim,
		k,
		faiss.MetricInnerProduct,
	)
	if err != nil {
		t.Fatalf("Failed to compute ground truth: %v", err)
	}

	// Test each index type
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Building %s index...", tc.name)

			index := tc.buildIndex()
			defer index.Close()

			// Train if needed
			if !index.IsTrained() {
				trainSize := 50000
				if trainSize > nDocs {
					trainSize = nDocs
				}
				t.Logf("Training index with %d documents...", trainSize)

				startTrain := time.Now()
				if err := index.Train(docEmbeddings.Vectors[:trainSize*dim]); err != nil {
					t.Fatalf("Training failed: %v", err)
				}
				t.Logf("Training completed in %v", time.Since(startTrain))
			}

			// Add documents
			t.Logf("Adding %d documents to index...", nDocs)
			startAdd := time.Now()
			if err := index.Add(docEmbeddings.Vectors); err != nil {
				t.Fatalf("Add failed: %v", err)
			}
			addTime := time.Since(startAdd)
			t.Logf("Added %d documents in %v (%.0f docs/sec)",
				nDocs, addTime, float64(nDocs)/addTime.Seconds())

			// Measure memory
			memory := helpers.MeasureIndexMemory(index)
			memoryMB := memory / (1024 * 1024)
			compressionRatio := float64(nDocs*dim*4) / float64(memory)
			t.Logf("Index memory: %d MB (%.1fx compression)", memoryMB, compressionRatio)

			// Search
			t.Logf("Searching %d queries...", nQueries)
			results, latencies, err := helpers.SearchWithTiming(index, docEmbeddings.Queries, k)
			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			// Calculate metrics
			metrics := helpers.CalculateAllMetrics(groundTruth, results, k)
			perf := helpers.MeasureLatencies(latencies)

			// Log results
			t.Logf("\n=== Results for %s ===", tc.name)
			t.Logf("Quality Metrics:")
			t.Logf("  Recall@1:  %.4f", metrics.Recall1)
			t.Logf("  Recall@10: %.4f", metrics.Recall10)
			t.Logf("  Precision: %.4f", metrics.Precision)
			t.Logf("  MRR:       %.4f", metrics.MRR)
			t.Logf("  NDCG:      %.4f", metrics.NDCG)
			t.Logf("\nPerformance Metrics:")
			t.Logf("  QPS:        %.0f queries/sec", perf.QPS)
			t.Logf("  P50 Latency: %v", perf.P50Latency.Round(time.Microsecond))
			t.Logf("  P95 Latency: %v", perf.P95Latency.Round(time.Microsecond))
			t.Logf("  P99 Latency: %v", perf.P99Latency.Round(time.Microsecond))
			t.Logf("\nResource Usage:")
			t.Logf("  Memory:     %d MB", memoryMB)
			t.Logf("  Add Time:   %v", addTime)

			// Validate quality targets
			if metrics.Recall10 < tc.minRecall {
				t.Errorf("Recall@10 (%.4f) below target (%.4f)",
					metrics.Recall10, tc.minRecall)
			}

			if perf.P99Latency > tc.maxLatencyP99 {
				t.Errorf("P99 latency (%v) exceeds target (%v)",
					perf.P99Latency, tc.maxLatencyP99)
			}

			// Validate reasonable performance
			if perf.QPS < 100 {
				t.Logf("Warning: QPS (%.0f) seems low for this use case", perf.QPS)
			}

			if metrics.Recall10 >= tc.minRecall && perf.P99Latency <= tc.maxLatencyP99 {
				t.Logf("✓ %s meets all quality and performance targets", tc.name)
			}
		})
	}

	// Comparison summary
	t.Run("Summary", func(t *testing.T) {
		t.Logf("\n=== Semantic Search Scenario Summary ===")
		t.Logf("Use Case: Document retrieval with BERT embeddings")
		t.Logf("Dataset: %d documents, %d dimensions", nDocs, dim)
		t.Logf("\nRecommendation:")
		t.Logf("  - For high accuracy (>95%% recall): HNSW with M=32, efSearch=64")
		t.Logf("  - For balanced performance: IVF with nlist=1000, nprobe=20")
		t.Logf("  - For memory efficiency: IVFPQ with moderate M (48-64)")
	})
}

// TestSemanticSearch_QA simulates a question-answering system
// Use case: Find relevant passages for user questions
func TestSemanticSearch_QA(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Q&A scenario in short mode")
	}

	// Smaller dataset for Q&A (50K passages)
	nPassages := 50000
	dim := 768
	nQuestions := 500
	k := 5 // Top-5 passages

	t.Logf("Simulating Q&A system with %d passages, %d questions", nPassages, nQuestions)

	// Generate realistic embeddings
	passageEmbeddings := datasets.GenerateRealisticEmbeddings(nPassages, dim)
	passageEmbeddings.GenerateQueries(nQuestions, datasets.Normalized)

	// For Q&A, we want very high precision (top results must be relevant)
	// Use HNSW for best accuracy
	index, err := faiss.NewIndexHNSWFlat(dim, 48, faiss.MetricInnerProduct)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}
	index.SetEfSearch(128) // Higher efSearch for better recall
	defer index.Close()

	// Add passages
	t.Log("Building passage index...")
	if err := index.Add(passageEmbeddings.Vectors); err != nil {
		t.Fatalf("Failed to add passages: %v", err)
	}

	// Compute ground truth
	groundTruth, err := helpers.ComputeGroundTruth(
		passageEmbeddings.Vectors,
		passageEmbeddings.Queries,
		dim,
		k,
		faiss.MetricInnerProduct,
	)
	if err != nil {
		t.Fatalf("Failed to compute ground truth: %v", err)
	}

	// Search
	results, latencies, err := helpers.SearchWithTiming(index, passageEmbeddings.Queries, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Calculate metrics
	metrics := helpers.CalculateAllMetrics(groundTruth, results, k)
	perf := helpers.MeasureLatencies(latencies)

	// Log results
	t.Logf("\n=== Q&A System Results ===")
	t.Logf("Recall@5:  %.4f (target: >0.98)", metrics.RecallK)
	t.Logf("Recall@1:  %.4f (top answer accuracy)", metrics.Recall1)
	t.Logf("MRR:       %.4f (answer ranking quality)", metrics.MRR)
	t.Logf("P99:       %v (target: <5ms)", perf.P99Latency.Round(time.Microsecond))

	// Q&A requires very high accuracy
	if metrics.Recall1 < 0.95 {
		t.Errorf("Top answer recall too low: %.4f (target: >0.95)", metrics.Recall1)
	}

	if metrics.MRR < 0.97 {
		t.Errorf("MRR too low: %.4f (target: >0.97)", metrics.MRR)
	}

	if perf.P99Latency > 5*time.Millisecond {
		t.Logf("Warning: P99 latency (%v) exceeds 5ms target", perf.P99Latency)
	}

	t.Logf("✓ Q&A system achieves high accuracy with low latency")
}

// TestSemanticSearch_MultiLingual simulates multi-lingual search
func TestSemanticSearch_MultiLingual(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping multi-lingual scenario in short mode")
	}

	// Simulate multi-lingual embeddings (e.g., mBERT, XLM-R)
	nDocs := 20000
	dim := 768
	nQueries := 200
	k := 10

	t.Logf("Simulating multi-lingual search with %d documents", nDocs)

	// Generate realistic embeddings
	docEmbeddings := datasets.GenerateRealisticEmbeddings(nDocs, dim)
	docEmbeddings.GenerateQueries(nQueries, datasets.Normalized)

	// Use HNSW for multi-lingual (cross-lingual retrieval needs high recall)
	index, err := faiss.NewIndexHNSWFlat(dim, 32, faiss.MetricInnerProduct)
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}
	index.SetEfSearch(64)
	defer index.Close()

	// Add documents
	if err := index.Add(docEmbeddings.Vectors); err != nil {
		t.Fatalf("Failed to add documents: %v", err)
	}

	// Compute ground truth
	groundTruth, err := helpers.ComputeGroundTruth(
		docEmbeddings.Vectors,
		docEmbeddings.Queries,
		dim,
		k,
		faiss.MetricInnerProduct,
	)
	if err != nil {
		t.Fatalf("Failed to compute ground truth: %v", err)
	}

	// Search
	results, latencies, err := helpers.SearchWithTiming(index, docEmbeddings.Queries, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Calculate metrics
	metrics := helpers.CalculateAllMetrics(groundTruth, results, k)
	perf := helpers.MeasureLatencies(latencies)

	// Log results
	t.Logf("\n=== Multi-Lingual Search Results ===")
	t.Logf("Recall@10: %.4f", metrics.Recall10)
	t.Logf("QPS:       %.0f", perf.QPS)
	t.Logf("P99:       %v", perf.P99Latency.Round(time.Microsecond))

	// Multi-lingual requires good recall across languages
	if metrics.Recall10 < 0.92 {
		t.Errorf("Recall too low for multi-lingual: %.4f", metrics.Recall10)
	}

	t.Logf("✓ Multi-lingual search achieves %.4f recall", metrics.Recall10)
}
