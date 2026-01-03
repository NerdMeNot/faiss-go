package scenarios

import (
	"math"
	"testing"

	faiss "github.com/NerdMeNot/faiss-go"
)

// =============================================================================
// STRUCTURED TEST DATA GENERATION
// =============================================================================
//
// These tests use DETERMINISTIC data with KNOWN properties, NOT random data.
// This ensures tests are meaningful and reproducible:
//
// - Clustered data: vectors grouped by "category" - we KNOW neighbors should be in same cluster
// - Duplicates: exact copies - we KNOW they should match with distance 0
// - Ordered data: vectors at known distances - we KNOW the exact ordering
//
// This approach eliminates flaky tests caused by random data variance.
// =============================================================================

// generateClusteredData creates vectors in well-separated clusters
// where we KNOW which vectors should be nearest neighbors.
//
// Each cluster has a strong signal along one dimension axis, making clusters
// well-separated. Points within a cluster have small perturbations.
func generateClusteredData(nClusters, pointsPerCluster, dim int) (vectors []float32, clusterIDs []int) {
	totalPoints := nClusters * pointsPerCluster
	vectors = make([]float32, totalPoints*dim)
	clusterIDs = make([]int, totalPoints)

	for c := 0; c < nClusters; c++ {
		// Each cluster's centroid points along a different axis
		centroidAxis := c % dim

		for p := 0; p < pointsPerCluster; p++ {
			idx := c*pointsPerCluster + p
			clusterIDs[idx] = c

			// Initialize to zero
			for d := 0; d < dim; d++ {
				vectors[idx*dim+d] = 0
			}

			// Strong signal on cluster axis (value = 10.0)
			vectors[idx*dim+centroidAxis] = 10.0

			// Small deterministic perturbation for variation within cluster
			for d := 0; d < dim; d++ {
				noise := float32(math.Sin(float64(idx*dim+d))) * 0.1
				vectors[idx*dim+d] += noise
			}
		}
	}

	return vectors, clusterIDs
}

// generateHierarchicalClusters creates a two-level hierarchy (categories -> subcategories)
// simulating real-world product catalogs or document taxonomies.
func generateHierarchicalClusters(nCategories, nSubcategories, pointsPerSub, dim int) (vectors []float32, categoryIDs, subcategoryIDs []int) {
	totalPoints := nCategories * nSubcategories * pointsPerSub
	vectors = make([]float32, totalPoints*dim)
	categoryIDs = make([]int, totalPoints)
	subcategoryIDs = make([]int, totalPoints)

	idx := 0
	for cat := 0; cat < nCategories; cat++ {
		catAxis := cat % dim

		for sub := 0; sub < nSubcategories; sub++ {
			subAxis := (cat*nSubcategories + sub) % dim

			for p := 0; p < pointsPerSub; p++ {
				categoryIDs[idx] = cat
				subcategoryIDs[idx] = cat*nSubcategories + sub

				// Initialize
				for d := 0; d < dim; d++ {
					vectors[idx*dim+d] = 0
				}

				// Category signal (strong)
				vectors[idx*dim+catAxis] = 10.0

				// Subcategory signal (medium)
				vectors[idx*dim+subAxis] += 3.0

				// Small noise
				for d := 0; d < dim; d++ {
					vectors[idx*dim+d] += float32(math.Sin(float64(idx*dim+d))) * 0.05
				}

				idx++
			}
		}
	}

	return vectors, categoryIDs, subcategoryIDs
}

// =============================================================================
// CORE FUNCTIONALITY TESTS
// =============================================================================

// TestStructuredSearch_ClusteredData verifies that nearest neighbor search
// correctly finds vectors from the same cluster (simulates finding similar items).
func TestStructuredSearch_ClusteredData(t *testing.T) {
	nClusters := 20
	pointsPerCluster := 500
	dim := 64
	k := 10

	t.Logf("Creating %d clusters with %d points each (%d total)",
		nClusters, pointsPerCluster, nClusters*pointsPerCluster)

	vectors, clusterIDs := generateClusteredData(nClusters, pointsPerCluster, dim)

	testCases := []struct {
		name       string
		buildIndex func() (faiss.Index, error)
		minRecall  float64
	}{
		{
			name: "Flat_Exact",
			buildIndex: func() (faiss.Index, error) {
				return faiss.NewIndexFlatL2(dim)
			},
			minRecall: 1.0, // Exact search must be perfect
		},
		{
			name: "HNSW_M32",
			buildIndex: func() (faiss.Index, error) {
				idx, err := faiss.NewIndexHNSWFlat(dim, 32, faiss.MetricL2)
				if err != nil {
					return nil, err
				}
				idx.SetEfSearch(64)
				return idx, nil
			},
			minRecall: 0.95,
		},
		{
			name: "IVF_nlist20",
			buildIndex: func() (faiss.Index, error) {
				quantizer, err := faiss.NewIndexFlatL2(dim)
				if err != nil {
					return nil, err
				}
				idx, err := faiss.NewIndexIVFFlat(quantizer, dim, nClusters, faiss.MetricL2)
				if err != nil {
					return nil, err
				}
				idx.SetNprobe(5)
				return idx, nil
			},
			minRecall: 0.90,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			index, err := tc.buildIndex()
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}
			defer index.Close()

			if !index.IsTrained() {
				if err := index.Train(vectors); err != nil {
					t.Fatalf("Training failed: %v", err)
				}
			}

			if err := index.Add(vectors); err != nil {
				t.Fatalf("Add failed: %v", err)
			}

			// Query from each cluster, verify neighbors are from same cluster
			totalCorrect := 0
			totalNeighbors := 0
			nQueries := nClusters * 5

			for q := 0; q < nQueries; q++ {
				queryIdx := (q / 5) * pointsPerCluster + (q % 5)
				queryCluster := clusterIDs[queryIdx]
				query := vectors[queryIdx*dim : (queryIdx+1)*dim]

				_, indices, err := index.Search(query, k)
				if err != nil {
					t.Fatalf("Search failed: %v", err)
				}

				for i := 0; i < k; i++ {
					if indices[i] >= 0 && indices[i] < int64(len(clusterIDs)) {
						if clusterIDs[indices[i]] == queryCluster {
							totalCorrect++
						}
					}
					totalNeighbors++
				}
			}

			recall := float64(totalCorrect) / float64(totalNeighbors)
			t.Logf("Cluster recall: %.2f%% (%d/%d)", recall*100, totalCorrect, totalNeighbors)

			if recall < tc.minRecall {
				t.Errorf("Recall %.2f%% below minimum %.2f%%", recall*100, tc.minRecall*100)
			} else {
				t.Logf("✓ %s achieves %.2f%% cluster recall", tc.name, recall*100)
			}
		})
	}
}

// TestStructuredSearch_KnownDuplicates verifies exact duplicates are found with distance 0.
func TestStructuredSearch_KnownDuplicates(t *testing.T) {
	dim := 128
	nBase := 1000
	nDuplicates := 50

	vectors := make([]float32, (nBase+nDuplicates)*dim)
	for i := 0; i < nBase*dim; i++ {
		vectors[i] = float32(math.Sin(float64(i)))
	}

	// Create exact duplicates
	duplicateOf := make(map[int]int)
	for i := 0; i < nDuplicates; i++ {
		srcIdx := i
		dstIdx := nBase + i
		duplicateOf[dstIdx] = srcIdx
		copy(vectors[dstIdx*dim:(dstIdx+1)*dim], vectors[srcIdx*dim:(srcIdx+1)*dim])
	}

	t.Logf("Created %d base vectors + %d exact duplicates", nBase, nDuplicates)

	index, err := faiss.NewIndexFlatL2(dim)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	foundOriginals := 0
	for dupIdx, origIdx := range duplicateOf {
		query := vectors[dupIdx*dim : (dupIdx+1)*dim]
		distances, indices, err := index.Search(query, 2)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		for i := 0; i < 2; i++ {
			if indices[i] == int64(origIdx) && distances[i] < 1e-6 {
				foundOriginals++
				break
			}
		}
	}

	if foundOriginals != nDuplicates {
		t.Errorf("Only found %d/%d originals", foundOriginals, nDuplicates)
	} else {
		t.Logf("✓ All %d duplicates correctly matched to originals", nDuplicates)
	}
}

// TestStructuredSearch_DistanceOrdering verifies results are correctly ordered by distance.
func TestStructuredSearch_DistanceOrdering(t *testing.T) {
	dim := 32
	n := 100
	k := 10

	// Create vectors at known distances from origin
	vectors := make([]float32, n*dim)
	for i := 0; i < n; i++ {
		vectors[i*dim] = float32(i + 1) // Distance = i+1
	}

	index, err := faiss.NewIndexFlatL2(dim)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Query from origin
	query := make([]float32, dim)
	distances, indices, err := index.Search(query, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	for i := 0; i < k; i++ {
		expectedIdx := int64(i)
		expectedDist := float32((i + 1) * (i + 1))

		if indices[i] != expectedIdx {
			t.Errorf("Position %d: expected index %d, got %d", i, expectedIdx, indices[i])
		}
		if math.Abs(float64(distances[i]-expectedDist)) > 0.001 {
			t.Errorf("Position %d: expected distance %.3f, got %.3f", i, expectedDist, distances[i])
		}
	}

	t.Logf("✓ Distance ordering verified for k=%d results", k)
}

// TestStructuredSearch_InnerProduct verifies inner product metric works correctly.
func TestStructuredSearch_InnerProduct(t *testing.T) {
	dim := 64
	n := 100

	// Unit vectors pointing in different directions
	vectors := make([]float32, n*dim)
	for i := 0; i < n; i++ {
		axis := i % dim
		vectors[i*dim+axis] = 1.0
	}

	index, err := faiss.NewIndexFlatIP(dim)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Query with vector along axis 0
	query := make([]float32, dim)
	query[0] = 1.0

	distances, indices, err := index.Search(query, 5)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if distances[0] < 0.99 {
		t.Errorf("Best match has inner product %.3f, expected ~1.0", distances[0])
	}

	t.Logf("✓ Inner product search verified (best IP: %.3f at index %d)", distances[0], indices[0])
}

// =============================================================================
// REAL-WORLD SCENARIO TESTS (with structured data)
// =============================================================================

// TestScenario_ProductRecommendations simulates e-commerce product recommendations
// using hierarchical clusters (categories -> products).
func TestScenario_ProductRecommendations(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}

	nCategories := 10      // Electronics, Clothing, etc.
	nSubcategories := 5    // Phones, Laptops, etc.
	productsPerSub := 100  // Products per subcategory
	dim := 64
	k := 20

	totalProducts := nCategories * nSubcategories * productsPerSub
	t.Logf("Simulating %d products in %d categories", totalProducts, nCategories)

	vectors, categoryIDs, subcategoryIDs := generateHierarchicalClusters(
		nCategories, nSubcategories, productsPerSub, dim)

	// Use HNSW for fast recommendations
	index, err := faiss.NewIndexHNSWFlat(dim, 32, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()
	index.SetEfSearch(64)

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Test: recommendations should be from same category/subcategory
	nQueries := 50
	sameCategoryCount := 0
	sameSubcategoryCount := 0
	totalResults := 0

	for q := 0; q < nQueries; q++ {
		queryIdx := q * (totalProducts / nQueries)
		query := vectors[queryIdx*dim : (queryIdx+1)*dim]
		queryCat := categoryIDs[queryIdx]
		querySub := subcategoryIDs[queryIdx]

		_, indices, err := index.Search(query, k)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		for i := 1; i < k; i++ { // Skip self (i=0)
			idx := int(indices[i])
			if idx >= 0 && idx < len(categoryIDs) {
				totalResults++
				if categoryIDs[idx] == queryCat {
					sameCategoryCount++
				}
				if subcategoryIDs[idx] == querySub {
					sameSubcategoryCount++
				}
			}
		}
	}

	categoryRecall := float64(sameCategoryCount) / float64(totalResults)
	subcategoryRecall := float64(sameSubcategoryCount) / float64(totalResults)

	t.Logf("Category recall: %.1f%% (recommendations from same category)", categoryRecall*100)
	t.Logf("Subcategory recall: %.1f%% (recommendations from same subcategory)", subcategoryRecall*100)

	// With hierarchical structure, most results should be from same category
	if categoryRecall < 0.80 {
		t.Errorf("Category recall %.1f%% too low (expected >80%%)", categoryRecall*100)
	}
	if subcategoryRecall < 0.50 {
		t.Errorf("Subcategory recall %.1f%% too low (expected >50%%)", subcategoryRecall*100)
	}

	t.Logf("✓ Product recommendation quality verified")
}

// TestScenario_DocumentDeduplication simulates finding duplicate documents.
func TestScenario_DocumentDeduplication(t *testing.T) {
	dim := 256
	nDocuments := 5000
	nDuplicates := 100     // 2% duplicates
	nNearDuplicates := 100 // Slight variations

	vectors := make([]float32, (nDocuments+nDuplicates+nNearDuplicates)*dim)

	// Generate base documents
	for i := 0; i < nDocuments*dim; i++ {
		vectors[i] = float32(math.Sin(float64(i*7))) * 0.5
	}

	// Create exact duplicates (copies of first 100 docs)
	exactDups := make([][2]int, nDuplicates)
	for i := 0; i < nDuplicates; i++ {
		srcIdx := i
		dstIdx := nDocuments + i
		exactDups[i] = [2]int{dstIdx, srcIdx}
		copy(vectors[dstIdx*dim:(dstIdx+1)*dim], vectors[srcIdx*dim:(srcIdx+1)*dim])
	}

	// Create near-duplicates (slight modifications)
	nearDups := make([][2]int, nNearDuplicates)
	for i := 0; i < nNearDuplicates; i++ {
		srcIdx := 100 + i // Different set of originals
		dstIdx := nDocuments + nDuplicates + i
		nearDups[i] = [2]int{dstIdx, srcIdx}

		// Copy with small noise
		for d := 0; d < dim; d++ {
			vectors[dstIdx*dim+d] = vectors[srcIdx*dim+d] + float32(math.Sin(float64(d)))*0.01
		}
	}

	t.Logf("Created %d documents + %d exact dups + %d near-dups", nDocuments, nDuplicates, nNearDuplicates)

	index, err := faiss.NewIndexFlatL2(dim)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Find exact duplicates
	foundExact := 0
	for _, pair := range exactDups {
		query := vectors[pair[0]*dim : (pair[0]+1)*dim]
		distances, indices, _ := index.Search(query, 2)

		for i := 0; i < 2; i++ {
			if indices[i] == int64(pair[1]) && distances[i] < 1e-6 {
				foundExact++
				break
			}
		}
	}

	// Find near-duplicates (distance < threshold)
	foundNear := 0
	threshold := float32(0.1) // Very close but not exact
	for _, pair := range nearDups {
		query := vectors[pair[0]*dim : (pair[0]+1)*dim]
		distances, indices, _ := index.Search(query, 5)

		for i := 0; i < 5; i++ {
			if indices[i] == int64(pair[1]) && distances[i] < threshold {
				foundNear++
				break
			}
		}
	}

	t.Logf("Found %d/%d exact duplicates", foundExact, nDuplicates)
	t.Logf("Found %d/%d near-duplicates", foundNear, nNearDuplicates)

	if foundExact != nDuplicates {
		t.Errorf("Should find all exact duplicates")
	}
	if foundNear < nNearDuplicates*90/100 {
		t.Errorf("Should find most near-duplicates")
	}

	t.Logf("✓ Document deduplication verified")
}

// TestScenario_ScaleTest verifies index works at larger scale.
func TestScenario_ScaleTest(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping scale test in short mode")
	}

	dim := 128
	nVectors := 100000 // 100K vectors
	nClusters := 50
	pointsPerCluster := nVectors / nClusters
	k := 10

	t.Logf("Scale test: %d vectors, %d dimensions", nVectors, dim)

	vectors, clusterIDs := generateClusteredData(nClusters, pointsPerCluster, dim)

	// Test IVF at scale
	quantizer, err := faiss.NewIndexFlatL2(dim)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}

	index, err := faiss.NewIndexIVFFlat(quantizer, dim, 100, faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()
	index.SetNprobe(10)

	if err := index.Train(vectors); err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Verify search quality
	nQueries := 100
	correctCount := 0
	totalCount := 0

	for q := 0; q < nQueries; q++ {
		queryIdx := q * (nVectors / nQueries)
		query := vectors[queryIdx*dim : (queryIdx+1)*dim]
		queryCluster := clusterIDs[queryIdx]

		_, indices, _ := index.Search(query, k)

		for i := 0; i < k; i++ {
			idx := int(indices[i])
			if idx >= 0 && idx < len(clusterIDs) {
				totalCount++
				if clusterIDs[idx] == queryCluster {
					correctCount++
				}
			}
		}
	}

	recall := float64(correctCount) / float64(totalCount)
	t.Logf("Cluster recall at 100K scale: %.1f%%", recall*100)

	if recall < 0.85 {
		t.Errorf("Recall %.1f%% too low at scale", recall*100)
	}

	t.Logf("✓ Scale test passed with %.1f%% recall", recall*100)
}
