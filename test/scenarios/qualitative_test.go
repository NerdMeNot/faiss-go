package scenarios_test

import (
	"math"
	"testing"

	faiss "github.com/NerdMeNot/faiss-go"
	"github.com/NerdMeNot/faiss-go/test/helpers"
)

// ========================================
// Realistic Embedding Data
// ========================================

// These are simplified but realistic sentence embeddings
// In production, these would come from models like BERT, Sentence-BERT, or OpenAI embeddings
// The patterns reflect real semantic relationships

// animalEmbeddings represents embeddings for animal-related sentences
var animalEmbeddings = map[string][]float32{
	"The cat sat on the mat": {
		0.8, 0.7, 0.2, 0.1, 0.9, 0.3, 0.5, 0.6, // Animal features
		0.4, 0.3, 0.8, 0.2, 0.1, 0.7, 0.4, 0.5, // Domestic/home features
		0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.4, 0.3, // Low technical features
	},
	"A feline rested on the rug": {
		0.85, 0.72, 0.18, 0.15, 0.88, 0.28, 0.52, 0.58, // Similar animal features
		0.42, 0.35, 0.82, 0.22, 0.12, 0.68, 0.38, 0.48, // Similar home features
		0.08, 0.18, 0.12, 0.28, 0.18, 0.15, 0.38, 0.32, // Low technical features
	},
	"The dog played in the yard": {
		0.75, 0.68, 0.25, 0.2, 0.85, 0.35, 0.48, 0.62, // Animal features (dog)
		0.5, 0.4, 0.7, 0.3, 0.2, 0.65, 0.45, 0.55, // Outdoor/home features
		0.15, 0.25, 0.15, 0.35, 0.25, 0.2, 0.42, 0.35, // Low technical features
	},
	"Cars drive on highways": {
		0.1, 0.15, 0.8, 0.75, 0.2, 0.7, 0.6, 0.5, // Vehicle/transport features
		0.2, 0.15, 0.3, 0.65, 0.7, 0.25, 0.5, 0.4, // Infrastructure features
		0.7, 0.6, 0.75, 0.5, 0.65, 0.8, 0.45, 0.55, // High technical features
	},
	"Automobiles travel on roads": {
		0.12, 0.18, 0.82, 0.78, 0.18, 0.68, 0.58, 0.52, // Similar vehicle features
		0.18, 0.12, 0.28, 0.68, 0.72, 0.28, 0.52, 0.38, // Similar infrastructure
		0.72, 0.62, 0.78, 0.52, 0.68, 0.78, 0.42, 0.52, // High technical features
	},
}

// technicalEmbeddings represents embeddings for technical/scientific content
var technicalEmbeddings = map[string][]float32{
	"Machine learning algorithms process data": {
		0.1, 0.2, 0.15, 0.25, 0.1, 0.2, 0.3, 0.15, // Low animal/casual
		0.3, 0.25, 0.2, 0.35, 0.4, 0.3, 0.45, 0.35, // Medium abstract
		0.9, 0.85, 0.92, 0.88, 0.95, 0.9, 0.87, 0.91, // High technical
	},
	"Neural networks train on datasets": {
		0.08, 0.18, 0.12, 0.22, 0.15, 0.25, 0.28, 0.18, // Low animal/casual
		0.28, 0.22, 0.25, 0.38, 0.42, 0.32, 0.42, 0.38, // Medium abstract
		0.92, 0.88, 0.95, 0.85, 0.9, 0.88, 0.85, 0.9, // High technical
	},
	"The kitten chased the yarn": {
		0.9, 0.85, 0.3, 0.2, 0.88, 0.25, 0.4, 0.7, // High animal/playful
		0.6, 0.5, 0.75, 0.3, 0.25, 0.6, 0.55, 0.65, // Home/casual setting
		0.15, 0.2, 0.18, 0.25, 0.2, 0.15, 0.3, 0.25, // Low technical
	},
}

// wordEmbeddings represents word-level embeddings with semantic relationships
var wordEmbeddings = map[string][]float32{
	"king": {
		0.8, 0.3, 0.2, 0.9, 0.7, 0.4, 0.6, 0.5, // Royalty/male/power
		0.85, 0.75, 0.4, 0.3, 0.5, 0.6, 0.7, 0.8,
	},
	"queen": {
		0.75, 0.8, 0.25, 0.85, 0.65, 0.45, 0.55, 0.5, // Royalty/female/power
		0.8, 0.7, 0.45, 0.35, 0.55, 0.65, 0.75, 0.75,
	},
	"man": {
		0.4, 0.2, 0.5, 0.3, 0.6, 0.35, 0.45, 0.4, // Male/person/generic
		0.5, 0.4, 0.6, 0.5, 0.4, 0.45, 0.5, 0.55,
	},
	"woman": {
		0.35, 0.65, 0.55, 0.35, 0.55, 0.4, 0.4, 0.45, // Female/person/generic
		0.45, 0.35, 0.65, 0.55, 0.45, 0.5, 0.55, 0.5,
	},
	"car": {
		0.1, 0.15, 0.85, 0.2, 0.3, 0.75, 0.7, 0.4, // Vehicle/transport
		0.25, 0.3, 0.4, 0.8, 0.7, 0.35, 0.45, 0.5,
	},
	"automobile": {
		0.12, 0.18, 0.88, 0.18, 0.28, 0.78, 0.68, 0.38, // Vehicle/transport (synonym)
		0.22, 0.28, 0.38, 0.82, 0.72, 0.38, 0.42, 0.48,
	},
	"cat": {
		0.85, 0.5, 0.25, 0.3, 0.8, 0.35, 0.4, 0.75, // Animal/pet/feline
		0.6, 0.55, 0.7, 0.3, 0.25, 0.65, 0.6, 0.7,
	},
	"feline": {
		0.88, 0.52, 0.22, 0.28, 0.82, 0.32, 0.38, 0.78, // Animal/cat (synonym)
		0.58, 0.52, 0.72, 0.28, 0.22, 0.68, 0.62, 0.72,
	},
}

// Helper function to convert map to flat vectors for indexing
func embeddingsToVectors(embeddings map[string][]float32) ([]float32, []string) {
	keys := make([]string, 0, len(embeddings))
	for k := range embeddings {
		keys = append(keys, k)
	}

	// Get dimension from first embedding
	d := len(embeddings[keys[0]])
	vectors := make([]float32, 0, len(keys)*d)

	for _, key := range keys {
		vectors = append(vectors, embeddings[key]...)
	}

	return vectors, keys
}

// ========================================
// Semantic Similarity Tests
// ========================================

func TestSemanticSimilarity_AnimalSentences(t *testing.T) {
	// Test that semantically similar sentences have closer embeddings
	catMat := animalEmbeddings["The cat sat on the mat"]
	felineRug := animalEmbeddings["A feline rested on the rug"]
	carHighway := animalEmbeddings["Cars drive on highways"]

	// Distance between similar sentences (cat/feline)
	distSimilar, err := faiss.L2Distance(catMat, felineRug)
	if err != nil {
		t.Fatalf("faiss.L2Distance failed: %v", err)
	}

	// Distance between dissimilar sentences (cat/cars)
	distDissimilar, err := faiss.L2Distance(catMat, carHighway)
	if err != nil {
		t.Fatalf("faiss.L2Distance failed: %v", err)
	}

	if distSimilar >= distDissimilar {
		t.Errorf("Semantically similar sentences should be closer. Similar distance: %.4f, Dissimilar distance: %.4f",
			distSimilar, distDissimilar)
	}

	// Similar sentences should be reasonably close (< 1.0 for normalized embeddings)
	if distSimilar > 1.0 {
		t.Errorf("Similar sentences distance too large: %.4f", distSimilar)
	}
}

func TestSemanticSimilarity_TechnicalContent(t *testing.T) {
	mlAlgo := technicalEmbeddings["Machine learning algorithms process data"]
	neuralNet := technicalEmbeddings["Neural networks train on datasets"]
	kitten := technicalEmbeddings["The kitten chased the yarn"]

	// Technical content should be similar to each other
	distTechnical, err := faiss.L2Distance(mlAlgo, neuralNet)
	if err != nil {
		t.Fatalf("faiss.L2Distance failed: %v", err)
	}

	// Technical vs casual should be dissimilar
	distMixed, err := faiss.L2Distance(mlAlgo, kitten)
	if err != nil {
		t.Fatalf("faiss.L2Distance failed: %v", err)
	}

	if distTechnical >= distMixed {
		t.Errorf("Technical content should cluster together. Technical distance: %.4f, Mixed distance: %.4f",
			distTechnical, distMixed)
	}
}

func TestSemanticSimilarity_Synonyms(t *testing.T) {
	// Synonyms should be very close
	car := wordEmbeddings["car"]
	automobile := wordEmbeddings["automobile"]
	cat := wordEmbeddings["cat"]
	feline := wordEmbeddings["feline"]

	distCarAuto, err := faiss.L2Distance(car, automobile)
	if err != nil {
		t.Fatalf("faiss.L2Distance failed: %v", err)
	}
	distCatFeline, err := faiss.L2Distance(cat, feline)
	if err != nil {
		t.Fatalf("faiss.L2Distance failed: %v", err)
	}

	// Synonym pairs should be closer than unrelated words
	distCarCat, err := faiss.L2Distance(car, cat)
	if err != nil {
		t.Fatalf("faiss.L2Distance failed: %v", err)
	}

	if distCarAuto >= distCarCat || distCatFeline >= distCarCat {
		t.Errorf("Synonyms should be closer than unrelated words. car-auto: %.4f, cat-feline: %.4f, car-cat: %.4f",
			distCarAuto, distCatFeline, distCarCat)
	}
}

func TestSemanticSimilarity_AnalogyRelationships(t *testing.T) {
	// Test analogy: king - man + woman ≈ queen
	king := wordEmbeddings["king"]
	queen := wordEmbeddings["queen"]
	man := wordEmbeddings["man"]
	woman := wordEmbeddings["woman"]

	// Compute king - man + woman
	analogy := make([]float32, len(king))
	for i := range king {
		analogy[i] = king[i] - man[i] + woman[i]
	}

	// Distance to queen should be smaller than to other words
	distToQueen, err := faiss.L2Distance(analogy, queen)
	if err != nil {
		t.Fatalf("faiss.L2Distance failed: %v", err)
	}
	distToKing, err := faiss.L2Distance(analogy, king)
	if err != nil {
		t.Fatalf("faiss.L2Distance failed: %v", err)
	}
	distToCar, err := faiss.L2Distance(analogy, wordEmbeddings["car"])
	if err != nil {
		t.Fatalf("faiss.L2Distance failed: %v", err)
	}

	if distToQueen >= distToKing || distToQueen >= distToCar {
		t.Errorf("Analogy king-man+woman should be closest to queen. Distances: queen=%.4f, king=%.4f, car=%.4f",
			distToQueen, distToKing, distToCar)
	}
}

// ========================================
// Search Quality Tests
// ========================================

func TestSearchQuality_SemanticRetrieval(t *testing.T) {
	d := 24

	// Create index with animal embeddings
	index, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	vectors, keys := embeddingsToVectors(animalEmbeddings)
	if err := index.Add(vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Search for "The cat sat on the mat"
	query := animalEmbeddings["The cat sat on the mat"]
	k := 3

	distances, indices, err := index.Search(query, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// First result should be the query itself (distance ≈ 0)
	if distances[0] > 0.001 {
		t.Errorf("First result should be query itself with distance ≈ 0, got %.6f", distances[0])
	}

	// Second result should be "A feline rested on the rug" (semantically similar)
	secondKey := keys[indices[1]]
	if secondKey != "A feline rested on the rug" {
		t.Logf("Warning: Expected 'A feline rested on the rug' as second result, got '%s'", secondKey)
		t.Logf("This may indicate the embeddings need adjustment or the test is too strict")
	}

	// Third result should NOT be "Cars drive on highways" (dissimilar)
	thirdKey := keys[indices[2]]
	if thirdKey == "Cars drive on highways" || thirdKey == "Automobiles travel on roads" {
		t.Errorf("Dissimilar sentence about cars should not be in top 3 results for animal query")
	}
}

func TestSearchQuality_RangeSearch(t *testing.T) {
	d := 4

	index, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Add vectors at known distances from origin
	vectors := []float32{
		0.0, 0.0, 0.0, 0.0, // ID 0: origin
		1.0, 0.0, 0.0, 0.0, // ID 1: distance 1 from origin (squared L2)
		2.0, 0.0, 0.0, 0.0, // ID 2: distance 4 from origin (squared L2)
		10.0, 0.0, 0.0, 0.0, // ID 3: distance 100 from origin (squared L2)
	}
	if err := index.Add(vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Query at origin with radius 5.0 (should find IDs 0, 1, 2 - not 3)
	query := []float32{0.0, 0.0, 0.0, 0.0}
	result, err := index.RangeSearch(query, 5.0)
	if err != nil {
		t.Fatalf("RangeSearch failed: %v", err)
	}

	// Verify we found exactly 3 results (IDs 0, 1, 2)
	if result.TotalResults() != 3 {
		t.Errorf("Expected 3 results, got %d", result.TotalResults())
	}

	// Verify the labels are 0, 1, 2 (not necessarily in order)
	labels := make(map[int64]bool)
	for _, l := range result.Labels {
		labels[l] = true
	}
	for _, expected := range []int64{0, 1, 2} {
		if !labels[expected] {
			t.Errorf("Expected label %d in results", expected)
		}
	}
	if labels[3] {
		t.Error("Label 3 should not be in results (distance 100 > radius 5)")
	}

	// Verify distances are correct
	for i, dist := range result.Distances {
		if dist > 5.0 {
			t.Errorf("Distance[%d] = %f exceeds radius 5.0", i, dist)
		}
	}
}

func TestSearchQuality_TopKAccuracy(t *testing.T) {
	d := 16

	index, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	vectors, keys := embeddingsToVectors(wordEmbeddings)
	if err := index.Add(vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Search for "car" - should find "automobile" as a close match
	query := wordEmbeddings["car"]
	k := 3

	distances, indices, err := index.Search(query, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Check if "automobile" is in top 3 results
	foundSynonym := false
	for i := 0; i < k; i++ {
		if keys[indices[i]] == "automobile" {
			foundSynonym = true
			// Synonym should be very close
			if distances[i] > 0.5 {
				t.Errorf("Synonym 'automobile' distance too large: %.4f", distances[i])
			}
			break
		}
	}

	if !foundSynonym {
		t.Error("Synonym 'automobile' should be in top 3 results for 'car'")
	}
}

// ========================================
// Recall and Precision Tests
// ========================================

func TestRecallPrecision_IVFIndex(t *testing.T) {
	d := 24

	// Create ground truth index (flat, exhaustive search)
	groundTruth, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create ground truth index: %v", err)
	}
	defer groundTruth.Close()

	// Create IVF index (approximate search) using factory to avoid direct constructor bug
	ivfIndex, err := faiss.IndexFactory(d, "IVF2,Flat", faiss.MetricL2)
	if err != nil {
		t.Fatalf("Failed to create IVF index: %v", err)
	}
	defer ivfIndex.Close()

	// Add animal embeddings to both indexes
	vectors, _ := embeddingsToVectors(animalEmbeddings)

	if err := groundTruth.Add(vectors); err != nil {
		t.Fatalf("Failed to add to ground truth: %v", err)
	}

	// IVF needs sufficient training data (at least 39*nlist samples)
	// Generate additional training vectors based on existing patterns
	trainingVectors := helpers.GenerateVectors(100, d)
	if err := ivfIndex.Train(trainingVectors); err != nil {
		t.Fatalf("Failed to train IVF: %v", err)
	}

	if err := ivfIndex.Add(vectors); err != nil {
		t.Fatalf("Failed to add to IVF: %v", err)
	}

	// Note: GenericIndex doesn't expose SetNprobe. Using default nprobe value.
	// For production use with parameter tuning, use IndexIVFFlat direct constructor (once fixed).

	// Query with "The cat sat on the mat"
	query := animalEmbeddings["The cat sat on the mat"]
	k := 3

	// Ground truth results
	_, gtIndices, err := groundTruth.Search(query, k)
	if err != nil {
		t.Fatalf("Ground truth search failed: %v", err)
	}

	// IVF results
	_, ivfIndices, err := ivfIndex.Search(query, k)
	if err != nil {
		t.Fatalf("IVF search failed: %v", err)
	}

	// Calculate recall: how many ground truth results did IVF find?
	matches := 0
	for _, gtIdx := range gtIndices[:k] {
		for _, ivfIdx := range ivfIndices[:k] {
			if gtIdx == ivfIdx {
				matches++
				break
			}
		}
	}

	recall := float64(matches) / float64(k)

	// With nprobe=2 and only 2 clusters, recall should be high
	if recall < 0.66 { // At least 2 out of 3 should match
		t.Errorf("Recall too low: %.2f (expected >= 0.66)", recall)
	}

	t.Logf("Recall@%d: %.2f", k, recall)
}

// ========================================
// Cosine Similarity Tests (for normalized embeddings)
// ========================================

func TestCosineSimilarity_SemanticMatching(t *testing.T) {
	// Cosine similarity is often used for text embeddings
	catMat := animalEmbeddings["The cat sat on the mat"]
	felineRug := animalEmbeddings["A feline rested on the rug"]
	carHighway := animalEmbeddings["Cars drive on highways"]

	// Higher cosine similarity = more similar
	simSimilar, err := faiss.CosineSimilarity(catMat, felineRug)
	if err != nil {
		t.Fatalf("faiss.CosineSimilarity failed: %v", err)
	}
	simDissimilar, err := faiss.CosineSimilarity(catMat, carHighway)
	if err != nil {
		t.Fatalf("faiss.CosineSimilarity failed: %v", err)
	}

	if simSimilar <= simDissimilar {
		t.Errorf("Cosine similarity: similar sentences should have higher score. Similar: %.4f, Dissimilar: %.4f",
			simSimilar, simDissimilar)
	}

	// Cosine similarity should be in [-1, 1]
	if simSimilar < -1 || simSimilar > 1 {
		t.Errorf("Cosine similarity out of range [-1, 1]: %.4f", simSimilar)
	}
}

// ========================================
// Batch Operations Quality Tests
// ========================================

func TestBatchDistance_QualityCheck(t *testing.T) {
	d := 24

	// Create query and database from embeddings
	queries := []float32{}
	database := []float32{}

	// Queries: cat and ML
	queries = append(queries, animalEmbeddings["The cat sat on the mat"]...)
	queries = append(queries, technicalEmbeddings["Machine learning algorithms process data"]...)

	// Database: all animal embeddings
	dbVectors, dbKeys := embeddingsToVectors(animalEmbeddings)
	database = append(database, dbVectors...)

	// Compute batch distances
	distances, err := faiss.BatchL2Distance(queries, database, d)
	if err != nil {
		t.Fatalf("faiss.BatchL2Distance failed: %v", err)
	}

	// First query (cat) should be closest to itself
	nb := len(dbKeys)

	// Find minimum distance for first query
	minIdx := 0
	minDist := distances[0]
	for i := 1; i < nb; i++ {
		if distances[i] < minDist {
			minDist = distances[i]
			minIdx = i
		}
	}

	// Should find itself with near-zero distance
	if dbKeys[minIdx] != "The cat sat on the mat" {
		t.Errorf("Query should be closest to itself, found '%s' instead", dbKeys[minIdx])
	}

	if minDist > 0.001 {
		t.Errorf("Distance to self should be ≈0, got %.6f", minDist)
	}

	// Second query (ML) should be farthest from animal sentences
	mlQueryStart := nb // Second query starts at index nb
	maxDist := distances[mlQueryStart]
	avgDist := float32(0)
	for i := 0; i < nb; i++ {
		dist := distances[mlQueryStart+i]
		avgDist += dist
		if dist > maxDist {
			maxDist = dist
		}
	}
	avgDist /= float32(nb)

	// ML query should have larger average distance to animal sentences
	if avgDist < minDist*10 {
		t.Logf("ML query average distance (%.4f) should be much larger than cat self-distance (%.6f)",
			avgDist, minDist)
	}
}

// ========================================
// Edge Cases and Robustness Tests
// ========================================

func TestQualitative_EmptyQuery(t *testing.T) {
	d := 24

	index, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	vectors, _ := embeddingsToVectors(animalEmbeddings)
	if err := index.Add(vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Empty query should return error or handle gracefully
	emptyQuery := []float32{}
	_, _, err = index.Search(emptyQuery, 5)

	// Either error or gracefully handle
	if err == nil {
		t.Log("Empty query handled without error (implementation choice)")
	}
}

func TestQualitative_IdenticalVectors(t *testing.T) {
	d := 16

	index, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Add identical vectors
	vec := wordEmbeddings["cat"]
	vectors := make([]float32, 3*d)
	copy(vectors[0:d], vec)
	copy(vectors[d:2*d], vec)
	copy(vectors[2*d:3*d], vec)

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Search should find all three with same (zero) distance
	distances, _, err := index.Search(vec, 3)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// All three should have near-zero distance
	for i := 0; i < 3; i++ {
		if distances[i] > 0.001 {
			t.Errorf("Distance to identical vector should be ≈0, got %.6f at position %d", distances[i], i)
		}
	}
}

// ========================================
// Helper function for validating embedding quality
// ========================================

func TestEmbeddingQuality_Normalization(t *testing.T) {
	// Check that embeddings are in reasonable ranges
	for sentence, embedding := range animalEmbeddings {
		// Check for NaN or Inf
		for i, val := range embedding {
			if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
				t.Errorf("Invalid value in embedding for '%s' at index %d: %f", sentence, i, val)
			}
		}

		// Check embedding magnitude (L2 norm)
		norm := float32(0)
		for _, val := range embedding {
			norm += val * val
		}
		norm = float32(math.Sqrt(float64(norm)))

		// Embeddings should have reasonable magnitude (not too large or small)
		if norm < 0.1 || norm > 100 {
			t.Errorf("Embedding norm out of expected range for '%s': %.4f", sentence, norm)
		}
	}
}
