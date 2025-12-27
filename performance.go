package faiss

import (
	"runtime"
)

// PerformanceHints provides guidance on optimizing FAISS operations
//
// Zero-Copy Operations:
// This library uses unsafe pointers (&vectors[0]) to pass data directly to C++
// without copying, minimizing memory overhead and maximizing performance.
//
// Batching:
// Always batch your operations when possible:
//  - Search: Pass multiple query vectors in one call instead of looping
//  - Add: Accumulate vectors and add in large batches
//  - Train: Use as many training vectors as feasible
//
// Scheduler Release:
// For operations on large batches (>100 queries), the library automatically
// locks the goroutine to an OS thread during C++ computation to prevent
// scheduler overhead and optimize cache locality.

// SearchBatch is a helper to demonstrate optimal batch searching
// Use this pattern when searching multiple queries
func SearchBatch(index Index, queries []float32, k int) ([]float32, []int64, error) {
	// Validate batch size
	nq := len(queries) / index.D()
	if nq == 0 {
		return nil, nil, nil
	}

	// For very large batches, consider splitting to manage memory
	// FAISS is optimized for batches of 100-10000 queries
	const optimalBatchSize = 10000

	if nq <= optimalBatchSize {
		// Single batch - optimal path
		return index.Search(queries, k)
	}

	// Split into optimal-sized batches
	d := index.D()
	allDistances := make([]float32, 0, nq*k)
	allIndices := make([]int64, 0, nq*k)

	for i := 0; i < nq; i += optimalBatchSize {
		end := i + optimalBatchSize
		if end > nq {
			end = nq
		}

		batch := queries[i*d : end*d]
		distances, indices, err := index.Search(batch, k)
		if err != nil {
			return nil, nil, err
		}

		allDistances = append(allDistances, distances...)
		allIndices = append(allIndices, indices...)
	}

	return allDistances, allIndices, nil
}

// AddBatch is a helper to demonstrate optimal batch addition
// Use this pattern when adding multiple vectors
func AddBatch(index Index, vectors []float32) error {
	// Validate batch size
	n := len(vectors) / index.D()
	if n == 0 {
		return nil
	}

	// For very large batches, consider splitting to manage memory
	// FAISS is optimized for batches of 1000-100000 vectors
	const optimalBatchSize = 100000

	if n <= optimalBatchSize {
		// Single batch - optimal path
		return index.Add(vectors)
	}

	// Split into optimal-sized batches
	d := index.D()
	for i := 0; i < n; i += optimalBatchSize {
		end := i + optimalBatchSize
		if end > n {
			end = n
		}

		batch := vectors[i*d : end*d]
		if err := index.Add(batch); err != nil {
			return err
		}
	}

	return nil
}

// withOSThreadLock executes a function with the goroutine locked to an OS thread
// This is useful for long-running C++ operations that benefit from cache locality
// and want to avoid Go scheduler migration overhead
func withOSThreadLock(fn func()) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	fn()
}

// BatchConfig provides configuration for batch operations
type BatchConfig struct {
	// BatchSize is the number of vectors per batch
	// Default: 10000 for search, 100000 for add
	BatchSize int

	// UseThreadLock controls whether to lock goroutine to OS thread
	// Default: auto (true for batches > 100)
	UseThreadLock *bool
}

// DefaultSearchBatchSize is the recommended batch size for search operations
const DefaultSearchBatchSize = 10000

// DefaultAddBatchSize is the recommended batch size for add operations
const DefaultAddBatchSize = 100000
