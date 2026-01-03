package faiss

import (
	"sync"
	"sync/atomic"
	"time"
)

// Metrics provides performance tracking for FAISS operations.
// Enable with EnableMetrics() to start collecting data.
// Metrics collection has minimal overhead (~50ns per operation when enabled).
type Metrics struct {
	enabled atomic.Bool

	mu sync.RWMutex

	// Operation counts
	trainCount  int64
	addCount    int64
	searchCount int64
	resetCount  int64

	// Vector counts
	vectorsTrained int64
	vectorsAdded   int64
	queriesRun     int64

	// Timing (nanoseconds)
	trainTimeNs  int64
	addTimeNs    int64
	searchTimeNs int64

	// Search result counts
	resultsReturned int64
}

// OperationStats contains statistics for a single operation type.
type OperationStats struct {
	Count       int64         // Number of times operation was called
	TotalTime   time.Duration // Total time spent in operation
	AvgTime     time.Duration // Average time per operation
	VectorCount int64         // Total vectors processed (for train/add) or queries (for search)
}

// MetricsSnapshot contains a point-in-time snapshot of all metrics.
type MetricsSnapshot struct {
	Train  OperationStats
	Add    OperationStats
	Search OperationStats
	Reset  int64 // Number of reset operations

	// Aggregate stats
	TotalOperations int64
	TotalTime       time.Duration
	ResultsReturned int64 // Total search results returned
}

// Global metrics instance
var globalMetrics = &Metrics{}

// EnableMetrics enables global metrics collection.
func EnableMetrics() {
	globalMetrics.enabled.Store(true)
}

// DisableMetrics disables global metrics collection.
func DisableMetrics() {
	globalMetrics.enabled.Store(false)
}

// MetricsEnabled returns true if metrics collection is enabled.
func MetricsEnabled() bool {
	return globalMetrics.enabled.Load()
}

// GetMetrics returns a snapshot of the current metrics.
func GetMetrics() MetricsSnapshot {
	return globalMetrics.Snapshot()
}

// ResetMetrics clears all collected metrics.
func ResetMetrics() {
	globalMetrics.Reset()
}

// Snapshot returns a point-in-time copy of all metrics.
func (m *Metrics) Snapshot() MetricsSnapshot {
	m.mu.RLock()
	defer m.mu.RUnlock()

	snapshot := MetricsSnapshot{
		Train: OperationStats{
			Count:       m.trainCount,
			TotalTime:   time.Duration(m.trainTimeNs),
			VectorCount: m.vectorsTrained,
		},
		Add: OperationStats{
			Count:       m.addCount,
			TotalTime:   time.Duration(m.addTimeNs),
			VectorCount: m.vectorsAdded,
		},
		Search: OperationStats{
			Count:       m.searchCount,
			TotalTime:   time.Duration(m.searchTimeNs),
			VectorCount: m.queriesRun,
		},
		Reset:           m.resetCount,
		ResultsReturned: m.resultsReturned,
	}

	// Calculate averages
	if snapshot.Train.Count > 0 {
		snapshot.Train.AvgTime = snapshot.Train.TotalTime / time.Duration(snapshot.Train.Count)
	}
	if snapshot.Add.Count > 0 {
		snapshot.Add.AvgTime = snapshot.Add.TotalTime / time.Duration(snapshot.Add.Count)
	}
	if snapshot.Search.Count > 0 {
		snapshot.Search.AvgTime = snapshot.Search.TotalTime / time.Duration(snapshot.Search.Count)
	}

	snapshot.TotalOperations = m.trainCount + m.addCount + m.searchCount + m.resetCount
	snapshot.TotalTime = time.Duration(m.trainTimeNs + m.addTimeNs + m.searchTimeNs)

	return snapshot
}

// Reset clears all metrics.
func (m *Metrics) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.trainCount = 0
	m.addCount = 0
	m.searchCount = 0
	m.resetCount = 0
	m.vectorsTrained = 0
	m.vectorsAdded = 0
	m.queriesRun = 0
	m.trainTimeNs = 0
	m.addTimeNs = 0
	m.searchTimeNs = 0
	m.resultsReturned = 0
}

// recordTrain records a training operation.
func (m *Metrics) recordTrain(duration time.Duration, nVectors int) {
	if !m.enabled.Load() {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.trainCount++
	m.trainTimeNs += int64(duration)
	m.vectorsTrained += int64(nVectors)
}

// recordAdd records an add operation.
func (m *Metrics) recordAdd(duration time.Duration, nVectors int) {
	if !m.enabled.Load() {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.addCount++
	m.addTimeNs += int64(duration)
	m.vectorsAdded += int64(nVectors)
}

// recordSearch records a search operation.
func (m *Metrics) recordSearch(duration time.Duration, nQueries int, nResults int) {
	if !m.enabled.Load() {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.searchCount++
	m.searchTimeNs += int64(duration)
	m.queriesRun += int64(nQueries)
	m.resultsReturned += int64(nResults)
}

// recordReset records a reset operation.
func (m *Metrics) recordReset() {
	if !m.enabled.Load() {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.resetCount++
}

// Timer is a helper for timing operations.
// Usage:
//
//	timer := StartTimer()
//	// ... do work ...
//	timer.RecordTrain(nVectors)
type Timer struct {
	start time.Time
}

// StartTimer begins timing an operation.
func StartTimer() Timer {
	return Timer{start: time.Now()}
}

// RecordTrain records the elapsed time as a training operation.
func (t Timer) RecordTrain(nVectors int) {
	globalMetrics.recordTrain(time.Since(t.start), nVectors)
}

// RecordAdd records the elapsed time as an add operation.
func (t Timer) RecordAdd(nVectors int) {
	globalMetrics.recordAdd(time.Since(t.start), nVectors)
}

// RecordSearch records the elapsed time as a search operation.
func (t Timer) RecordSearch(nQueries, nResults int) {
	globalMetrics.recordSearch(time.Since(t.start), nQueries, nResults)
}

// RecordReset records a reset operation.
func (t Timer) RecordReset() {
	globalMetrics.recordReset()
}

// Elapsed returns the time elapsed since the timer started.
func (t Timer) Elapsed() time.Duration {
	return time.Since(t.start)
}

// QPS calculates queries per second from a search snapshot.
func (s OperationStats) QPS() float64 {
	if s.TotalTime == 0 {
		return 0
	}
	return float64(s.VectorCount) / s.TotalTime.Seconds()
}

// VectorsPerSecond calculates throughput for add operations.
func (s OperationStats) VectorsPerSecond() float64 {
	if s.TotalTime == 0 {
		return 0
	}
	return float64(s.VectorCount) / s.TotalTime.Seconds()
}
