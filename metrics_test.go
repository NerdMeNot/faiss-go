package faiss

import (
	"testing"
	"time"
)

func TestMetrics_EnableDisable(t *testing.T) {
	// Start fresh
	ResetMetrics()
	DisableMetrics()

	if MetricsEnabled() {
		t.Error("Metrics should be disabled by default after DisableMetrics()")
	}

	EnableMetrics()
	if !MetricsEnabled() {
		t.Error("Metrics should be enabled after EnableMetrics()")
	}

	DisableMetrics()
	if MetricsEnabled() {
		t.Error("Metrics should be disabled after DisableMetrics()")
	}
}

func TestMetrics_Recording(t *testing.T) {
	ResetMetrics()
	EnableMetrics()
	defer DisableMetrics()

	// Record some operations
	globalMetrics.recordTrain(100*time.Millisecond, 1000)
	globalMetrics.recordAdd(50*time.Millisecond, 500)
	globalMetrics.recordSearch(10*time.Millisecond, 10, 100)
	globalMetrics.recordReset()

	snapshot := GetMetrics()

	// Verify counts
	if snapshot.Train.Count != 1 {
		t.Errorf("Train count = %d, want 1", snapshot.Train.Count)
	}
	if snapshot.Add.Count != 1 {
		t.Errorf("Add count = %d, want 1", snapshot.Add.Count)
	}
	if snapshot.Search.Count != 1 {
		t.Errorf("Search count = %d, want 1", snapshot.Search.Count)
	}
	if snapshot.Reset != 1 {
		t.Errorf("Reset count = %d, want 1", snapshot.Reset)
	}

	// Verify vector counts
	if snapshot.Train.VectorCount != 1000 {
		t.Errorf("Vectors trained = %d, want 1000", snapshot.Train.VectorCount)
	}
	if snapshot.Add.VectorCount != 500 {
		t.Errorf("Vectors added = %d, want 500", snapshot.Add.VectorCount)
	}
	if snapshot.Search.VectorCount != 10 {
		t.Errorf("Queries run = %d, want 10", snapshot.Search.VectorCount)
	}

	// Verify timing
	if snapshot.Train.TotalTime != 100*time.Millisecond {
		t.Errorf("Train time = %v, want 100ms", snapshot.Train.TotalTime)
	}

	// Verify aggregates
	if snapshot.TotalOperations != 4 {
		t.Errorf("Total operations = %d, want 4", snapshot.TotalOperations)
	}
	if snapshot.ResultsReturned != 100 {
		t.Errorf("Results returned = %d, want 100", snapshot.ResultsReturned)
	}
}

func TestMetrics_DisabledNoRecording(t *testing.T) {
	ResetMetrics()
	DisableMetrics()

	// Record operations while disabled
	globalMetrics.recordTrain(100*time.Millisecond, 1000)
	globalMetrics.recordAdd(50*time.Millisecond, 500)
	globalMetrics.recordSearch(10*time.Millisecond, 10, 100)

	snapshot := GetMetrics()

	if snapshot.TotalOperations != 0 {
		t.Errorf("Operations recorded while disabled: %d", snapshot.TotalOperations)
	}
}

func TestMetrics_Timer(t *testing.T) {
	ResetMetrics()
	EnableMetrics()
	defer DisableMetrics()

	timer := StartTimer()
	time.Sleep(10 * time.Millisecond)
	timer.RecordSearch(5, 50)

	snapshot := GetMetrics()

	if snapshot.Search.Count != 1 {
		t.Errorf("Search count = %d, want 1", snapshot.Search.Count)
	}
	if snapshot.Search.TotalTime < 10*time.Millisecond {
		t.Errorf("Search time = %v, want >= 10ms", snapshot.Search.TotalTime)
	}
}

func TestMetrics_Reset(t *testing.T) {
	ResetMetrics()
	EnableMetrics()
	defer DisableMetrics()

	globalMetrics.recordAdd(50*time.Millisecond, 500)
	globalMetrics.recordSearch(10*time.Millisecond, 10, 100)

	ResetMetrics()
	snapshot := GetMetrics()

	if snapshot.TotalOperations != 0 {
		t.Errorf("Operations after reset: %d, want 0", snapshot.TotalOperations)
	}
}

func TestMetrics_QPS(t *testing.T) {
	stats := OperationStats{
		Count:       100,
		TotalTime:   1 * time.Second,
		VectorCount: 1000,
	}

	qps := stats.QPS()
	if qps != 1000.0 {
		t.Errorf("QPS = %f, want 1000", qps)
	}

	vps := stats.VectorsPerSecond()
	if vps != 1000.0 {
		t.Errorf("VectorsPerSecond = %f, want 1000", vps)
	}
}

func TestMetrics_WithRealIndex(t *testing.T) {
	ResetMetrics()
	EnableMetrics()
	defer DisableMetrics()

	d := 64
	index, err := NewIndexFlatL2(d)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Add vectors
	vectors := make([]float32, d*100)
	for i := range vectors {
		vectors[i] = float32(i)
	}

	if err := index.Add(vectors); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Search
	query := vectors[:d]
	_, _, err = index.Search(query, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Check metrics were recorded
	snapshot := GetMetrics()

	if snapshot.Add.Count < 1 {
		t.Errorf("Add operations = %d, want >= 1", snapshot.Add.Count)
	}
	if snapshot.Search.Count < 1 {
		t.Errorf("Search operations = %d, want >= 1", snapshot.Search.Count)
	}
	if snapshot.Add.VectorCount < 100 {
		t.Errorf("Vectors added = %d, want >= 100", snapshot.Add.VectorCount)
	}

	t.Logf("Metrics snapshot:")
	t.Logf("  Add: %d ops, %d vectors, avg %v", snapshot.Add.Count, snapshot.Add.VectorCount, snapshot.Add.AvgTime)
	t.Logf("  Search: %d ops, %d queries, avg %v, %.0f QPS",
		snapshot.Search.Count, snapshot.Search.VectorCount, snapshot.Search.AvgTime, snapshot.Search.QPS())
}

func BenchmarkMetrics_Overhead(b *testing.B) {
	ResetMetrics()
	EnableMetrics()
	defer DisableMetrics()

	b.Run("WithMetrics", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			timer := StartTimer()
			timer.RecordSearch(1, 10)
		}
	})

	DisableMetrics()

	b.Run("WithoutMetrics", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			timer := StartTimer()
			timer.RecordSearch(1, 10)
		}
	})
}
