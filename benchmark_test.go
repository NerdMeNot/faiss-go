package faiss

import (
	"testing"
)

// ========================================
// Index Creation Benchmarks
// ========================================

func BenchmarkIndexFlatL2_Create(b *testing.B) {
	d := 128
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, _ := NewIndexFlatL2(d)
		idx.Close()
	}
}

func BenchmarkIndexIVFFlat_Create(b *testing.B) {
	d := 128
	nlist := 100
	quantizer, _ := NewIndexFlatL2(d)
	defer quantizer.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, _ := NewIndexIVFFlat(quantizer, d, nlist, MetricL2)
		idx.Close()
	}
}

func BenchmarkIndexHNSW_Create(b *testing.B) {
	d := 128
	M := 16
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, _ := NewIndexHNSWFlat(d, M, MetricL2)
		idx.Close()
	}
}

func BenchmarkIndexPQ_Create(b *testing.B) {
	d := 128
	M := 16
	nbits := 8
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, _ := NewIndexPQ(d, M, nbits, MetricL2)
		idx.Close()
	}
}

// ========================================
// Vector Addition Benchmarks
// ========================================

func BenchmarkIndexFlatL2_Add_1K(b *testing.B) {
	benchmarkIndexAdd(b, "IndexFlatL2", 128, 1000)
}

func BenchmarkIndexFlatL2_Add_10K(b *testing.B) {
	benchmarkIndexAdd(b, "IndexFlatL2", 128, 10000)
}

func BenchmarkIndexFlatL2_Add_100K(b *testing.B) {
	benchmarkIndexAdd(b, "IndexFlatL2", 128, 100000)
}

func BenchmarkIndexHNSW_Add_1K(b *testing.B) {
	benchmarkIndexAdd(b, "IndexHNSW", 128, 1000)
}

func BenchmarkIndexHNSW_Add_10K(b *testing.B) {
	benchmarkIndexAdd(b, "IndexHNSW", 128, 10000)
}

func benchmarkIndexAdd(b *testing.B, indexType string, d, nb int) {
	vectors := generateVectors(nb, d)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		var idx Index
		switch indexType {
		case "IndexFlatL2":
			idx, _ = NewIndexFlatL2(d)
		case "IndexHNSW":
			idx, _ = NewIndexHNSWFlat(d, 16, MetricL2)
		}
		b.StartTimer()

		idx.Add(vectors)

		b.StopTimer()
		idx.Close()
		b.StartTimer()
	}

	// Report throughput
	b.ReportMetric(float64(nb*b.N)/b.Elapsed().Seconds(), "vectors/sec")
}

// ========================================
// Search Benchmarks
// ========================================

func BenchmarkIndexFlatL2_Search_1K_K1(b *testing.B) {
	benchmarkIndexSearch(b, "IndexFlatL2", 128, 1000, 1, 10)
}

func BenchmarkIndexFlatL2_Search_1K_K10(b *testing.B) {
	benchmarkIndexSearch(b, "IndexFlatL2", 128, 1000, 10, 10)
}

func BenchmarkIndexFlatL2_Search_10K_K10(b *testing.B) {
	benchmarkIndexSearch(b, "IndexFlatL2", 128, 10000, 10, 10)
}

func BenchmarkIndexFlatL2_Search_100K_K10(b *testing.B) {
	benchmarkIndexSearch(b, "IndexFlatL2", 128, 100000, 10, 10)
}

func BenchmarkIndexHNSW_Search_10K_K10(b *testing.B) {
	benchmarkIndexSearch(b, "IndexHNSW", 128, 10000, 10, 10)
}

func BenchmarkIndexHNSW_Search_100K_K10(b *testing.B) {
	benchmarkIndexSearch(b, "IndexHNSW", 128, 100000, 10, 10)
}

func BenchmarkIndexIVFFlat_Search_10K_K10_Nprobe1(b *testing.B) {
	benchmarkIndexIVFFlatSearch(b, 128, 10000, 10, 10, 1)
}

func BenchmarkIndexIVFFlat_Search_10K_K10_Nprobe10(b *testing.B) {
	benchmarkIndexIVFFlatSearch(b, 128, 10000, 10, 10, 10)
}

func BenchmarkIndexIVFFlat_Search_100K_K10_Nprobe1(b *testing.B) {
	benchmarkIndexIVFFlatSearch(b, 128, 100000, 10, 10, 1)
}

func BenchmarkIndexIVFFlat_Search_100K_K10_Nprobe10(b *testing.B) {
	benchmarkIndexIVFFlatSearch(b, 128, 100000, 10, 10, 10)
}

func benchmarkIndexSearch(b *testing.B, indexType string, d, nb, k, nq int) {
	// Setup
	vectors := generateVectors(nb, d)
	queries := generateVectors(nq, d)

	var idx Index
	switch indexType {
	case "IndexFlatL2":
		idx, _ = NewIndexFlatL2(d)
	case "IndexHNSW":
		idx, _ = NewIndexHNSWFlat(d, 16, MetricL2)
	}
	defer idx.Close()

	idx.Add(vectors)

	// Benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(queries, k)
	}

	// Report QPS (queries per second)
	totalQueries := nq * b.N
	b.ReportMetric(float64(totalQueries)/b.Elapsed().Seconds(), "qps")
}

func benchmarkIndexIVFFlatSearch(b *testing.B, d, nb, k, nq, nprobe int) {
	// Setup
	nlist := 100
	vectors := generateVectors(nb, d)
	queries := generateVectors(nq, d)

	quantizer, _ := NewIndexFlatL2(d)
	defer quantizer.Close()

	idx, _ := NewIndexIVFFlat(quantizer, d, nlist, MetricL2)
	defer idx.Close()

	idx.Train(vectors)
	idx.Add(vectors)
	idx.SetNprobe(nprobe)

	// Benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(queries, k)
	}

	// Report QPS
	totalQueries := nq * b.N
	b.ReportMetric(float64(totalQueries)/b.Elapsed().Seconds(), "qps")
}

// ========================================
// Training Benchmarks
// ========================================

func BenchmarkIndexIVFFlat_Train_10K(b *testing.B) {
	d := 128
	nlist := 100
	nb := 10000
	vectors := generateVectors(nb, d)

	quantizer, _ := NewIndexFlatL2(d)
	defer quantizer.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		idx, _ := NewIndexIVFFlat(quantizer, d, nlist, MetricL2)
		b.StartTimer()

		idx.Train(vectors)

		b.StopTimer()
		idx.Close()
		b.StartTimer()
	}
}

func BenchmarkIndexPQ_Train_10K(b *testing.B) {
	d := 128
	M := 16
	nbits := 8
	nb := 10000
	vectors := generateVectors(nb, d)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		idx, _ := NewIndexPQ(d, M, nbits, MetricL2)
		b.StartTimer()

		idx.Train(vectors)

		b.StopTimer()
		idx.Close()
		b.StartTimer()
	}
}

// ========================================
// Range Search Benchmarks
// ========================================

func BenchmarkIndexFlatL2_RangeSearch_1K(b *testing.B) {
	d := 128
	nb := 1000
	nq := 10
	radius := float32(10.0)

	vectors := generateVectors(nb, d)
	queries := generateVectors(nq, d)

	idx, _ := NewIndexFlatL2(d)
	defer idx.Close()
	idx.Add(vectors)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.RangeSearch(queries, radius)
	}

	totalQueries := nq * b.N
	b.ReportMetric(float64(totalQueries)/b.Elapsed().Seconds(), "qps")
}

func BenchmarkIndexFlatL2_RangeSearch_10K(b *testing.B) {
	d := 128
	nb := 10000
	nq := 10
	radius := float32(10.0)

	vectors := generateVectors(nb, d)
	queries := generateVectors(nq, d)

	idx, _ := NewIndexFlatL2(d)
	defer idx.Close()
	idx.Add(vectors)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.RangeSearch(queries, radius)
	}

	totalQueries := nq * b.N
	b.ReportMetric(float64(totalQueries)/b.Elapsed().Seconds(), "qps")
}

// ========================================
// Binary Index Benchmarks
// ========================================

func BenchmarkIndexBinaryFlat_Add_1K(b *testing.B) {
	d := 256
	nb := 1000
	vectors := generateBinaryVectors(nb, d)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		idx, _ := NewIndexBinaryFlat(d)
		b.StartTimer()

		idx.Add(vectors)

		b.StopTimer()
		idx.Close()
		b.StartTimer()
	}

	b.ReportMetric(float64(nb*b.N)/b.Elapsed().Seconds(), "vectors/sec")
}

func BenchmarkIndexBinaryFlat_Search_10K(b *testing.B) {
	d := 256
	nb := 10000
	nq := 10
	k := 10

	vectors := generateBinaryVectors(nb, d)
	queries := generateBinaryVectors(nq, d)

	idx, _ := NewIndexBinaryFlat(d)
	defer idx.Close()
	idx.Add(vectors)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(queries, k)
	}

	totalQueries := nq * b.N
	b.ReportMetric(float64(totalQueries)/b.Elapsed().Seconds(), "qps")
}

// ========================================
// Transform Benchmarks
// ========================================

func BenchmarkPCAMatrix_Train(b *testing.B) {
	dIn := 128
	dOut := 64
	nb := 5000
	vectors := generateVectors(nb, dIn)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		pca, _ := NewPCAMatrix(dIn, dOut)
		b.StartTimer()

		pca.Train(vectors)

		b.StopTimer()
		pca.Close()
		b.StartTimer()
	}
}

func BenchmarkPCAMatrix_Apply(b *testing.B) {
	dIn := 128
	dOut := 64
	nb := 5000
	nApply := 1000

	trainingVectors := generateVectors(nb, dIn)
	applyVectors := generateVectors(nApply, dIn)

	pca, _ := NewPCAMatrix(dIn, dOut)
	defer pca.Close()
	pca.Train(trainingVectors)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pca.Apply(applyVectors)
	}

	b.ReportMetric(float64(nApply*b.N)/b.Elapsed().Seconds(), "vectors/sec")
}

func BenchmarkOPQMatrix_Train(b *testing.B) {
	d := 128
	M := 16
	nb := 5000
	vectors := generateVectors(nb, d)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		opq, _ := NewOPQMatrix(d, M)
		b.StartTimer()

		opq.Train(vectors)

		b.StopTimer()
		opq.Close()
		b.StartTimer()
	}
}

// ========================================
// Utility Benchmarks
// ========================================

func BenchmarkKMin_1K(b *testing.B) {
	n := 1000
	k := 10
	vals := RandUniform(n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		KMin(vals, k)
	}
}

func BenchmarkKMax_1K(b *testing.B) {
	n := 1000
	k := 10
	vals := RandUniform(n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		KMax(vals, k)
	}
}

func BenchmarkL2Distance(b *testing.B) {
	d := 128
	a := RandUniform(d)
	b_vec := RandUniform(d)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		L2Distance(a, b_vec)
	}
}

func BenchmarkInnerProduct(b *testing.B) {
	d := 128
	a := RandUniform(d)
	b_vec := RandUniform(d)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		InnerProduct(a, b_vec)
	}
}

func BenchmarkBatchL2Distance_1K_Queries(b *testing.B) {
	d := 128
	nq := 100
	nb := 1000

	queries := generateVectors(nq, d)
	database := generateVectors(nb, d)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BatchL2Distance(queries, database, d)
	}

	// Report throughput in distance computations per second
	totalComps := nq * nb * b.N
	b.ReportMetric(float64(totalComps)/b.Elapsed().Seconds(), "comps/sec")
}

func BenchmarkBatchInnerProduct_1K_Queries(b *testing.B) {
	d := 128
	nq := 100
	nb := 1000

	queries := generateVectors(nq, d)
	database := generateVectors(nb, d)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BatchInnerProduct(queries, database, d)
	}

	totalComps := nq * nb * b.N
	b.ReportMetric(float64(totalComps)/b.Elapsed().Seconds(), "comps/sec")
}

// ========================================
// Clustering Benchmarks
// ========================================

func BenchmarkKmeans_Train_1K(b *testing.B) {
	d := 64
	k := 10
	nb := 1000
	vectors := generateVectors(nb, d)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		km, _ := NewKmeans(d, k, 25)
		b.StartTimer()

		km.Train(vectors)

		b.StopTimer()
		km.Close()
		b.StartTimer()
	}
}

func BenchmarkKmeans_Assign_1K(b *testing.B) {
	d := 64
	k := 10
	nb := 1000
	nAssign := 1000

	trainingVectors := generateVectors(nb, d)
	assignVectors := generateVectors(nAssign, d)

	km, _ := NewKmeans(d, k, 25)
	defer km.Close()
	km.Train(trainingVectors)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Assign(assignVectors)
	}

	b.ReportMetric(float64(nAssign*b.N)/b.Elapsed().Seconds(), "vectors/sec")
}

// ========================================
// Serialization Benchmarks
// ========================================

func BenchmarkIndexFlat_Serialize(b *testing.B) {
	d := 128
	nb := 10000
	vectors := generateVectors(nb, d)

	idx, _ := NewIndexFlatL2(d)
	defer idx.Close()
	idx.Add(vectors)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, _ := SerializeIndex(idx)
		b.SetBytes(int64(len(data)))
	}
}

func BenchmarkIndexFlat_Deserialize(b *testing.B) {
	d := 128
	nb := 10000
	vectors := generateVectors(nb, d)

	idx, _ := NewIndexFlatL2(d)
	idx.Add(vectors)
	data, _ := SerializeIndex(idx)
	idx.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		newIdx, _ := DeserializeIndex(data)
		b.StartTimer()

		b.StopTimer()
		newIdx.Close()
		b.StartTimer()
	}
	b.SetBytes(int64(len(data)))
}

// ========================================
// Memory Benchmarks
// ========================================

func BenchmarkIndexFlat_MemoryAllocation(b *testing.B) {
	d := 128
	nb := 1000
	vectors := generateVectors(nb, d)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx, _ := NewIndexFlatL2(d)
		idx.Add(vectors)
		idx.Close()
	}
}

func BenchmarkSearch_MemoryAllocation(b *testing.B) {
	d := 128
	nb := 10000
	nq := 10
	k := 10

	vectors := generateVectors(nb, d)
	queries := generateVectors(nq, d)

	idx, _ := NewIndexFlatL2(d)
	defer idx.Close()
	idx.Add(vectors)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		idx.Search(queries, k)
	}
}
