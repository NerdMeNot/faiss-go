# Benchmark CI Strategy

This document outlines the continuous benchmarking strategy for faiss-go.

## Overview

The benchmark CI is designed to:
1. **Track performance** across Go versions and CPU architectures
2. **Catch regressions** early with automated alerts
3. **Provide insights** on AMD64 vs ARM64 performance
4. **Balance speed and thoroughness** with quick and comprehensive modes

## Build Modes

faiss-go supports three build modes:

| Mode | Tag | Build Time | Use Case |
|------|-----|------------|----------|
| **Amalgamation** | `(default)` | ~2-5 min | CI benchmarks (most reliable) |
| **Static Libraries** | `-tags=faiss_use_lib` | ~30 sec | Development (when libs available) |
| **System FAISS** | `-tags=nogpu` | ~15-30 min | Legacy/testing |

**Current CI uses:** Amalgamation mode (no build tag) because static libraries aren't fully populated yet.

## Two-Tier Strategy

### 1. Quick Benchmarks (Every push to main)

**When:** On every push to `main` branch
**Duration:** ~10-15 minutes
**Go Versions:** 1.23, 1.24, 1.25 (latest stable versions)
**Architectures:** AMD64, ARM64
**Benchtime:** 1 second
**Scope:** Fast benchmarks only (IndexFlatL2, IndexIVFFlat_Search, IndexHNSW)

**Purpose:** Rapid feedback on performance-critical paths

```bash
go test -tags nogpu \
  -bench='BenchmarkIndexFlatL2|BenchmarkIndexIVFFlat_Search|BenchmarkIndexHNSW' \
  -benchmem -benchtime=1s -run=^$ ./...
```

### 2. Comprehensive Benchmarks (Weekly + Manual)

**When:**
- Weekly on Sunday at 00:00 UTC (cron schedule)
- Manual trigger via GitHub Actions UI

**Duration:** ~25-30 minutes
**Go Versions:** 1.21, 1.22, 1.23, 1.24, 1.25 (all supported versions)
**Architectures:** AMD64, ARM64
**Benchtime:** 5 seconds (configurable via workflow input)
**Scope:** All benchmarks

**Purpose:** Complete performance profiling and regression detection

```bash
go test -tags nogpu \
  -bench=. \
  -benchmem -benchtime=5s -run=^$ ./...
```

## Matrix Strategy

The workflow uses GitHub Actions matrix strategy to test all combinations:

```yaml
matrix:
  go-version: ['1.21', '1.22', '1.23', '1.24', '1.25']
  arch: ['amd64', 'arm64']
```

This creates **10 parallel jobs** for comprehensive benchmarks:
- Go 1.21 on AMD64
- Go 1.21 on ARM64
- Go 1.22 on AMD64
- Go 1.22 on ARM64
- ... and so on

## Results & Artifacts

### Artifacts Stored

1. **Quick benchmarks** (30 days retention)
   - `benchmark-quick-go<version>-<arch>-<sha>.txt`

2. **Comprehensive benchmarks** (90 days retention)
   - `benchmark-comprehensive-go<version>-<arch>-<sha>.txt`

3. **Comparison report** (90 days retention)
   - `benchmark-comparison-<sha>/comparison.md`

### Benchmark Comparison

The workflow automatically generates comparisons using `benchstat`:

1. **AMD64 vs ARM64** (Go 1.23)
   - Shows performance differences between architectures

2. **Go 1.21 vs Go 1.25** (AMD64)
   - Shows performance impact of Go version upgrades

Example output:
```
name                              old time/op    new time/op    delta
IndexFlatL2_Search_1K_K1-4          149µs ± 2%     142µs ± 3%   -4.70%
IndexHNSW_Search_100K_K10-4         159µs ± 1%     155µs ± 2%   -2.52%
```

## Performance Regression Detection

- **Threshold:** 150% of baseline performance
- **Action:** Automated comment on commit
- **Behavior:** Non-blocking (doesn't fail the workflow)

## Platform-Specific Notes

### Linux ARM64

The workflow uses `ubuntu-24.04-arm` for native ARM64 execution. If unavailable, you can use QEMU:

```yaml
- name: Set up QEMU
  uses: docker/setup-qemu-action@v3

- name: Run benchmarks in ARM64 container
  run: |
    docker run --rm --platform linux/arm64 \
      -v $PWD:/workspace -w /workspace \
      arm64v8/ubuntu:24.04 \
      bash -c "apt-get update && ..."
```

### Dependencies

All platforms require:
- **libopenblas-dev** - BLAS linear algebra library
- **libgomp1** - OpenMP runtime for parallel processing
- **libomp-dev** - OpenMP development headers

## Future Optimizations

### When Static Libraries are Ready

Once `libs/linux_amd64/libfaiss.a` and `libs/linux_arm64/libfaiss.a` are populated:

```yaml
- name: Run benchmarks (static lib build)
  env:
    CGO_LDFLAGS: "-lopenblas -lgomp -lstdc++ -lm"
  run: |
    go test -tags=faiss_use_lib,nogpu \
      -bench=. -benchmem -benchtime=5s -run=^$ ./...
```

**Benefits:**
- Build time: 30 seconds (vs 2-5 minutes)
- No compilation overhead
- Consistent FAISS version across builds

### To Populate Static Libraries

Run the build-static-libs workflow:

```bash
gh workflow run build-static-libs.yml \
  -f faiss_version=v1.13.2 \
  -f platforms=all
```

This will create a PR with pre-built libraries for all platforms.

## Monitoring & Maintenance

### Check Benchmark Results

```bash
# Download latest artifacts
gh run download <run-id>

# Compare with benchstat
benchstat old.txt new.txt
```

### Manual Trigger

```bash
# Run comprehensive benchmarks with custom benchtime
gh workflow run benchmark.yml \
  -f benchtime=10s
```

### View Historical Trends

Benchmark results are uploaded to GitHub Actions artifacts and can be tracked over time using the benchmark-action integration (when gh-pages is set up).

## Troubleshooting

### Benchmark Hangs

**Fixed in commit 1040189** - Timer usage bugs have been corrected. If benchmarks still hang:

1. Check timeout settings (currently 25-30 minutes)
2. Reduce benchtime: `-f benchtime=1s`
3. Run specific benchmarks: `-bench=BenchmarkIndexFlatL2`

### Out of Memory

ARM64 runners may have less memory. If OOM occurs:

1. Reduce parallel jobs: `fail-fast: true`
2. Limit benchmark scope
3. Use smaller datasets in tests

### Go Version Compatibility

Update the matrix when new Go versions are released:

```yaml
matrix:
  go-version: ['1.22', '1.23', '1.24']  # Update as needed
```

## Summary

This strategy provides:

✅ **Fast feedback** - Quick benchmarks on every push (10-15 min)
✅ **Comprehensive coverage** - Weekly full benchmarks (25-30 min)
✅ **Multi-platform** - AMD64 and ARM64 support
✅ **Multi-version** - Go 1.21 through 1.25 (5 versions tested)
✅ **Automated comparison** - benchstat reports
✅ **Regression detection** - Alerts on 150% slowdowns
✅ **Artifact retention** - 30-90 days of history

The benchmarking is now reliable, scalable, and provides actionable insights!
