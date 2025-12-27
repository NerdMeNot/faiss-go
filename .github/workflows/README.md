# GitHub Actions Workflows

This directory contains automated CI/CD workflows for the faiss-go project.

## Workflows

### 1. CI (`ci.yml`)

**Triggers:** Push to main or `claude/*` branches, Pull Requests

**Purpose:** Main continuous integration pipeline

**What it does:**
- Tests across multiple Go versions (1.21, 1.22, 1.23)
- Tests with different FAISS versions (1.7.4, 1.8.0)
- Builds FAISS from source and caches it for faster subsequent runs
- Runs all tests with coverage reporting
- Uploads coverage to Codecov
- Runs sample benchmarks
- Tests on both Ubuntu and macOS

**Matrix:**
- Go versions: 1.21, 1.22, 1.23
- FAISS versions: 1.7.4, 1.8.0
- OS: Ubuntu (all combinations), macOS (Go 1.22, 1.23)

**Cache Strategy:**
FAISS builds are cached by version and OS to speed up CI runs from ~15 minutes to ~3 minutes.

---

### 2. GPU CI (`gpu-ci.yml`)

**Triggers:** Push to main, Pull Requests, Manual dispatch

**Purpose:** Test GPU-accelerated functionality

**What it does:**
- Checks for CUDA availability
- Builds FAISS with GPU support (CUDA 12.3)
- Compiles Go code with GPU flags
- Runs GPU-specific tests and benchmarks
- Uploads benchmark results

**Requirements:**
- GitHub runner with NVIDIA GPU
- CUDA toolkit installed
- Uncomment `runs-on: [self-hosted, linux, gpu]` if you have GPU runners

**Note:** Currently configured for standard runners (will skip if no GPU). Update the `runs-on` field if you have self-hosted GPU runners.

---

### 3. Continuous Benchmarking (`benchmark.yml`)

**Triggers:**
- Push to main
- Weekly schedule (Sunday 00:00 UTC)
- Manual dispatch

**Purpose:** Track performance over time and detect regressions

**What it does:**
- Runs comprehensive benchmarks with 5-second runs
- Tracks results using `github-action-benchmark`
- Alerts on performance regressions (>150% slowdown)
- Compares PR benchmarks with base branch
- Stores benchmark history for 90 days

**Benchmark Comparison:**
On pull requests, uses `benchstat` to compare current vs base performance:
```
name                    old time/op    new time/op    delta
IndexFlatL2_Add_1K-8      125µs ± 2%     118µs ± 1%   -5.60%
IndexFlatL2_Search-8      45.3µs ± 1%    44.1µs ± 2%   -2.65%
```

**Performance Alerts:**
- Detects regressions automatically
- Comments on PR with performance issues
- Threshold: 150% (configurable in workflow)

---

## Setup Requirements

### For Basic CI (Ubuntu/macOS)

No additional setup required. The workflow automatically:
1. Installs system dependencies
2. Downloads and builds FAISS
3. Caches the build for reuse

### For GPU CI

**Option 1: Self-hosted GPU runner**
1. Set up a self-hosted runner with NVIDIA GPU
2. Install CUDA toolkit 12.3+
3. Update `gpu-ci.yml`:
   ```yaml
   runs-on: [self-hosted, linux, gpu]
   ```

**Option 2: Use GitHub-hosted GPU runners** (when available)
- Update to use GitHub's GPU runners when they become available

**Option 3: Skip GPU tests**
- Keep current configuration (will gracefully skip if no GPU)

### For Benchmark Tracking

**Optional: Enable benchmark visualization**
1. Create `gh-pages` branch:
   ```bash
   git checkout --orphan gh-pages
   git reset --hard
   git commit --allow-empty -m "Initial gh-pages commit"
   git push origin gh-pages
   ```

2. Enable GitHub Pages in repository settings:
   - Settings → Pages → Source: `gh-pages` branch

3. View benchmark graphs at:
   `https://<username>.github.io/<repo>/dev/bench/`

### For Coverage Reporting

**Optional: Set up Codecov**
1. Sign up at https://codecov.io
2. Add repository
3. Add `CODECOV_TOKEN` to repository secrets (if private repo)

---

## Environment Variables

The workflows set these environment variables for building:

### CPU-only builds (`ci.yml`):
```bash
CGO_LDFLAGS="-L/usr/local/lib -lfaiss -lgomp -lstdc++ -lm -lopenblas"
CGO_CFLAGS="-I/usr/local/include"
LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
```

### GPU builds (`gpu-ci.yml`):
```bash
CGO_LDFLAGS="-L/usr/local/lib -lfaiss -lfaiss_gpu -lcudart -lcublas -lgomp -lstdc++ -lm -lopenblas"
CGO_CFLAGS="-I/usr/local/include -DFAISS_GPU"
LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

---

## Local Development

To build locally with the same configuration as CI:

### Ubuntu/Debian:
```bash
# Install dependencies
sudo apt-get install cmake g++ libopenblas-dev libgomp1 libomp-dev

# Build FAISS
cd /tmp
wget https://github.com/facebookresearch/faiss/archive/refs/tags/v1.8.0.tar.gz
tar -xzf v1.8.0.tar.gz
cd faiss-1.8.0
cmake -B build -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=OFF -DBUILD_SHARED_LIBS=ON .
make -C build -j$(nproc)
sudo make -C build install
sudo ldconfig

# Build Go project
export CGO_LDFLAGS="-L/usr/local/lib -lfaiss -lgomp -lstdc++ -lm -lopenblas"
export CGO_CFLAGS="-I/usr/local/include"
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
go build -tags nogpu -v ./...
```

### macOS:
```bash
# Install dependencies
brew install cmake openblas libomp faiss

# Build Go project
export CGO_LDFLAGS="-L/opt/homebrew/lib -lfaiss -lomp"
export CGO_CFLAGS="-I/opt/homebrew/include"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
go build -tags nogpu -v ./...
```

---

## Troubleshooting

### "cannot find -lfaiss"
- FAISS library not installed or not in library path
- Check: `ldconfig -p | grep faiss`
- Verify: `ls /usr/local/lib/libfaiss*`

### Tests timeout
- Increase timeout in workflow: `timeout: 20m`
- Some tests (especially IVF training) can be slow

### Benchmarks show high variance
- GitHub runners have variable performance
- Use larger `-benchtime` for more stable results
- Compare trends over multiple runs, not single values

### Cache not working
- Check cache key matches exactly
- GitHub cache has 10GB limit per repository
- Caches expire after 7 days of no use

---

## Workflow Status Badges

Add to your README.md:

```markdown
![CI](https://github.com/USERNAME/REPO/workflows/CI/badge.svg)
![GPU CI](https://github.com/USERNAME/REPO/workflows/GPU%20CI/badge.svg)
![Benchmarks](https://github.com/USERNAME/REPO/workflows/Continuous%20Benchmarking/badge.svg)
[![codecov](https://codecov.io/gh/USERNAME/REPO/branch/main/graph/badge.svg)](https://codecov.io/gh/USERNAME/REPO)
```

---

## Performance Expectations

Typical CI run times:

| Workflow | First Run | Cached Run |
|----------|-----------|------------|
| CI (Ubuntu) | ~15 min | ~3 min |
| CI (macOS) | ~8 min | ~2 min |
| GPU CI | ~20 min | ~5 min |
| Benchmarks | ~8 min | ~4 min |

Cache hit rate: ~90% for subsequent runs on same FAISS version.
