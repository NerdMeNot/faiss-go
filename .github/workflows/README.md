# GitHub Actions Workflows

This document describes the CI/CD workflows for the faiss-go project following Go best practices.

## Workflow Overview

| Workflow | Triggers | Purpose | Est. Duration |
|----------|----------|---------|---------------|
| **CI** | PRs, main pushes, version tags, manual | Main build, test, and lint pipeline | 3-8 min |
| **GPU CI** | PRs (code changes), manual | GPU acceleration testing | 15-20 min |
| **Continuous Benchmarking** | Main pushes, weekly, manual | Performance tracking | 5-10 min |
| **Release** | Manual only | Version management and releases | 5-8 min |

---

## Workflows

### 1. CI Workflow (`ci.yml`)

**Purpose:** Validates code quality, builds, and tests across multiple platforms and Go versions.

**Triggers:**
- ✅ Pull requests to `main` - Ensures quality before merge
- ✅ Pushes to `main` - Validates merged code
- ✅ Version tags (`v*`) - Validates releases
- ✅ Manual dispatch - On-demand testing
- ❌ ~~Feature branch pushes~~ - Use manual dispatch if needed

**Jobs:**

1. **build-and-test (Ubuntu)**
   - Go versions: 1.22, 1.23
   - FAISS version: 1.8.0
   - Runs: Build, tests, coverage reporting
   - Coverage uploaded to Codecov
   - Sample benchmarks

2. **build-macos**
   - Go versions: 1.22, 1.23
   - Tests on macOS with homebrew FAISS

3. **lint**
   - golangci-lint v1.62.2
   - Build tags: `nogpu`
   - Timeout: 5 minutes

**Features:**
- FAISS build caching (reduces 15 min → 3 min)
- Multi-platform testing
- Comprehensive environment setup for CGO

---

### 2. GPU CI Workflow (`gpu-ci.yml`)

**Purpose:** Tests GPU-accelerated FAISS functionality.

**Triggers:**
- ✅ Pull requests to `main` (only when code changes: `**.go`, `go.mod`, `go.sum`, workflow file)
- ✅ Manual dispatch - On-demand GPU testing
- ❌ ~~Automatic pushes~~ - Too expensive, use manual trigger

**Jobs:**

1. **gpu-test**
   - CUDA 12.3 support
   - Graceful skip if CUDA unavailable
   - GPU-specific tests and benchmarks
   - Uploads benchmark results

**Notes:**
- Currently uses standard runners (CUDA not available)
- Requires self-hosted GPU runner for actual GPU testing
- Path filtering prevents unnecessary runs

---

### 3. Continuous Benchmarking Workflow (`benchmark.yml`)

**Purpose:** Tracks performance over time and detects regressions.

**Triggers:**
- ✅ Pushes to `main` - Track performance changes
- ✅ Weekly schedule (Sundays 00:00 UTC) - Regular monitoring
- ✅ Manual dispatch - On-demand benchmarking

**Jobs:**

1. **benchmark**
   - Runs comprehensive benchmarks (5-second timing)
   - Uses `github-action-benchmark` for tracking
   - Alerts on 150% performance regression
   - Stores results for 90 days

2. **compare-benchmarks** (PRs only)
   - Uses `benchstat` for detailed comparisons
   - Generates PR comments with performance deltas

**Example Output:**
```
name                    old time/op    new time/op    delta
IndexFlatL2_Add_1K-8      125µs ± 2%     118µs ± 1%   -5.60%
IndexFlatL2_Search-8      45.3µs ± 1%    44.1µs ± 2%   -2.65%
```

---

### 4. Release Workflow (`release.yml`) 🆕

**Purpose:** Automates version management and creates GitHub releases following Go module best practices.

**Triggers:**
- ✅ Manual dispatch ONLY - Controlled release process

**Input Parameters:**
- `version_type`: `patch` | `minor` | `major`
- `prerelease_suffix`: Optional (e.g., `alpha.1`, `beta.1`, `rc.1`)

**Process:**
1. ✅ Validates current state (clean main branch)
2. ✅ Calculates next version number based on semver
3. ✅ Updates version constant in `faiss.go`
4. ✅ Generates categorized changelog from commits
5. ✅ Runs full test suite and linting
6. ✅ Commits version bump to main
7. ✅ Creates and pushes git tag (e.g., `v0.2.0`)
8. ✅ Creates GitHub release with auto-generated notes

**Changelog Categories:**
- ✨ Features (`feat:`)
- 🐛 Bug Fixes (`fix:`)
- ⚡ Performance (`perf:`)
- 📚 Documentation (`docs:`)
- 🔧 Other Changes

**Usage:**
```bash
# Navigate to: Actions → Release → Run workflow
# Select version type: patch/minor/major
# Optional: Add pre-release suffix (e.g., alpha.1)
# Click "Run workflow"
```

**Version Examples:**
- `v0.1.0` → `v0.1.1` (patch) - Bug fixes
- `v0.1.0` → `v0.2.0` (minor) - New features
- `v0.9.0` → `v1.0.0` (major) - Breaking changes or API stability
- `v0.2.0-alpha.1` (pre-release) - Testing before official release

**Go Module Integration:**
Users can immediately use the new version:
```bash
go get github.com/NerdMeNot/faiss-go@v0.2.0
```

---

## Versioning Strategy

See [VERSIONING.md](../../VERSIONING.md) for complete details.

**Summary:**
- Follows [Semantic Versioning 2.0.0](https://semver.org/)
- Git tags for releases (no release branches - Go best practice)
- Manual release process for quality control
- [Conventional Commits](https://www.conventionalcommits.org/) for changelog generation

**Branching:**
- `main` - Always stable, protected
- `claude/*` or `feature/*` - Short-lived development branches
- Tags: `v0.1.0`, `v0.2.0`, `v1.0.0`, etc.

**Version Format:**
- `v0.x.y` - Pre-1.0 development (breaking changes allowed)
- `v1.x.y` - Stable API (semver guarantees)
- `v2.x.y+` - Major versions (requires `/v2` in module path)

---

## CI Optimization Strategy

### Reduced Workflow Frequency 🎯

**Before:**
- ❌ CI ran on every push to `main` AND `claude/*` branches
- ❌ GPU CI ran on every push to `main`
- ❌ High CI minutes usage (~500-800 min/week)

**After:**
- ✅ CI only on PRs, main pushes, version tags, and manual dispatch
- ✅ GPU CI only on PRs with code changes, and manual dispatch
- ✅ Benchmarks only on main pushes, weekly, and manual dispatch
- ✅ **Result:** ~60-70% reduction in CI runs (~150-250 min/week)

### Manual Triggers

All workflows support `workflow_dispatch` for on-demand execution:

**How to Run:**
1. Navigate to: `Actions` → Select workflow (e.g., "CI")
2. Click `Run workflow` button
3. Select branch (for CI/GPU CI)
4. Select options (for Release)
5. Click `Run workflow`

**Use Cases:**
- Testing feature branches before creating PR
- Re-running failed workflows without new commits
- On-demand GPU testing for performance validation
- Benchmark comparisons for specific commits
- Creating releases when ready

---

## Setup Requirements

### For Local Development

#### CPU-only Build (Ubuntu/Debian)

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

#### CPU-only Build (macOS)

```bash
# Install dependencies
brew install cmake openblas libomp faiss

# Build Go project
export CGO_LDFLAGS="-L/opt/homebrew/lib -lfaiss -lomp"
export CGO_CFLAGS="-I/opt/homebrew/include"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
go build -tags nogpu -v ./...
```

#### GPU Build (Optional)

```bash
# Requires CUDA toolkit
sudo apt-get install cuda-toolkit-12-3

# Build FAISS with GPU support (see gpu-ci.yml for complete steps)
cmake -B build -DFAISS_ENABLE_GPU=ON \
  -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86" ...

# Build project with GPU flags
export CGO_LDFLAGS="-L/usr/local/lib -lfaiss -lfaiss_gpu -lcudart -lcublas ..."
go build -v ./...
```

### For GPU CI Workflow

To enable actual GPU testing:

1. Set up self-hosted runner with NVIDIA GPU
2. Install CUDA toolkit 12.3+
3. Update `gpu-ci.yml`:
   ```yaml
   runs-on: [self-hosted, linux, gpu]
   ```

### For Coverage Reporting

**Optional: Set up Codecov**
1. Sign up at https://codecov.io
2. Add repository
3. Add `CODECOV_TOKEN` to repository secrets (if private)

---

## Environment Variables

### CPU-only builds:
```bash
CGO_LDFLAGS="-L/usr/local/lib -lfaiss -lgomp -lstdc++ -lm -lopenblas"
CGO_CFLAGS="-I/usr/local/include"
LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
```

### GPU builds:
```bash
CGO_LDFLAGS="-L/usr/local/lib -lfaiss -lfaiss_gpu -lcudart -lcublas -lgomp -lstdc++ -lm -lopenblas"
CGO_CFLAGS="-I/usr/local/include -DFAISS_GPU"
LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

---

## Troubleshooting

### CI Failures

**Build errors:**
- Check FAISS installation logs in workflow
- Verify CGO environment variables are set correctly
- Clear cache if corrupted: Delete and re-run workflow

**Test failures:**
- Review test logs for specific error messages
- Verify FAISS version compatibility
- Run locally with same Go version: `go test -v ./...`

**Lint failures:**
- Run `golangci-lint run` locally
- Check `.golangci.yml` configuration
- Review linter documentation: https://golangci-lint.run/

### Common Issues

**"cannot find -lfaiss"**
- FAISS library not installed or not in library path
- Check: `ldconfig -p | grep faiss` (Linux)
- Check: `ls /usr/local/lib/libfaiss*`

**Tests timeout**
- Increase timeout in workflow if needed
- Some tests (IVF training) can be slow on CI runners

**Benchmark variance**
- GitHub runners have variable performance
- Compare trends over multiple runs, not single values
- Use larger `-benchtime` for stability

**Cache issues**
- Cache key must match exactly
- GitHub cache limit: 10GB per repository
- Caches expire after 7 days of inactivity

### Release Issues

**Version conflicts:**
- Ensure tag doesn't already exist: `git tag -l`
- Verify version format: `vMAJOR.MINOR.PATCH`
- Check current version in `faiss.go`

**Failed release:**
- Review workflow logs for specific error
- Ensure all tests pass locally first
- Verify branch is up-to-date with remote
- Check repository permissions

---

## Best Practices

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) for automated changelog:

```bash
# Features (minor version bump)
feat: Add IVF index clustering support
feat(gpu): Enable multi-GPU training

# Bug Fixes (patch version bump)
fix: Resolve memory leak in Index.Search
fix(serialization): Handle empty index case

# Other types (no version bump, but included in changelog)
docs: Update installation instructions
perf: Optimize distance calculations (10% faster)
refactor: Simplify index factory logic
test: Add benchmarks for 1M vectors
chore: Update dependencies to latest versions
```

**Breaking Changes (major version bump):**
```bash
feat!: Remove deprecated Index.AddVector method

BREAKING CHANGE: Use Index.Add() instead of Index.AddVector().
This provides better batch performance and clearer semantics.
```

### Pull Request Workflow

1. **Create feature branch** from `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/my-feature
   ```

2. **Make changes** with conventional commits
   ```bash
   git commit -m "feat: Add new feature"
   ```

3. **Push and create PR**
   ```bash
   git push origin feature/my-feature
   ```

4. **Wait for CI** - All workflows must pass:
   - ✅ Build and Test (Ubuntu & macOS)
   - ✅ Linting
   - ✅ GPU CI (if code changes)

5. **Address reviews** and update PR

6. **Merge** when approved and all checks pass

### Release Workflow

1. **Ensure `main` is stable**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Review recent changes**
   ```bash
   # See commits since last tag
   git log $(git describe --tags --abbrev=0)..HEAD --oneline
   ```

3. **Run tests locally**
   ```bash
   go test -v ./...
   go test -race ./...
   golangci-lint run
   ```

4. **Trigger release workflow**
   - Actions → Release → Run workflow
   - Select version type: patch/minor/major
   - Add pre-release suffix if needed
   - Click "Run workflow"

5. **Monitor execution**
   - Watch workflow progress
   - Verify all steps complete successfully

6. **Verify release**
   - Check GitHub release page
   - Test installation: `go get github.com/NerdMeNot/faiss-go@vX.Y.Z`
   - Verify version: `faiss.Version` in code

### Testing Feature Branches

**Option 1: Manual Dispatch (Recommended)**
```
Actions → CI → Run workflow → Select branch → Run
```

**Option 2: Create Draft PR**
```
Create PR with "draft" status → CI runs automatically
```

**Option 3: Push to branch matching trigger pattern**
- Not recommended anymore (increases CI costs)
- Use manual dispatch instead

---

## Performance Expectations

Typical CI run times (with cache):

| Workflow | First Run | Cached Run | Frequency |
|----------|-----------|------------|-----------|
| CI (Ubuntu) | ~15 min | ~3 min | PRs, main pushes, tags |
| CI (macOS) | ~8 min | ~2 min | PRs, main pushes, tags |
| GPU CI | ~20 min | ~5 min | PRs (code changes only) |
| Benchmarks | ~8 min | ~4 min | Main pushes, weekly |
| Release | ~8 min | ~5 min | Manual only |

**Cache hit rate:** ~90% for subsequent runs with same FAISS version

---

## Workflow Status Badges

Add to your README.md:

```markdown
![CI](https://github.com/NerdMeNot/faiss-go/workflows/CI/badge.svg)
![GPU CI](https://github.com/NerdMeNot/faiss-go/workflows/GPU%20CI/badge.svg)
![Benchmarks](https://github.com/NerdMeNot/faiss-go/workflows/Continuous%20Benchmarking/badge.svg)
[![codecov](https://codecov.io/gh/NerdMeNot/faiss-go/branch/main/graph/badge.svg)](https://codecov.io/gh/NerdMeNot/faiss-go)
```

---

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Go Modules Reference](https://go.dev/ref/mod)
- [Semantic Versioning](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [golangci-lint](https://golangci-lint.run/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Project Versioning Guide](../../VERSIONING.md)

---

## Questions or Issues?

- Review workflow logs in the Actions tab
- Check [VERSIONING.md](../../VERSIONING.md) for release process
- Review this documentation
- Open an issue with `[CI]` or `[Release]` prefix
