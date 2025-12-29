# GitHub Actions Workflow Usage Guide

All workflows are now **manual-only** (workflow_dispatch). They will not run automatically on push, PR, or schedule.

## Quick Reference

| Workflow | Purpose | When to Run |
|----------|---------|-------------|
| **benchmark.yml** | Performance benchmarks | After performance changes |
| **ci.yml** | Build & test suite | Before merging PRs |
| **gpu-ci.yml** | GPU-specific tests | When changing GPU code |
| **build-static-libs.yml** | Build static libraries | When updating FAISS version |
| **release.yml** | Create releases | When ready to release |

## Benchmark Workflows

### Quick Benchmarks (1-2 minutes)
Runs fast benchmarks on recent Go versions (1.23, 1.24, 1.25).

```bash
# Via CLI
gh workflow run benchmark.yml \
  -f mode=quick \
  -f benchtime=1s

# Via GitHub UI
Actions → Continuous Benchmarking → Run workflow
  mode: quick
  benchtime: 1s
```

### Comprehensive Benchmarks (25-30 minutes)
Runs all benchmarks on all Go versions (1.21-1.25) with detailed comparisons.

```bash
# Via CLI
gh workflow run benchmark.yml \
  -f mode=comprehensive \
  -f benchtime=5s

# Via GitHub UI
Actions → Continuous Benchmarking → Run workflow
  mode: comprehensive
  benchtime: 5s
```

**Outputs:**
- Benchmark results for each Go version + architecture
- AMD64 vs ARM64 comparison
- Go 1.21 vs 1.25 comparison
- Artifacts retained for 30-90 days

## CI Workflows

### Full CI Suite
Runs build and tests on Ubuntu + macOS for all Go versions (1.21-1.25).

```bash
# Test all Go versions with static libraries
gh workflow run ci.yml
```

**What it tests:**

**Static Libraries Build** (10 parallel jobs):
- Go versions: 1.21, 1.22, 1.23, 1.24, 1.25
- Platforms: Ubuntu + macOS
- Build time: ~30 seconds
- Uses pre-built static libraries from `libs/`
- Runs full test suite with coverage
- Sample benchmarks

**Lint**:
- Go 1.25
- golangci-lint with 5 minute timeout

**Total**: 11 parallel jobs

The default build mode uses pre-built static libraries for fast ~30 second builds. No build tags needed!

### GPU CI
Tests GPU-specific code (requires GPU runner or manual setup).

```bash
gh workflow run gpu-ci.yml
```

**Note:** Currently configured for CPU-only mode. Update `runs-on` for GPU runners.

## Build Workflows

### Build Static Libraries
Creates pre-compiled static libraries for all platforms.

```bash
# Build all platforms
gh workflow run build-static-libs.yml \
  -f faiss_version=v1.13.2 \
  -f platforms=all

# Build specific platform
gh workflow run build-static-libs.yml \
  -f faiss_version=v1.13.2 \
  -f platforms=linux-amd64
```

**Platforms supported:**
- `linux-amd64`, `linux-arm64`
- `darwin-amd64`, `darwin-arm64`
- `windows-amd64`
- `all` (builds everything)

**Output:**
- Individual platform artifacts (per platform)
- Combined `all-static-libraries` artifact
- INSTALL.md with copy instructions

**Next step:** Download artifact and follow `MANUAL_STATIC_LIBS_INSTALL.md`

## Release Workflow

Creates a new release with version bumping and changelog.

```bash
# Patch release (0.0.X)
gh workflow run release.yml -f version_type=patch

# Minor release (0.X.0)
gh workflow run release.yml -f version_type=minor

# Major release (X.0.0)
gh workflow run release.yml -f version_type=major

# Pre-release (e.g., 0.1.0-alpha.1)
gh workflow run release.yml \
  -f version_type=minor \
  -f prerelease_suffix=alpha.1
```

## Common Scenarios

### Before Merging a PR

```bash
# Run full CI (tests static libs + lint)
gh workflow run ci.yml

# Run quick benchmarks to check for regressions
gh workflow run benchmark.yml -f mode=quick -f benchtime=1s

# Wait for results
gh run watch
```

**Total time**: ~5-10 minutes (static libs are fast!)

### After Performance Optimizations

```bash
# Run comprehensive benchmarks
gh workflow run benchmark.yml -f mode=comprehensive -f benchtime=5s

# Compare results in artifacts
```

### Updating FAISS Version

```bash
# 1. Build static libraries
gh workflow run build-static-libs.yml \
  -f faiss_version=v1.14.0 \
  -f platforms=all

# 2. Download artifact and install
# Follow instructions in MANUAL_STATIC_LIBS_INSTALL.md from the artifact

# 3. Commit the new library files
git add libs/
git commit -m "feat: Update FAISS to v1.14.0"
git push

# 4. Test with new version
gh workflow run ci.yml

# 5. Run benchmarks
gh workflow run benchmark.yml -f mode=comprehensive
```

### Preparing a Release

```bash
# 1. Run full test suite
gh workflow run ci.yml

# 2. Run comprehensive benchmarks
gh workflow run benchmark.yml -f mode=comprehensive

# 3. Review results, then create release
gh workflow run release.yml -f version_type=patch
```

## Monitoring Workflow Runs

### Via CLI

```bash
# List recent runs
gh run list

# Watch a specific run
gh run watch <run-id>

# View run logs
gh run view <run-id> --log

# Download artifacts
gh run download <run-id>
```

### Via GitHub UI

1. Go to **Actions** tab
2. Select workflow from left sidebar
3. Click on run to see details
4. Download artifacts from run summary

## Troubleshooting

### Workflow Doesn't Appear in UI

Make sure you've pushed the workflow file to the branch:
```bash
git push origin <branch-name>
```

### "No workflow_dispatch inputs" Error

This is expected - some workflows don't have required inputs. Just click "Run workflow" directly.

### Benchmark Hangs

All benchmark timer bugs have been fixed (commit 1040189). If issues persist:
- Use shorter benchtime: `-f benchtime=1s`
- Run quick mode instead of comprehensive
- Check timeout settings (currently 25-30 min)

### Static Library Build Fails for a Platform

Check the individual platform artifact in the run:
```bash
gh run view <run-id>
# Look for "faiss-static-<platform>" artifact
```

Platform builds are independent - one can fail without affecting others.

## Re-enabling Automatic Triggers (Future)

To restore automatic runs, edit workflow files:

### For CI on Pull Requests

```yaml
# .github/workflows/ci.yml
on:
  pull_request:
    branches: [ main ]
  workflow_dispatch:
    # ... keep existing inputs
```

### For Benchmarks on Push

```yaml
# .github/workflows/benchmark.yml
on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:
    # ... keep existing inputs
```

### For GPU CI on PR

```yaml
# .github/workflows/gpu-ci.yml
on:
  pull_request:
    branches: [ main ]
    paths:
      - '**.go'
      - 'go.mod'
      - 'go.sum'
  workflow_dispatch:
```

## Summary

✅ **All workflows are manual-only**
✅ **Use `gh workflow run` or GitHub UI**
✅ **No automatic CI runs**
✅ **Full control over execution**
✅ **Easy to re-enable automatic triggers later**

This gives you complete control while workflows are being stabilized! 🎯
