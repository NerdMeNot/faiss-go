# CI Strategy: Static Libraries, ARM Support, and Go 1.21-1.25 Compatibility

**Date:** 2025-12-28
**Branch:** `claude/ci-static-arm-strategy-JQIGR`

## Executive Summary

This document outlines the strategy to enhance the faiss-go CI workflows to:
1. Use pre-built static libraries instead of building FAISS from source
2. Support both AMD64 and ARM64 platforms
3. Expand Go version compatibility from 1.21 to 1.25
4. Test both amalgamation and static library build modes

---

## Current State Analysis

### 1. **Static Libraries Status**

**What Exists:**
```
libs/
├── windows_amd64/    ✅ COMPLETE (faiss.lib 26MB, faiss_c.lib 1.2MB)
├── linux_amd64/      ❌ MISSING (only headers + metadata, no libfaiss.a)
├── linux_arm64/      ❌ MISSING (only headers + metadata, no libfaiss.a)
├── darwin_amd64/     ❌ MISSING (only headers + metadata, no libfaiss.a)
└── darwin_arm64/     ❌ MISSING (only headers + metadata, no libfaiss.a)
```

**Issue:** The static library files (`.a` for Linux/macOS) are missing from the repository. Only the Windows `.lib` files exist. This is why the previous attempt to use static libraries in CI (commit 4d2bd89) was reverted in commit ca14677.

### 2. **Amalgamation Build Status**

**What Exists:**
- `faiss/faiss.cpp` (1.3 KB) - Stub implementation
- `faiss/faiss.h` (1.1 KB) - Header stubs
- `faiss/BUILD_INFO.txt` - Metadata

**Issue:** The amalgamated source is just a placeholder/stub that returns `-1` (not implemented) for all operations. It's not functional for production use.

### 3. **Current CI Behavior**

The current CI workflow (`.github/workflows/ci.yml`):
- ✅ Builds FAISS from source (downloads v1.8.0, runs cmake + make, 30+ min build time)
- ✅ Tests on Ubuntu AMD64 and macOS ARM64
- ✅ Tests Go versions: 1.22, 1.23
- ❌ Does NOT test ARM64 Linux
- ❌ Does NOT use pre-built static libraries
- ❌ Does NOT use amalgamation build
- ❌ Does NOT test Go 1.21 or 1.24+

### 4. **Build Modes**

The project supports two build modes via Go build tags:

**Mode 1: System FAISS (Current CI Default)**
```bash
go build -tags=nogpu
```
- Uses system-installed FAISS
- Requires FAISS to be installed via apt/brew/build
- Defined in: `faiss_source.go` (build tag: `!faiss_use_lib`)

**Mode 2: Pre-built Static Libraries**
```bash
go build -tags=faiss_use_lib,nogpu
```
- Uses pre-built static libraries from `libs/{platform}/`
- Defined in: `faiss_lib.go` (build tag: `faiss_use_lib`)
- Platform selection via CGO LDFLAGS based on GOOS/GOARCH

---

## Problems to Solve

### P1: Missing Static Library Files
The static libraries for Linux/macOS platforms are not in the repository. Without these, we cannot use the `faiss_use_lib` build mode.

### P2: Non-functional Amalgamation
The amalgamated source is just a stub. It cannot be used for testing.

### P3: No ARM64 Testing
Current CI only tests AMD64 (Ubuntu) and ARM64 (macOS). No ARM64 Linux testing.

### P4: Limited Go Version Coverage
User requirements: Go 1.21 - 1.25
Current testing: Go 1.22, 1.23

### P5: Slow CI (30+ minutes)
Building FAISS from source on every CI run is slow and resource-intensive.

---

## Proposed Strategy

### Phase 1: Build and Commit Static Libraries ⚠️ **PREREQUISITE**

**Goal:** Generate all missing static library files and commit them to the repository.

**Actions:**
1. Run the `build-static-libs.yml` workflow manually for all platforms:
   - Linux AMD64
   - Linux ARM64
   - macOS AMD64
   - macOS ARM64

2. Download artifacts and commit to repository under `libs/`

3. Verify file sizes (expected: ~15-40 MB per platform):
   ```bash
   libs/linux_amd64/libfaiss.a
   libs/linux_arm64/libfaiss.a
   libs/darwin_amd64/libfaiss.a
   libs/darwin_arm64/libfaiss.a
   libs/windows_amd64/faiss.lib  # Already exists
   ```

**Why This First?**
Without the actual library files, we cannot test the `faiss_use_lib` build mode in CI. This is blocking.

**Workflow Location:** `.github/workflows/build-static-libs.yml`
**Execution:** Manual workflow dispatch

### Phase 2: Update CI Matrix

**Goal:** Restructure CI to test both build modes, multiple Go versions, and multiple architectures.

**Proposed Test Matrix:**

| OS | Architecture | Go Version | Build Mode | Notes |
|----|-------------|------------|------------|-------|
| Ubuntu | AMD64 | 1.21, 1.22, 1.23, 1.24, 1.25 | static-lib | Uses libs/linux_amd64/ |
| Ubuntu | AMD64 | 1.21, 1.22, 1.23, 1.24, 1.25 | source-build | System FAISS (current approach) |
| Ubuntu | ARM64 | 1.23, 1.25 | static-lib | QEMU emulation, uses libs/linux_arm64/ |
| Ubuntu | ARM64 | 1.23, 1.25 | source-build | QEMU emulation |
| macOS | ARM64 | 1.21, 1.22, 1.23, 1.24, 1.25 | static-lib | M1/M2 runners, uses libs/darwin_arm64/ |

**Total Jobs:** ~20-25 test jobs (can be optimized with fail-fast: false)

**Rationale:**
- **Go 1.21-1.25:** Covers user requirements
- **Both build modes:** Validates both user experience paths
- **ARM64 coverage:** Future-proofs for ARM adoption
- **Ubuntu source-build:** Keep as regression test to ensure compatibility

### Phase 3: Implement Build Mode Testing

**Changes to `.github/workflows/ci.yml`:**

```yaml
jobs:
  test-static-libs:
    name: Test Static Libs (${{ matrix.os }}, Go ${{ matrix.go-version }}, ${{ matrix.arch }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        include:
          # Linux AMD64
          - os: ubuntu-latest
            arch: amd64
            go-version: ['1.21', '1.22', '1.23', '1.24', '1.25']
            platform: linux/amd64

          # Linux ARM64 (QEMU)
          - os: ubuntu-latest
            arch: arm64
            go-version: ['1.23', '1.25']  # Reduced for speed
            platform: linux/arm64

          # macOS ARM64
          - os: macos-14  # M1 runners
            arch: arm64
            go-version: ['1.21', '1.22', '1.23', '1.24', '1.25']
            platform: darwin/arm64

    steps:
      - uses: actions/checkout@v4

      - name: Setup Go
        uses: actions/setup-go@v5
        with:
          go-version: ${{ matrix.go-version }}

      - name: Setup QEMU (ARM64 Linux only)
        if: matrix.arch == 'arm64' && matrix.os == 'ubuntu-latest'
        uses: docker/setup-qemu-action@v3

      - name: Install dependencies (Linux)
        if: runner.os == 'Linux' && matrix.arch == 'amd64'
        run: |
          sudo apt-get update
          sudo apt-get install -y libopenblas-dev libgomp1

      - name: Install dependencies (macOS)
        if: runner.os == 'macOS'
        run: brew install openblas libomp

      - name: Verify static libraries exist
        run: |
          echo "Checking for static library..."
          if [ "${{ runner.os }}" = "Linux" ]; then
            ls -lh libs/linux_${{ matrix.arch }}/libfaiss.a || exit 1
          elif [ "${{ runner.os }}" = "macOS" ]; then
            ls -lh libs/darwin_${{ matrix.arch }}/libfaiss.a || exit 1
          fi

      - name: Build with static libraries (AMD64)
        if: matrix.arch == 'amd64'
        run: go build -v -tags=faiss_use_lib,nogpu ./...

      - name: Test with static libraries (AMD64)
        if: matrix.arch == 'amd64'
        run: |
          go test -short -v -tags=faiss_use_lib,nogpu \
            -timeout 10m -coverprofile=coverage.out ./...

      - name: Build and test (ARM64 via QEMU)
        if: matrix.arch == 'arm64' && matrix.os == 'ubuntu-latest'
        run: |
          docker run --rm --platform linux/arm64 \
            -v $PWD:/workspace -w /workspace \
            golang:${{ matrix.go-version }}-alpine \
            sh -c "
              apk add --no-cache gcc musl-dev openblas-dev libc6-compat && \
              go build -v -tags=faiss_use_lib,nogpu ./... && \
              go test -short -v -tags=faiss_use_lib,nogpu -timeout 10m ./...
            "

      - name: Upload coverage
        if: matrix.arch == 'amd64'
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.out
          flags: static-lib,${{ matrix.go-version }}

  test-source-build:
    name: Test Source Build (Ubuntu AMD64, Go ${{ matrix.go-version }})
    runs-on: ubuntu-latest
    strategy:
      matrix:
        go-version: ['1.21', '1.23', '1.25']  # Reduced matrix

    steps:
      # ... Keep current FAISS build-from-source approach ...
      # This serves as regression testing
```

### Phase 4: Optimize Build Workflows

**build-static-libs.yml:**
- Already well-structured
- Uses artifacts for distribution
- Supports selective platform builds

**Recommendation:** No changes needed, but ensure libraries are committed to main branch after building.

**build-amalgamation.yml:**
- Currently produces stub files
- Can be left as-is (future work to create real amalgamation)
- OR disable until full amalgamation is implemented

---

## Implementation Plan

### Step 1: Generate Static Libraries (Blocking) 🔴

**Owner:** Manual workflow trigger
**Duration:** ~2-3 hours (parallel builds)
**Dependencies:** None

**Actions:**
1. Go to GitHub Actions → Build Static Libraries workflow
2. Run manually with inputs:
   - FAISS version: `v1.8.0`
   - Platforms: `all`
3. Wait for all platform builds to complete
4. Download all artifacts
5. Extract and organize into `libs/` directory structure
6. Verify file integrity with checksums
7. Commit to a new branch (e.g., `add-static-libs-all-platforms`)
8. Create PR and merge to main

**Verification:**
```bash
# After downloading artifacts
find libs -name "*.a" -o -name "*.lib" | xargs ls -lh

# Expected output:
# libs/linux_amd64/libfaiss.a (15-25 MB)
# libs/linux_arm64/libfaiss.a (15-25 MB)
# libs/darwin_amd64/libfaiss.a (15-25 MB)
# libs/darwin_arm64/libfaiss.a (15-25 MB)
# libs/windows_amd64/faiss.lib (26 MB) - already exists
```

### Step 2: Update CI Workflow 🟡

**Owner:** Developer
**Duration:** 2-3 hours
**Dependencies:** Step 1 complete, libraries merged to main

**Actions:**
1. Create new branch from main (with static libs)
2. Rewrite `.github/workflows/ci.yml` based on Phase 3 design
3. Add Go versions: 1.21, 1.24, 1.25 to matrix
4. Add ARM64 Linux testing via QEMU
5. Add static library verification step
6. Test locally with act (if possible) or create PR for testing

**Testing Strategy:**
- Initial PR should test with limited matrix (e.g., Go 1.23 only)
- Once stable, expand to full matrix
- Monitor CI run times and costs

### Step 3: Documentation Updates 🟢

**Owner:** Developer
**Duration:** 1 hour
**Dependencies:** Steps 1-2 complete

**Actions:**
1. Update README.md with build modes and Go version support
2. Update libs/README.md with verification instructions
3. Add CI badge showing build status
4. Document how to run tests locally with different build modes

---

## Risk Assessment

### High Risks 🔴

**R1: Static Library Size**
- **Risk:** Committing ~100MB of binary files to git repo
- **Impact:** Repo size growth, clone times
- **Mitigation:**
  - Use Git LFS for large binary files
  - OR: Use GitHub Releases to host libraries, download in CI
  - OR: Accept size increase (100MB is manageable)

**R2: Cross-compilation Issues**
- **Risk:** ARM64 libraries may have ABI incompatibilities
- **Impact:** Runtime failures on ARM platforms
- **Mitigation:**
  - Test on actual ARM64 hardware (macOS M1/M2)
  - Use QEMU for Linux ARM64 testing
  - Provide fallback to source build if static lib fails

### Medium Risks 🟡

**R3: Go Version Compatibility**
- **Risk:** Go 1.21 or 1.25 may have CGO changes
- **Impact:** Build failures on certain versions
- **Mitigation:**
  - Test each version individually
  - Use Go version-specific build tags if needed
  - Document minimum CGO requirements

**R4: CI Cost Increase**
- **Risk:** Larger test matrix = more runner minutes
- **Impact:** GitHub Actions costs
- **Mitigation:**
  - Use matrix with fail-fast: false strategically
  - Limit ARM64 tests to subset of Go versions
  - Use caching aggressively

### Low Risks 🟢

**R5: Platform-specific Dependencies**
- **Risk:** Missing system libraries on runners
- **Impact:** Build failures
- **Mitigation:** Document and install in CI (openblas, libomp, etc.)

---

## Alternative Approaches

### Alternative 1: GitHub Releases for Libraries
Instead of committing static libs to repo, host them as GitHub Release assets.

**Pros:**
- No repo size bloat
- Easy to version libraries separately
- Can update libraries without code changes

**Cons:**
- Extra download step in CI (network dependency)
- More complex workflow
- Release management overhead

**Recommendation:** Use if repo size is a concern (>50 MB total).

### Alternative 2: Docker-based CI
Build a custom Docker image with pre-built FAISS, use in CI.

**Pros:**
- Fast CI (no FAISS build, no lib download)
- Reproducible environment
- Can test multiple platforms

**Cons:**
- Docker image maintenance
- Still need to build image initially
- Less flexible than native runners

**Recommendation:** Consider for future optimization, not initial implementation.

### Alternative 3: Reduce Test Matrix
Only test Go 1.21, 1.23, 1.25 (odd versions).

**Pros:**
- Fewer CI jobs
- Lower costs
- Still covers range

**Cons:**
- Less coverage
- May miss version-specific issues

**Recommendation:** Start with full matrix, reduce if CI time/cost is prohibitive.

---

## Success Criteria

✅ **SC1:** All static library files exist and are committed to repository
✅ **SC2:** CI successfully builds with `faiss_use_lib` tag on all platforms
✅ **SC3:** CI tests pass on Go 1.21, 1.22, 1.23, 1.24, 1.25
✅ **SC4:** ARM64 Linux tests pass via QEMU
✅ **SC5:** macOS ARM64 tests pass on native runners
✅ **SC6:** CI run time is < 15 minutes (vs current 30+ min)
✅ **SC7:** All tests pass with >80% code coverage

---

## Timeline Estimate

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| 1 | Build static libraries (all platforms) | 2-3 hours | None |
| 1 | Download, verify, commit libraries | 1 hour | Build complete |
| 1 | PR review and merge | 1-2 days | Commit ready |
| 2 | Write new CI workflow | 2-3 hours | Libraries in main |
| 2 | Test CI workflow (trial PR) | 2-4 hours | Workflow written |
| 2 | Fix issues, iterate | 2-6 hours | Testing done |
| 3 | Documentation updates | 1 hour | CI stable |

**Total Estimated Time:** 10-20 hours over 3-5 days (including review time)

---

## Recommendations

### Immediate Actions (This Week)

1. **Build all missing static libraries** using the existing workflow
2. **Commit libraries to repository** (accept repo size increase)
3. **Create baseline tests** with Go 1.23 + static libs on AMD64

### Short-term Actions (Next 2 Weeks)

4. **Expand test matrix** to include Go 1.21-1.25
5. **Add ARM64 testing** via QEMU for Linux
6. **Update documentation** with build modes and supported versions

### Long-term Actions (Future)

7. **Consider Git LFS** if repo size becomes problematic
8. **Implement full amalgamation** to reduce dependency on external libs
9. **Add benchmark CI** to track performance across versions
10. **Explore GPU CI** when GPU runners become available

---

## Open Questions

**Q1:** Should we use Git LFS for the static library files?
**A:** Recommend starting without LFS. If repo exceeds 200MB, revisit.

**Q2:** Should we continue testing the source-build mode?
**A:** Yes, as regression testing. Users may still build from source.

**Q3:** Which Go versions should be prioritized if we need to reduce matrix?
**A:** Keep 1.21 (min), 1.23 (stable), 1.25 (latest). Drop 1.22, 1.24 if needed.

**Q4:** Should we test on Windows in CI?
**A:** Not initially (no native Windows runners for free tier). Windows libs exist and can be tested manually.

**Q5:** Do we need to update go.mod for version constraints?
**A:** Check current go.mod. If it specifies `go 1.22`, update to `go 1.21` as minimum.

---

## Appendix: Key Files

**CI Workflows:**
- `.github/workflows/ci.yml` - Main CI (needs update)
- `.github/workflows/build-static-libs.yml` - Library builder (OK)
- `.github/workflows/build-amalgamation.yml` - Stub builder (future)

**Build Mode Files:**
- `faiss_lib.go` - Static library build (tag: faiss_use_lib)
- `faiss_source.go` - System FAISS build (tag: !faiss_use_lib)

**Library Locations:**
- `libs/{platform}/libfaiss.a` - Static libraries (MISSING for non-Windows)
- `libs/{platform}/include/` - FAISS C headers (EXISTS)

**Go Version:**
- Check `go.mod` for current version constraint

---

## Conclusion

The strategy is **feasible** with the following critical path:

1. **Generate missing static libraries** (blocking, ~3 hours)
2. **Commit to repository** (1 hour + review time)
3. **Update CI matrix** (3-5 hours development + testing)
4. **Expand Go version support** (part of CI matrix update)

**Key Decision Points:**
- ✅ Use pre-built static libraries (not build from source in CI)
- ✅ Test both AMD64 and ARM64
- ✅ Support Go 1.21-1.25
- ⚠️ Handle repo size increase (~100MB) - use Git LFS if needed
- ⚠️ Amalgamation build is NOT ready - skip for now

**Estimated Total Effort:** 10-20 hours
**Estimated Calendar Time:** 3-5 days (with reviews)
**Complexity:** Medium
**Risk Level:** Low-Medium

**Recommendation:** ✅ **PROCEED** with Phase 1 immediately.
