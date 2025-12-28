# Building Static FAISS Libraries

This guide explains how to generate the pre-built static FAISS libraries for all platforms.

## Why Build Static Libraries?

Static libraries enable:
- ✅ Fast CI (no FAISS compilation, saves 25+ minutes per run)
- ✅ Easy local development (no system FAISS installation required)
- ✅ Cross-platform consistency (same FAISS version everywhere)
- ✅ ARM64 support (pre-built for both AMD64 and ARM64)

## Prerequisites

- Access to GitHub Actions on this repository
- Permissions to trigger workflows and create PRs

## Step-by-Step Guide

### Step 1: Trigger the Workflow

1. Go to the GitHub Actions tab:
   ```
   https://github.com/NerdMeNot/faiss-go/actions/workflows/build-static-libs.yml
   ```

2. Click **"Run workflow"** button (top right)

3. Configure the workflow inputs:
   - **FAISS version**: `v1.8.0` (or desired version)
   - **Platforms**: `all` (to build all platforms)

4. Click **"Run workflow"** to start the build

### Step 2: Monitor the Build

The workflow will run 5 parallel jobs, one for each platform:

| Platform | Runner | Duration | Output |
|----------|--------|----------|--------|
| Linux AMD64 | ubuntu-latest | ~20-30 min | libfaiss.a (~20 MB) |
| Linux ARM64 | ubuntu-latest (QEMU) | ~40-60 min | libfaiss.a (~20 MB) |
| macOS AMD64 | macos-15-intel | ~25-35 min | libfaiss.a (~18 MB) |
| macOS ARM64 | macos-14 (M1) | ~20-30 min | libfaiss.a (~18 MB) |
| Windows AMD64 | windows-latest | ~30-40 min | faiss.lib (~26 MB) |

**Total time:** ~60-90 minutes (jobs run in parallel)

### Step 3: Review the Pull Request

After all builds complete successfully:

1. A PR will be automatically created titled:
   ```
   Add pre-built FAISS static libraries v1.8.0
   ```

2. The PR will include:
   - All platform static libraries in `libs/{platform}/`
   - Checksums for verification
   - Build metadata (build_info.json)

3. Review the PR to ensure:
   - ✅ All 5 platforms have library files
   - ✅ File sizes are reasonable (15-30 MB per platform)
   - ✅ Checksums are present
   - ✅ Build info shows correct FAISS version and build date

### Step 4: Merge the Libraries

Once the PR is reviewed and approved:

1. Merge the PR to main branch
2. The libraries will now be available in the repository
3. CI workflows can use these libraries with the `faiss_use_lib` build tag

## Alternative: Manual Build and Commit

If you prefer to build and commit manually (or the workflow fails):

### Linux AMD64

```bash
# On Linux AMD64 machine
git clone https://github.com/NerdMeNot/faiss-go.git
cd faiss-go
chmod +x scripts/build_static_lib.sh
./scripts/build_static_lib.sh linux-amd64 v1.8.0

# Verify
ls -lh libs/linux_amd64/libfaiss.a
```

### Linux ARM64

```bash
# Using QEMU on any Linux machine
docker run --rm --platform linux/arm64 \
  -v $PWD:/workspace \
  -w /workspace \
  arm64v8/ubuntu:24.04 \
  bash -c "
    apt-get update && \
    apt-get install -y build-essential cmake git libopenblas-dev libomp-dev && \
    chmod +x scripts/build_static_lib.sh && \
    ./scripts/build_static_lib.sh linux-arm64 v1.8.0
  "

# Verify
ls -lh libs/linux_arm64/libfaiss.a
```

### macOS ARM64 (Apple Silicon)

```bash
# On macOS ARM64 (M1/M2/M3) machine
git clone https://github.com/NerdMeNot/faiss-go.git
cd faiss-go
brew install cmake openblas libomp
chmod +x scripts/build_static_lib.sh
./scripts/build_static_lib.sh darwin-arm64 v1.8.0

# Verify
ls -lh libs/darwin_arm64/libfaiss.a
file libs/darwin_arm64/libfaiss.a
```

### macOS AMD64 (Intel)

```bash
# On macOS Intel machine (or use GitHub Actions)
brew install cmake openblas libomp
chmod +x scripts/build_static_lib.sh
./scripts/build_static_lib.sh darwin-amd64 v1.8.0

# Verify
ls -lh libs/darwin_amd64/libfaiss.a
```

### Windows AMD64

```bash
# On Windows with MSVC and vcpkg
vcpkg install openblas:x64-windows lapack:x64-windows
vcpkg integrate install

# Using Git Bash or WSL
chmod +x scripts/build_static_lib.sh
./scripts/build_static_lib.sh windows-amd64 v1.8.0

# Verify
ls -lh libs/windows_amd64/faiss.lib
```

### Commit All Libraries

```bash
# After building all platforms (or downloading artifacts)
git checkout -b add-static-libs-all-platforms
git add libs/
git commit -m "feat: Add pre-built FAISS static libraries v1.8.0 for all platforms"
git push origin add-static-libs-all-platforms

# Create PR via GitHub UI
```

## Verifying Built Libraries

### Check File Integrity

```bash
# Verify checksums
for platform in linux_amd64 linux_arm64 darwin_amd64 darwin_arm64 windows_amd64; do
  echo "Checking $platform..."
  if [ -f "libs/$platform/checksums.txt" ]; then
    (cd "libs/$platform" && sha256sum -c checksums.txt) || true
  fi
done
```

### Check Library Contents

```bash
# Linux/macOS: Check symbols in static library
nm -g libs/linux_amd64/libfaiss.a | grep -i "faiss_Index" | head -10

# Expected output (example):
# 0000000000000000 T faiss_IndexFlatL2_new
# 0000000000000000 T faiss_IndexFlatIP_new
# 0000000000000000 T faiss_Index_add
# 0000000000000000 T faiss_Index_search
# ...
```

### Test with Go Build

```bash
# Try building with the static libraries
go build -v -tags=faiss_use_lib,nogpu ./...

# If successful, you should see:
# Building with pre-built static FAISS libraries...
```

## File Structure

After building, you should have:

```
libs/
├── README.md
├── linux_amd64/
│   ├── libfaiss.a          # ~20 MB
│   ├── build_info.json
│   ├── checksums.txt
│   └── include/
│       └── faiss/          # C API headers
├── linux_arm64/
│   ├── libfaiss.a          # ~20 MB
│   ├── build_info.json
│   ├── checksums.txt
│   └── include/
├── darwin_amd64/
│   ├── libfaiss.a          # ~18 MB
│   ├── build_info.json
│   ├── checksums.txt
│   └── include/
├── darwin_arm64/
│   ├── libfaiss.a          # ~18 MB
│   ├── build_info.json
│   ├── checksums.txt
│   └── include/
└── windows_amd64/
    ├── faiss.lib           # ~26 MB
    ├── faiss_c.lib         # ~1.2 MB
    ├── build_info.json
    ├── checksums.txt
    └── include/
```

**Total repository size increase:** ~100-110 MB

## Troubleshooting

### Build Fails on ARM64

**Problem:** QEMU emulation timeout or OOM

**Solution:**
- Increase workflow timeout in `.github/workflows/build-static-libs.yml`
- Or build on actual ARM64 hardware (Raspberry Pi, AWS Graviton, etc.)

### Missing Dependencies

**Problem:** Build fails with "library not found"

**Solution:**
```bash
# Linux
sudo apt-get install -y build-essential cmake libopenblas-dev libomp-dev

# macOS
brew install cmake openblas libomp

# Windows
vcpkg install openblas:x64-windows lapack:x64-windows
```

### Checksum Mismatch

**Problem:** Checksum verification fails

**Solution:**
- Rebuild the library
- Or regenerate checksums:
  ```bash
  cd libs/{platform}
  sha256sum libfaiss.a build_info.json > checksums.txt
  ```

### Git LFS Needed?

**Problem:** Repository size too large (>200 MB)

**Solution:**
- Consider using Git LFS for `.a` and `.lib` files
- Or host libraries in GitHub Releases instead
- Current size (~100 MB) should be acceptable for most workflows

## Using the Static Libraries

Once committed to the repository, use them in your builds:

### Local Development

```bash
# Build with static libraries (fast)
go build -tags=faiss_use_lib,nogpu

# Build with system FAISS (requires installation)
go build -tags=nogpu
```

### CI/CD

The new CI workflow (`.github/workflows/ci.yml`) automatically:
- ✅ Verifies static libraries exist for all platforms
- ✅ Tests with static libraries on Go 1.21-1.25
- ✅ Tests on AMD64 and ARM64
- ✅ Falls back to source build for regression testing

## Updating FAISS Version

To update to a newer FAISS version (e.g., v1.9.0):

1. Trigger the workflow with new version:
   - FAISS version: `v1.9.0`
   - Platforms: `all`

2. Review and merge the auto-generated PR

3. Update `FAISSVersion` constant in Go code if needed

4. Test thoroughly before releasing

## Resources

- **Build script:** `scripts/build_static_lib.sh`
- **Workflow:** `.github/workflows/build-static-libs.yml`
- **CI workflow:** `.github/workflows/ci.yml`
- **FAISS releases:** https://github.com/facebookresearch/faiss/releases

## FAQ

**Q: Why are the libraries not in the repo already?**
A: They need to be built fresh for each FAISS version. The workflow builds them on-demand.

**Q: Can I use my own FAISS libraries?**
A: Yes, replace files in `libs/{platform}/` with your custom builds, but ensure they're compatible.

**Q: Do I need to build all platforms?**
A: For CI to pass, yes. For local development, only your platform is needed.

**Q: How often should we rebuild?**
A: When FAISS releases a new version, or when build configurations change.

**Q: What about GPU support?**
A: GPU libraries are not included by default. Build with `FAISS_ENABLE_GPU=ON` if needed (requires CUDA).
