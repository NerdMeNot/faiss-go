# Static Libraries PR Creation Fix

## Problem

The Build Static Libraries workflow was creating artifacts with `.a` files (e.g., `libfaiss.a`, `libfaiss_c.a`) but these files were **not being committed to the PR**.

### Root Cause

**Incorrect directory nesting** in the artifact organization step:

```bash
# OLD (BROKEN) - Line 190
cp -r "libs-artifacts/faiss-static-$platform" "libs/${platform//-/_}"
```

This created a **nested structure**:
```
libs/
└── darwin_amd64/
    └── faiss-static-darwin-amd64/    ← Extra nested directory!
        ├── libfaiss.a                 (10M)
        ├── libfaiss_c.a               (325K)
        ├── build_info.json
        ├── checksums.txt
        └── include/
```

The `peter-evans/create-pull-request` action was looking for files directly in `libs/darwin_amd64/` but they were one level deeper, so **nothing was committed**.

### Expected Structure

```
libs/
└── darwin_amd64/
    ├── libfaiss.a          ✅ (10M)
    ├── libfaiss_c.a        ✅ (325K)
    ├── build_info.json
    ├── checksums.txt
    └── include/
```

## Solution

### 1. Fixed File Organization

**NEW (FIXED)** - Updated the copy command to copy **contents** instead of the directory:

```bash
# Copy contents (not the directory itself) to avoid nesting
cp -rv "$artifact_dir/"* "libs/$platform_dir/"
```

Key changes:
- Added `/*` to copy contents only
- Used `-v` (verbose) for debugging
- Added error handling for missing files
- Added verification output to confirm library files exist

### 2. Explicit Path Addition

Added `add-paths` to the PR creation action to explicitly tell it what to commit:

```yaml
- name: Create Pull Request
  uses: peter-evans/create-pull-request@v5
  with:
    add-paths: |
      libs/**/*
    commit-message: "feat: Add pre-built FAISS static libraries ..."
```

### 3. Enhanced Verification

Added diagnostic output to help debug future issues:

```bash
# Show what we have
echo "=== Libraries organized ==="
ls -lR libs/

# Verify critical files exist
echo "=== Verifying library files ==="
for platform_dir in libs/*/; do
  if [ -f "$platform_dir/libfaiss.a" ] || [ -f "$platform_dir/faiss.lib" ]; then
    echo "✓ Found library in $platform_dir"
  else
    echo "✗ No library found in $platform_dir"
  fi
done
```

## Impact on Static Library Builds

### Before Fix

❌ **Static library builds would fail** because:
- `faiss_lib.go` expects libraries at `libs/darwin_amd64/libfaiss.a`
- Files were actually at `libs/darwin_amd64/faiss-static-darwin-amd64/libfaiss.a`
- CGO linker would fail with "library not found"

### After Fix

✅ **Static library builds will work**:
```bash
go build -tags=faiss_use_lib -v ./...
```

Expected linker paths (from `faiss_lib.go`):
```go
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/libs/linux_amd64 -lfaiss ...
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/libs/darwin_amd64 -lfaiss ...
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/libs/windows_amd64 -lfaiss ...
```

## How to Verify the Fix

### 1. Re-run the Build Static Libraries Workflow

```bash
gh workflow run build-static-libs.yml \
  -f faiss_version=v1.13.2 \
  -f platforms=all
```

### 2. Check the Generated PR

The PR should now include:
```
libs/
├── darwin_amd64/
│   ├── libfaiss.a
│   ├── libfaiss_c.a
│   ├── build_info.json
│   ├── checksums.txt
│   └── include/
├── darwin_arm64/
│   ├── libfaiss.a
│   └── ...
├── linux_amd64/
│   ├── libfaiss.a
│   └── ...
├── linux_arm64/
│   ├── libfaiss.a
│   └── ...
└── windows_amd64/
    ├── faiss.lib
    └── ...
```

### 3. Test Static Library Build

After merging the PR:

```bash
# Should build in ~30 seconds (vs 2-5 minutes for amalgamation)
go build -tags=faiss_use_lib,nogpu -v ./...

# Run tests
go test -tags=faiss_use_lib,nogpu -v ./...
```

## File Sizes

Expected library sizes per platform:

| Platform | `libfaiss.a` | `libfaiss_c.a` | Total |
|----------|--------------|----------------|-------|
| darwin_amd64 | ~10 MB | ~325 KB | ~10.3 MB |
| darwin_arm64 | ~10 MB | ~325 KB | ~10.3 MB |
| linux_amd64 | ~15 MB | ~325 KB | ~15.3 MB |
| linux_arm64 | ~15 MB | ~325 KB | ~15.3 MB |
| windows_amd64 | ~15 MB (faiss.lib) | ~325 KB | ~15.3 MB |

**Total repository size increase:** ~60-70 MB for all platforms

## Next Steps

1. ✅ **Fix committed** - Workflow updated to organize files correctly
2. ⏳ **Re-run workflow** - Generate new PR with correct file structure
3. ⏳ **Merge PR** - Add static libraries to repository
4. ✅ **Update CI** - Benchmark workflow ready to use static libs
5. ⏳ **Switch build mode** - Change CI to use `-tags=faiss_use_lib`

## Benefits of Static Libraries

Once the PR is merged:

| Aspect | Amalgamation | Static Libs |
|--------|--------------|-------------|
| **Build time** | 2-5 minutes | ~30 seconds |
| **Reliability** | Source compilation | Pre-built binaries |
| **Consistency** | Version varies | Fixed FAISS v1.13.2 |
| **CI speed** | Slower | 4-8x faster |
| **Dependencies** | libopenblas-dev | None (static linked) |

## Troubleshooting

### If Static Build Still Fails

1. **Verify file location:**
   ```bash
   find libs/ -name "libfaiss.a" -o -name "faiss.lib"
   ```

2. **Check file permissions:**
   ```bash
   ls -l libs/*/libfaiss.a
   ```

3. **Verify build tag:**
   ```bash
   go build -tags=faiss_use_lib -v -x ./... 2>&1 | grep -i ldflags
   ```

4. **Check CGO environment:**
   ```bash
   go env CGO_ENABLED  # Should be "1"
   ```

### If Libraries Are Missing

The workflow might have failed for a specific platform. Check:
```bash
gh run list --workflow=build-static-libs.yml
gh run view <run-id> --log
```

## Related Files

- `.github/workflows/build-static-libs.yml` - Build workflow
- `faiss_lib.go` - Static library integration (build tag: `faiss_use_lib`)
- `faiss_source.go` - Amalgamation build (build tag: `!faiss_use_lib`)
- `libs/README.md` - Library documentation

## Summary

The fix ensures that:
- ✅ Library files are organized in the correct directory structure
- ✅ Files are explicitly added to git commits
- ✅ Verification output helps debug future issues
- ✅ Static library builds will work once PR is merged
- ✅ CI can switch to faster build mode

This was a **critical bug** that prevented the static library build mode from working. The fix is now in place and ready for the next workflow run.
