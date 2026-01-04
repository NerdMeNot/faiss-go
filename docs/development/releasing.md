# Release Process

Guide for creating releases and managing versions of faiss-go.

---

## Versioning Scheme

faiss-go uses a compound versioning scheme that tracks both the upstream FAISS version and the Go bindings version:

```
v{FAISS_VERSION}-{BINDING_MAJOR}.{BINDING_MINOR}
```

### Components

| Component | Description | When to Increment |
|-----------|-------------|-------------------|
| **FAISS_VERSION** | Upstream FAISS library version | When upgrading to a new FAISS release |
| **BINDING_MAJOR** | faiss-go feature version | When adding new Go-specific features or interfaces |
| **BINDING_MINOR** | Patch version | For bug fixes and minor improvements |

### Examples

| Version | Meaning |
|---------|---------|
| `v1.13.2-0.1` | First release for FAISS 1.13.2 |
| `v1.13.2-0.2` | Bug fix release |
| `v1.13.2-0.3` | Another bug fix |
| `v1.13.2-1.0` | New faiss-go feature added (not in upstream FAISS) |
| `v1.13.2-1.1` | Bug fix after feature release |
| `v1.14.0-0.1` | First release for FAISS 1.14.0 (reset binding version) |

### Version Constants

Version information is stored in `faiss.go`:

```go
const (
    FAISSVersion = "1.13.2"  // Upstream FAISS version
    BindingMajor = 0         // faiss-go feature version
    BindingMinor = 1         // Patch version
    Version = "1.13.2-0.1"   // Full version string
)

func FullVersion() string    // Returns "v1.13.2-0.1"
```

---

## Creating a Release

### Prerequisites

1. All tests pass on main branch
2. All workflows have been verified
3. CHANGELOG.md is updated (optional but recommended)

### Using the Release Workflow

1. Go to **Actions** > **Release** workflow
2. Click **Run workflow**
3. Select bump type:
   - `binding_minor` - Bug fixes (0.1 → 0.2)
   - `binding_major` - New features (0.1 → 1.0)
4. Optionally add a pre-release suffix (e.g., `alpha`, `beta`, `rc1`)
5. Click **Run workflow**

The workflow will:
1. Update version constants in `faiss.go`
2. Run full test suite
3. Run linting
4. Commit the version bump
5. Create and push a git tag
6. Create a GitHub Release with auto-generated changelog

### Manual Release (if needed)

```bash
# 1. Update version constants in faiss.go
# Edit FAISSVersion, BindingMajor, BindingMinor, Version

# 2. Commit the change
git add faiss.go
git commit -m "chore: Bump version to v1.13.2-0.2"

# 3. Create and push tag
git tag -a v1.13.2-0.2 -m "Release v1.13.2-0.2"
git push origin main
git push origin v1.13.2-0.2

# 4. Create GitHub Release (or use gh cli)
gh release create v1.13.2-0.2 --generate-notes
```

---

## Upgrading to a New FAISS Version

When a new FAISS version is released, the nightly checker workflow creates an issue with an upgrade checklist.

### Automated Detection

The `check-faiss-releases` workflow runs nightly and:
1. Compares current `FAISSVersion` in code vs latest GitHub release
2. Creates an issue if a new version is available
3. Provides a complete upgrade checklist

### Upgrade Process

1. **Review Release Notes**
   - Check [FAISS releases](https://github.com/facebookresearch/faiss/releases) for breaking changes
   - Review C API changes that might affect bindings

2. **Build New Static Libraries**

   Libraries are built in the [faiss-go-bindings](https://github.com/NerdMeNot/faiss-go-bindings) repository.
   See [Building Libraries](building-libs.md) for details.

3. **Update Version Constants**
   ```go
   // In faiss.go
   FAISSVersion = "1.14.0"  // New FAISS version
   BindingMajor = 0         // Reset to 0
   BindingMinor = 1         // Reset to 1
   Version = "1.14.0-0.1"   // Update string
   ```

4. **Update C API Bindings** (if needed)
   - Check for new functions in `faiss_lib.go`
   - Update `faiss_system.go` for system FAISS builds
   - Add/remove/modify function signatures as needed

5. **Test All Platforms**
   ```bash
   # Run CI workflow
   gh workflow run ci.yml

   # Run system FAISS workflow
   gh workflow run ci-system-faiss.yml -f faiss_version=v1.14.0
   ```

6. **Create Release**
   ```bash
   gh workflow run release.yml -f bump_type=binding_minor
   ```

---

## Workflows Reference

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `release.yml` | Manual | Create a new release |
| `check-faiss-releases.yml` | Nightly + Manual | Check for new FAISS versions |
| `ci.yml` | PR to main | Run tests with prebuilt libs |
| `ci-system-faiss.yml` | Manual | Run tests with system FAISS |
| `benchmark.yml` | Manual | Run performance benchmarks |
| `gpu-ci.yml` | Manual | Run GPU tests (requires CUDA) |

---

## Pre-release Versions

For testing before a stable release:

```bash
# Create alpha release
gh workflow run release.yml \
  -f bump_type=binding_minor \
  -f prerelease=alpha

# Creates: v1.13.2-0.2-alpha
```

Pre-release suffixes:
- `alpha` - Early testing, may have bugs
- `beta` - Feature complete, needs testing
- `rc1`, `rc2` - Release candidates

---

## Checklist for Releases

### Bug Fix Release (binding_minor)

- [ ] Fix is tested and merged to main
- [ ] CI passes
- [ ] Run release workflow with `binding_minor`

### Feature Release (binding_major)

- [ ] Feature is implemented and tested
- [ ] Documentation is updated
- [ ] Examples are added (if applicable)
- [ ] CI passes
- [ ] Run release workflow with `binding_major`

### FAISS Upgrade

- [ ] Review FAISS changelog for breaking changes
- [ ] Build new static libraries
- [ ] Update version constants (reset binding version)
- [ ] Update C API bindings if needed
- [ ] Test on all platforms (Linux AMD64/ARM64, macOS Intel/ARM64)
- [ ] Run full test suite
- [ ] Create release

---

## Troubleshooting

### Release workflow fails

1. Check that you're on the main branch
2. Verify the tag doesn't already exist
3. Ensure tests pass locally:
   ```bash
   go test -v ./...
   golangci-lint run
   ```

### Version mismatch

If `Version` string doesn't match `BindingMajor.BindingMinor`:
```go
// These must be consistent:
BindingMajor = 1
BindingMinor = 2
Version = "1.13.2-1.2"  // Must match above
```

### Tag already exists

```bash
# Check existing tags
git tag -l "v1.13.2-*"

# If you need to delete a tag (be careful!)
git tag -d v1.13.2-0.2
git push origin :refs/tags/v1.13.2-0.2
```
