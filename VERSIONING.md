# Versioning and Release Strategy

## Overview

This project follows Go best practices for versioning and releases, using semantic versioning and git tags for version management.

## Semantic Versioning

We follow [Semantic Versioning 2.0.0](https://semver.org/):

- **v0.x.y** - Pre-1.0 development phase (current)
  - Breaking changes allowed
  - API may change
  - Minor version bumps for new features
  - Patch version bumps for bug fixes

- **v1.x.y** - Stable release
  - MAJOR version for incompatible API changes
  - MINOR version for backwards-compatible functionality
  - PATCH version for backwards-compatible bug fixes

- **v2.x.y+** - Major version changes
  - Requires `/v2` suffix in `go.mod` module path
  - Used for breaking changes after v1

## Branching Strategy

### Main Branch

- **`main`** - Always stable and deployable
  - Protected branch requiring PR reviews
  - All tests must pass before merge
  - Represents the latest stable development state

### Feature Branches

- **`claude/*`** or **`feature/*`** - Short-lived development branches
  - Branched from `main`
  - Merged back to `main` via Pull Request
  - Deleted after merge
  - No direct pushes to `main`

### Version Tags

- **`v0.1.0`, `v0.2.0`, `v1.0.0`, etc.** - Release tags
  - Created automatically by release workflow
  - Immutable references to specific commits
  - Used by Go modules for dependency management
  - Follow `vMAJOR.MINOR.PATCH` format

### No Release Branches

Following Go best practices, we **do not** maintain long-lived release branches. Instead:
- Version tags mark specific release points
- Hotfixes are applied to `main` and tagged as new patch versions
- Go modules use git tags for versioning

## Release Process

### 1. Manual Release Workflow

Releases are created manually to ensure quality and control:

```bash
# Trigger via GitHub Actions UI:
# Actions → Release → Run workflow
# Select: major | minor | patch
```

**The workflow automatically:**
1. Validates the current state (clean main branch)
2. Determines the next version number
3. Updates version in `faiss.go`
4. Generates changelog from commits
5. Commits version bump
6. Creates and pushes git tag
7. Creates GitHub release with notes
8. Triggers CI validation on the new tag

### 2. Version Bumping Rules

**Patch Release (v0.1.0 → v0.1.1)**
- Bug fixes
- Documentation updates
- Performance improvements (no API changes)
- Security patches

**Minor Release (v0.1.0 → v0.2.0)**
- New features (backwards-compatible)
- New API additions
- Deprecations (with backwards compatibility)
- Significant internal improvements

**Major Release (v0.9.0 → v1.0.0 or v1.5.0 → v2.0.0)**
- Breaking API changes
- Removal of deprecated features
- Major architectural changes
- v0.x.x → v1.0.0 signifies API stability commitment

### 3. Pre-release Versions

For testing before official release:

```
v0.2.0-alpha.1    # Alpha release
v0.2.0-beta.2     # Beta release
v0.2.0-rc.1       # Release candidate
```

These can be created by appending the suffix when triggering the release workflow.

## CI/CD Pipeline

### CI Workflow (`ci.yml`)

**Runs on:**
- ✅ Pull requests to `main` - Ensures quality before merge
- ✅ Pushes to `main` - Validates merged code
- ✅ Version tags (`v*`) - Validates releases
- ✅ Manual dispatch - On-demand testing

**Does NOT run on:**
- ❌ Feature branch pushes - Reduces noise, use manual dispatch if needed

**Jobs:**
- Build and test (Ubuntu, Go 1.22 & 1.23)
- Build and test (macOS, Go 1.22 & 1.23)
- Linting (golangci-lint)
- Coverage reporting (Codecov)

### GPU CI Workflow (`gpu-ci.yml`)

**Runs on:**
- ✅ Pull requests - If GPU code changes
- ✅ Manual dispatch - On-demand GPU testing

**Jobs:**
- GPU build and test (CUDA 12.3)
- GPU benchmarks

### Benchmark Workflow (`benchmark.yml`)

**Runs on:**
- ✅ Pushes to `main` - Track performance over time
- ✅ Weekly schedule (Sundays) - Regular performance monitoring
- ✅ Manual dispatch - On-demand benchmarking

**Jobs:**
- Continuous benchmarking with regression detection
- PR benchmark comparisons (when on PR)

### Release Workflow (`release.yml`)

**Runs on:**
- ✅ Manual dispatch ONLY - Controlled release process

**Jobs:**
- Version validation and bumping
- Changelog generation
- Git tag creation
- GitHub release publication

## Changelog

We automatically generate changelogs from commit messages using conventional commits format:

```
feat: Add new indexing method
fix: Resolve memory leak in Index.Search
docs: Update installation instructions
perf: Optimize distance calculations
refactor: Simplify serialization logic
test: Add benchmarks for large datasets
chore: Update dependencies
```

**Commit Message Format:**
```
<type>: <description>

[optional body]

[optional footer]
```

**Types:**
- `feat` - New feature (minor version bump)
- `fix` - Bug fix (patch version bump)
- `docs` - Documentation only
- `perf` - Performance improvement
- `refactor` - Code refactoring
- `test` - Adding tests
- `chore` - Maintenance tasks

**Breaking Changes:**
Add `BREAKING CHANGE:` in footer or `!` after type:
```
feat!: Remove deprecated Index.AddVector method

BREAKING CHANGE: Use Index.Add() instead of Index.AddVector()
```

## Go Module Versioning

### How Go Uses Our Tags

When users import this module:

```go
import "github.com/NerdMeNot/faiss-go"
```

And specify a version in `go.mod`:

```go
require github.com/NerdMeNot/faiss-go v0.2.0
```

Go automatically:
1. Resolves `v0.2.0` to our git tag
2. Downloads the specific tagged version
3. Caches it locally
4. Ensures reproducible builds

### Updating Module Path for v2+

When we release v2.0.0, we must update `go.mod`:

```go
// Current (v0.x and v1.x)
module github.com/NerdMeNot/faiss-go

// Future v2+
module github.com/NerdMeNot/faiss-go/v2
```

This allows users to:
- Use v1 and v2 simultaneously in the same project
- Migrate gradually to new versions

## Manual Testing Before Release

Before triggering a release:

1. **Ensure `main` is stable**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Run tests locally**
   ```bash
   go test ./...
   go test -race ./...
   ```

3. **Run linting**
   ```bash
   golangci-lint run
   ```

4. **Test examples**
   ```bash
   go run examples/basic/main.go
   ```

5. **Review recent changes**
   ```bash
   git log --oneline $(git describe --tags --abbrev=0)..HEAD
   ```

6. **Trigger release workflow** via GitHub Actions UI

## Hotfix Process

For critical bugs in released versions:

1. Create fix on `main`
2. Merge via PR (normal process)
3. Immediately trigger patch release workflow
4. New tag is created (e.g., v0.1.1 → v0.1.2)
5. Users can update with `go get -u github.com/NerdMeNot/faiss-go`

## Version File Location

The canonical version is stored in:

```
faiss.go:59-64
```

```go
const (
    Version = "0.1.0-alpha"
    FAISSVersion = "1.8.0"
)
```

The release workflow automatically updates this constant.

## Future Considerations

### v1.0.0 Release Criteria

Before releasing v1.0.0, we commit to:
- [ ] API stability (no breaking changes in v1.x.y)
- [ ] Comprehensive documentation
- [ ] 80%+ test coverage
- [ ] Performance benchmarks established
- [ ] Production usage validation
- [ ] Security audit

### Maintenance Policy

- **Current version (v0.x)**: Active development
- **v1.x**: Long-term support with security updates
- **Older versions**: Community support only

## Resources

- [Semantic Versioning](https://semver.org/)
- [Go Modules Reference](https://go.dev/ref/mod)
- [Go Module Versioning](https://go.dev/doc/modules/version-numbers)
- [Conventional Commits](https://www.conventionalcommits.org/)
