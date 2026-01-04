# Building Static Libraries

This guide is for maintainers who need to build or update the pre-built FAISS static libraries.

## Overview

faiss-go uses pre-built static libraries provided by the [faiss-go-bindings](https://github.com/NerdMeNot/faiss-go-bindings) module. The libraries are built separately and published as Go module releases.

**Most users don't need this** - just use `go get github.com/NerdMeNot/faiss-go` and the bindings are fetched automatically.

## Architecture

See [Bindings Architecture](bindings-architecture.md) for details on how faiss-go uses the separate bindings module.

## Building Libraries

Libraries are built in the [faiss-go-bindings](https://github.com/NerdMeNot/faiss-go-bindings) repository:

```bash
# Clone the bindings repo
git clone https://github.com/NerdMeNot/faiss-go-bindings.git
cd faiss-go-bindings

# Build for current platform
./c_api_ext/build.sh

# Libraries are output to lib/<platform>/
```

### Supported Platforms

| Platform | Architecture | Directory |
|----------|--------------|-----------|
| Linux | AMD64 | `lib/linux_amd64/` |
| Linux | ARM64 | `lib/linux_arm64/` |
| macOS | Intel | `lib/darwin_amd64/` |
| macOS | Apple Silicon | `lib/darwin_arm64/` |

### Platform Requirements

**Linux (AMD64 / ARM64)**:
```bash
# Ubuntu/Debian
sudo apt-get install cmake build-essential gfortran git

# Fedora/RHEL
sudo dnf install cmake gcc gcc-c++ gcc-gfortran git
```

**macOS (Intel / Apple Silicon)**:
```bash
brew install cmake gcc git
```

## Updating c_api_ext Extensions

The custom C extensions source code lives in the bindings repo at `c_api_ext/`:

1. Make changes in `faiss-go-bindings/c_api_ext/`
2. Rebuild: `./c_api_ext/build.sh`
3. Test with faiss-go using replace directive
4. Tag and release new bindings version

## Testing Local Changes

Use a replace directive to test with local bindings:

```go
// go.mod
replace github.com/NerdMeNot/faiss-go-bindings => ../faiss-go-bindings
```

```bash
go build ./...
go test ./...
```

## GPU Support

GPU builds require CUDA toolkit and aren't included in pre-built libraries. See [GPU Setup](../getting-started/gpu-setup.md).

## See Also

- [Bindings Architecture](bindings-architecture.md) - How the two-module system works
- [faiss-go-bindings](https://github.com/NerdMeNot/faiss-go-bindings) - Pre-built libraries repository
