# Pre-built FAISS Libraries

This directory contains pre-compiled static FAISS libraries for different platforms.

## Available Platforms

| Platform | Architecture | Compiler | BLAS |
|----------|-------------|----------|------|
| `linux_amd64` | x86_64 | GCC 11 | OpenBLAS 0.3.21 |
| `linux_arm64` | ARM64 | GCC 11 | OpenBLAS 0.3.21 |
| `darwin_amd64` | x86_64 | Clang 15 | Accelerate Framework |
| `darwin_arm64` | ARM64 (M1/M2/M3) | Clang 15 | Accelerate Framework |
| `windows_amd64` | x86_64 | MSVC 2022 | OpenBLAS 0.3.21 |

## File Structure

Each platform directory contains:
- `libfaiss.a` - Core FAISS library (C++)
- `libfaiss_c.a` - FAISS C API wrapper
- `libfaiss_go_ext.a` - Go-specific extensions (macOS only, for HNSW accessors)
- `build_info.json` - Build metadata and configuration

## Runtime Dependencies

| Platform | Dependencies |
|----------|-------------|
| macOS | None (uses system Accelerate framework) |
| Linux | `libopenblas`, `libgfortran`, `libgomp` (install via apt/yum) |
| Windows | `libopenblas`, `libgfortran`, `libquadmath` |

## Usage

These libraries are used by default (no build tags needed):

```bash
go build ./...
```

The appropriate library is selected based on `GOOS` and `GOARCH`.

## Building Libraries

To rebuild the pre-compiled libraries (maintainers only):

```bash
cd scripts
./build_static_libs.sh [platform]
```

For example:
```bash
./build_static_libs.sh linux_amd64
./build_static_libs.sh all  # Build for all platforms
```

## Size Information

Approximate sizes per platform:
- `libfaiss.a`: ~15-25 MB
- `libopenblas.a`: ~20-40 MB (if bundled)
- **Total per platform**: ~35-65 MB

## Verification

Each library includes SHA256 checksums in `checksums.txt` for verification:

```bash
cd libs/linux_amd64
sha256sum -c checksums.txt
```

## Using Custom Libraries

To use your own FAISS build instead of these pre-built libraries:

```bash
# Use system-installed FAISS
go build -tags=faiss_use_system ./...

# Or set custom library paths
export CGO_LDFLAGS="-L/path/to/your/libs -lfaiss_c -lfaiss"
go build -tags=faiss_use_system ./...
```

## Future: Separate Bindings Module

This directory may be extracted into a separate Go module
(`github.com/NerdMeNot/faiss-go-bindings`) following the pattern
used by [duckdb-go-bindings](https://github.com/duckdb/duckdb-go-bindings).

Benefits of separation:
- Independent versioning of bindings vs library code
- Smaller main module download size
- Easier to provide custom bindings
- Better caching in CI/CD pipelines

## License

The pre-built libraries are compiled from:
- FAISS (MIT License) - Copyright Meta Platforms, Inc.
- OpenBLAS (BSD License) - Copyright OpenBLAS developers

See LICENSE files in each directory for full license text.
