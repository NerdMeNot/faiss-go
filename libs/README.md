# Pre-built FAISS Libraries

This directory contains pre-compiled static FAISS libraries for different platforms.

## Available Platforms

| Platform | Architecture | Compiler | BLAS |
|----------|-------------|----------|------|
| `linux_amd64` | x86_64 | GCC 11 | OpenBLAS 0.3.21 |
| `linux_arm64` | ARM64 | GCC 11 | OpenBLAS 0.3.21 |
| `darwin_amd64` | x86_64 | Clang 15 | Accelerate Framework |
| `darwin_arm64` | ARM64 (M1/M2) | Clang 15 | Accelerate Framework |
| `windows_amd64` | x86_64 | MSVC 2022 | OpenBLAS 0.3.21 |

## File Structure

Each platform directory contains:
- `libfaiss.a` (or `.lib` on Windows) - Static FAISS library
- `libopenblas.a` - Static OpenBLAS library (Linux/Windows only)
- `build_info.json` - Build metadata and configuration

## Usage

These libraries are automatically used when building with the `faiss_use_lib` tag:

```bash
go build -tags=faiss_use_lib
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

## License

The pre-built libraries are compiled from:
- FAISS (MIT License) - Copyright Meta Platforms, Inc.
- OpenBLAS (BSD License) - Copyright OpenBLAS developers

See LICENSE files in each directory for full license text.
