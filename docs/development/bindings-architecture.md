# Bindings Architecture

This document describes the architecture of faiss-go's C bindings, which are provided by a separate module.

## Architecture Overview

```
faiss-go/                           # Main module (Go bindings)
├── faiss_lib.go                    # CGO bindings (default, imports bindings module)
└── faiss_system.go                 # CGO bindings (system FAISS, -tags=faiss_use_system)

faiss-go-bindings/                  # Bindings module (pre-built libraries)
├── bindings.go                     # CGO declarations
├── cgo_darwin_amd64.go             # Platform-specific LDFLAGS
├── cgo_darwin_arm64.go
├── cgo_linux_amd64.go
├── cgo_linux_arm64.go
├── include/                        # C headers
│   ├── faiss_c.h
│   └── faiss_go_ext.h
└── lib/                            # Pre-built libraries
    ├── darwin_amd64/
    ├── darwin_arm64/
    ├── linux_amd64/
    └── linux_arm64/
```

## Build Tags

| Tag | Description |
|-----|-------------|
| (default) | Uses pre-built static libs from `faiss-go-bindings` module |
| `faiss_use_system` | Links against system-installed FAISS |
| `gpu` | Enables GPU support (requires CUDA) |

## Library Components

The bindings module provides three static libraries per platform:

1. **libfaiss.a** - Core FAISS library (C++)
2. **libfaiss_c.a** - Official FAISS C API wrapper
3. **libfaiss_go_ext.a** - Go-specific extensions (HNSW accessors, binary index constructor)

## How It Works

The main `faiss-go` module imports the bindings module with a blank import:

```go
import _ "github.com/NerdMeNot/faiss-go-bindings"
```

This triggers CGO compilation of the bindings module, which provides:
- Platform-specific LDFLAGS pointing to the pre-built libraries
- CFLAGS for header file locations

The linker then combines:
- Go wrapper code from `faiss-go`
- Pre-built FAISS libraries from `faiss-go-bindings`

## Benefits

1. **Independent Versioning**
   - Bindings: `v1.13.2-1` (FAISS version + patch)
   - Library: `v0.1.0`, `v0.2.0`, etc. (semantic versioning)

2. **Smaller Downloads**
   - Main module: ~500KB (code only)
   - Bindings module: ~75MB (libraries, downloaded once and cached)

3. **Custom Bindings**
   - Users can provide their own bindings module
   - Useful for custom FAISS builds or unsupported platforms

4. **CI/CD Efficiency**
   - Bindings cached separately from code
   - Code changes don't re-download large binaries

## Custom Extensions

The bindings module includes custom C extensions (`libfaiss_go_ext.a`) that fill gaps in the FAISS C API:

- HNSW property accessors (efConstruction, efSearch)
- Binary index constructor
- Vector transform wrappers with ABI safety

Source code for these extensions lives in the [faiss-go-bindings](https://github.com/NerdMeNot/faiss-go-bindings) repository under `c_api_ext/`.

## Using System FAISS

For custom FAISS builds or unsupported platforms:

```bash
go build -tags=faiss_use_system ./...
```

This bypasses the bindings module and links against system-installed FAISS libraries.

## References

- [faiss-go-bindings](https://github.com/NerdMeNot/faiss-go-bindings) - Pre-built FAISS libraries
- [duckdb-go-bindings](https://github.com/duckdb/duckdb-go-bindings) - Similar pattern for DuckDB
