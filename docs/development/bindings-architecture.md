# Bindings Architecture

This document describes the architecture of faiss-go's C bindings and how they could be extracted into a separate module.

## Current Architecture

```
faiss-go/
├── faiss_lib.go          # CGO bindings (default, uses pre-built libs)
├── faiss_system.go       # CGO bindings (system FAISS, -tags=faiss_use_system)
├── libs/                 # Pre-built static libraries
│   ├── darwin_amd64/
│   ├── darwin_arm64/
│   ├── linux_amd64/
│   ├── linux_arm64/
│   └── windows_amd64/
└── c_api_ext/           # Custom C extensions for missing FAISS C API
```

### Build Tags

| Tag | Description |
|-----|-------------|
| (default) | Uses pre-built static libs from `libs/` |
| `faiss_use_system` | Links against system-installed FAISS |
| `gpu` | Enables GPU support (requires CUDA) |

### Library Components

1. **libfaiss.a** - Core FAISS library (C++)
2. **libfaiss_c.a** - Official FAISS C API wrapper
3. **libfaiss_go_ext.a** - Go-specific extensions (HNSW accessors, binary index constructor)

## Future: Separate Bindings Module

Following the [duckdb-go-bindings](https://github.com/duckdb/duckdb-go-bindings) pattern,
the pre-built libraries could be extracted into a separate module.

### Proposed Structure

```
faiss-go-bindings/
├── go.mod                          # module github.com/NerdMeNot/faiss-go-bindings
├── bindings.go                     # CGO declarations
├── cgo_static.go                   # Static linking config
├── cgo_dynamic.go                  # Dynamic linking config (-tags=dynamic)
├── prebuilt_darwin_amd64.go        # Platform-specific LDFLAGS
├── prebuilt_darwin_arm64.go
├── prebuilt_linux_amd64.go
├── prebuilt_linux_arm64.go
├── prebuilt_windows_amd64.go
├── include/                        # C headers
│   ├── faiss_c.h
│   └── faiss_go_ext.h
└── lib/                            # Pre-built libraries
    ├── darwin_amd64/
    ├── darwin_arm64/
    ├── linux_amd64/
    ├── linux_arm64/
    └── windows_amd64/
```

### Main Module Changes

The main `faiss-go` module would:

1. Import the bindings module:
   ```go
   import _ "github.com/NerdMeNot/faiss-go-bindings"
   ```

2. Remove the `libs/` directory
3. Remove platform-specific LDFLAGS from `faiss_lib.go`
4. Reference headers from the bindings module

### Benefits

1. **Independent Versioning**
   - Bindings: `v1.13.2-0.1` (FAISS version + patch)
   - Library: `v0.1.0`, `v0.2.0`, etc. (semantic versioning)

2. **Smaller Downloads**
   - Main module: ~500KB (code only)
   - Bindings module: ~100MB (libraries, downloaded once and cached)

3. **Custom Bindings**
   - Users can provide their own bindings module
   - Useful for custom FAISS builds or unsupported platforms

4. **CI/CD Efficiency**
   - Bindings cached separately from code
   - Code changes don't re-download large binaries

### Migration Path

1. Create `faiss-go-bindings` repository
2. Move `libs/` and CGO declarations
3. Update `faiss-go` to depend on bindings module
4. Maintain backward compatibility with `faiss_use_system` tag

### Considerations

- **Module Proxy Caching**: Go module proxy caches bindings, reducing download times
- **Version Coordination**: Bindings version must match expected FAISS version
- **Build Tag Compatibility**: Both modules must respect the same build tags

## Current Status

The bindings are currently embedded in the main module. This document serves as
a reference for future extraction if/when the benefits outweigh the maintenance
cost of managing two modules.

## References

- [duckdb-go-bindings](https://github.com/duckdb/duckdb-go-bindings) - Similar pattern for DuckDB
- [Go Modules: Separating Large Files](https://go.dev/blog/module-compatibility) - Go blog on module design
