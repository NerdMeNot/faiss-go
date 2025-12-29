# Building faiss-go

This project supports two build modes to accommodate different platforms and use cases.

## TL;DR

```bash
# Default build (fast, recommended)
go build

# For unsupported platforms or custom FAISS builds
go build -tags=faiss_use_system
```

## Build Modes

### Mode 1: Static Libraries (Default) ⚡

**Recommended for most users** - Fast builds using pre-built static libraries.

```bash
go build
go test
```

**Features:**
- ✅ **Fast builds**: ~30 seconds (8x faster than compiling FAISS)
- ✅ **No dependencies**: Works out of the box
- ✅ **No build tags needed**: This is the default!
- ✅ **Cross-platform**: Pre-built for common platforms

**Supported Platforms:**
- `linux/amd64`, `linux/arm64`
- `darwin/amd64` (Intel Mac), `darwin/arm64` (Apple Silicon)
- `windows/amd64`

**How it works:**
- Uses pre-compiled FAISS static libraries (`.a` files) from the `libs/` directory
- No C++ compilation needed - just links the pre-built libraries
- Libraries are built from FAISS v1.13.2

**When to use:**
- ✅ You're on a supported platform
- ✅ You want fast builds
- ✅ You don't need a custom FAISS build

---

### Mode 2: System FAISS (Fallback) 🔧

For platforms without pre-built static libraries, or when you need a custom FAISS build.

```bash
# Install FAISS first
# Ubuntu/Debian:
sudo apt-get install libfaiss-dev

# macOS:
brew install faiss

# Then build with the tag:
go build -tags=faiss_use_system
go test -tags=faiss_use_system
```

**Features:**
- ✅ **Any platform**: Works on any platform where FAISS can be installed
- ✅ **Custom builds**: Use your own FAISS build with specific features
- ✅ **Latest FAISS**: Can use the latest FAISS version from your package manager
- ⚠️ **Requires installation**: Must install FAISS system-wide first
- ⚠️ **Slower builds**: Compiles C++ bridge code

**How it works:**
- Compiles `faiss_c_impl.cpp` (our C++ → C bridge) during the Go build
- Links against your system-installed FAISS library (`-lfaiss`)
- Requires FAISS headers to be available on your system

**When to use:**
- ✅ Your platform doesn't have pre-built static libraries
- ✅ You need a specific FAISS version or features
- ✅ You're building on a custom/unusual platform
- ✅ You want to use your own optimized FAISS build

---

## Choosing a Build Mode

```
┌─────────────────────────────────────┐
│ Are you on Linux/macOS/Windows      │
│ with amd64 or arm64?                │
└─────────────────────────────────────┘
           │
           ├─ YES → Use default mode (no tags needed) ⚡
           │        Fast builds, works out of the box
           │
           └─ NO  → Use system FAISS mode
                    go build -tags=faiss_use_system
```

## For Package Users

If you're importing this package in your Go project:

```go
import "github.com/NerdMeNot/faiss-go"
```

It will **automatically use static libraries** if your platform is supported. No action needed!

For unsupported platforms, add this to your build commands:
```bash
go build -tags=faiss_use_system ./...
go test -tags=faiss_use_system ./...
```

## CI/CD

Our CI uses the default static library mode for fast ~5-10 minute test runs across all supported Go versions (1.21-1.25) and platforms.

## Troubleshooting

### Static library mode fails with "unsupported platform"

Your platform doesn't have pre-built static libraries. Use system FAISS mode:
```bash
go build -tags=faiss_use_system
```

### System FAISS mode fails with "faiss/IndexFlat.h: No such file or directory"

Install FAISS development headers:
```bash
# Ubuntu/Debian
sudo apt-get install libfaiss-dev

# macOS
brew install faiss
```

### Link errors about missing symbols

Make sure you're using the correct build mode:
- Static mode: `go build` (no tags)
- System mode: `go build -tags=faiss_use_system`

Don't mix them!

## Platform Matrix

| Platform | Static Libs | System FAISS |
|----------|------------|--------------|
| Linux AMD64 | ✅ Default | ✅ Fallback |
| Linux ARM64 | ✅ Default | ✅ Fallback |
| macOS Intel | ✅ Default | ✅ Fallback |
| macOS ARM64 | ✅ Default | ✅ Fallback |
| Windows AMD64 | ✅ Default | ⚠️ Complex |
| Other platforms | ❌ Not available | ✅ Use this |

## GPU Support

GPU support is available in both modes but requires:
- NVIDIA GPU with CUDA
- FAISS built with GPU support
- Remove the `nogpu` tag from imports

This is an advanced use case - see [GPU documentation](docs/gpu.md) for details.

## Build Performance

| Mode | First Build | Subsequent Builds | Dependencies |
|------|-------------|-------------------|--------------|
| Static Libs | ~30 sec | ~5 sec | None |
| System FAISS | ~2-5 min | ~30 sec | libfaiss-dev |

## Questions?

- **Which mode should I use?** → Default (static libs) unless your platform isn't supported
- **Can I switch between modes?** → Yes, just use or remove the build tag
- **Do I need to install anything?** → Not for static libs mode!
- **What if my platform isn't supported?** → Use system FAISS mode with `-tags=faiss_use_system`

For more details, see:
- [WORKFLOW_USAGE.md](WORKFLOW_USAGE.md) - CI/CD workflows
- [.github/workflows/README.md](.github/workflows/README.md) - Workflow reference
- [examples/](examples/) - Code examples
