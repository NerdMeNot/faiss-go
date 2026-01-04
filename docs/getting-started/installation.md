# Installation

faiss-go includes pre-built static libraries for major platforms, so installation is simple.

## Quick Install

```bash
go get github.com/NerdMeNot/faiss-go
```

That's it! Pre-built binaries are included for:
- Linux (AMD64, ARM64)
- macOS (Intel, Apple Silicon)

## Verify Installation

Create a test file:

```go
package main

import (
    "fmt"
    faiss "github.com/NerdMeNot/faiss-go"
)

func main() {
    index, err := faiss.IndexFactory(128, "Flat", faiss.MetricL2)
    if err != nil {
        panic(err)
    }
    defer index.Close()

    fmt.Printf("Created index with dimension %d\n", index.D())
}
```

Run it:

```bash
go run main.go
# Output: Created index with dimension 128
```

## Requirements

- **Go 1.21 or later**
- **Supported platform** (see above)

## Alternative: System FAISS

For platforms without pre-built binaries, or if you need a custom FAISS build:

```bash
go build -tags=faiss_use_system ./...
```

This requires FAISS to be installed on your system.

### Installing System FAISS

**Ubuntu/Debian:**
```bash
sudo apt-get install -y libfaiss-dev libopenblas-dev
```

**macOS:**
```bash
brew install faiss
```

**From source:**
```bash
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
sudo cmake --install build
```

## Troubleshooting

### "cannot find -lfaiss"

FAISS library not found. Either:
- Use pre-built binaries (default, no tags needed)
- Install system FAISS and use `-tags=faiss_use_system`

### Build takes too long

Make sure you're using pre-built binaries (the default). If building from source, the first build takes longer but subsequent builds are cached.

### Missing dependencies on Linux

```bash
sudo apt-get install -y build-essential
```

## Next Steps

- [Quickstart](quickstart.md) - Build your first search
- [Choosing an Index](choosing-an-index.md) - Select the right index type
