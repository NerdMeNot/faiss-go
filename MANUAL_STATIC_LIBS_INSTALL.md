# Manual Static Library Installation Guide

The automated PR creation was having issues, so we've simplified to manual installation.

## Quick Steps

### 1. Download the Artifact

After running the `build-static-libs.yml` workflow:

1. Go to GitHub Actions → Build Static Libraries → Latest run
2. Download the **`all-static-libraries`** artifact (it will be a `.zip` file)
3. Extract it - you'll get a `combined-static-libs/` directory

### 2. Copy Files to Your Repo

Based on your artifact structure shown, here's what needs to be copied:

```bash
# From: combined-static-libs/faiss-static-darwin-amd64/
# To:   libs/darwin_amd64/

cd combined-static-libs/

# For EACH platform directory (faiss-static-*), copy these files:
```

## Files to Copy Per Platform

### What You Have (from artifact):
```
faiss-static-darwin-amd64/
├── build_info.json
├── checksums.txt
├── include/           (22 header files)
├── libfaiss.a        ← CRITICAL (10 MB)
└── libfaiss_c.a      ← CRITICAL (325 KB)
```

### Where They Go (in your repo):
```
libs/darwin_amd64/
├── build_info.json
├── checksums.txt
├── include/
├── libfaiss.a
└── libfaiss_c.a
```

## Copy Script

Save this as `install-libs.sh` in the extracted `combined-static-libs/` directory:

```bash
#!/bin/bash
set -e

# Get the repo directory (assuming faiss-go is one level up)
REPO_DIR="${1:-../faiss-go}"

if [ ! -d "$REPO_DIR" ]; then
    echo "Error: Repository directory not found at $REPO_DIR"
    echo "Usage: $0 /path/to/faiss-go"
    exit 1
fi

echo "Installing static libraries to: $REPO_DIR/libs/"

# Platform mappings: artifact-name → repo-directory
declare -A platforms=(
    ["faiss-static-linux-amd64"]="linux_amd64"
    ["faiss-static-linux-arm64"]="linux_arm64"
    ["faiss-static-darwin-amd64"]="darwin_amd64"
    ["faiss-static-darwin-arm64"]="darwin_arm64"
    ["faiss-static-windows-amd64"]="windows_amd64"
)

# Copy files for each platform
for artifact_dir in "${!platforms[@]}"; do
    target_dir="${platforms[$artifact_dir]}"

    if [ ! -d "$artifact_dir" ]; then
        echo "⚠️  Warning: $artifact_dir not found (build may have failed)"
        continue
    fi

    echo "📦 Installing $target_dir..."

    # Create target directory
    mkdir -p "$REPO_DIR/libs/$target_dir"

    # Copy all files from artifact to target
    # This handles both .a files and .lib files
    cp -v "$artifact_dir"/* "$REPO_DIR/libs/$target_dir/" 2>/dev/null || true

    # Copy include directory if it exists
    if [ -d "$artifact_dir/include" ]; then
        cp -rv "$artifact_dir/include" "$REPO_DIR/libs/$target_dir/"
    fi

    # Verify the critical library file exists
    if [ -f "$REPO_DIR/libs/$target_dir/libfaiss.a" ] || \
       [ -f "$REPO_DIR/libs/$target_dir/faiss.lib" ]; then
        echo "✅ $target_dir installed successfully"
    else
        echo "❌ $target_dir: library file not found!"
    fi
done

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📊 Library sizes:"
du -sh "$REPO_DIR/libs/"*/*.a "$REPO_DIR/libs/"*/*.lib 2>/dev/null || true

echo ""
echo "📁 Directory structure:"
tree -L 2 "$REPO_DIR/libs/" || ls -lR "$REPO_DIR/libs/"

echo ""
echo "✨ Test the static library build:"
echo "   cd $REPO_DIR"
echo "   go build -tags=faiss_use_lib,nogpu -v ./..."
echo "   go test -tags=faiss_use_lib,nogpu -v ./..."
```

## Usage

```bash
# Extract the downloaded artifact
unzip all-static-libraries.zip
cd combined-static-libs/

# Run the install script
chmod +x install-libs.sh
./install-libs.sh /path/to/your/faiss-go

# Or if faiss-go is one directory up:
./install-libs.sh
```

## Expected Result

After installation, your `libs/` directory should look like:

```
libs/
├── README.md
├── darwin_amd64/
│   ├── build_info.json
│   ├── checksums.txt
│   ├── include/          (22 .h files)
│   ├── libfaiss.a        (~10 MB)
│   └── libfaiss_c.a      (~325 KB)
├── darwin_arm64/
│   └── (same structure)
├── linux_amd64/
│   └── (same structure)
├── linux_arm64/
│   └── (same structure)
└── windows_amd64/
    ├── build_info.json
    ├── checksums.txt
    ├── include/
    ├── faiss.lib         (~15 MB)
    └── faiss_c.lib       (~325 KB)
```

## Verify Installation

```bash
# Check files exist
find libs/ -name "*.a" -o -name "*.lib"

# Should output:
# libs/darwin_amd64/libfaiss.a
# libs/darwin_amd64/libfaiss_c.a
# libs/darwin_arm64/libfaiss.a
# libs/darwin_arm64/libfaiss_c.a
# libs/linux_amd64/libfaiss.a
# libs/linux_amd64/libfaiss_c.a
# libs/linux_arm64/libfaiss.a
# libs/linux_arm64/libfaiss_c.a
# libs/windows_amd64/faiss.lib
# libs/windows_amd64/faiss_c.lib
```

## Test the Build

```bash
cd /path/to/faiss-go

# Build with static libraries
go build -tags=faiss_use_lib,nogpu -v ./...

# Run tests
go test -tags=faiss_use_lib,nogpu -v ./...
```

**Expected build time:** ~30 seconds (vs 2-5 minutes for amalgamation)

## Commit the Files

```bash
git add libs/
git commit -m "feat: Add pre-built FAISS static libraries v1.13.2

Adds pre-compiled static libraries for all platforms:
- Linux AMD64/ARM64
- macOS AMD64/ARM64 (Intel/Apple Silicon)
- Windows AMD64

Enables fast builds with: go build -tags=faiss_use_lib

Library sizes:
- libfaiss.a: ~10-15 MB per platform
- libfaiss_c.a: ~325 KB per platform
- Total: ~60-70 MB

Built from FAISS v1.13.2 with static linking."

git push
```

## Troubleshooting

### Library file missing after copying

Check the artifact structure:
```bash
ls -lh combined-static-libs/faiss-static-darwin-amd64/
```

Make sure `libfaiss.a` exists in the artifact.

### Build fails with "library not found"

Verify file permissions:
```bash
ls -l libs/*/libfaiss.a
chmod 644 libs/*/libfaiss.a  # Fix if needed
```

### Wrong directory structure

The files should be **directly** in `libs/darwin_amd64/`, not nested:
```
✅ libs/darwin_amd64/libfaiss.a
❌ libs/darwin_amd64/faiss-static-darwin-amd64/libfaiss.a
```

## Future Automation

Once this manual process is confirmed working, we can:
1. Re-enable automated PR creation with the correct file structure
2. Or create a GitHub Action that runs the install script automatically

For now, manual is simpler and more reliable! 🎯
