# Build Scripts

This directory contains scripts for building faiss-go components.

## Available Scripts

### `build-static-libs.sh`

Builds fully self-contained static FAISS libraries for the current platform.

**Purpose**: Create static libraries with all dependencies (OpenBLAS, gfortran, OpenMP) bundled, eliminating the need for runtime dependencies.

**Usage**:
```bash
# Build for current platform (merged strategy - single libfaiss.a)
./scripts/build-static-libs.sh

# Build with bundled libraries (separate .a files)
./scripts/build-static-libs.sh --bundled
```

**Output**:
- Default: `libs/<platform>/libfaiss.a` (50-70MB, fully self-contained)
- Bundled: Multiple `.a` files in `libs/<platform>/`

**Requirements**:
- Linux: cmake, gcc, g++, gfortran, git
- macOS: cmake, gcc (via brew), git
- Windows: MSYS2 with mingw-w64 toolchain

**See**: `docs/building-static-libs.md` for comprehensive documentation.

## Directory Structure

```
scripts/
├── README.md                 # This file
└── build-static-libs.sh      # Build fully static libraries
```

## Future Scripts

Planned scripts for future development:

- `build-all-platforms.sh` - Cross-compile for all platforms
- `update-libraries.sh` - Automated library update workflow
- `verify-static-libs.sh` - Verify libraries are fully self-contained
- `benchmark-build-times.sh` - Measure build performance

## Contributing

When adding new build scripts:

1. **Make them executable**: `chmod +x scripts/your-script.sh`
2. **Add shebang**: `#!/bin/bash`
3. **Use strict mode**: `set -e` (exit on error)
4. **Add documentation**: Update this README
5. **Test on all platforms**: Linux, macOS, Windows (MSYS2)
6. **Add error handling**: Check for missing dependencies
7. **Use consistent paths**: Reference `$SCRIPT_DIR` and `$PROJECT_ROOT`

## Support

For questions about build scripts:
- See documentation: `docs/building-static-libs.md`
- Open an issue: https://github.com/NerdMeNot/faiss-go/issues
