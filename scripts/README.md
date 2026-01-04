# Scripts

This directory contains utility scripts for faiss-go development.

## Available Scripts

### `download_test_datasets.sh`

Downloads test datasets for benchmarking and testing.

**Usage**:
```bash
./scripts/download_test_datasets.sh
```

## Library Building

Static libraries are now built in the separate [faiss-go-bindings](https://github.com/NerdMeNot/faiss-go-bindings) repository. See [docs/development/building-libs.md](../docs/development/building-libs.md) for details.

## Contributing

When adding new scripts:

1. **Make them executable**: `chmod +x scripts/your-script.sh`
2. **Add shebang**: `#!/bin/bash`
3. **Use strict mode**: `set -e` (exit on error)
4. **Add documentation**: Update this README
5. **Test on supported platforms**: Linux, macOS
6. **Add error handling**: Check for missing dependencies

## Support

For questions about scripts:
- See documentation: `docs/development/`
- Open an issue: https://github.com/NerdMeNot/faiss-go/issues
