# Project Maintenance

## Reporting Issues

If you discover an issue with faiss-go, please report it through our [GitHub Issues](https://github.com/NerdMeNot/faiss-go/issues).

### How to Report

1. Go to [Issues](https://github.com/NerdMeNot/faiss-go/issues)
2. Click "New Issue"
3. Provide details about the issue
4. Include reproduction steps if applicable

### What to Include

- Go version (`go version`)
- Operating system and version
- Build mode (source or pre-built libraries)
- Steps to reproduce
- Expected vs actual behavior
- Error messages or stack traces

## Supported Versions

We maintain the latest release. Please update to the most recent version before reporting issues.

| Version | Supported |
|---------|-----------|
| Latest  | ✅ Yes    |
| Older   | ❌ No     |

## Best Practices

When using faiss-go:
- Always call `Close()` on indexes to prevent memory leaks
- Validate input dimensions match index dimensions
- Use appropriate error handling
- Follow examples in documentation

## Contact

For questions or discussions:
- [GitHub Discussions](https://github.com/NerdMeNot/faiss-go/discussions)
- [GitHub Issues](https://github.com/NerdMeNot/faiss-go/issues)
