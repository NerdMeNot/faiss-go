# Contributing to faiss-go

Thank you for your interest in contributing to faiss-go!

## Getting Started

### Prerequisites

- Go 1.21 or later
- Git

### Setup

```bash
git clone https://github.com/NerdMeNot/faiss-go.git
cd faiss-go
go build ./...
go test ./...
```

Pre-built static libraries are included, so no additional dependencies are needed on supported platforms.

## Development Workflow

### Building

```bash
# Build all packages
go build ./...

# Build with verbose output
go build -v ./...
```

### Testing

```bash
# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run specific test
go test -v -run TestIndexFactory

# Run with coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

### Linting

```bash
# Format code
go fmt ./...

# Run linter (install: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest)
golangci-lint run
```

## Making Changes

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for IndexPQFastScan

fix: prevent crash when searching empty index

docs: update installation instructions
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `go test ./...`
5. Run linter: `golangci-lint run`
6. Push and create a pull request

### PR Requirements

- All tests pass
- Linter passes
- New code has tests
- Documentation updated if needed

## Code Guidelines

### Style

- Follow [Effective Go](https://go.dev/doc/effective_go)
- Use `gofmt` for formatting
- Keep functions focused and small
- Add comments for non-obvious logic

### Error Handling

```go
// Good: wrap errors with context
if err := index.Add(vectors); err != nil {
    return fmt.Errorf("failed to add vectors: %w", err)
}

// Good: always check errors
index, err := faiss.IndexFactory(128, "Flat", faiss.MetricL2)
if err != nil {
    return err
}
```

### Resource Management

```go
// Good: always close indexes
index, err := faiss.IndexFactory(128, "Flat", faiss.MetricL2)
if err != nil {
    return err
}
defer index.Close()
```

### CGO Guidelines

When working with CGO code:

1. Minimize CGO calls (they have overhead)
2. Always check return codes from C functions
3. Free C memory with `C.free()`
4. Validate inputs before CGO calls

See [Programming Guide](docs/development/programming-guide.md) for detailed CGO patterns.

## Project Structure

```
faiss-go/
├── *.go                 # Main package source
├── *_test.go            # Tests
├── docs/                # Documentation
│   ├── getting-started/ # Installation, quickstart
│   ├── guides/          # API reference, tutorials
│   ├── development/     # Architecture, contributing
│   └── reference/       # FAQ, glossary
├── examples/            # Example code
├── test/                # Integration tests
└── scripts/             # Build scripts
```

## Testing

### Test Organization

- `*_test.go` - Unit tests alongside source
- `test/recall/` - Recall validation tests
- `test/scenarios/` - Real-world scenario tests
- `test/integration/` - Integration tests

### Writing Tests

```go
func TestIndexFactory_Flat(t *testing.T) {
    index, err := faiss.IndexFactory(128, "Flat", faiss.MetricL2)
    if err != nil {
        t.Fatalf("failed to create index: %v", err)
    }
    defer index.Close()

    // Test operations...
}
```

## Building Static Libraries

If you need to rebuild the static libraries:

```bash
./scripts/build-static-libs.sh
```

See [Building Libraries](docs/development/building-libs.md) for details.

## Getting Help

- [Documentation](docs/)
- [GitHub Issues](https://github.com/NerdMeNot/faiss-go/issues)
- [GitHub Discussions](https://github.com/NerdMeNot/faiss-go/discussions)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
