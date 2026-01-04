# Makefile for faiss-go
#
# Build modes:
#   Default: Uses pre-built static libraries (fast, recommended)
#   -tags=faiss_use_system: Links against system-installed FAISS
#   -tags=gpu: Enables GPU support (requires CUDA)

.PHONY: all build test test-short test-coverage test-scenarios test-recall \
        bench clean fmt lint vet version info help \
        example-basic example-ivf example-hnsw example-pq example-gpu example-pretransform \
        install-deps install-deps-linux install-deps-macos

# Environment
export CGO_LDFLAGS_ALLOW=.*

# Default target
all: build

# ============================================
# Build Targets
# ============================================

## Build using pre-built static libraries (default, fast)
build:
	@echo "Building faiss-go..."
	go build -v ./...
	@echo "Build complete!"

## Build with system-installed FAISS
build-system:
	@echo "Building with system FAISS..."
	go build -tags=faiss_use_system -v ./...
	@echo "Build complete!"

## Build with GPU support (requires CUDA)
build-gpu:
	@echo "Building with GPU support..."
	go build -tags=gpu -v ./...
	@echo "Build complete!"

# ============================================
# Test Targets
# ============================================

## Run all tests
test:
	go test -v -timeout 10m ./...

## Run fast tests only (for development)
test-short:
	go test -short -v -timeout 5m .

## Run tests with coverage report
test-coverage:
	go test -coverprofile=coverage.out -covermode=atomic -timeout 10m .
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

## Run scenario tests (real-world use cases)
test-scenarios:
	go test -v -timeout 10m ./test/scenarios/...

## Run recall tests (search quality validation)
test-recall:
	go test -v -timeout 10m ./test/recall/...

## Run all qualitative tests
test-qualitative: test-scenarios test-recall

## Run tests with race detector
test-race:
	go test -race -short -v -timeout 10m .

# ============================================
# Benchmark Targets
# ============================================

## Run all benchmarks
bench:
	go test -bench=. -benchmem -run=^$$ ./...

## Run quick benchmarks (shorter duration)
bench-quick:
	go test -bench=. -benchmem -benchtime=100ms -run=^$$ ./...

## Run specific benchmark (usage: make bench-one NAME=BenchmarkIndexFlatL2)
bench-one:
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make bench-one NAME=BenchmarkName"; \
		exit 1; \
	fi
	go test -bench=$(NAME) -benchmem -benchtime=3s -run=^$$ ./...

# ============================================
# Code Quality
# ============================================

## Format code
fmt:
	@echo "Formatting code..."
	go fmt ./...
	@echo "Done!"

## Run go vet
vet:
	@echo "Running go vet..."
	go vet ./...
	@echo "Done!"

## Run linters (requires golangci-lint)
lint:
	@echo "Running linters..."
	@if command -v golangci-lint > /dev/null; then \
		golangci-lint run --timeout=5m; \
	else \
		echo "golangci-lint not installed. Install with:"; \
		echo "  go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest"; \
		exit 1; \
	fi

## Run all code quality checks
check: fmt vet lint

# ============================================
# Examples
# ============================================

## Run basic search example
example-basic:
	go run ./examples/01_basic_search/

## Run IVF clustering example
example-ivf:
	go run ./examples/02_ivf_clustering/

## Run HNSW graph example
example-hnsw:
	go run ./examples/03_hnsw_graph/

## Run PQ compression example
example-pq:
	go run ./examples/04_pq_compression/

## Run GPU acceleration example (requires CUDA)
example-gpu:
	go run -tags=gpu ./examples/05_gpu_acceleration/

## Run pretransform example
example-pretransform:
	go run ./examples/06_pretransform/

## Run all examples (except GPU)
examples: example-basic example-ivf example-hnsw example-pq example-pretransform

# ============================================
# Dependencies
# ============================================

## Install system dependencies (auto-detect OS)
install-deps:
	@echo "Installing dependencies..."
	@if [ "$$(uname)" = "Darwin" ]; then \
		$(MAKE) install-deps-macos; \
	elif [ "$$(uname)" = "Linux" ]; then \
		$(MAKE) install-deps-linux; \
	else \
		echo "Unsupported OS. Please install dependencies manually."; \
		exit 1; \
	fi

## Install dependencies on Linux
install-deps-linux:
	@echo "Installing Linux dependencies..."
	@if command -v apt-get > /dev/null; then \
		sudo apt-get update && sudo apt-get install -y libopenblas-dev libgomp1 libomp-dev; \
	elif command -v dnf > /dev/null; then \
		sudo dnf install -y openblas-devel libomp-devel; \
	elif command -v yum > /dev/null; then \
		sudo yum install -y openblas-devel libomp-devel; \
	else \
		echo "Package manager not found. Install OpenBLAS and OpenMP manually."; \
		exit 1; \
	fi
	@echo "Done!"

## Install dependencies on macOS
install-deps-macos:
	@echo "Installing macOS dependencies..."
	brew install openblas libomp
	@echo "Done!"

# ============================================
# Utilities
# ============================================

## Show version information
version:
	@echo "faiss-go version information:"
	@grep -E "^\s*(FAISSVersion|BindingMajor|BindingMinor)\s*=" faiss.go | sed 's/\/\/.*//' | sed 's/\t/  /g'

## Show build and environment information
info:
	@echo "=== Environment ==="
	@go version
	@echo ""
	@echo "GOOS=$$(go env GOOS)"
	@echo "GOARCH=$$(go env GOARCH)"
	@echo "CGO_ENABLED=$$(go env CGO_ENABLED)"
	@echo ""
	@echo "=== Version ==="
	@$(MAKE) -s version

## Clean build artifacts and caches
clean:
	@echo "Cleaning..."
	go clean -cache -testcache
	rm -f coverage.out coverage.html
	rm -rf tmp/ bin/ dist/
	rm -f faiss-go-test
	@echo "Done!"

## Deep clean (includes module cache)
clean-all: clean
	go clean -modcache
	@echo "Module cache cleaned!"

# ============================================
# Help
# ============================================

## Show help
help:
	@echo "faiss-go - Go bindings for FAISS"
	@echo ""
	@echo "Build:"
	@echo "  make build            Build with pre-built static libs (default)"
	@echo "  make build-system     Build with system-installed FAISS"
	@echo "  make build-gpu        Build with GPU support"
	@echo ""
	@echo "Test:"
	@echo "  make test             Run all tests"
	@echo "  make test-short       Run fast tests only"
	@echo "  make test-coverage    Run tests with coverage report"
	@echo "  make test-scenarios   Run scenario tests"
	@echo "  make test-recall      Run recall quality tests"
	@echo "  make test-race        Run tests with race detector"
	@echo ""
	@echo "Benchmark:"
	@echo "  make bench            Run all benchmarks"
	@echo "  make bench-quick      Run quick benchmarks"
	@echo "  make bench-one NAME=X Run specific benchmark"
	@echo ""
	@echo "Code Quality:"
	@echo "  make fmt              Format code"
	@echo "  make vet              Run go vet"
	@echo "  make lint             Run golangci-lint"
	@echo "  make check            Run all quality checks"
	@echo ""
	@echo "Examples:"
	@echo "  make example-basic    Basic vector search"
	@echo "  make example-ivf      IVF clustering"
	@echo "  make example-hnsw     HNSW graph search"
	@echo "  make example-pq       Product quantization"
	@echo "  make example-gpu      GPU acceleration"
	@echo "  make examples         Run all examples (except GPU)"
	@echo ""
	@echo "Utilities:"
	@echo "  make install-deps     Install system dependencies"
	@echo "  make version          Show version info"
	@echo "  make info             Show build info"
	@echo "  make clean            Clean build artifacts"
	@echo "  make help             Show this help"
