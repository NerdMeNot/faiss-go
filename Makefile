# Makefile for faiss-go

.PHONY: all build build-source build-prebuilt test bench clean install help

# Default target
all: help

## Build from amalgamated source (default)
build: build-source

## Build from amalgamated source
build-source:
	@echo "Building from source (this may take 5-10 minutes on first build)..."
	go build -v ./...
	@echo "Build complete!"

## Build using pre-built libraries
build-prebuilt:
	@echo "Building with pre-built libraries..."
	go build -tags=faiss_use_lib -v ./...
	@echo "Build complete!"

## Run all tests
test:
	go test -v ./...

## Run tests with pre-built libraries
test-prebuilt:
	go test -tags=faiss_use_lib -v ./...

## Run benchmarks
bench:
	go test -bench=. -benchmem ./...

## Run benchmarks with pre-built libraries
bench-prebuilt:
	go test -tags=faiss_use_lib -bench=. -benchmem ./...

## Clean build artifacts
clean:
	go clean -cache -testcache -modcache
	rm -rf bin/ dist/ tmp/
	@echo "Clean complete!"

## Install dependencies
install-deps:
	@echo "Installing dependencies..."
	@if [ "$(shell uname)" = "Darwin" ]; then \
		echo "macOS detected..."; \
		brew install openblas cmake || true; \
	elif [ "$(shell uname)" = "Linux" ]; then \
		echo "Linux detected..."; \
		if command -v apt-get > /dev/null; then \
			sudo apt-get install -y libopenblas-dev cmake build-essential; \
		elif command -v dnf > /dev/null; then \
			sudo dnf install -y openblas-devel cmake gcc-c++; \
		fi; \
	fi
	@echo "Dependencies installed!"

## Generate FAISS amalgamation
generate-amalgamation:
	@echo "Generating FAISS amalgamation..."
	cd scripts && ./generate_amalgamation.sh
	@echo "Amalgamation generation complete!"

## Update FAISS to specific version (e.g., make update-faiss VERSION=v1.8.0)
update-faiss:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION not specified. Usage: make update-faiss VERSION=v1.8.0"; \
		exit 1; \
	fi
	@echo "Updating FAISS to $(VERSION)..."
	cd scripts && ./generate_amalgamation.sh $(VERSION)

## Run example: basic search
example-basic:
	@echo "Running basic search example..."
	cd examples && go run basic_search.go

## Run example: inner product
example-ip:
	@echo "Running inner product example..."
	cd examples && go run inner_product.go

## Format code
fmt:
	@echo "Formatting code..."
	go fmt ./...
	@echo "Format complete!"

## Run linters
lint:
	@echo "Running linters..."
	go vet ./...
	@if command -v staticcheck > /dev/null; then \
		staticcheck ./...; \
	else \
		echo "staticcheck not installed, skipping..."; \
	fi
	@echo "Lint complete!"

## Show Go environment
info:
	@echo "Go Information:"
	@go version
	@echo ""
	@go env | grep -E "(GOOS|GOARCH|CGO_ENABLED)"
	@echo ""
	@echo "Build Info:"
	@go run -tags=faiss_use_lib -e 'package main; import ("fmt"; f "github.com/NerdMeNot/faiss-go"); func main() { fmt.Println(f.GetBuildInfo()) }' 2>/dev/null || echo "Run 'make build' first"

## Show help
help:
	@echo "faiss-go - FAISS Go Bindings"
	@echo ""
	@echo "Available targets:"
	@echo "  make build              Build from source (default)"
	@echo "  make build-prebuilt     Build using pre-built libraries"
	@echo "  make test               Run tests"
	@echo "  make bench              Run benchmarks"
	@echo "  make clean              Clean build artifacts"
	@echo "  make install-deps       Install system dependencies"
	@echo "  make generate-amalgamation  Generate FAISS amalgamation"
	@echo "  make update-faiss       Update FAISS version (VERSION=vX.Y.Z)"
	@echo "  make example-basic      Run basic search example"
	@echo "  make example-ip         Run inner product example"
	@echo "  make fmt                Format code"
	@echo "  make lint               Run linters"
	@echo "  make info               Show build information"
	@echo "  make help               Show this help"
	@echo ""
	@echo "Build modes:"
	@echo "  Default: Compiles from amalgamated source"
	@echo "  -tags=faiss_use_lib: Uses pre-built static libraries"
	@echo ""
	@echo "Examples:"
	@echo "  make build              # Build from source"
	@echo "  make build-prebuilt     # Build with pre-built libs"
	@echo "  make test               # Run tests"
	@echo "  make update-faiss VERSION=v1.8.0  # Update FAISS"
