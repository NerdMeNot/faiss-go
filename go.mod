module github.com/NerdMeNot/faiss-go

go 1.21

// This module provides Go bindings for FAISS (Facebook AI Similarity Search)
// with embedded FAISS - no separate compilation required!
//
// Build options:
//   - Default: Compiles from amalgamated source (requires C++17 compiler + BLAS)
//   - Tag 'faiss_use_lib': Uses pre-built static libraries (fastest, no compilation)
//
// Example:
//   go build                          # Compile from source
//   go build -tags=faiss_use_lib      # Use pre-built libraries
