module github.com/NerdMeNot/faiss-go

go 1.21

// This module provides Go bindings for FAISS (Facebook AI Similarity Search)
// with pre-built static libraries - no separate compilation required!
//
// Build options:
//   - Default: Uses pre-built static libraries from faiss-go-bindings module
//   - Tag 'faiss_use_system': Links against system-installed FAISS
//
// Example:
//   go build                            # Use pre-built libraries (default)
//   go build -tags=faiss_use_system     # Use system FAISS

require github.com/NerdMeNot/faiss-go-bindings v1.13.2-1

replace github.com/NerdMeNot/faiss-go-bindings => ../faiss-go-bindings
