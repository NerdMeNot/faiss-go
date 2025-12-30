//go:build !faiss_use_system && faiss_phase3
// +build !faiss_use_system,faiss_phase3

package faiss

/*
// Phase 3 Static Library Build Mode (Experimental)
// Uses Phase 3 unified builds with ALL dependencies merged including runtime libs
// This is the most aggressive static linking approach
//
// Build with: go build -tags="nogpu,faiss_phase3"
//
// Supported platforms:
//   - linux/amd64, linux/arm64 (Phase 3 unified: ZERO deps)
//   - windows/amd64 (Phase 3 unified: ZERO deps)
//   - darwin/amd64, darwin/arm64 (Standard: uses Accelerate)
//
// Phase 3 Innovation:
//   Not only is OpenBLAS merged, but also runtime libraries:
//   - libgomp.a (OpenMP runtime) → merged
//   - libgfortran.a (Fortran runtime) → merged
//   - libquadmath.a (Quad math) → merged
//
// Result: Single libfaiss.a file (~50-60MB) with ZERO external dependencies
//
// If this works, it's the ultimate static build. If it fails, fall back
// to the standard approach with runtime dependencies.
//
// ============================================================================

// Platform-specific library paths and flags
// Linux/Windows - Phase 3: Truly ZERO dependencies!
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/libs/linux_amd64 -lfaiss_c -lfaiss -lm -lpthread -ldl
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/libs/linux_arm64 -lfaiss_c -lfaiss -lm -lpthread -ldl
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/libs/windows_amd64 -lfaiss_c -lfaiss -lm -lpthread

// macOS (Phase 3 not applicable, uses standard Accelerate)
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/libs/darwin_amd64 -lfaiss_c -lfaiss -Wl,-framework,Accelerate
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/libs/darwin_arm64 -lfaiss_c -lfaiss -Wl,-framework,Accelerate

// Note: NO -lgomp, NO -lgfortran, NO -lstdc++!
// Everything is merged into libfaiss.a

#include <stdlib.h>
#include <stdint.h>

// Forward declarations for FAISS C API
// These match the actual FAISS C API but are declared here
// until we generate the real amalgamation

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer types
typedef void* FaissIndex;
typedef void* FaissIndexBinary;
typedef void* FaissVectorTransform;
typedef void* FaissKmeans;

// ==== Flat Index Functions ====
extern int faiss_IndexFlatL2_new(FaissIndex* p_index, int64_t d);
extern int faiss_IndexFlatIP_new(FaissIndex* p_index, int64_t d);

// ... (rest of the declarations would be identical to faiss_lib.go)

#ifdef __cplusplus
}
#endif

*/
import "C"

// The implementation is identical to faiss_lib.go
// Just the CGO LDFLAGS are different - testing if Phase 3 merging worked
