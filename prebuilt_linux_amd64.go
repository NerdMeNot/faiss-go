//go:build !faiss_use_system && !faiss_phase3 && linux && amd64
// +build !faiss_use_system,!faiss_phase3,linux,amd64

package faiss

// Standard build for Linux AMD64
// This file automatically loads the correct static library for this platform
//
// Build mode: Unified (OpenBLAS merged, runtime deps: gomp + gfortran)
// Size: ~45MB
// Dependencies: libgomp1, libgfortran5
//
// For Phase 3 builds (experimental zero-dep), use: go build -tags="nogpu,faiss_phase3"

/*
#cgo LDFLAGS: ${SRCDIR}/libs/linux_amd64/libfaiss_c.a ${SRCDIR}/libs/linux_amd64/libfaiss.a -lgomp -lgfortran -lm -lstdc++ -lpthread -ldl
*/
import "C"
