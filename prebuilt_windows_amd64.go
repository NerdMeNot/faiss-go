//go:build !faiss_use_system && !faiss_phase3 && windows && amd64
// +build !faiss_use_system,!faiss_phase3,windows,amd64

package faiss

// Standard build for Windows AMD64
// This file automatically loads the correct static library for this platform
//
// Build mode: Unified (OpenBLAS merged, runtime deps: gomp + gfortran)
// Size: ~45MB
// Dependencies: libgomp, libgfortran (provided by MinGW)
//
// For Phase 3 builds (experimental zero-dep), use: go build -tags="nogpu,faiss_phase3"

/*
#cgo LDFLAGS: -L${SRCDIR}/libs/windows_amd64 -lfaiss_c -lfaiss -lgomp -lgfortran -lm -lstdc++ -lpthread
*/
import "C"
