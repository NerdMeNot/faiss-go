//go:build !faiss_use_system && windows && amd64
// +build !faiss_use_system,windows,amd64

package faiss

// Standard build for Windows AMD64
// This file automatically loads the correct static library for this platform
//
// Build mode: Unified (OpenBLAS merged, runtime deps: gomp + gfortran)
// Size: ~45MB
// Dependencies: libgomp, libgfortran (provided by MinGW)

/*
#cgo LDFLAGS: ${SRCDIR}/libs/windows_amd64/faiss_c.lib ${SRCDIR}/libs/windows_amd64/faiss.lib -lgomp -lgfortran -lm -lstdc++ -lpthread
*/
import "C"
