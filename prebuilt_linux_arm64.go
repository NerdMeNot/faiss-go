//go:build !faiss_use_system && linux && arm64
// +build !faiss_use_system,linux,arm64

package faiss

// Standard build for Linux ARM64
// This file automatically loads the correct static library for this platform
//
// Build mode: Unified (OpenBLAS merged, runtime deps: gomp + gfortran)
// Size: ~45MB
// Dependencies: libgomp1, libgfortran5

/*
#cgo LDFLAGS: ${SRCDIR}/libs/linux_arm64/libfaiss_c.a ${SRCDIR}/libs/linux_arm64/libfaiss.a -lgomp -lgfortran -lm -lstdc++ -lpthread -ldl
*/
import "C"
