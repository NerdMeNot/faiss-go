//go:build !faiss_use_system && darwin && arm64
// +build !faiss_use_system,darwin,arm64

package faiss

// Standard build for macOS Apple Silicon (ARM64)
// This file automatically loads the correct static library for this platform
//
// Build mode: Standard (uses system Accelerate framework)
// Size: ~9MB
// Dependencies: Accelerate.framework (built-in), libomp (brew install libomp)
//
// Note: macOS cannot do unified builds - Accelerate framework cannot be statically linked
//       This is the optimal configuration for macOS
//       Accelerate is highly optimized for Apple Silicon

/*
#cgo LDFLAGS: ${SRCDIR}/libs/darwin_arm64/libfaiss_c.a ${SRCDIR}/libs/darwin_arm64/libfaiss.a -Wl,-framework,Accelerate -L/opt/homebrew/opt/libomp/lib -L/usr/local/opt/libomp/lib -lomp -lm -lstdc++
*/
import "C"
