//go:build !faiss_use_system && darwin && amd64
// +build !faiss_use_system,darwin,amd64

package faiss

// Standard build for macOS Intel (x86_64)
// This file automatically loads the correct static library for this platform
//
// Build mode: Standard (uses system Accelerate framework)
// Size: ~9MB
// Dependencies: Accelerate.framework (built-in), libomp (brew install libomp)
//
// Note: macOS cannot do unified builds - Accelerate framework cannot be statically linked
//       This is the optimal configuration for macOS

/*
#cgo LDFLAGS: -L${SRCDIR}/libs/darwin_amd64 -lfaiss_c -lfaiss -Wl,-framework,Accelerate -L/usr/local/opt/libomp/lib -L/opt/homebrew/opt/libomp/lib -lomp -lm -lstdc++
*/
import "C"
