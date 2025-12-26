//go:build faiss_use_lib
// +build faiss_use_lib

package faiss

/*
#cgo CPPFLAGS: -I${SRCDIR}/faiss -std=c++17
#cgo LDFLAGS: -lstdc++ -lm

// Platform-specific library paths and flags
// Linux
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/libs/linux_amd64 -lfaiss -lopenblas -lgfortran -lgomp -lpthread
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/libs/linux_arm64 -lfaiss -lopenblas -lgfortran -lgomp -lpthread

// macOS
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/libs/darwin_amd64 -lfaiss -Wl,-framework,Accelerate
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/libs/darwin_arm64 -lfaiss -Wl,-framework,Accelerate

// Windows
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/libs/windows_amd64 -lfaiss -lopenblas -lgfortran -lquadmath -lpthread

#include <stdlib.h>
#include <stdint.h>

// Forward declarations for FAISS C API
#ifdef __cplusplus
extern "C" {
#endif

typedef void* FaissIndex;

// Same API as source build
extern int faiss_IndexFlatL2_new(FaissIndex* p_index, int64_t d);
extern int faiss_IndexFlatIP_new(FaissIndex* p_index, int64_t d);
extern int faiss_Index_add(FaissIndex index, int64_t n, const float* x);
extern int faiss_Index_search(FaissIndex index, int64_t n, const float* x, int64_t k, float* distances, int64_t* labels);
extern int faiss_Index_reset(FaissIndex index);
extern void faiss_Index_free(FaissIndex index);
extern int64_t faiss_Index_ntotal(FaissIndex index);
extern int faiss_Index_is_trained(FaissIndex index);
extern int faiss_Index_d(FaissIndex index);

#ifdef __cplusplus
}
#endif

*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// faissIndexFlatL2New creates a new IndexFlatL2
func faissIndexFlatL2New(d int) (uintptr, error) {
	var idx C.FaissIndex
	ret := C.faiss_IndexFlatL2_new(&idx, C.int64_t(d))
	if ret != 0 {
		return 0, fmt.Errorf("FAISS error code: %d", ret)
	}
	if idx == nil {
		return 0, errors.New("null index pointer")
	}
	return uintptr(unsafe.Pointer(idx)), nil
}

// faissIndexFlatIPNew creates a new IndexFlatIP
func faissIndexFlatIPNew(d int) (uintptr, error) {
	var idx C.FaissIndex
	ret := C.faiss_IndexFlatIP_new(&idx, C.int64_t(d))
	if ret != 0 {
		return 0, fmt.Errorf("FAISS error code: %d", ret)
	}
	if idx == nil {
		return 0, errors.New("null index pointer")
	}
	return uintptr(unsafe.Pointer(idx)), nil
}

// faissIndexAdd adds vectors to an index
func faissIndexAdd(ptr uintptr, vectors []float32, n int) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	vecPtr := (*C.float)(unsafe.Pointer(&vectors[0]))
	ret := C.faiss_Index_add(idx, C.int64_t(n), vecPtr)
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

// faissIndexSearch searches for nearest neighbors
func faissIndexSearch(ptr uintptr, queries []float32, nq, k int, distances []float32, indices []int64) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	queryPtr := (*C.float)(unsafe.Pointer(&queries[0]))
	distPtr := (*C.float)(unsafe.Pointer(&distances[0]))
	idxPtr := (*C.int64_t)(unsafe.Pointer(&indices[0]))

	ret := C.faiss_Index_search(idx, C.int64_t(nq), queryPtr, C.int64_t(k), distPtr, idxPtr)
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

// faissIndexReset resets an index
func faissIndexReset(ptr uintptr) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	ret := C.faiss_Index_reset(idx)
	if ret != 0 {
		return fmt.Errorf("FAISS error code: %d", ret)
	}
	return nil
}

// faissIndexFree frees an index
func faissIndexFree(ptr uintptr) error {
	idx := C.FaissIndex(unsafe.Pointer(ptr))
	C.faiss_Index_free(idx)
	return nil
}

// getBuildInfo returns build information for pre-built library build
func getBuildInfo() BuildInfo {
	return BuildInfo{
		Version:      Version,
		FAISSVersion: FAISSVersion,
		BuildMode:    "prebuilt",
		Compiler:     getPrebuiltCompiler(),
		Platform:     fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
		BLASBackend:  getPrebuiltBLAS(),
	}
}

func getPrebuiltCompiler() string {
	switch runtime.GOOS {
	case "darwin":
		return "Clang 15"
	case "linux":
		return "GCC 11"
	case "windows":
		return "MSVC 2022"
	default:
		return "unknown"
	}
}

func getPrebuiltBLAS() string {
	switch runtime.GOOS {
	case "darwin":
		return "Accelerate Framework"
	case "linux":
		return "OpenBLAS 0.3.21 (static)"
	case "windows":
		return "OpenBLAS 0.3.21 (static)"
	default:
		return "unknown"
	}
}
