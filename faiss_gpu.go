// +build !nogpu

package faiss

/*
#cgo LDFLAGS: -lfaiss -lfaiss_gpu -lcudart -lcublas
#cgo CFLAGS: -DFAISS_GPU

#include <stdlib.h>
#include <stdint.h>

// Forward declarations for FAISS GPU C API
typedef void* FaissStandardGpuResources;
typedef void* FaissGpuIndex;
typedef void* FaissIndex;

// StandardGpuResources functions
extern int faiss_StandardGpuResources_new(FaissStandardGpuResources* p_res);
extern void faiss_StandardGpuResources_free(FaissStandardGpuResources res);
extern int faiss_StandardGpuResources_setTempMemory(FaissStandardGpuResources res, size_t size);
extern int faiss_StandardGpuResources_setDefaultNullStreamAllDevices(FaissStandardGpuResources res);

// GPU index creation functions
extern int faiss_GpuIndexFlat_new(FaissGpuIndex* p_index, FaissStandardGpuResources res, int device, int64_t d, int metric_type);
extern int faiss_GpuIndexIVFFlat_new(FaissGpuIndex* p_index, FaissStandardGpuResources res, int device, FaissIndex coarse_quantizer, int64_t d, int64_t nlist, int metric_type);

// CPU <-> GPU conversion functions
extern int faiss_index_cpu_to_gpu(FaissStandardGpuResources res, int device, FaissIndex cpu_index, FaissGpuIndex* p_gpu_index);
extern int faiss_index_gpu_to_cpu(FaissGpuIndex gpu_index, FaissIndex* p_cpu_index);
extern int faiss_index_cpu_to_all_gpus(FaissStandardGpuResources res, FaissIndex cpu_index, FaissGpuIndex* p_gpu_index);

// GPU utility functions
extern int faiss_get_num_gpus(int* num_gpus);

*/
import "C"
import (
	"fmt"
	"unsafe"
)

// ========================================
// StandardGpuResources Wrapper Functions
// ========================================

func faiss_StandardGpuResources_new(p_res *uintptr) error {
	var res C.FaissStandardGpuResources
	ret := C.faiss_StandardGpuResources_new(&res)
	if ret != 0 {
		return fmt.Errorf("failed to create GPU resources")
	}
	*p_res = uintptr(unsafe.Pointer(res))
	return nil
}

func faiss_StandardGpuResources_free(res uintptr) {
	r := C.FaissStandardGpuResources(unsafe.Pointer(res))
	C.faiss_StandardGpuResources_free(r)
}

func faiss_StandardGpuResources_setTempMemory(res uintptr, size int64) error {
	r := C.FaissStandardGpuResources(unsafe.Pointer(res))
	ret := C.faiss_StandardGpuResources_setTempMemory(r, C.size_t(size))
	if ret != 0 {
		return fmt.Errorf("failed to set temp memory")
	}
	return nil
}

func faiss_StandardGpuResources_setDefaultNullStreamAllDevices(res uintptr) error {
	r := C.FaissStandardGpuResources(unsafe.Pointer(res))
	ret := C.faiss_StandardGpuResources_setDefaultNullStreamAllDevices(r)
	if ret != 0 {
		return fmt.Errorf("failed to set default null stream")
	}
	return nil
}

// ========================================
// GPU Index Creation Wrapper Functions
// ========================================

func faiss_GpuIndexFlat_new(p_index *uintptr, res uintptr, device int, d int64, metric_type int) error {
	var idx C.FaissGpuIndex
	r := C.FaissStandardGpuResources(unsafe.Pointer(res))
	ret := C.faiss_GpuIndexFlat_new(&idx, r, C.int(device), C.int64_t(d), C.int(metric_type))
	if ret != 0 {
		return fmt.Errorf("failed to create GpuIndexFlat")
	}
	*p_index = uintptr(unsafe.Pointer(idx))
	return nil
}

func faiss_GpuIndexIVFFlat_new(p_index *uintptr, res uintptr, device int, coarse_quantizer uintptr, d, nlist int64, metric_type int) error {
	var idx C.FaissGpuIndex
	r := C.FaissStandardGpuResources(unsafe.Pointer(res))
	q := C.FaissIndex(unsafe.Pointer(coarse_quantizer))
	ret := C.faiss_GpuIndexIVFFlat_new(&idx, r, C.int(device), q, C.int64_t(d), C.int64_t(nlist), C.int(metric_type))
	if ret != 0 {
		return fmt.Errorf("failed to create GpuIndexIVFFlat")
	}
	*p_index = uintptr(unsafe.Pointer(idx))
	return nil
}

// ========================================
// CPU <-> GPU Conversion Wrapper Functions
// ========================================

func faiss_index_cpu_to_gpu(res uintptr, device int, cpu_index uintptr, p_gpu_index *uintptr) error {
	var gpu_idx C.FaissGpuIndex
	r := C.FaissStandardGpuResources(unsafe.Pointer(res))
	cpu_idx := C.FaissIndex(unsafe.Pointer(cpu_index))

	ret := C.faiss_index_cpu_to_gpu(r, C.int(device), cpu_idx, &gpu_idx)
	if ret != 0 {
		return fmt.Errorf("failed to transfer index to GPU")
	}
	*p_gpu_index = uintptr(unsafe.Pointer(gpu_idx))
	return nil
}

func faiss_index_gpu_to_cpu(gpu_index uintptr, p_cpu_index *uintptr) error {
	var cpu_idx C.FaissIndex
	gpu_idx := C.FaissGpuIndex(unsafe.Pointer(gpu_index))

	ret := C.faiss_index_gpu_to_cpu(gpu_idx, &cpu_idx)
	if ret != 0 {
		return fmt.Errorf("failed to transfer index to CPU")
	}
	*p_cpu_index = uintptr(unsafe.Pointer(cpu_idx))
	return nil
}

func faiss_index_cpu_to_all_gpus(res uintptr, cpu_index uintptr, p_gpu_index *uintptr) error {
	var gpu_idx C.FaissGpuIndex
	r := C.FaissStandardGpuResources(unsafe.Pointer(res))
	cpu_idx := C.FaissIndex(unsafe.Pointer(cpu_index))

	ret := C.faiss_index_cpu_to_all_gpus(r, cpu_idx, &gpu_idx)
	if ret != 0 {
		return fmt.Errorf("failed to transfer index to all GPUs")
	}
	*p_gpu_index = uintptr(unsafe.Pointer(gpu_idx))
	return nil
}

// ========================================
// GPU Utility Wrapper Functions
// ========================================

func faiss_get_num_gpus() (int, error) {
	var num_gpus C.int
	ret := C.faiss_get_num_gpus(&num_gpus)
	if ret != 0 {
		return 0, fmt.Errorf("failed to get number of GPUs")
	}
	return int(num_gpus), nil
}
