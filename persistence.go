package faiss

/*
#include <stdlib.h>
#include <stdint.h>

// Forward declarations for FAISS C API
typedef void* FaissIndex;
typedef int64_t idx_t;

// Index I/O functions
int faiss_write_index_fname(const FaissIndex* idx, const char* fname);
int faiss_read_index_fname(const char* fname, int io_flags, FaissIndex** p_out);

// Index property getters (take FaissIndex by value, which is void*)
int faiss_Index_d(FaissIndex index);
idx_t faiss_Index_ntotal(FaissIndex index);
int faiss_Index_is_trained(FaissIndex index);
int faiss_Index_metric_type(FaissIndex index);
*/
import "C"
import (
	"fmt"
	"os"
	"runtime"
	"unsafe"
)

// WriteIndexToFile saves the index to a file
//
// Python equivalent: faiss.write_index(index, filename)
//
// Example:
//
//	index, _ := faiss.NewIndexFlatL2(128)
//	index.Add(vectors)
//	faiss.WriteIndexToFile(index, "my_index.faiss")
func WriteIndexToFile(index Index, filename string) error {
	if index == nil {
		return fmt.Errorf("faiss: index cannot be nil")
	}

	var ptr uintptr

	// Extract pointer from any Index type
	switch idx := index.(type) {
	case *IndexFlat:
		ptr = idx.ptr
	case *IndexIVFFlat:
		ptr = idx.ptr
	case *IndexLSH:
		ptr = idx.ptr
	case *IndexScalarQuantizer:
		ptr = idx.ptr
	case *IndexIVFScalarQuantizer:
		ptr = idx.ptr
	case *IndexIDMap:
		ptr = idx.ptr
	case *IndexRefine:
		ptr = idx.ptr
	case *IndexPreTransform:
		ptr = idx.ptr
	case *IndexShards:
		ptr = idx.ptr
	case *GenericIndex:
		ptr = idx.ptr
	default:
		return fmt.Errorf("faiss: unsupported index type for serialization: %T", index)
	}

	if ptr == 0 {
		return fmt.Errorf("faiss: index pointer is null")
	}

	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	// Pass pointer to FaissIndex
	idxPtr := (*C.FaissIndex)(unsafe.Pointer(ptr))
	ret := C.faiss_write_index_fname(idxPtr, cFilename)
	if ret != 0 {
		return fmt.Errorf("faiss: failed to write index to %s (error code: %d)", filename, ret)
	}

	return nil
}

// ReadIndexFromFile loads an index from a file
//
// Python equivalent: faiss.read_index(filename)
//
// Example:
//
//	index, err := faiss.ReadIndexFromFile("my_index.faiss")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer index.Close()
func ReadIndexFromFile(filename string) (Index, error) {
	// Check if file exists
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		return nil, fmt.Errorf("faiss: index file not found: %s", filename)
	}

	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	var idx *C.FaissIndex
	// io_flags = 0 means no special flags (no mmap, not read-only)
	ret := C.faiss_read_index_fname(cFilename, 0, &idx)
	if ret != 0 {
		return nil, fmt.Errorf("faiss: failed to read index from %s (error code: %d)", filename, ret)
	}

	ptr := uintptr(unsafe.Pointer(idx))
	if ptr == 0 {
		return nil, fmt.Errorf("faiss: loaded index pointer is null")
	}

	// Get index properties using C functions
	// Convert *FaissIndex to FaissIndex (void*) for value-type functions
	idxVal := C.FaissIndex(idx)
	d := int(C.faiss_Index_d(idxVal))
	ntotal := int64(C.faiss_Index_ntotal(idxVal))
	isTrained := int(C.faiss_Index_is_trained(idxVal)) != 0
	metricType := MetricType(C.faiss_Index_metric_type(idxVal))

	// Wrap in GenericIndex since we don't know the specific type
	genericIdx := &GenericIndex{
		ptr:       ptr,
		d:         d,
		ntotal:    ntotal,
		isTrained: isTrained,
		metric:    metricType,
	}

	runtime.SetFinalizer(genericIdx, func(i *GenericIndex) {
		if i.ptr != 0 {
			_ = i.Close()
		}
	})

	return genericIdx, nil
}
