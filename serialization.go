//nolint:goconst // Type identifiers match serialized format, not config values
package faiss

import (
	"fmt"
	"os"
	"runtime"
)

// WriteIndex saves an index to a file
//
// Python equivalent: faiss.write_index(index, filename)
//
// Example:
//   index, _ := faiss.NewIndexFlatL2(128)
//   index.Add(vectors)
//   faiss.WriteIndex(index, "my_index.faiss")
func WriteIndex(index Index, filename string) error {
	if index == nil {
		return fmt.Errorf("faiss: index cannot be nil")
	}

	var ptr uintptr
	switch idx := index.(type) {
	case *IndexFlat:
		ptr = idx.ptr
	case *IndexIVFFlat:
		ptr = idx.ptr
	case *IndexHNSW:
		ptr = idx.ptr
	case *IndexIDMap:
		ptr = idx.ptr
	default:
		return fmt.Errorf("faiss: unsupported index type for serialization")
	}

	if ptr == 0 {
		return ErrNullPointer
	}

	if err := faissWriteIndex(ptr, filename); err != nil {
		return fmt.Errorf("faiss: failed to write index to %s: %w", filename, err)
	}

	return nil
}

// ReadIndex loads an index from a file
//
// Python equivalent: faiss.read_index(filename)
//
// Example:
//   index, err := faiss.ReadIndex("my_index.faiss")
//   if err != nil {
//       log.Fatal(err)
//   }
//   defer index.Close()
func ReadIndex(filename string) (Index, error) {
	// Check if file exists
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		return nil, fmt.Errorf("faiss: index file not found: %s", filename)
	}

	ptr, indexType, d, metric, ntotal, err := faissReadIndex(filename)
	if err != nil {
		return nil, fmt.Errorf("faiss: failed to read index from %s: %w", filename, err)
	}

	// Create appropriate Go wrapper based on index type
	switch indexType {
	case "IndexFlatL2", "IndexFlatIP":
		idx := &IndexFlat{
			ptr:       ptr,
			d:         d,
			metric:    MetricType(metric),
			ntotal:    ntotal,
			isTrained: true,
		}
		runtime.SetFinalizer(idx, func(i *IndexFlat) {
			if i.ptr != 0 {
				_ = i.Close()
			}
		})
		return idx, nil

	case "IndexIVFFlat":
		idx := &IndexIVFFlat{
			ptr:       ptr,
			d:         d,
			metric:    MetricType(metric),
			ntotal:    ntotal,
			isTrained: true, // Loaded indexes are already trained
			nlist:     0,    // Will be populated from C
			nprobe:    1,
		}
		runtime.SetFinalizer(idx, func(i *IndexIVFFlat) {
			if i.ptr != 0 {
				_ = i.Close()
			}
		})
		return idx, nil

	case "IndexHNSWFlat":
		idx := &IndexHNSW{
			ptr:       ptr,
			d:         d,
			metric:    MetricType(metric),
			ntotal:    ntotal,
			isTrained: true,
			M:         0,  // Will be populated from C
			efSearch:  16,
		}
		runtime.SetFinalizer(idx, func(i *IndexHNSW) {
			if i.ptr != 0 {
				_ = i.Close()
			}
		})
		return idx, nil

	default:
		// Unknown index type, wrap as generic index
		return nil, fmt.Errorf("faiss: unsupported index type: %s", indexType)
	}
}

// SerializeIndex serializes an index to bytes
//
// Python equivalent: faiss.serialize_index(index)
//
// Example:
//   data, err := faiss.SerializeIndex(index)
//   // Send data over network, store in database, etc.
func SerializeIndex(index Index) ([]byte, error) {
	if index == nil {
		return nil, fmt.Errorf("faiss: index cannot be nil")
	}

	var ptr uintptr
	switch idx := index.(type) {
	case *IndexFlat:
		ptr = idx.ptr
	case *IndexIVFFlat:
		ptr = idx.ptr
	case *IndexHNSW:
		ptr = idx.ptr
	case *IndexIDMap:
		ptr = idx.ptr
	default:
		return nil, fmt.Errorf("faiss: unsupported index type for serialization")
	}

	if ptr == 0 {
		return nil, ErrNullPointer
	}

	data, err := faissSerializeIndex(ptr)
	if err != nil {
		return nil, fmt.Errorf("faiss: failed to serialize index: %w", err)
	}

	return data, nil
}

// DeserializeIndex deserializes an index from bytes
//
// Python equivalent: faiss.deserialize_index(data)
//
// Example:
//   // data received from network, database, etc.
//   index, err := faiss.DeserializeIndex(data)
//   if err != nil {
//       log.Fatal(err)
//   }
//   defer index.Close()
func DeserializeIndex(data []byte) (Index, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("faiss: empty data")
	}

	ptr, indexType, d, metric, ntotal, err := faissDeserializeIndex(data)
	if err != nil {
		return nil, fmt.Errorf("faiss: failed to deserialize index: %w", err)
	}

	// Create appropriate Go wrapper (same as ReadIndex)
	switch indexType {
	case "IndexFlatL2", "IndexFlatIP":
		idx := &IndexFlat{
			ptr:       ptr,
			d:         d,
			metric:    MetricType(metric),
			ntotal:    ntotal,
			isTrained: true,
		}
		runtime.SetFinalizer(idx, func(i *IndexFlat) {
			if i.ptr != 0 {
				_ = i.Close()
			}
		})
		return idx, nil

	case "IndexIVFFlat":
		idx := &IndexIVFFlat{
			ptr:       ptr,
			d:         d,
			metric:    MetricType(metric),
			ntotal:    ntotal,
			isTrained: true,
			nlist:     0,
			nprobe:    1,
		}
		runtime.SetFinalizer(idx, func(i *IndexIVFFlat) {
			if i.ptr != 0 {
				_ = i.Close()
			}
		})
		return idx, nil

	case "IndexHNSWFlat":
		idx := &IndexHNSW{
			ptr:       ptr,
			d:         d,
			metric:    MetricType(metric),
			ntotal:    ntotal,
			isTrained: true,
			M:         0,
			efSearch:  16,
		}
		runtime.SetFinalizer(idx, func(i *IndexHNSW) {
			if i.ptr != 0 {
				_ = i.Close()
			}
		})
		return idx, nil

	default:
		return nil, fmt.Errorf("faiss: unsupported index type: %s", indexType)
	}
}

// CloneIndex creates a deep copy of an index
//
// Python equivalent: faiss.clone_index(index)
func CloneIndex(index Index) (Index, error) {
	// Serialize and deserialize to create a deep copy
	data, err := SerializeIndex(index)
	if err != nil {
		return nil, err
	}

	return DeserializeIndex(data)
}
