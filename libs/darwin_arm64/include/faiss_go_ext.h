/**
 * FAISS Go Extensions - Additional C API functions for faiss-go
 *
 * This header declares functions missing from the standard FAISS C API
 * that are needed for full faiss-go functionality.
 *
 * Copyright (c) 2024 faiss-go contributors
 * Licensed under MIT License
 */

#ifndef FAISS_GO_EXT_H
#define FAISS_GO_EXT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations - these match FAISS C API types */
typedef void* FaissIndex;
typedef void* FaissIndexBinary;
typedef void* FaissRangeSearchResult;

/* ============================================================
 * Index Assign Extension
 * ============================================================ */

/**
 * Assign vectors to their nearest neighbors.
 * This is a custom wrapper that works reliably with all index types.
 *
 * @param index   The index pointer
 * @param n       Number of vectors
 * @param x       Input vectors (n * d floats)
 * @param labels  Output labels (n * k int64_t)
 * @param k       Number of nearest neighbors to find
 * @return 0 on success, -1 on error
 */
int faiss_Index_assign_ext(
    FaissIndex index,
    int64_t n,
    const float* x,
    int64_t* labels,
    int64_t k);

/* ============================================================
 * Range Search Result Extensions
 * ============================================================ */

/**
 * Get the distances array from a RangeSearchResult.
 *
 * @param result    The RangeSearchResult pointer
 * @param distances Output pointer to the distances array
 * @return 0 on success, -1 on error
 */
int faiss_RangeSearchResult_distances(
    FaissRangeSearchResult result,
    float** distances);

/**
 * Get all arrays from a RangeSearchResult at once.
 * This is more efficient than calling individual accessors.
 *
 * @param result    The RangeSearchResult pointer
 * @param lims      Output: array of size nq+1 with result offsets
 * @param labels    Output: array of result labels
 * @param distances Output: array of result distances
 * @return 0 on success, -1 on error
 */
int faiss_RangeSearchResult_get(
    FaissRangeSearchResult result,
    int64_t** lims,
    int64_t** labels,
    float** distances);

/* ============================================================
 * Binary Index Constructors
 * ============================================================ */

/**
 * Create a new IndexBinaryFlat.
 *
 * @param p_index Output pointer to the new index
 * @param d       Dimension of binary vectors (in bits, must be multiple of 8)
 * @return 0 on success, -1 on error
 */
int faiss_IndexBinaryFlat_new(
    FaissIndexBinary* p_index,
    int64_t d);

/**
 * Create a new IndexBinaryIVF.
 *
 * @param p_index   Output pointer to the new index
 * @param quantizer The quantizer index (must be a binary index)
 * @param d         Dimension of binary vectors (in bits)
 * @param nlist     Number of inverted lists
 * @return 0 on success, -1 on error
 */
int faiss_IndexBinaryIVF_new(
    FaissIndexBinary* p_index,
    FaissIndexBinary quantizer,
    int64_t d,
    int64_t nlist);

/**
 * Create a new IndexBinaryHash.
 *
 * @param p_index Output pointer to the new index
 * @param d       Dimension of binary vectors (in bits)
 * @param b       Number of hash bits
 * @return 0 on success, -1 on error
 */
int faiss_IndexBinaryHash_new(
    FaissIndexBinary* p_index,
    int64_t d,
    int64_t b);

/* ============================================================
 * Serialization Extensions
 * ============================================================ */

/**
 * Serialize an index to a byte buffer.
 *
 * @param index The index to serialize
 * @param data  Output: pointer to allocated buffer (caller must free)
 * @param size  Output: size of the buffer in bytes
 * @return 0 on success, -1 on error
 */
int faiss_serialize_index(
    FaissIndex index,
    uint8_t** data,
    size_t* size);

/**
 * Deserialize an index from a byte buffer.
 *
 * @param data       Input buffer containing serialized index
 * @param size       Size of the input buffer
 * @param p_index    Output: pointer to the deserialized index
 * @param index_type Output: string buffer for index type (min 64 chars)
 * @param d          Output: dimension of the index
 * @param metric     Output: metric type
 * @param ntotal     Output: number of vectors in the index
 * @return 0 on success, -1 on error
 */
int faiss_deserialize_index(
    const uint8_t* data,
    size_t size,
    FaissIndex* p_index,
    char* index_type,
    int* d,
    int* metric,
    int64_t* ntotal);

/**
 * Free a buffer allocated by faiss_serialize_index.
 *
 * @param data The buffer to free
 */
void faiss_serialize_free(uint8_t* data);

/* ============================================================
 * HNSW Index Extensions (for direct creation, not via factory)
 * ============================================================ */

/**
 * Create a new IndexHNSWFlat.
 *
 * @param p_index     Output pointer to the new index
 * @param d           Dimension of vectors
 * @param M           Number of connections per layer
 * @param metric_type Metric type (0=IP, 1=L2)
 * @return 0 on success, -1 on error
 */
int faiss_IndexHNSWFlat_new(
    FaissIndex* p_index,
    int64_t d,
    int M,
    int metric_type);

/**
 * Set efConstruction parameter for HNSW index.
 *
 * @param index The HNSW index
 * @param ef    The efConstruction value
 * @return 0 on success, -1 on error
 */
int faiss_IndexHNSW_set_efConstruction(FaissIndex index, int ef);

/**
 * Set efSearch parameter for HNSW index.
 *
 * @param index The HNSW index
 * @param ef    The efSearch value
 * @return 0 on success, -1 on error
 */
int faiss_IndexHNSW_set_efSearch(FaissIndex index, int ef);

/**
 * Get efConstruction parameter from HNSW index.
 *
 * @param index The HNSW index
 * @param ef    Output: the efConstruction value
 * @return 0 on success, -1 on error
 */
int faiss_IndexHNSW_get_efConstruction(FaissIndex index, int* ef);

/**
 * Get efSearch parameter from HNSW index.
 *
 * @param index The HNSW index
 * @param ef    Output: the efSearch value
 * @return 0 on success, -1 on error
 */
int faiss_IndexHNSW_get_efSearch(FaissIndex index, int* ef);

/* ============================================================
 * PQ Index Extensions
 * ============================================================ */

/**
 * Create a new IndexPQ.
 *
 * @param p_index     Output pointer to the new index
 * @param d           Dimension of vectors
 * @param M           Number of subquantizers
 * @param nbits       Number of bits per subquantizer code
 * @param metric_type Metric type (0=IP, 1=L2)
 * @return 0 on success, -1 on error
 */
int faiss_IndexPQ_new(
    FaissIndex* p_index,
    int64_t d,
    int64_t M,
    int64_t nbits,
    int metric_type);

/**
 * Create a new IndexIVFPQ.
 *
 * @param p_index   Output pointer to the new index
 * @param quantizer The quantizer index
 * @param d         Dimension of vectors
 * @param nlist     Number of inverted lists
 * @param M         Number of subquantizers
 * @param nbits     Number of bits per subquantizer code
 * @return 0 on success, -1 on error
 */
int faiss_IndexIVFPQ_new(
    FaissIndex* p_index,
    FaissIndex quantizer,
    int64_t d,
    int64_t nlist,
    int64_t M,
    int64_t nbits);

#ifdef __cplusplus
}
#endif

#endif /* FAISS_GO_EXT_H */
