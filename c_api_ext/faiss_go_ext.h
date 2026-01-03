/**
 * FAISS Go Extensions - Additional C API functions for faiss-go
 *
 * This header declares functions missing from the standard FAISS C API
 * that are needed for full faiss-go functionality.
 *
 * NOTE: Only includes functions that access fields/properties.
 * Constructor wrappers are excluded due to ABI compatibility issues
 * with different FAISS library versions.
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
 * Binary Index Constructor
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

/* ============================================================
 * HNSW Index Extensions (property accessors)
 * ============================================================ */

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

#ifdef __cplusplus
}
#endif

#endif /* FAISS_GO_EXT_H */
