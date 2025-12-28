/**
 * FAISS C API - Amalgamated Header
 * Auto-generated from FAISS C API
 */

#ifndef FAISS_AMALGAMATION_H
#define FAISS_AMALGAMATION_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct FaissIndex_H FaissIndex;
typedef struct FaissIndexBinary_H FaissIndexBinary;

// Index creation
int faiss_IndexFlatL2_new(FaissIndex** p_index, int64_t d);
int faiss_IndexFlatIP_new(FaissIndex** p_index, int64_t d);

// Index operations
int faiss_Index_add(FaissIndex* index, int64_t n, const float* x);
int faiss_Index_search(FaissIndex* index, int64_t n, const float* x, int64_t k, float* distances, int64_t* labels);
int faiss_Index_train(FaissIndex* index, int64_t n, const float* x);
int faiss_Index_reset(FaissIndex* index);
void faiss_Index_free(FaissIndex* index);

// Index properties
int64_t faiss_Index_ntotal(FaissIndex* index);
int faiss_Index_d(FaissIndex* index);
int faiss_Index_is_trained(FaissIndex* index);

// More comprehensive API - to be expanded

#ifdef __cplusplus
}
#endif

#endif /* FAISS_AMALGAMATION_H */
