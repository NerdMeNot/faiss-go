/**
 * FAISS Go Extensions - C++ Implementation
 *
 * This file implements additional C API functions for faiss-go
 * that are missing from the standard FAISS C API.
 *
 * NOTE: Only includes functions that access fields/properties.
 * Constructor wrappers are excluded due to ABI compatibility issues
 * with different FAISS library versions.
 *
 * Copyright (c) 2024 faiss-go contributors
 * Licensed under MIT License
 */

#include "faiss_go_ext.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/impl/AuxIndexStructures.h>

#include <cstring>
#include <exception>

// Error handling macro
#define CATCH_AND_HANDLE() \
    catch (const std::exception& e) { \
        return -1; \
    } \
    catch (...) { \
        return -1; \
    }

extern "C" {

/* ============================================================
 * Range Search Result Extensions
 * ============================================================ */

int faiss_RangeSearchResult_distances(
    FaissRangeSearchResult result,
    float** distances)
{
    try {
        auto* res = static_cast<faiss::RangeSearchResult*>(result);
        *distances = res->distances;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_RangeSearchResult_get(
    FaissRangeSearchResult result,
    int64_t** lims,
    int64_t** labels,
    float** distances)
{
    try {
        auto* res = static_cast<faiss::RangeSearchResult*>(result);
        *lims = reinterpret_cast<int64_t*>(res->lims);
        *labels = res->labels;
        *distances = res->distances;
        return 0;
    }
    CATCH_AND_HANDLE()
}

/* ============================================================
 * Binary Index - Simple constructor (no version-specific params)
 * ============================================================ */

int faiss_IndexBinaryFlat_new(
    FaissIndexBinary* p_index,
    int64_t d)
{
    try {
        *p_index = new faiss::IndexBinaryFlat(d);
        return 0;
    }
    CATCH_AND_HANDLE()
}

/* ============================================================
 * HNSW Index Extensions - Property accessors only
 * ============================================================ */

int faiss_IndexHNSW_set_efConstruction(FaissIndex index, int ef) {
    try {
        auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(static_cast<faiss::Index*>(index));
        if (!hnsw) return -1;
        hnsw->hnsw.efConstruction = ef;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexHNSW_set_efSearch(FaissIndex index, int ef) {
    try {
        auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(static_cast<faiss::Index*>(index));
        if (!hnsw) return -1;
        hnsw->hnsw.efSearch = ef;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexHNSW_get_efConstruction(FaissIndex index, int* ef) {
    try {
        auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(static_cast<faiss::Index*>(index));
        if (!hnsw) return -1;
        *ef = hnsw->hnsw.efConstruction;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexHNSW_get_efSearch(FaissIndex index, int* ef) {
    try {
        auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(static_cast<faiss::Index*>(index));
        if (!hnsw) return -1;
        *ef = hnsw->hnsw.efSearch;
        return 0;
    }
    CATCH_AND_HANDLE()
}

} // extern "C"
