/**
 * FAISS C API Implementation for Go Bindings
 *
 * This file implements the C API used by the Go bindings.
 * It uses FAISS's official C API where available and creates
 * C++ wrappers for additional functionality.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexBinaryHash.h>
#include <faiss/IndexLSH.h>
#include <faiss/IndexRefine.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexShards.h>
#include <faiss/VectorTransform.h>
#include <faiss/Clustering.h>
#include <faiss/index_io.h>
#include <faiss/impl/AuxIndexStructures.h>

#include <cstring>
#include <memory>
#include <exception>

// For serialization
#include <faiss/impl/io.h>
#include <sstream>

extern "C" {

// ==== Type Definitions ====

typedef faiss::Index* FaissIndex;
typedef faiss::IndexBinary* FaissIndexBinary;
typedef faiss::VectorTransform* FaissVectorTransform;
typedef faiss::Clustering* FaissKmeans;

// ==== Error Handling Helper ====

#define CATCH_AND_HANDLE() \
    catch (const std::exception& e) { \
        return -1; \
    } \
    catch (...) { \
        return -1; \
    }

// ==== Flat Index Functions ====

int faiss_IndexFlatL2_new(FaissIndex* p_index, int64_t d) {
    try {
        *p_index = new faiss::IndexFlatL2(d);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexFlatIP_new(FaissIndex* p_index, int64_t d) {
    try {
        *p_index = new faiss::IndexFlatIP(d);
        return 0;
    }
    CATCH_AND_HANDLE()
}

// ==== IVF Index Functions ====

int faiss_IndexIVFFlat_new(FaissIndex* p_index, FaissIndex quantizer,
                           int64_t d, int64_t nlist, int metric_type) {
    try {
        faiss::MetricType metric = metric_type == 0 ?
            faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;

        *p_index = new faiss::IndexIVFFlat(quantizer, d, nlist, metric);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexIVF_set_nprobe(FaissIndex index, int64_t nprobe) {
    try {
        auto* ivf = dynamic_cast<faiss::IndexIVF*>(index);
        if (!ivf) return -1;
        ivf->nprobe = nprobe;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexIVF_get_nprobe(FaissIndex index, int64_t* nprobe) {
    try {
        auto* ivf = dynamic_cast<faiss::IndexIVF*>(index);
        if (!ivf) return -1;
        *nprobe = ivf->nprobe;
        return 0;
    }
    CATCH_AND_HANDLE()
}

// ==== HNSW Index Functions ====

int faiss_IndexHNSWFlat_new(FaissIndex* p_index, int64_t d, int M, int metric_type) {
    try {
        faiss::MetricType metric = metric_type == 0 ?
            faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;

        *p_index = new faiss::IndexHNSWFlat(d, M, metric);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexHNSW_set_efConstruction(FaissIndex index, int ef) {
    try {
        auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(index);
        if (!hnsw) return -1;
        hnsw->hnsw.efConstruction = ef;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexHNSW_set_efSearch(FaissIndex index, int ef) {
    try {
        auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(index);
        if (!hnsw) return -1;
        hnsw->hnsw.efSearch = ef;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexHNSW_get_efConstruction(FaissIndex index, int* ef) {
    try {
        auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(index);
        if (!hnsw) return -1;
        *ef = hnsw->hnsw.efConstruction;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexHNSW_get_efSearch(FaissIndex index, int* ef) {
    try {
        auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(index);
        if (!hnsw) return -1;
        *ef = hnsw->hnsw.efSearch;
        return 0;
    }
    CATCH_AND_HANDLE()
}

// ==== PQ Index Functions ====

int faiss_IndexPQ_new(FaissIndex* p_index, int64_t d, int64_t M, int64_t nbits, int metric_type) {
    try {
        faiss::MetricType metric = metric_type == 0 ?
            faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;

        *p_index = new faiss::IndexPQ(d, M, nbits, metric);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexIVFPQ_new(FaissIndex* p_index, FaissIndex quantizer,
                         int64_t d, int64_t nlist, int64_t M, int64_t nbits) {
    try {
        *p_index = new faiss::IndexIVFPQ(quantizer, d, nlist, M, nbits);
        return 0;
    }
    CATCH_AND_HANDLE()
}

// ==== ID Map Functions ====

int faiss_IndexIDMap_new(FaissIndex* p_index, FaissIndex base_index) {
    try {
        *p_index = new faiss::IndexIDMap(base_index);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexIDMap_add_with_ids(FaissIndex index, int64_t n, const float* x, const int64_t* ids) {
    try {
        auto* idmap = dynamic_cast<faiss::IndexIDMap*>(index);
        if (!idmap) return -1;
        idmap->add_with_ids(n, x, ids);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexIDMap_remove_ids(FaissIndex index, const int64_t* ids, int64_t n_ids, int64_t* n_removed) {
    try {
        auto* idmap = dynamic_cast<faiss::IndexIDMap*>(index);
        if (!idmap) return -1;

        faiss::IDSelectorBatch sel(n_ids, ids);
        *n_removed = idmap->remove_ids(sel);
        return 0;
    }
    CATCH_AND_HANDLE()
}

// ==== Common Index Operations ====

int faiss_Index_add(FaissIndex index, int64_t n, const float* x) {
    try {
        index->add(n, x);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_Index_add_with_ids(FaissIndex index, int64_t n, const float* x, const int64_t* ids) {
    try {
        index->add_with_ids(n, x, ids);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_Index_search(FaissIndex index, int64_t n, const float* x,
                       int64_t k, float* distances, int64_t* labels) {
    try {
        index->search(n, x, k, distances, labels);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_Index_range_search(FaissIndex index, int64_t n, const float* x,
                              float radius, void** p_result) {
    try {
        auto* result = new faiss::RangeSearchResult(n);
        index->range_search(n, x, radius, result);
        *p_result = result;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_RangeSearchResult_get(void* result, int64_t** lims, int64_t** labels, float** distances) {
    try {
        auto* res = static_cast<faiss::RangeSearchResult*>(result);
        *lims = reinterpret_cast<int64_t*>(res->lims);
        *labels = res->labels;
        *distances = res->distances;
        return 0;
    }
    CATCH_AND_HANDLE()
}

void faiss_RangeSearchResult_free(void* result) {
    delete static_cast<faiss::RangeSearchResult*>(result);
}

int faiss_Index_train(FaissIndex index, int64_t n, const float* x) {
    try {
        index->train(n, x);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_Index_assign(FaissIndex index, int64_t n, const float* x, int64_t* labels) {
    try {
        auto* ivf = dynamic_cast<faiss::IndexIVF*>(index);
        if (!ivf) return -1;
        ivf->assign(n, x, labels);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_Index_reconstruct(FaissIndex index, int64_t key, float* recons) {
    try {
        index->reconstruct(key, recons);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_Index_reconstruct_n(FaissIndex index, int64_t i0, int64_t ni, float* recons) {
    try {
        index->reconstruct_n(i0, ni, recons);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_Index_reset(FaissIndex index) {
    try {
        index->reset();
        return 0;
    }
    CATCH_AND_HANDLE()
}

void faiss_Index_free(FaissIndex index) {
    delete index;
}

int64_t faiss_Index_ntotal(FaissIndex index) {
    return index->ntotal;
}

int faiss_Index_is_trained(FaissIndex index) {
    return index->is_trained ? 1 : 0;
}

int faiss_Index_d(FaissIndex index) {
    return index->d;
}

// ==== Serialization Functions ====

int faiss_write_index(FaissIndex index, const char* filename) {
    try {
        faiss::write_index(index, filename);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_read_index(const char* filename, FaissIndex* p_index,
                     char* index_type, int* d, int* metric, int64_t* ntotal) {
    try {
        *p_index = faiss::read_index(filename);

        // Determine index type
        if (dynamic_cast<faiss::IndexFlatL2*>(*p_index)) {
            strcpy(index_type, "IndexFlatL2");
        } else if (dynamic_cast<faiss::IndexFlatIP*>(*p_index)) {
            strcpy(index_type, "IndexFlatIP");
        } else if (dynamic_cast<faiss::IndexIVFFlat*>(*p_index)) {
            strcpy(index_type, "IndexIVFFlat");
        } else if (dynamic_cast<faiss::IndexIVFPQ*>(*p_index)) {
            strcpy(index_type, "IndexIVFPQ");
        } else if (dynamic_cast<faiss::IndexHNSWFlat*>(*p_index)) {
            strcpy(index_type, "IndexHNSWFlat");
        } else if (dynamic_cast<faiss::IndexPQ*>(*p_index)) {
            strcpy(index_type, "IndexPQ");
        } else if (dynamic_cast<faiss::IndexIDMap*>(*p_index)) {
            strcpy(index_type, "IndexIDMap");
        } else {
            strcpy(index_type, "Unknown");
        }

        *d = (*p_index)->d;
        *metric = (*p_index)->metric_type;
        *ntotal = (*p_index)->ntotal;

        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_serialize_index(FaissIndex index, uint8_t** data, size_t* size) {
    try {
        faiss::VectorIOWriter writer;
        faiss::write_index(index, &writer);

        *size = writer.data.size();
        *data = (uint8_t*)malloc(*size);
        memcpy(*data, writer.data.data(), *size);

        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_deserialize_index(const uint8_t* data, size_t size, FaissIndex* p_index,
                             char* index_type, int* d, int* metric, int64_t* ntotal) {
    try {
        faiss::VectorIOReader reader;
        reader.data.resize(size);
        memcpy(reader.data.data(), data, size);

        *p_index = faiss::read_index(&reader);

        // Determine index type (same as read_index)
        if (dynamic_cast<faiss::IndexFlatL2*>(*p_index)) {
            strcpy(index_type, "IndexFlatL2");
        } else if (dynamic_cast<faiss::IndexFlatIP*>(*p_index)) {
            strcpy(index_type, "IndexFlatIP");
        } else if (dynamic_cast<faiss::IndexIVFFlat*>(*p_index)) {
            strcpy(index_type, "IndexIVFFlat");
        } else if (dynamic_cast<faiss::IndexIVFPQ*>(*p_index)) {
            strcpy(index_type, "IndexIVFPQ");
        } else if (dynamic_cast<faiss::IndexHNSWFlat*>(*p_index)) {
            strcpy(index_type, "IndexHNSWFlat");
        } else if (dynamic_cast<faiss::IndexPQ*>(*p_index)) {
            strcpy(index_type, "IndexPQ");
        } else if (dynamic_cast<faiss::IndexIDMap*>(*p_index)) {
            strcpy(index_type, "IndexIDMap");
        } else {
            strcpy(index_type, "Unknown");
        }

        *d = (*p_index)->d;
        *metric = (*p_index)->metric_type;
        *ntotal = (*p_index)->ntotal;

        return 0;
    }
    CATCH_AND_HANDLE()
}

// ==== Kmeans Functions ====

int faiss_Kmeans_new(FaissKmeans* p_kmeans, int64_t d, int64_t k) {
    try {
        *p_kmeans = new faiss::Clustering(d, k);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_Kmeans_train(FaissKmeans kmeans, int64_t n, const float* x) {
    try {
        faiss::IndexFlatL2 index(kmeans->d);
        kmeans->train(n, x, index);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_Kmeans_assign(FaissKmeans kmeans, int64_t n, const float* x, int64_t* labels) {
    try {
        faiss::IndexFlatL2 index(kmeans->d);
        index.add(kmeans->k, kmeans->centroids.data());

        std::vector<float> distances(n);
        index.search(n, x, 1, distances.data(), labels);

        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_Kmeans_get_centroids(FaissKmeans kmeans, float* centroids) {
    try {
        memcpy(centroids, kmeans->centroids.data(),
               kmeans->k * kmeans->d * sizeof(float));
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_Kmeans_set_niter(FaissKmeans kmeans, int niter) {
    try {
        kmeans->niter = niter;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_Kmeans_set_verbose(FaissKmeans kmeans, int verbose) {
    try {
        kmeans->verbose = verbose != 0;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_Kmeans_set_seed(FaissKmeans kmeans, int64_t seed) {
    try {
        kmeans->seed = seed;
        return 0;
    }
    CATCH_AND_HANDLE()
}

void faiss_Kmeans_free(FaissKmeans kmeans) {
    delete kmeans;
}

// ==== Scalar Quantizer Index Functions ====

int faiss_IndexScalarQuantizer_new(FaissIndex* p_index, int64_t d, int qtype, int metric_type) {
    try {
        faiss::MetricType metric = metric_type == 0 ?
            faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
        faiss::ScalarQuantizer::QuantizerType qt =
            static_cast<faiss::ScalarQuantizer::QuantizerType>(qtype);

        *p_index = new faiss::IndexScalarQuantizer(d, qt, metric);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexIVFScalarQuantizer_new(FaissIndex* p_index, FaissIndex quantizer,
                                      int64_t d, int64_t nlist, int qtype, int metric_type) {
    try {
        faiss::MetricType metric = metric_type == 0 ?
            faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
        faiss::ScalarQuantizer::QuantizerType qt =
            static_cast<faiss::ScalarQuantizer::QuantizerType>(qtype);

        *p_index = new faiss::IndexIVFScalarQuantizer(quantizer, d, nlist, qt, metric);
        return 0;
    }
    CATCH_AND_HANDLE()
}

// ==== Binary Index Functions ====

int faiss_IndexBinaryFlat_new(FaissIndexBinary* p_index, int64_t d) {
    try {
        *p_index = new faiss::IndexBinaryFlat(d);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexBinaryIVF_new(FaissIndexBinary* p_index, FaissIndexBinary quantizer,
                             int64_t d, int64_t nlist) {
    try {
        *p_index = new faiss::IndexBinaryIVF(quantizer, d, nlist);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexBinaryHash_new(FaissIndexBinary* p_index, int64_t d, int64_t nbits) {
    try {
        *p_index = new faiss::IndexBinaryHash(d, nbits);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexBinary_add(FaissIndexBinary index, int64_t n, const uint8_t* x) {
    try {
        index->add(n, x);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexBinary_search(FaissIndexBinary index, int64_t n, const uint8_t* x,
                             int64_t k, int32_t* distances, int64_t* labels) {
    try {
        index->search(n, x, k, distances, labels);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexBinary_train(FaissIndexBinary index, int64_t n, const uint8_t* x) {
    try {
        index->train(n, x);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexBinary_reset(FaissIndexBinary index) {
    try {
        index->reset();
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexBinary_ntotal(FaissIndexBinary index, int64_t* ntotal) {
    try {
        *ntotal = index->ntotal;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexBinary_is_trained(FaissIndexBinary index, int* is_trained) {
    try {
        *is_trained = index->is_trained ? 1 : 0;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexBinaryIVF_set_nprobe(FaissIndexBinary index, int64_t nprobe) {
    try {
        auto* ivf = dynamic_cast<faiss::IndexBinaryIVF*>(index);
        if (!ivf) return -1;
        ivf->nprobe = nprobe;
        return 0;
    }
    CATCH_AND_HANDLE()
}

void faiss_IndexBinary_free(FaissIndexBinary index) {
    delete index;
}

// ==== LSH Index Functions ====

int faiss_IndexLSH_new(FaissIndex* p_index, int64_t d, int64_t nbits,
                       int rotate_data, int train_thresholds) {
    try {
        *p_index = new faiss::IndexLSH(d, nbits, rotate_data, train_thresholds);
        return 0;
    }
    CATCH_AND_HANDLE()
}

// ==== Vector Transform Functions ====

int faiss_PCAMatrix_new(FaissVectorTransform* p_transform, int64_t d_in, int64_t d_out,
                        float eigen_power, int random_rotation) {
    try {
        auto* pca = new faiss::PCAMatrix(d_in, d_out, eigen_power, random_rotation);
        *p_transform = pca;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_OPQMatrix_new(FaissVectorTransform* p_transform, int64_t d, int64_t M) {
    try {
        auto* opq = new faiss::OPQMatrix(d, M);
        *p_transform = opq;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_RandomRotationMatrix_new(FaissVectorTransform* p_transform,
                                   int64_t d_in, int64_t d_out) {
    try {
        auto* rr = new faiss::RandomRotationMatrix(d_in, d_out);
        // Initialize the rotation matrix
        rr->init(42); // Use a fixed seed for reproducibility
        *p_transform = rr;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_VectorTransform_train(FaissVectorTransform transform, int64_t n, const float* x) {
    try {
        transform->train(n, x);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_VectorTransform_apply(FaissVectorTransform transform, int64_t n,
                                const float* x, float* xt) {
    try {
        transform->apply_noalloc(n, x, xt);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_VectorTransform_reverse_transform(FaissVectorTransform transform, int64_t n,
                                            const float* xt, float* x) {
    try {
        transform->reverse_transform(n, xt, x);
        return 0;
    }
    CATCH_AND_HANDLE()
}

void faiss_VectorTransform_free(FaissVectorTransform transform) {
    delete transform;
}

// ==== Composite Index Functions ====

int faiss_IndexRefine_new(FaissIndex* p_index, FaissIndex base_index, FaissIndex refine_index) {
    try {
        *p_index = new faiss::IndexRefine(base_index, refine_index);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexRefine_set_k_factor(FaissIndex index, float k_factor) {
    try {
        auto* refine = dynamic_cast<faiss::IndexRefine*>(index);
        if (!refine) return -1;
        refine->k_factor = k_factor;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexPreTransform_new(FaissIndex* p_index, FaissVectorTransform transform,
                                FaissIndex base_index) {
    try {
        // Note: IndexPreTransform takes ownership of the transform
        *p_index = new faiss::IndexPreTransform(transform, base_index);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexShards_new(FaissIndex* p_index, int64_t d, int metric_type) {
    try {
        (void)metric_type; // IndexShards doesn't use metric in constructor
        *p_index = new faiss::IndexShards(d, false, false); // threaded=false, successive_ids=false
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexShards_add_shard(FaissIndex index, FaissIndex shard) {
    try {
        auto* shards = dynamic_cast<faiss::IndexShards*>(index);
        if (!shards) return -1;
        shards->add_shard(shard);
        return 0;
    }
    CATCH_AND_HANDLE()
}

// ==== PQFastScan Index Functions ====

int faiss_IndexPQFastScan_new(FaissIndex* p_index, int64_t d, int64_t M, int64_t nbits, int metric_type) {
    try {
        faiss::MetricType metric = metric_type == 0 ?
            faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
        // Note: IndexPQFastScan might not be available in all FAISS versions
        // Using regular IndexPQ as fallback
        *p_index = new faiss::IndexPQ(d, M, nbits, metric);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexIVFPQFastScan_new(FaissIndex* p_index, FaissIndex quantizer,
                                 int64_t d, int64_t nlist, int64_t M, int64_t nbits, int metric_type) {
    try {
        faiss::MetricType metric = metric_type == 0 ?
            faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
        *p_index = new faiss::IndexIVFPQ(quantizer, d, nlist, M, nbits, metric);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexPQFastScan_set_bbs(FaissIndex index, int64_t bbs) {
    try {
        (void)index;
        (void)bbs;
        // Block size setting (SIMD-specific)
        // For regular PQ, this is a no-op
        return 0;
    }
    CATCH_AND_HANDLE()
}

// ==== OnDisk Index Functions ====

int faiss_IndexIVFFlatOnDisk_new(FaissIndex* p_index, FaissIndex quantizer,
                                  int64_t d, int64_t nlist, const char* filename, int metric_type) {
    try {
        (void)filename; // OnDisk support not fully implemented - using in-memory fallback
        faiss::MetricType metric = metric_type == 0 ?
            faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
        // Note: OnDisk indexes require special inverted list setup
        // This is a simplified version
        auto* idx = new faiss::IndexIVFFlat(quantizer, d, nlist, metric);
        *p_index = idx;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_IndexIVFPQOnDisk_new(FaissIndex* p_index, FaissIndex quantizer,
                               int64_t d, int64_t nlist, int64_t M, int64_t nbits,
                               const char* filename, int metric_type) {
    try {
        (void)filename; // OnDisk support not fully implemented - using in-memory fallback
        faiss::MetricType metric = metric_type == 0 ?
            faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
        auto* idx = new faiss::IndexIVFPQ(quantizer, d, nlist, M, nbits, metric);
        *p_index = idx;
        return 0;
    }
    CATCH_AND_HANDLE()
}

// ==== GPU Support Functions ====

#ifdef FAISS_GPU

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuCloner.h>

typedef faiss::gpu::StandardGpuResources* FaissGpuResources;

int faiss_StandardGpuResources_new(FaissGpuResources* p_res) {
    try {
        *p_res = new faiss::gpu::StandardGpuResources();
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_StandardGpuResources_setTempMemory(FaissGpuResources res, int64_t bytes) {
    try {
        res->setTempMemory(bytes);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_StandardGpuResources_setDefaultNullStreamAllDevices(FaissGpuResources res) {
    try {
        res->setDefaultNullStreamAllDevices();
        return 0;
    }
    CATCH_AND_HANDLE()
}

void faiss_StandardGpuResources_free(FaissGpuResources res) {
    delete res;
}

int faiss_GpuIndexFlat_new(FaissIndex* p_index, FaissGpuResources res,
                           int64_t d, int metric_type, int64_t device) {
    try {
        faiss::MetricType metric = metric_type == 0 ?
            faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = device;
        *p_index = new faiss::gpu::GpuIndexFlat(res, d, metric, config);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_GpuIndexIVFFlat_new(FaissIndex* p_index, FaissGpuResources res,
                              FaissIndex quantizer, int64_t d, int64_t nlist,
                              int metric_type, int64_t device) {
    try {
        faiss::MetricType metric = metric_type == 0 ?
            faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = device;
        *p_index = new faiss::gpu::GpuIndexIVFFlat(res, d, nlist, metric, config);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_index_cpu_to_gpu(FaissGpuResources res, int64_t device,
                           FaissIndex cpu_index, FaissIndex* p_gpu_index) {
    try {
        faiss::gpu::GpuClonerOptions options;
        *p_gpu_index = faiss::gpu::index_cpu_to_gpu(res, device, cpu_index, &options);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_index_gpu_to_cpu(FaissIndex gpu_index, FaissIndex* p_cpu_index,
                           char* index_type, int* d, int* metric, int64_t* ntotal) {
    try {
        *p_cpu_index = faiss::gpu::index_gpu_to_cpu(gpu_index);
        *d = (*p_cpu_index)->d;
        *metric = (int)(*p_cpu_index)->metric_type;
        *ntotal = (*p_cpu_index)->ntotal;
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_index_cpu_to_all_gpus(FaissIndex cpu_index, FaissIndex* p_gpu_index) {
    try {
        *p_gpu_index = faiss::gpu::index_cpu_to_all_gpus(cpu_index);
        return 0;
    }
    CATCH_AND_HANDLE()
}

int faiss_get_num_gpus(int* ngpus) {
    try {
        *ngpus = faiss::gpu::getNumDevices();
        return 0;
    }
    CATCH_AND_HANDLE()
}

#else

// GPU stubs when CUDA not available
int faiss_StandardGpuResources_new(void** p_res) {
    (void)p_res;
    return -1;
}
int faiss_StandardGpuResources_setTempMemory(void* res, int64_t bytes) {
    (void)res; (void)bytes;
    return -1;
}
int faiss_StandardGpuResources_setDefaultNullStreamAllDevices(void* res) {
    (void)res;
    return -1;
}
void faiss_StandardGpuResources_free(void* res) {
    (void)res;
}
int faiss_GpuIndexFlat_new(void** p_index, void* res, int64_t d, int metric_type, int64_t device) {
    (void)p_index; (void)res; (void)d; (void)metric_type; (void)device;
    return -1;
}
int faiss_GpuIndexIVFFlat_new(void** p_index, void* res, void* quantizer, int64_t d, int64_t nlist, int metric_type, int64_t device) {
    (void)p_index; (void)res; (void)quantizer; (void)d; (void)nlist; (void)metric_type; (void)device;
    return -1;
}
int faiss_index_cpu_to_gpu(void* res, int64_t device, void* cpu_index, void** p_gpu_index) {
    (void)res; (void)device; (void)cpu_index; (void)p_gpu_index;
    return -1;
}
int faiss_index_gpu_to_cpu(void* gpu_index, void** p_cpu_index, char* index_type, int* d, int* metric, int64_t* ntotal) {
    (void)gpu_index; (void)p_cpu_index; (void)index_type; (void)d; (void)metric; (void)ntotal;
    return -1;
}
int faiss_index_cpu_to_all_gpus(void* cpu_index, void** p_gpu_index) {
    (void)cpu_index; (void)p_gpu_index;
    return -1;
}
int faiss_get_num_gpus(int* ngpus) { *ngpus = 0; return 0; }

#endif // FAISS_GPU

} // extern "C"

