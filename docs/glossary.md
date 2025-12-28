# FAISS Glossary

Key terms and concepts used in FAISS and faiss-go.

---

## Core Concepts

### Vector / Embedding
A numerical representation of data (text, images, etc.) as an array of floating-point numbers. Example: a 128-dimensional vector represents data as 128 numbers.

### Dimension (d)
The length of each vector. All vectors in an index must have the same dimension.

### Index
The data structure that stores vectors and enables fast similarity search.

### Similarity Search / Nearest Neighbor Search
Finding vectors in an index that are most similar to a query vector.

### Distance Metric
The method used to measure similarity between vectors:
- **L2 (Euclidean)**: Geometric distance
- **Inner Product**: Dot product similarity

---

## Index Types

### Flat Index
Exact search using brute-force comparison. 100% recall but slow for large datasets.

### IVF (Inverted File)
Partitions vectors into clusters for faster approximate search.

### HNSW (Hierarchical Navigable Small World)
Graph-based index with excellent recall/speed tradeoff.

### PQ (Product Quantization)
Compression technique that reduces memory usage 8-32x.

### Scalar Quantization
Compression using reduced precision (e.g., 8-bit instead of 32-bit floats).

### GPU Index
Index that runs on GPU for massive speedup (10-100x).

### OnDisk Index
Index stored on disk for datasets larger than RAM.

---

## Search Concepts

### k-NN (k-Nearest Neighbors)
Finding the k closest vectors to a query.

### Recall
Percentage of true nearest neighbors found by approximate search.

### Precision
Accuracy of search results.

### QPS (Queries Per Second)
Search throughput measure.

### Range Search
Finding all vectors within a distance threshold.

---

## Training & Configuration

### Training
Process of optimizing an index for a specific data distribution (required for IVF, PQ indexes).

### nlist
Number of clusters in IVF indexes.

### nprobe
Number of clusters to search in IVF indexes (higher = better recall, slower).

### M (HNSW)
Number of connections per node in HNSW graph.

### efConstruction (HNSW)
Build quality parameter (higher = better graph, slower build).

### efSearch (HNSW)
Search quality parameter (higher = better recall, slower search).

---

## Memory & Compression

### Quantizer
Index used to partition space in IVF indexes.

### Centroid
Center point of a cluster.

### Code / PQ Code
Compressed representation of a vector.

### Reconstruction
Retrieving approximate original vector from compressed representation.

---

## Operations

### Add
Insert vectors into an index.

### Search
Query for nearest neighbors.

### Train
Optimize index for data distribution.

### Serialize
Save index to disk.

### Deserialize
Load index from disk.

---

## Advanced

### Index Factory
Pattern for creating indexes from string descriptors (e.g., "IVF100,PQ16").

### Preprocessing / Transform
Operations on vectors before indexing (normalization, PCA, etc.).

### ID Mapping
Associating custom IDs with vectors.

### Sharding
Distributing index across multiple instances.

---

## Metrics

### L2 Distance
Euclidean distance: sqrt(sum((a-b)²))

### Inner Product
Dot product: sum(a×b)

### Cosine Similarity
Inner product of normalized vectors.

---

## Performance Terms

### Throughput
Number of operations per second.

### Latency
Time to complete a single operation.

### Batch Size
Number of queries processed together.

### SIMD (Single Instruction Multiple Data)
CPU feature for parallel processing (used in PQFastScan).

---

## Abbreviations

- **ANN**: Approximate Nearest Neighbor
- **BLAS**: Basic Linear Algebra Subprograms
- **CGO**: C/Go interoperability
- **CUDA**: GPU computing platform
- **IVF**: Inverted File
- **HNSW**: Hierarchical Navigable Small World
- **LSH**: Locality-Sensitive Hashing
- **PQ**: Product Quantization
- **QPS**: Queries Per Second
- **RAM**: Random Access Memory
- **SQ**: Scalar Quantization
- **VRAM**: Video RAM (GPU memory)

---

## Related Terms

### Embedding Model
ML model that converts data to vectors (e.g., BERT for text, ResNet for images).

### Vector Database
Database optimized for storing and searching vectors.

### Semantic Search
Finding documents by meaning rather than keywords.

### RAG (Retrieval Augmented Generation)
LLM pattern using similarity search to retrieve relevant context.

---

For more details, see:
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Index Types Guide](guides/index-types.md)
- [Performance Tuning](guides/performance-tuning.md)
