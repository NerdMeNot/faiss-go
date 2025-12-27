#!/bin/bash

# Script to download standard benchmark datasets for testing
# Usage: ./scripts/download_test_datasets.sh [dataset_name]
# If no dataset_name is provided, downloads all datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TESTDATA_DIR="$PROJECT_ROOT/testdata/embeddings"

# Create directory if it doesn't exist
mkdir -p "$TESTDATA_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to download and extract SIFT1M dataset
download_sift1m() {
    log_info "Downloading SIFT1M dataset..."

    cd "$TESTDATA_DIR"

    if [ -f "sift1m_base.fvecs" ] && [ -f "sift1m_query.fvecs" ] && [ -f "sift1m_groundtruth.ivecs" ]; then
        log_warn "SIFT1M dataset already exists, skipping..."
        return 0
    fi

    # Download
    if [ ! -f "sift.tar.gz" ]; then
        log_info "Downloading sift.tar.gz (approx. 160MB)..."
        wget -q --show-progress ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz || {
            log_error "Failed to download SIFT1M"
            return 1
        }
    fi

    # Extract
    log_info "Extracting SIFT1M..."
    tar -xzf sift.tar.gz

    # Rename files to standard names
    mv sift/sift_base.fvecs sift1m_base.fvecs
    mv sift/sift_query.fvecs sift1m_query.fvecs
    mv sift/sift_groundtruth.ivecs sift1m_groundtruth.ivecs
    mv sift/sift_learn.fvecs sift1m_learn.fvecs

    # Cleanup
    rm -rf sift
    rm sift.tar.gz

    log_info "SIFT1M dataset ready!"
    log_info "  - Base: 1M vectors, 128-dim"
    log_info "  - Query: 10K vectors"
    log_info "  - Ground truth: 100-NN for each query"
}

# Function to download and extract GIST1M dataset
download_gist1m() {
    log_info "Downloading GIST1M dataset..."

    cd "$TESTDATA_DIR"

    if [ -f "gist1m_base.fvecs" ] && [ -f "gist1m_query.fvecs" ] && [ -f "gist1m_groundtruth.ivecs" ]; then
        log_warn "GIST1M dataset already exists, skipping..."
        return 0
    fi

    # Download
    if [ ! -f "gist.tar.gz" ]; then
        log_info "Downloading gist.tar.gz (approx. 3.6GB - this will take a while)..."
        wget -q --show-progress ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz || {
            log_error "Failed to download GIST1M"
            return 1
        }
    fi

    # Extract
    log_info "Extracting GIST1M..."
    tar -xzf gist.tar.gz

    # Rename files
    mv gist/gist_base.fvecs gist1m_base.fvecs
    mv gist/gist_query.fvecs gist1m_query.fvecs
    mv gist/gist_groundtruth.ivecs gist1m_groundtruth.ivecs
    mv gist/gist_learn.fvecs gist1m_learn.fvecs

    # Cleanup
    rm -rf gist
    rm gist.tar.gz

    log_info "GIST1M dataset ready!"
    log_info "  - Base: 1M vectors, 960-dim"
    log_info "  - Query: 1K vectors"
    log_info "  - Ground truth: 100-NN for each query"
}

# Function to create SIFT10K subset from SIFT1M
create_sift10k() {
    log_info "Creating SIFT10K subset..."

    cd "$TESTDATA_DIR"

    if [ ! -f "sift1m_base.fvecs" ]; then
        log_error "SIFT1M not found. Download it first."
        return 1
    fi

    if [ -f "sift10k_base.fvecs" ]; then
        log_warn "SIFT10K already exists, skipping..."
        return 0
    fi

    # Use Go program to create subset (need to implement this)
    log_info "Creating 10K subset from SIFT1M..."

    # For now, use dd to extract first 10K vectors
    # Each vector: 4 bytes (dim) + 128*4 bytes (data) = 516 bytes
    dd if=sift1m_base.fvecs of=sift10k_base.fvecs bs=516 count=10000 2>/dev/null

    # Query: first 100 queries
    dd if=sift1m_query.fvecs of=sift10k_query.fvecs bs=516 count=100 2>/dev/null

    # Ground truth: first 100 entries
    # Each entry: 4 bytes (k=100) + 100*4 bytes (ids) = 404 bytes
    dd if=sift1m_groundtruth.ivecs of=sift10k_groundtruth.ivecs bs=404 count=100 2>/dev/null

    log_info "SIFT10K dataset ready!"
    log_info "  - Base: 10K vectors, 128-dim"
    log_info "  - Query: 100 vectors"
}

# Function to show dataset info
show_info() {
    log_info "Available datasets:"
    echo ""
    echo "1. SIFT1M - 1M SIFT descriptors (128-dim)"
    echo "   Size: ~160MB compressed"
    echo "   Use: ./download_test_datasets.sh sift1m"
    echo ""
    echo "2. SIFT10K - 10K SIFT descriptors (subset of SIFT1M)"
    echo "   Size: ~5MB"
    echo "   Use: ./download_test_datasets.sh sift10k"
    echo "   Note: Requires SIFT1M to be downloaded first"
    echo ""
    echo "3. GIST1M - 1M GIST descriptors (960-dim)"
    echo "   Size: ~3.6GB compressed"
    echo "   Use: ./download_test_datasets.sh gist1m"
    echo ""
    echo "Download all:"
    echo "   ./download_test_datasets.sh all"
}

# Main script
main() {
    DATASET="${1:-all}"

    case "$DATASET" in
        sift1m)
            download_sift1m
            ;;
        sift10k)
            download_sift1m
            create_sift10k
            ;;
        gist1m)
            download_gist1m
            ;;
        all)
            download_sift1m
            create_sift10k
            download_gist1m
            ;;
        info|--help|-h)
            show_info
            ;;
        *)
            log_error "Unknown dataset: $DATASET"
            echo ""
            show_info
            exit 1
            ;;
    esac

    log_info "Done! Datasets are in: $TESTDATA_DIR"
}

main "$@"
