# GitHub Actions Workflows

This directory contains all CI/CD workflows for faiss-go.

## Quick Reference

| Workflow | Trigger | Duration | Purpose |
|----------|---------|----------|---------|
| **ci.yml** | Manual | ~5-10 min | Fast testing with static libs (Go 1.21-1.25) |
| **benchmark.yml** | Manual | 10-30 min | Performance benchmarks (quick/comprehensive) |
| **build-static-libs.yml** | Manual | ~60 min | Build static libraries for all platforms |
| **gpu-ci.yml** | Manual | Varies | GPU-specific tests |
| **release.yml** | Manual | ~5 min | Create versioned releases |

All workflows are **manual-only** (workflow_dispatch). Nothing runs automatically.

## CI Workflow Overview

**Fast CI using pre-built static libraries!**

- **11 parallel jobs** testing Go 1.21-1.25
- **~30 second builds** (8x faster than compiling FAISS)
- **Build mode**: Static libraries (default, no build tags needed)
- **Platforms**: Ubuntu + macOS

### Jobs:
1. **test-static-libs** (10 jobs) - Go 1.21-1.25 on Ubuntu + macOS
2. **lint** (1 job) - Go 1.25 linting
3. **ci-success** - Summary check

**Total CI time: ~5-10 minutes** 🚀

See [WORKFLOW_USAGE.md](../../WORKFLOW_USAGE.md) for complete usage guide.
