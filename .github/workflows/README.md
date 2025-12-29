# GitHub Actions Workflows

This directory contains all CI/CD workflows for faiss-go.

## Quick Reference

| Workflow | Trigger | Duration | Purpose |
|----------|---------|----------|---------|
| **ci.yml** | Manual | ~5-10 min | Fast testing with static libs (Go 1.21-1.25) |
| **benchmark.yml** | Manual | 10-30 min | Performance benchmarks (quick/comprehensive) |
| **build-static-libs.yml** | Manual | ~60 min | Build static libraries for all platforms |
| **build-amalgamation.yml** | Manual | ~10 min | Build amalgamated FAISS source |
| **gpu-ci.yml** | Manual | Varies | GPU-specific tests |
| **release.yml** | Manual | ~5 min | Create versioned releases |

All workflows are **manual-only** (workflow_dispatch). Nothing runs automatically.

## CI Workflow Overview

**New fast CI using static libraries!**

- **11 parallel jobs** testing Go 1.21-1.25
- **~30 second builds** (was 15-30 minutes)
- **Build mode**: Static libraries (amalgamation not yet implemented)
- **Platforms**: Ubuntu + macOS

### Jobs:
1. **test-static-libs** (10 jobs) - Go 1.21-1.25 on Ubuntu + macOS
2. **lint** (1 job) - Go 1.25 linting
3. **ci-success** - Summary check

**Total CI time: ~5-10 minutes** 🚀

**Note:** Amalgamation build mode is not yet implemented (stub files only). CI currently tests static library mode only.

See [WORKFLOW_USAGE.md](../../WORKFLOW_USAGE.md) for complete usage guide.
