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

- **13 parallel jobs** testing Go 1.21-1.25
- **~30 second builds** (was 15-30 minutes)
- **Both build modes**: Static libs + Amalgamation
- **Platforms**: Ubuntu + macOS

### Jobs:
1. **test-static-libs** (10 jobs) - Go 1.21-1.25 on Ubuntu + macOS
2. **test-amalgamation** (2 jobs) - Go 1.23 + 1.25 on Ubuntu
3. **lint** (1 job) - Go 1.25 linting
4. **ci-success** - Summary check

**Total CI time: ~5-10 minutes** 🚀

See [WORKFLOW_USAGE.md](../../WORKFLOW_USAGE.md) for complete usage guide.
