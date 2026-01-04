# GitHub Actions Workflows

This document describes the GitHub Actions workflows configured for this repository and their execution policies.

## Workflow Policies

### 1. **CI Workflow** (`ci.yml`)

**Trigger**: Pull Requests to `main` branch only

**Purpose**: Validates code quality and tests before merging to main

**Behavior**:
- ✅ Runs automatically on PRs targeting `main`
- ✅ **BLOCKS merging if tests fail** (required status check)
- ❌ Does NOT run on direct pushes to main
- ❌ Does NOT run on tags
- ❌ Cannot be manually triggered

**What it does**:
- Builds the project with multiple Go versions
- Runs all unit tests
- Performs linting checks
- Tests both CPU and GPU builds

### 2. **Release Workflow** (`release.yml`)

**Trigger**: Manual dispatch only (`workflow_dispatch`)

**Purpose**: Creates official releases with version bumping and Git tags

**Behavior**:
- ✅ Can ONLY be triggered manually via GitHub UI
- ✅ **REQUIRES running on `main` branch** (enforced by workflow)
- ✅ **PREVENTS overwriting existing tags** (enforced by workflow)
- ❌ Does NOT run automatically on any event

**Safety Checks**:
1. Verifies current branch is `main` (fails if not)
2. Checks if calculated tag already exists (fails if yes)
3. Runs all tests before creating release
4. Runs linting before creating release

**What it does**:
1. Calculates next version based on type (major/minor/patch)
2. Updates version in `faiss.go`
3. Generates changelog from commits
4. Runs tests and linting
5. Creates version bump commit
6. Creates and pushes Git tag
7. Creates GitHub release with notes

**How to trigger**:
1. Ensure you're on `main` branch
2. Go to Actions tab
3. Select "Release"
4. Click "Run workflow"
5. Choose version bump type (major/minor/patch)
6. Optionally add pre-release suffix (e.g., `beta.1`)

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Developer Workflow                       │
└─────────────────────────────────────────────────────────────┘

Feature Branch ──────┐
                     │
                     ├──> Pull Request to main
                     │         │
                     │         ├──> CI Workflow (REQUIRED)
                     │         │    ├─ Build & Test ✓
                     │         │    ├─ Lint ✓
                     │         │    └─ BLOCKS if fails ✗
                     │         │
                     │         ├──> Manual Review
                     │         │
                     └─────────┼──> Merge to main
                               │
                               ▼
                          main branch
                               │
                               └──> Manual: Release Workflow
                                    ├─ Verify on main ✓
                                    ├─ Check tag exists ✓
                                    ├─ Run tests ✓
                                    ├─ Create tag
                                    └─ Create release
```

## Branch Protection Recommendations

To enforce these policies at the repository level, configure the following branch protection rules for `main`:

1. **Require status checks to pass before merging**
   - Require: `build-and-test` (from CI workflow)

2. **Require pull request reviews**
   - Recommended: At least 1 approval

3. **Do not allow bypassing the above settings**
   - Includes administrators

4. **Require linear history**
   - Prevents merge commits

## Tag Protection Recommendations

Configure tag protection rules in repository settings:

1. **Pattern**: `v*`
2. **Restrict tag creation**: Require workflow (Release workflow only)
3. **Prevent tag deletion**: Enabled
4. **Prevent tag overwrites**: Enabled (enforced by Release workflow)

## Manual Workflow Execution

All manual workflows (`workflow_dispatch`) can be triggered via:

```bash
# Using GitHub CLI
gh workflow run <workflow-name>.yml

# Example: Trigger release
gh workflow run release.yml -f version_type=patch
```

Or via the GitHub web interface:
- Navigate to **Actions** tab
- Select desired workflow from the left sidebar
- Click **Run workflow** button
- Fill in required inputs
- Click **Run workflow**
