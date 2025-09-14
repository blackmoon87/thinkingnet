# ThinkingNet-Go Cleanup Guide

This guide explains the cleanup scripts available to reduce the project size while keeping essential functionality.

## Current Project Size: ~57MB

The project currently contains many files that may not be essential for the core library functionality.

## Available Cleanup Scripts

### 1. `cleanup_dry_run.sh` - Preview Changes
**Recommended first step** - Shows what would be removed without making any changes.

```bash
./cleanup_dry_run.sh
```

**What it does:**
- Lists all files that would be removed
- Shows file sizes
- Explains why each file would be removed
- Shows what would be kept

### 2. `cleanup_conservative.sh` - Safe Cleanup
**Recommended for most users** - Removes only obviously unused files.

```bash
./cleanup_conservative.sh
```

**What it removes:**
- MNIST data files (53MB) - Large binary datasets
- Original code directory - Already refactored code
- IDE-specific files (.vscode, .github)
- Redundant analysis markdown files

**What it keeps:**
- All library code and essential tests
- All examples and demos
- All documentation
- All algorithm implementations

**Estimated size reduction:** ~55MB → ~25MB

### 3. `cleanup.sh` - Aggressive Cleanup
**For production/minimal installs** - Removes more files for a leaner library.

```bash
./cleanup.sh
```

**Additional removals beyond conservative:**
- Redundant demo examples (keeps essential ones)
- Redundant test files (keeps core functionality tests)
- Unused algorithm implementations (DBSCAN, Random Forest)
- Additional analysis files

**What it keeps:**
- Core library functionality
- Essential tests for each package
- Key examples (simple_start.go, easy_*.go, helper demos)
- Important documentation (README.md, INTEGRATION_SUMMARY.md)
- Performance benchmarks

**Estimated size reduction:** ~55MB → ~15MB

## Essential Files That Are Always Kept

### Core Library Files
- `pkg/` - All essential .go files
- Essential test files for core functionality
- `go.mod`, `go.sum`

### Documentation
- `README.md` - Main documentation
- `INTEGRATION_SUMMARY.md` - Integration guide
- `docs/arabic-guide.md` - Arabic documentation

### Essential Examples
- `examples/simple_start.go` - Basic usage
- `examples/easy_*.go` - Helper function examples
- `examples/easy_usage_demo/` - Complete usage demo
- `examples/helper_functions_demo/` - Helper functions
- `examples/preprocessing_helpers_demo/` - Preprocessing helpers
- `examples/split_demo/` - Data splitting
- `examples/bilingual_errors_demo/` - Error handling

### Essential Tests
- `pkg/integration_test.go` - Integration tests
- Core functionality tests for each package
- Helper function tests
- Bilingual error tests

### Performance & Data
- `benchmarks/` - Performance benchmarks
- `data/intents.json` - Essential test data
- `data/test_real_world.json` - Real-world test data

## Safety Features

### Automatic Backups
All scripts create timestamped backups before removing files:
- Conservative: `backup_conservative_YYYYMMDD_HHMMSS/`
- Aggressive: `backup_YYYYMMDD_HHMMSS/`

### Restore Files
To restore a file from backup:
```bash
cp backup_*/path/to/file ./path/to/file
```

## Recommendations

1. **Start with dry run:** `./cleanup_dry_run.sh`
2. **Most users:** `./cleanup_conservative.sh`
3. **Production/minimal:** `./cleanup.sh`
4. **Keep backups** until you're sure everything works

## After Cleanup

### Verify Everything Works
```bash
# Run tests
go test ./...

# Run examples
go run examples/simple_start.go
go run examples/easy_usage_demo/main.go

# Run benchmarks
go test -bench=. ./benchmarks/
```

### Update Dependencies
```bash
go mod tidy
go mod verify
```

## File Size Breakdown

| Component | Before | Conservative | Aggressive |
|-----------|--------|--------------|------------|
| MNIST Data | 53MB | 0MB | 0MB |
| Examples | ~5MB | ~5MB | ~2MB |
| Tests | ~3MB | ~3MB | ~1MB |
| Core Library | ~2MB | ~2MB | ~2MB |
| Documentation | ~1MB | ~1MB | ~1MB |
| **Total** | **~57MB** | **~25MB** | **~15MB** |

## Need Help?

If you accidentally remove something important:
1. Check the backup directory
2. Restore the specific file you need
3. Run `go mod tidy` to update dependencies

The cleanup scripts are designed to be safe and reversible!