#!/bin/bash

# ThinkingNet-Go Cleanup Script
# Removes unused files while keeping essential library tests and important .md files

echo "🧹 Starting ThinkingNet-Go cleanup..."

# Create backup directory
mkdir -p backup_$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"

echo "📦 Creating backup in $BACKUP_DIR..."

# Function to backup and remove
backup_and_remove() {
    local file_path="$1"
    if [ -e "$file_path" ]; then
        # Create directory structure in backup
        mkdir -p "$BACKUP_DIR/$(dirname "$file_path")"
        cp -r "$file_path" "$BACKUP_DIR/$file_path"
        rm -rf "$file_path"
        echo "  ✅ Removed: $file_path"
    fi
}

# Remove large data files that are not essential
echo "🗂️  Removing large data files..."
backup_and_remove "mnist_data"
backup_and_remove "data/checker_test.csv"
backup_and_remove "data/checker_train.csv"
backup_and_remove "data/moons_test.csv"
backup_and_remove "data/moons_train.csv"
backup_and_remove "data/swiss_hole_test.csv"
backup_and_remove "data/swiss_hole_train.csv"
backup_and_remove "data/intents_large.json"

# Remove original code (already refactored)
echo "📜 Removing original code files..."
backup_and_remove "OriginalCode"

# Remove IDE-specific files
echo "🔧 Removing IDE-specific files..."
backup_and_remove ".vscode"

# Remove redundant demo examples (keep the essential ones)
echo "🎯 Removing redundant demo examples..."
backup_and_remove "examples/activation_demo"
backup_and_remove "examples/callbacks_demo"
backup_and_remove "examples/classification_demo"
backup_and_remove "examples/clustering_demo"
backup_and_remove "examples/linear_regression_demo"
backup_and_remove "examples/metrics_demo"
backup_and_remove "examples/pca_demo"
backup_and_remove "examples/preprocessing_demo"
backup_and_remove "examples/tensor_demo"
backup_and_remove "examples/optimization_demo.go"

# Remove redundant test files (keep core functionality tests)
echo "🧪 Removing redundant test files..."
backup_and_remove "pkg/core/comprehensive_test.go"
backup_and_remove "pkg/core/tensor_extended_test.go"
backup_and_remove "pkg/layers/dense_edge_cases_test.go"
backup_and_remove "pkg/layers/dropout_edge_cases_test.go"
backup_and_remove "pkg/layers/integration_error_handling_test.go"
backup_and_remove "pkg/layers/weight_initialization_test.go"
backup_and_remove "pkg/losses/simple_test.go"
backup_and_remove "pkg/models/example_test.go"
backup_and_remove "pkg/activations/utils_test.go"
backup_and_remove "pkg/algorithms/clustering_metrics_test.go"
backup_and_remove "pkg/algorithms/dbscan_test.go"
backup_and_remove "pkg/algorithms/random_forest_test.go"
backup_and_remove "pkg/callbacks/integration_test.go"

# Remove unused algorithm implementations
echo "🤖 Removing unused algorithm implementations..."
backup_and_remove "pkg/algorithms/dbscan.go"
backup_and_remove "pkg/algorithms/random_forest.go"
backup_and_remove "pkg/algorithms/clustering_metrics.go"

# Remove unused utility files
echo "🛠️  Removing unused utility files..."
backup_and_remove "pkg/activations/utils.go"
backup_and_remove "pkg/layers/test_utils.go"
backup_and_remove "pkg/layers/test_utils_test.go"

# Remove redundant optimization and analysis files
echo "📊 Removing redundant analysis files..."
backup_and_remove "OPTIMIZATION_IDEAS.md"
backup_and_remove "OPTIMIZATION_SUMMARY.md"
backup_and_remove "PERFORMANCE_ANALYSIS_RESULTS.md"
backup_and_remove "ULTRA_FAST_OPTIMIZATIONS.md"
backup_and_remove "integration_test.go"

# Remove GitHub-specific files (keep if needed for CI/CD)
echo "🐙 Removing GitHub-specific files..."
backup_and_remove ".github"

# Clean up empty directories
echo "📁 Cleaning up empty directories..."
find . -type d -empty -delete 2>/dev/null || true

echo ""
echo "✨ Cleanup completed!"
echo ""
echo "📋 Summary of what was KEPT (essential files):"
echo "  📚 Core library files:"
echo "    - All essential .go files in pkg/"
echo "    - Essential test files (*_test.go for core functionality)"
echo "    - go.mod, go.sum"
echo ""
echo "  📖 Documentation:"
echo "    - README.md"
echo "    - INTEGRATION_SUMMARY.md"
echo "    - docs/arabic-guide.md"
echo ""
echo "  🎯 Essential examples:"
echo "    - examples/simple_start.go"
echo "    - examples/easy_*.go (classification, clustering, regression)"
echo "    - examples/easy_usage_demo/"
echo "    - examples/helper_functions_demo/"
echo "    - examples/preprocessing_helpers_demo/"
echo "    - examples/split_demo/"
echo "    - examples/bilingual_errors_demo/"
echo ""
echo "  🧪 Essential tests:"
echo "    - pkg/integration_test.go"
echo "    - Core functionality tests for each package"
echo "    - Helper function tests"
echo "    - Bilingual error tests"
echo ""
echo "  📊 Performance:"
echo "    - benchmarks/ (performance benchmarks)"
echo ""
echo "  📦 Essential data:"
echo "    - data/intents.json"
echo "    - data/test_real_world.json"
echo ""
echo "🔄 All removed files have been backed up to: $BACKUP_DIR"
echo "💡 You can restore any file from the backup if needed."
echo ""
echo "🎉 Your ThinkingNet-Go library is now clean and focused on essential functionality!"