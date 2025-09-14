#!/bin/bash

# ThinkingNet-Go Conservative Cleanup Script
# Removes only obviously unused files while keeping most tests and examples

echo "🧹 Starting conservative ThinkingNet-Go cleanup..."

# Create backup directory
mkdir -p backup_conservative_$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backup_conservative_$(date +%Y%m%d_%H%M%S)"

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

# Remove only the most obvious unused files
echo "🗂️  Removing large data files (MNIST)..."
backup_and_remove "mnist_data"

echo "📜 Removing original code (already refactored)..."
backup_and_remove "OriginalCode"

echo "🔧 Removing IDE-specific files..."
backup_and_remove ".vscode"

echo "📊 Removing redundant analysis files..."
backup_and_remove "OPTIMIZATION_IDEAS.md"
backup_and_remove "OPTIMIZATION_SUMMARY.md" 
backup_and_remove "PERFORMANCE_ANALYSIS_RESULTS.md"
backup_and_remove "ULTRA_FAST_OPTIMIZATIONS.md"

echo "🐙 Removing GitHub-specific files..."
backup_and_remove ".github"

# Clean up empty directories
echo "📁 Cleaning up empty directories..."
find . -type d -empty -delete 2>/dev/null || true

echo ""
echo "✨ Conservative cleanup completed!"
echo ""
echo "📋 This conservative cleanup only removed:"
echo "  - MNIST data files (large binary files)"
echo "  - Original code directory (already refactored)"
echo "  - IDE-specific configuration files"
echo "  - Redundant analysis/optimization markdown files"
echo "  - GitHub workflow files"
echo ""
echo "📚 All library code, tests, examples, and essential docs were KEPT"
echo ""
echo "🔄 All removed files have been backed up to: $BACKUP_DIR"
echo "💡 Run ./cleanup.sh for more aggressive cleanup if needed."
echo ""
echo "🎉 Your ThinkingNet-Go library is now cleaner while preserving all functionality!"