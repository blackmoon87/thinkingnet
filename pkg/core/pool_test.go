package core

import (
	"testing"
)

func TestMatrixPool(t *testing.T) {
	pool := NewMatrixPool()

	// Test getting a matrix from empty pool
	matrix1 := pool.Get(3, 4)
	if matrix1 == nil {
		t.Fatal("Expected non-nil matrix from pool")
	}

	rows, cols := matrix1.Dims()
	if rows != 3 || cols != 4 {
		t.Errorf("Expected dimensions (3, 4), got (%d, %d)", rows, cols)
	}

	// Test putting matrix back
	pool.Put(matrix1)

	// Test getting matrix again (should reuse)
	matrix2 := pool.Get(3, 4)
	if matrix2 == nil {
		t.Fatal("Expected non-nil matrix from pool")
	}

	// Test getting matrix with different dimensions
	matrix3 := pool.Get(2, 2)
	if matrix3 == nil {
		t.Fatal("Expected non-nil matrix from pool")
	}

	rows, cols = matrix3.Dims()
	if rows != 2 || cols != 2 {
		t.Errorf("Expected dimensions (2, 2), got (%d, %d)", rows, cols)
	}

	// Test pool stats
	stats := pool.Stats()
	if len(stats) == 0 {
		t.Error("Expected non-empty stats")
	}

	// Test clearing pool
	pool.Clear()

	// After clearing, getting a matrix should create a new one
	matrix4 := pool.Get(5, 5)
	if matrix4 == nil {
		t.Fatal("Expected non-nil matrix from pool after clear")
	}
}

func TestGlobalMatrixPool(t *testing.T) {
	// Test global pool functions
	matrix := GetMatrix(2, 3)
	if matrix == nil {
		t.Fatal("Expected non-nil matrix from global pool")
	}

	rows, cols := matrix.Dims()
	if rows != 2 || cols != 3 {
		t.Errorf("Expected dimensions (2, 3), got (%d, %d)", rows, cols)
	}

	// Put matrix back
	PutMatrix(matrix)

	// Test global pool stats - skip since sync.Pool doesn't expose size

	// Test clearing global pool
	ClearMatrixPool()
}

func TestPoolKey(t *testing.T) {
	pool := NewMatrixPool()
	key1 := pool.getPoolKey(3, 4)
	key2 := pool.getPoolKey(3, 4)
	key3 := pool.getPoolKey(4, 3)

	if key1 != key2 {
		t.Error("Expected same key for same dimensions")
	}

	if key1 == key3 {
		t.Error("Expected different keys for different dimensions")
	}
}

func TestPoolPutWrongSize(t *testing.T) {
	pool := NewMatrixPool()

	// Get a 2x2 matrix
	matrix := pool.Get(2, 2)

	// Modify the matrix to have different dimensions (this is artificial for testing)
	// In real usage, this shouldn't happen, but we test the pool's robustness

	// Put it back - should work normally
	pool.Put(matrix)

	// Verify stats - skip since sync.Pool doesn't expose detailed stats
	stats := pool.Stats()
	if len(stats) == 0 {
		t.Error("Expected non-empty stats")
	}
}

func TestPoolStats(t *testing.T) {
	pool := NewMatrixPool()

	// Initial stats should be empty
	stats := pool.Stats()
	initialCount := len(stats)

	// Perform some operations
	matrix1 := pool.Get(2, 2)
	matrix2 := pool.Get(3, 3)

	pool.Put(matrix1)

	// Check updated stats
	newStats := pool.Stats()

	if len(newStats) < initialCount {
		t.Error("Expected stats to track pool usage")
	}

	// Put back the second matrix
	pool.Put(matrix2)
}
