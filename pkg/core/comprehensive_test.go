package core

import (
	"testing"
)

// TestComprehensiveLibrary tests the overall library functionality
func TestComprehensiveLibrary(t *testing.T) {
	t.Run("TensorOperations", func(t *testing.T) {
		// Test basic tensor operations
		tensor1 := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
		tensor2 := NewTensorFromSlice([][]float64{{2, 3}, {4, 5}})

		if tensor1 == nil || tensor2 == nil {
			t.Fatal("Failed to create tensors")
		}

		// Test dimensions
		rows, cols := tensor1.Dims()
		if rows != 2 || cols != 2 {
			t.Errorf("Expected dimensions (2, 2), got (%d, %d)", rows, cols)
		}

		// Test addition
		result := tensor1.Add(tensor2)
		if result == nil {
			t.Error("Addition should not return nil")
		}

		// Test multiplication
		result = tensor1.Mul(tensor2)
		if result == nil {
			t.Error("Multiplication should not return nil")
		}
	})

	t.Run("ErrorHandling", func(t *testing.T) {
		// Test error creation
		err := NewError(ErrInvalidInput, "test error")
		if err == nil {
			t.Error("NewError should return an error")
		}

		// Test error with context
		err = err.WithContext("key", "value")
		if err == nil {
			t.Error("WithContext should return an error")
		}
	})

	t.Run("Configuration", func(t *testing.T) {
		// Test default configuration
		config := DefaultConfig()
		if config == nil {
			t.Error("DefaultConfig should return a configuration")
		}

		// Test configuration values
		if config.Epsilon <= 0 {
			t.Error("Epsilon should be positive")
		}
	})

	t.Run("MatrixPool", func(t *testing.T) {
		// Test matrix pool
		pool := NewMatrixPool()
		if pool == nil {
			t.Error("NewMatrixPool should return a pool")
		}

		// Test getting and putting matrices
		matrix := pool.Get(3, 3)
		if matrix == nil {
			t.Error("Pool.Get should return a matrix")
		}

		pool.Put(matrix)
		// No error expected
	})
}

// TestLibraryCoverage tests various components for coverage
func TestLibraryCoverage(t *testing.T) {
	t.Run("ValidationFunctions", func(t *testing.T) {
		// Test positive validation
		err := ValidatePositive(1.0, "test value")
		if err != nil {
			t.Errorf("ValidatePositive should pass for positive value: %v", err)
		}

		err = ValidatePositive(-1.0, "test value")
		if err == nil {
			t.Error("ValidatePositive should fail for negative value")
		}

		// Test range validation
		err = ValidateRange(0.5, 0.0, 1.0, "test value")
		if err != nil {
			t.Errorf("ValidateRange should pass for value in range: %v", err)
		}

		err = ValidateRange(1.5, 0.0, 1.0, "test value")
		if err == nil {
			t.Error("ValidateRange should fail for value out of range")
		}
	})

	t.Run("TensorValidation", func(t *testing.T) {
		tensor := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})

		// Test non-empty validation
		err := ValidateNonEmpty(tensor, "test tensor")
		if err != nil {
			t.Errorf("ValidateNonEmpty should pass for valid tensor: %v", err)
		}

		// Test nil tensor
		err = ValidateNonEmpty(nil, "test tensor")
		if err == nil {
			t.Error("ValidateNonEmpty should fail for nil tensor")
		}
	})

	t.Run("SafeOperations", func(t *testing.T) {
		// Test safe execution
		err := SafeExecute("test operation", func() error {
			return nil
		})
		if err != nil {
			t.Errorf("SafeExecute should succeed for normal operation: %v", err)
		}

		err = SafeExecute("test operation", func() error {
			return NewError(ErrInvalidInput, "test error")
		})
		if err == nil {
			t.Error("SafeExecute should return error for failing operation")
		}
	})
}

// TestIntegrationScenarios tests real-world usage scenarios
func TestIntegrationScenarios(t *testing.T) {
	t.Run("BasicMLWorkflow", func(t *testing.T) {
		// Create sample data
		X := NewTensorFromSlice([][]float64{
			{1.0, 2.0},
			{2.0, 3.0},
			{3.0, 4.0},
			{4.0, 5.0},
		})

		y := NewTensorFromSlice([][]float64{
			{3.0},
			{5.0},
			{7.0},
			{9.0},
		})

		if X == nil || y == nil {
			t.Fatal("Failed to create training data")
		}

		// Validate data
		err := ValidateNonEmpty(X, "features")
		if err != nil {
			t.Errorf("Feature validation failed: %v", err)
		}

		err = ValidateNonEmpty(y, "targets")
		if err != nil {
			t.Errorf("Target validation failed: %v", err)
		}

		// Test dimensions match
		xRows, _ := X.Dims()
		yRows, _ := y.Dims()
		if xRows != yRows {
			t.Errorf("Sample count mismatch: X has %d samples, y has %d", xRows, yRows)
		}
	})

	t.Run("ErrorRecoveryScenario", func(t *testing.T) {
		// Test error recovery configuration
		config := DefaultErrorRecoveryConfig()
		if config.MaxRetries <= 0 {
			t.Error("MaxRetries should be positive")
		}

		recovery := NewErrorRecovery(config)
		if recovery == nil {
			t.Error("NewErrorRecovery should return a recovery instance")
		}

		// Test retry mechanism
		attempts := 0
		err := recovery.RetryWithBackoff("test operation", func() error {
			attempts++
			if attempts < 2 {
				return NewError(ErrNumericalInstability, "temporary failure")
			}
			return nil
		})

		if err != nil {
			t.Errorf("RetryWithBackoff should succeed after retries: %v", err)
		}

		if attempts < 2 {
			t.Errorf("Expected at least 2 attempts, got %d", attempts)
		}
	})
}

// BenchmarkTensorOperations benchmarks core tensor operations
func BenchmarkTensorOperations(b *testing.B) {
	tensor1 := NewTensorFromSlice([][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})
	tensor2 := NewTensorFromSlice([][]float64{{9, 8, 7}, {6, 5, 4}, {3, 2, 1}})

	b.Run("Addition", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = tensor1.Add(tensor2)
		}
	})

	b.Run("Multiplication", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = tensor1.Mul(tensor2)
		}
	})

	b.Run("Copy", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = tensor1.Copy()
		}
	})
}

// BenchmarkMatrixPool benchmarks matrix pool performance
func BenchmarkMatrixPool(b *testing.B) {
	pool := NewMatrixPool()

	b.Run("GetPut", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			matrix := pool.Get(10, 10)
			pool.Put(matrix)
		}
	})

	b.Run("GlobalPool", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			matrix := GetMatrix(10, 10)
			PutMatrix(matrix)
		}
	})
}
