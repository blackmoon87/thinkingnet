package core

import (
	"math"
	"testing"
)

func TestComprehensiveErrorHandling(t *testing.T) {
	// Test validation functions
	t.Run("ValidatePositive", func(t *testing.T) {
		err := ValidatePositive(1.0, "test_value")
		if err != nil {
			t.Errorf("Expected no error for positive value, got: %v", err)
		}

		err = ValidatePositive(-1.0, "test_value")
		if err == nil {
			t.Error("Expected error for negative value")
		}

		if !IsThinkingNetError(err) {
			t.Error("Expected ThinkingNetError")
		}

		errType, ok := GetErrorType(err)
		if !ok || errType != ErrInvalidInput {
			t.Errorf("Expected ErrInvalidInput, got: %v", errType)
		}
	})

	t.Run("ValidateRange", func(t *testing.T) {
		err := ValidateRange(0.5, 0.0, 1.0, "test_value")
		if err != nil {
			t.Errorf("Expected no error for value in range, got: %v", err)
		}

		err = ValidateRange(1.5, 0.0, 1.0, "test_value")
		if err == nil {
			t.Error("Expected error for value out of range")
		}
	})

	t.Run("ValidateTensorFinite", func(t *testing.T) {
		// Test with finite tensor
		tensor := NewTensorFromData(2, 2, []float64{1.0, 2.0, 3.0, 4.0})
		err := ValidateTensorFinite(tensor, "test_tensor")
		if err != nil {
			t.Errorf("Expected no error for finite tensor, got: %v", err)
		}

		// Test with NaN tensor
		nanTensor := NewTensorFromData(2, 2, []float64{1.0, math.NaN(), 3.0, 4.0})
		err = ValidateTensorFinite(nanTensor, "test_tensor")
		if err == nil {
			t.Error("Expected error for tensor with NaN")
		}

		// Test with Inf tensor
		infTensor := NewTensorFromData(2, 2, []float64{1.0, math.Inf(1), 3.0, 4.0})
		err = ValidateTensorFinite(infTensor, "test_tensor")
		if err == nil {
			t.Error("Expected error for tensor with Inf")
		}
	})

	t.Run("ValidateTrainingData", func(t *testing.T) {
		X := NewTensorFromData(3, 2, []float64{1, 2, 3, 4, 5, 6})
		y := NewTensorFromData(3, 1, []float64{1, 2, 3})

		err := ValidateTrainingData(X, y)
		if err != nil {
			t.Errorf("Expected no error for valid training data, got: %v", err)
		}

		// Test dimension mismatch
		yBad := NewTensorFromData(2, 1, []float64{1, 2})
		err = ValidateTrainingData(X, yBad)
		if err == nil {
			t.Error("Expected error for dimension mismatch")
		}
	})

	t.Run("ValidateIntRange", func(t *testing.T) {
		err := ValidateIntRange(5, 1, 10, "test_value")
		if err != nil {
			t.Errorf("Expected no error for value in range, got: %v", err)
		}

		err = ValidateIntRange(15, 1, 10, "test_value")
		if err == nil {
			t.Error("Expected error for value out of range")
		}
	})

	t.Run("ValidateStringInSet", func(t *testing.T) {
		allowed := []string{"option1", "option2", "option3"}

		err := ValidateStringInSet("option2", allowed, "test_option")
		if err != nil {
			t.Errorf("Expected no error for valid option, got: %v", err)
		}

		err = ValidateStringInSet("invalid", allowed, "test_option")
		if err == nil {
			t.Error("Expected error for invalid option")
		}
	})
}

func TestErrorRecovery(t *testing.T) {
	config := DefaultErrorRecoveryConfig()
	recovery := NewErrorRecovery(config)

	t.Run("RetryWithBackoff", func(t *testing.T) {
		attempts := 0
		err := recovery.RetryWithBackoff("test_operation", func() error {
			attempts++
			if attempts < 3 {
				return NewError(ErrNumericalInstability, "temporary failure")
			}
			return nil
		})

		if err != nil {
			t.Errorf("Expected success after retries, got: %v", err)
		}

		if attempts != 3 {
			t.Errorf("Expected 3 attempts, got: %d", attempts)
		}
	})

	t.Run("RetryWithBackoff_MaxRetriesExceeded", func(t *testing.T) {
		attempts := 0
		err := recovery.RetryWithBackoff("test_operation", func() error {
			attempts++
			return NewError(ErrNumericalInstability, "persistent failure")
		})

		if err == nil {
			t.Error("Expected error after max retries exceeded")
		}

		expectedAttempts := config.MaxRetries + 1
		if attempts != expectedAttempts {
			t.Errorf("Expected %d attempts, got: %d", expectedAttempts, attempts)
		}
	})

	t.Run("SafeTensorOperation", func(t *testing.T) {
		tensor := NewTensorFromData(2, 2, []float64{1.0, 2.0, 3.0, 4.0})

		result, err := recovery.SafeTensorOperation("test_op", func() Tensor {
			return tensor.Scale(2.0)
		})

		if err != nil {
			t.Errorf("Expected no error for safe operation, got: %v", err)
		}

		if result == nil {
			t.Error("Expected non-nil result")
		}
	})
}

func TestCleanNaNInf(t *testing.T) {
	t.Run("CleanWithZero", func(t *testing.T) {
		tensor := NewTensorFromData(2, 2, []float64{1.0, math.NaN(), math.Inf(1), 4.0})

		cleaned, err := CleanNaNInf(tensor, "zero")
		if err != nil {
			t.Errorf("Expected no error for cleaning, got: %v", err)
		}

		if cleaned.HasNaN() || cleaned.HasInf() {
			t.Error("Expected cleaned tensor to have no NaN or Inf values")
		}

		// Check that NaN and Inf were replaced with zero
		if cleaned.At(0, 1) != 0.0 || cleaned.At(1, 0) != 0.0 {
			t.Error("Expected NaN and Inf to be replaced with zero")
		}
	})

	t.Run("CleanWithMean", func(t *testing.T) {
		tensor := NewTensorFromData(3, 2, []float64{1.0, math.NaN(), 3.0, 4.0, 5.0, math.Inf(1)})

		cleaned, err := CleanNaNInf(tensor, "mean")
		if err != nil {
			t.Errorf("Expected no error for cleaning, got: %v", err)
		}

		if cleaned.HasNaN() || cleaned.HasInf() {
			t.Error("Expected cleaned tensor to have no NaN or Inf values")
		}
	})

	t.Run("CleanWithClamp", func(t *testing.T) {
		tensor := NewTensorFromData(2, 2, []float64{1.0, math.NaN(), math.Inf(1), math.Inf(-1)})

		cleaned, err := CleanNaNInf(tensor, "clamp")
		if err != nil {
			t.Errorf("Expected no error for cleaning, got: %v", err)
		}

		if cleaned.HasNaN() || cleaned.HasInf() {
			t.Error("Expected cleaned tensor to have no NaN or Inf values")
		}
	})

	t.Run("InvalidStrategy", func(t *testing.T) {
		tensor := NewTensorFromData(2, 2, []float64{1.0, math.NaN(), 3.0, 4.0})

		_, err := CleanNaNInf(tensor, "invalid_strategy")
		if err == nil {
			t.Error("Expected error for invalid cleaning strategy")
		}
	})
}

func TestValidateAndClean(t *testing.T) {
	t.Run("CleanTensor", func(t *testing.T) {
		tensor := NewTensorFromData(2, 2, []float64{1.0, math.NaN(), 3.0, 4.0})

		cleaned, err := ValidateAndClean(tensor, "test_tensor", "zero")
		if err != nil {
			t.Errorf("Expected no error for cleaning, got: %v", err)
		}

		if cleaned.HasNaN() {
			t.Error("Expected cleaned tensor to have no NaN values")
		}
	})

	t.Run("NoCleaningNeeded", func(t *testing.T) {
		tensor := NewTensorFromData(2, 2, []float64{1.0, 2.0, 3.0, 4.0})

		result, err := ValidateAndClean(tensor, "test_tensor", "")
		if err != nil {
			t.Errorf("Expected no error for clean tensor, got: %v", err)
		}

		if result != tensor {
			t.Error("Expected same tensor when no cleaning needed")
		}
	})

	t.Run("ErrorOnNaNWithoutCleaning", func(t *testing.T) {
		tensor := NewTensorFromData(2, 2, []float64{1.0, math.NaN(), 3.0, 4.0})

		_, err := ValidateAndClean(tensor, "test_tensor", "")
		if err == nil {
			t.Error("Expected error for NaN tensor without cleaning strategy")
		}
	})
}

func TestGradientClipping(t *testing.T) {
	t.Run("ClipByNorm", func(t *testing.T) {
		gc := &GradientClipping{
			MaxNorm:  1.0,
			Strategy: "norm",
		}

		gradients := []Tensor{
			NewTensorFromData(2, 2, []float64{2.0, 2.0, 2.0, 2.0}), // Large gradients
		}

		err := gc.ClipGradients(gradients)
		if err != nil {
			t.Errorf("Expected no error for gradient clipping, got: %v", err)
		}

		// Check that gradients were clipped
		norm := gradients[0].Norm()
		if norm > gc.MaxNorm+1e-6 { // Allow small numerical error
			t.Errorf("Expected gradient norm <= %f, got: %f", gc.MaxNorm, norm)
		}
	})

	t.Run("ClipByValue", func(t *testing.T) {
		gc := &GradientClipping{
			ClipValue: 1.0,
			Strategy:  "value",
		}

		gradients := []Tensor{
			NewTensorFromData(2, 2, []float64{2.0, -3.0, 0.5, 4.0}),
		}

		err := gc.ClipGradients(gradients)
		if err != nil {
			t.Errorf("Expected no error for gradient clipping, got: %v", err)
		}

		// Check that all values are within clip range
		rows, cols := gradients[0].Dims()
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				val := gradients[0].At(i, j)
				if val > gc.ClipValue || val < -gc.ClipValue {
					t.Errorf("Expected gradient value in range [-%f, %f], got: %f",
						gc.ClipValue, gc.ClipValue, val)
				}
			}
		}
	})

	t.Run("InvalidStrategy", func(t *testing.T) {
		gc := &GradientClipping{
			Strategy: "invalid",
		}

		gradients := []Tensor{
			NewTensorFromData(2, 2, []float64{1.0, 2.0, 3.0, 4.0}),
		}

		err := gc.ClipGradients(gradients)
		if err == nil {
			t.Error("Expected error for invalid clipping strategy")
		}
	})
}

func TestErrorContext(t *testing.T) {
	err := NewError(ErrInvalidInput, "test error")
	err = err.WithContext("param1", "value1")
	err = err.WithContext("param2", 42)

	if err.Context["param1"] != "value1" {
		t.Errorf("Expected context param1='value1', got: %v", err.Context["param1"])
	}

	if err.Context["param2"] != 42 {
		t.Errorf("Expected context param2=42, got: %v", err.Context["param2"])
	}

	// Test error with cause
	cause := NewError(ErrNumericalInstability, "underlying cause")
	wrappedErr := NewErrorWithCause(ErrConvergence, "wrapper error", cause)

	if wrappedErr.Unwrap() != cause {
		t.Error("Expected wrapped error to unwrap to original cause")
	}
}
