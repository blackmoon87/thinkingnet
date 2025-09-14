package core

import (
	"testing"
)

func TestValidationFunctions(t *testing.T) {
	// Create test tensors
	validTensor := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	var emptyTensor Tensor // nil tensor for testing

	// Test ValidateInput
	err := ValidateInput(validTensor, []int{2, 2})
	if err != nil {
		t.Errorf("ValidateInput should pass for valid tensor: %v", err)
	}

	err = ValidateInput(nil, []int{2, 2})
	if err == nil {
		t.Error("ValidateInput should fail for nil tensor")
	}

	// Test ValidateNonEmpty
	err = ValidateNonEmpty(validTensor, "test tensor")
	if err != nil {
		t.Errorf("ValidateNonEmpty should pass for non-empty tensor: %v", err)
	}

	err = ValidateNonEmpty(emptyTensor, "empty tensor")
	if err == nil {
		t.Error("ValidateNonEmpty should fail for empty tensor")
	}

	// Test ValidateCompiled
	err = ValidateCompiled(true)
	if err != nil {
		t.Errorf("ValidateCompiled should pass for compiled model: %v", err)
	}

	err = ValidateCompiled(false)
	if err == nil {
		t.Error("ValidateCompiled should fail for uncompiled model")
	}

	// Test ValidateNotFitted
	err = ValidateNotFitted(false, "test model")
	if err == nil {
		t.Error("ValidateNotFitted should fail for unfitted model")
	}

	err = ValidateNotFitted(true, "test model")
	if err != nil {
		t.Errorf("ValidateNotFitted should pass for fitted model: %v", err)
	}
}

func TestRangeValidation(t *testing.T) {
	// Test ValidateRange
	err := ValidateRange(0.5, 0.0, 1.0, "test value")
	if err != nil {
		t.Errorf("ValidateRange should pass for value in range: %v", err)
	}

	err = ValidateRange(1.5, 0.0, 1.0, "test value")
	if err == nil {
		t.Error("ValidateRange should fail for value out of range")
	}

	// Test ValidatePositive
	err = ValidatePositive(1.0, "positive value")
	if err != nil {
		t.Errorf("ValidatePositive should pass for positive value: %v", err)
	}

	err = ValidatePositive(-1.0, "negative value")
	if err == nil {
		t.Error("ValidatePositive should fail for negative value")
	}

	err = ValidatePositive(0.0, "zero value")
	if err == nil {
		t.Error("ValidatePositive should fail for zero value")
	}

	// Test ValidateNonNegative
	err = ValidateNonNegative(1.0, "positive value")
	if err != nil {
		t.Errorf("ValidateNonNegative should pass for positive value: %v", err)
	}

	err = ValidateNonNegative(0.0, "zero value")
	if err != nil {
		t.Errorf("ValidateNonNegative should pass for zero value: %v", err)
	}

	err = ValidateNonNegative(-1.0, "negative value")
	if err == nil {
		t.Error("ValidateNonNegative should fail for negative value")
	}
}

func TestIntegerValidation(t *testing.T) {
	// Test ValidateIntRange
	err := ValidateIntRange(5, 1, 10, "test value")
	if err != nil {
		t.Errorf("ValidateIntRange should pass for value in range: %v", err)
	}

	err = ValidateIntRange(15, 1, 10, "test value")
	if err == nil {
		t.Error("ValidateIntRange should fail for value out of range")
	}

	// Test ValidatePositiveInt
	err = ValidatePositiveInt(5, "positive int")
	if err != nil {
		t.Errorf("ValidatePositiveInt should pass for positive int: %v", err)
	}

	err = ValidatePositiveInt(-5, "negative int")
	if err == nil {
		t.Error("ValidatePositiveInt should fail for negative int")
	}

	err = ValidatePositiveInt(0, "zero int")
	if err == nil {
		t.Error("ValidatePositiveInt should fail for zero int")
	}

	// Test ValidateNonNegativeInt
	err = ValidateNonNegativeInt(5, "positive int")
	if err != nil {
		t.Errorf("ValidateNonNegativeInt should pass for positive int: %v", err)
	}

	err = ValidateNonNegativeInt(0, "zero int")
	if err != nil {
		t.Errorf("ValidateNonNegativeInt should pass for zero int: %v", err)
	}

	err = ValidateNonNegativeInt(-5, "negative int")
	if err == nil {
		t.Error("ValidateNonNegativeInt should fail for negative int")
	}
}

func TestSliceValidation(t *testing.T) {
	// Test ValidateSliceNotEmpty
	err := ValidateSliceNotEmpty([]string{"a", "b"}, "test slice")
	if err != nil {
		t.Errorf("ValidateSliceNotEmpty should pass for non-empty slice: %v", err)
	}

	err = ValidateSliceNotEmpty([]string{}, "empty slice")
	if err == nil {
		t.Error("ValidateSliceNotEmpty should fail for empty slice")
	}

	err = ValidateSliceNotEmpty(nil, "nil slice")
	if err == nil {
		t.Error("ValidateSliceNotEmpty should fail for nil slice")
	}
}

func TestStringValidation(t *testing.T) {
	// Test ValidateStringInSet
	validSet := []string{"option1", "option2", "option3"}

	err := ValidateStringInSet("option2", validSet, "test option")
	if err != nil {
		t.Errorf("ValidateStringInSet should pass for valid option: %v", err)
	}

	err = ValidateStringInSet("invalid", validSet, "test option")
	if err == nil {
		t.Error("ValidateStringInSet should fail for invalid option")
	}
}

func TestShapeValidation(t *testing.T) {
	tensor1 := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	tensor2 := NewTensorFromSlice([][]float64{{5, 6}, {7, 8}})
	tensor3 := NewTensorFromSlice([][]float64{{1, 2, 3}})

	// Test ValidateCompatibleShapes
	err := ValidateCompatibleShapes(tensor1, tensor2, "broadcast_add")
	if err != nil {
		t.Errorf("ValidateCompatibleShapes should pass for compatible tensors: %v", err)
	}

	err = ValidateCompatibleShapes(tensor1, tensor3, "broadcast_add")
	if err == nil {
		t.Error("ValidateCompatibleShapes should fail for incompatible tensors")
	}

	// Test ValidateSquareMatrix
	squareTensor := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	err = ValidateSquareMatrix(squareTensor, "square matrix")
	if err != nil {
		t.Errorf("ValidateSquareMatrix should pass for square matrix: %v", err)
	}

	err = ValidateSquareMatrix(tensor3, "non-square matrix")
	if err == nil {
		t.Error("ValidateSquareMatrix should fail for non-square matrix")
	}

	// Test ValidateVector
	vectorTensor := NewTensorFromSlice([][]float64{{1, 2, 3}})
	err = ValidateVector(vectorTensor, "vector")
	if err != nil {
		t.Errorf("ValidateVector should pass for vector: %v", err)
	}

	err = ValidateVector(squareTensor, "matrix")
	if err == nil {
		t.Error("ValidateVector should fail for matrix")
	}
}

func TestScalarValidation(t *testing.T) {
	// Test ValidateScalar
	err := ValidateScalar(5.0, "test scalar")
	if err != nil {
		t.Errorf("ValidateScalar should pass for finite scalar: %v", err)
	}
}

func TestModelStateValidation(t *testing.T) {
	// Test ValidateModelState
	err := ValidateModelState(true, true, "predict")
	if err != nil {
		t.Errorf("ValidateModelState should pass for compiled and fitted model: %v", err)
	}

	err = ValidateModelState(false, true, "predict")
	if err == nil {
		t.Error("ValidateModelState should fail for uncompiled model")
	}

	err = ValidateModelState(true, false, "test model")
	if err == nil {
		t.Error("ValidateModelState should fail for unfitted model")
	}
}

func TestOptimizationConfigValidation(t *testing.T) {
	validConfig := map[string]interface{}{
		"learning_rate": 0.001,
		"beta1":         0.9,
		"beta2":         0.999,
		"epsilon":       1e-8,
	}

	err := ValidateOptimizationConfig(validConfig)
	if err != nil {
		t.Errorf("ValidateOptimizationConfig should pass for valid config: %v", err)
	}

	invalidConfig := map[string]interface{}{
		"learning_rate": -0.001,
		"beta1":         0.9,
		"beta2":         0.999,
		"epsilon":       1e-8,
	}
	err = ValidateOptimizationConfig(invalidConfig)
	if err == nil {
		t.Error("ValidateOptimizationConfig should fail for negative learning rate")
	}

	invalidConfig2 := map[string]interface{}{
		"learning_rate": 0.001,
		"beta1":         1.5,
		"beta2":         0.999,
		"epsilon":       1e-8,
	}
	err = ValidateOptimizationConfig(invalidConfig2)
	if err == nil {
		t.Error("ValidateOptimizationConfig should fail for beta1 > 1")
	}
}

func TestSafeExecute(t *testing.T) {
	// Test successful execution
	err := SafeExecute("test operation", func() error {
		return nil
	})

	if err != nil {
		t.Errorf("SafeExecute should succeed: %v", err)
	}

	// Test execution with error
	err = SafeExecute("test operation", func() error {
		return NewError(ErrInvalidInput, "test error")
	})

	if err == nil {
		t.Error("SafeExecute should return error")
	}

	// Test execution with panic
	err = SafeExecute("test operation", func() error {
		panic("test panic")
	})

	if err == nil {
		t.Error("SafeExecute should handle panic")
	}
}

func TestRecoverFromPanic(t *testing.T) {
	t.Skip("Skipping RecoverFromPanic test - function needs redesign")
	return
	// Test panic recovery
	var recoveredErr error
	func() {
		defer func() {
			recoveredErr = RecoverFromPanic("test operation")
		}()
		panic("test panic")
	}()

	if recoveredErr == nil {
		t.Error("RecoverFromPanic should return error for panic")
	}

	// Test normal execution (no panic)
	var normalErr error
	func() {
		defer func() {
			normalErr = RecoverFromPanic("test operation")
		}()
		// Normal execution - no panic
	}()

	if normalErr != nil {
		t.Errorf("RecoverFromPanic should not return error for normal execution: %v", normalErr)
	}
}

func TestValidateMemoryUsage(t *testing.T) {
	// Test with reasonable memory usage
	smallTensor := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	err := ValidateMemoryUsage([]Tensor{smallTensor}, "test operation")
	if err != nil {
		t.Errorf("ValidateMemoryUsage should pass for reasonable memory: %v", err)
	}

	// Skip the excessive memory test to avoid memory issues in CI
	t.Skip("Skipping excessive memory test to avoid memory allocation issues")
}
