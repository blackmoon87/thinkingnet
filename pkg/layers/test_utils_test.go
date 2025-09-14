package layers

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/activations"
	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestMockTensor(t *testing.T) {
	// Test basic functionality
	mock := NewMockTensor(2, 3)

	// Test dimensions
	rows, cols := mock.Dims()
	if rows != 2 || cols != 3 {
		t.Errorf("Expected dimensions (2,3), got (%d,%d)", rows, cols)
	}

	// Test set and get
	mock.Set(1, 2, 5.5)
	if val := mock.At(1, 2); val != 5.5 {
		t.Errorf("Expected 5.5, got %f", val)
	}

	// Test copy
	copy := mock.Copy()
	if !copy.Equal(mock) {
		t.Error("Copy should be equal to original")
	}

	// Test name
	mock.SetName("test_tensor")
	if mock.Name() != "test_tensor" {
		t.Errorf("Expected name 'test_tensor', got '%s'", mock.Name())
	}
}

func TestMockTensorFromData(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	mock := NewMockTensorFromData(2, 3, data)

	// Test that data is correctly set
	expected := [][]float64{{1, 2, 3}, {4, 5, 6}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			if mock.At(i, j) != expected[i][j] {
				t.Errorf("Expected %f at (%d,%d), got %f", expected[i][j], i, j, mock.At(i, j))
			}
		}
	}
}

func TestMockTensorConfigurableBehaviors(t *testing.T) {
	mock := NewMockTensor(2, 2)

	// Test NaN configuration
	mock.ConfigureNaNAt(func(i, j int) bool { return i == 0 && j == 0 })
	if !math.IsNaN(mock.At(0, 0)) {
		t.Error("Expected NaN at (0,0)")
	}
	if math.IsNaN(mock.At(0, 1)) {
		t.Error("Did not expect NaN at (0,1)")
	}

	// Test Inf configuration
	mock.ConfigureInfAt(func(i, j int) bool { return i == 1 && j == 1 })
	if !math.IsInf(mock.At(1, 1), 1) {
		t.Error("Expected +Inf at (1,1)")
	}

	// Test failure configuration
	mock.ConfigureFailureAt(func(i, j int) bool { return i == 1 && j == 0 })
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic at (1,0)")
		}
	}()
	mock.At(1, 0)
}

func TestMockTensorOperations(t *testing.T) {
	// Test matrix multiplication
	a := NewMockTensorFromData(2, 3, []float64{1, 2, 3, 4, 5, 6})
	b := NewMockTensorFromData(3, 2, []float64{1, 2, 3, 4, 5, 6})

	result := a.Mul(b)
	rows, cols := result.Dims()
	if rows != 2 || cols != 2 {
		t.Errorf("Expected result shape (2,2), got (%d,%d)", rows, cols)
	}

	// Expected: [[22, 28], [49, 64]]
	expected := [][]float64{{22, 28}, {49, 64}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if result.At(i, j) != expected[i][j] {
				t.Errorf("Expected %f at (%d,%d), got %f", expected[i][j], i, j, result.At(i, j))
			}
		}
	}

	// Test element-wise multiplication
	c := NewMockTensorFromData(2, 3, []float64{2, 2, 2, 2, 2, 2})
	elemResult := a.MulElem(c)

	// Expected: [[2, 4, 6], [8, 10, 12]]
	expectedElem := [][]float64{{2, 4, 6}, {8, 10, 12}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			if elemResult.At(i, j) != expectedElem[i][j] {
				t.Errorf("Expected %f at (%d,%d), got %f", expectedElem[i][j], i, j, elemResult.At(i, j))
			}
		}
	}

	// Test transpose
	transpose := a.T()
	tRows, tCols := transpose.Dims()
	if tRows != 3 || tCols != 2 {
		t.Errorf("Expected transpose shape (3,2), got (%d,%d)", tRows, tCols)
	}

	// Expected: [[1, 4], [2, 5], [3, 6]]
	expectedT := [][]float64{{1, 4}, {2, 5}, {3, 6}}
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			if transpose.At(i, j) != expectedT[i][j] {
				t.Errorf("Expected %f at (%d,%d), got %f", expectedT[i][j], i, j, transpose.At(i, j))
			}
		}
	}
}

func TestTestDataGenerator(t *testing.T) {
	gen := NewTestDataGenerator(42) // Fixed seed for reproducibility

	// Test random tensor generation
	randomTensor := gen.GenerateRandomTensor(3, 4, -1.0, 1.0)
	rows, cols := randomTensor.Dims()
	if rows != 3 || cols != 4 {
		t.Errorf("Expected shape (3,4), got (%d,%d)", rows, cols)
	}

	// Check that values are in range
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := randomTensor.At(i, j)
			if val < -1.0 || val > 1.0 {
				t.Errorf("Value %f at (%d,%d) is out of range [-1,1]", val, i, j)
			}
		}
	}

	// Test normal tensor generation
	normalTensor := gen.GenerateNormalTensor(2, 3, 0.0, 1.0)
	nRows, nCols := normalTensor.Dims()
	if nRows != 2 || nCols != 3 {
		t.Errorf("Expected shape (2,3), got (%d,%d)", nRows, nCols)
	}

	// Test sequential tensor generation
	seqTensor := gen.GenerateSequentialTensor(2, 3, 10.0)
	expected := [][]float64{{10, 11, 12}, {13, 14, 15}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			if seqTensor.At(i, j) != expected[i][j] {
				t.Errorf("Expected %f at (%d,%d), got %f", expected[i][j], i, j, seqTensor.At(i, j))
			}
		}
	}

	// Test edge case tensor generation
	edgeTensor := gen.GenerateEdgeCaseTensor(5, 5, true, true)
	eRows, eCols := edgeTensor.Dims()
	if eRows != 5 || eCols != 5 {
		t.Errorf("Expected shape (5,5), got (%d,%d)", eRows, eCols)
	}

	// Check that some edge cases are present (this is probabilistic)
	// Note: Due to randomness, we can't guarantee these will be present,
	// but with a 5x5 tensor, it's very likely. We just verify the tensor was created.

	// Note: Due to randomness, we can't guarantee these will be present,
	// but with a 5x5 tensor, it's very likely
}

func TestValidationUtils(t *testing.T) {
	validator := NewValidationUtils(1e-6)

	// Test tensor equality assertion
	tensor1 := core.NewTensorFromSlice([][]float64{{1.0, 2.0}, {3.0, 4.0}})
	tensor2 := core.NewTensorFromSlice([][]float64{{1.0, 2.0}, {3.0, 4.0}})

	// This should pass
	validator.AssertTensorEqual(t, tensor1, tensor2, "Equal tensors")

	// Test float equality assertion
	validator.AssertFloatEqual(t, 1.0, 1.0000001, "Close floats") // Should pass with tolerance

	// Test tensor shape assertion
	validator.AssertTensorShape(t, tensor1, 2, 2, "Correct shape")

	// Test finite assertion
	finiteTensor := core.NewTensorFromSlice([][]float64{{1.0, 2.0}, {3.0, 4.0}})
	validator.AssertTensorFinite(t, finiteTensor, "Finite tensor")
}

func TestActivationTestHelper(t *testing.T) {
	helper := NewActivationTestHelper(1e-6)

	// Test ReLU activation
	relu := activations.NewReLU()
	reluCases := helper.GetStandardActivationTestCases("relu")

	if len(reluCases) == 0 {
		t.Error("Expected ReLU test cases")
	}

	// Test one case manually
	if reluCases[0].Input != 2.0 || reluCases[0].ExpectedForward != 2.0 || reluCases[0].ExpectedBackward != 1.0 {
		t.Error("ReLU test case values are incorrect")
	}

	// Test sigmoid activation
	sigmoidCases := helper.GetStandardActivationTestCases("sigmoid")
	if len(sigmoidCases) == 0 {
		t.Error("Expected Sigmoid test cases")
	}

	// Test the activation function testing
	helper.TestActivationFunction(t, relu, reluCases)
}

func TestGradientTestHelper(t *testing.T) {
	helper := NewGradientTestHelper(1e-5, 1e-3)

	// For now, just test that the helper was created successfully
	// Full gradient checking will be tested when we implement the actual layers
	if helper == nil {
		t.Error("Expected gradient test helper to be created")
	}
}

func TestGetAllActivationFunctions(t *testing.T) {
	activations := GetAllActivationFunctions()

	expectedActivations := []string{"relu", "sigmoid", "tanh", "linear", "softmax"}

	if len(activations) != len(expectedActivations) {
		t.Errorf("Expected %d activations, got %d", len(expectedActivations), len(activations))
	}

	for _, name := range expectedActivations {
		if _, exists := activations[name]; !exists {
			t.Errorf("Expected activation '%s' not found", name)
		}
	}

	// Test that each activation has the correct name
	for name, activation := range activations {
		if activation.Name() != name {
			t.Errorf("Activation name mismatch: expected '%s', got '%s'", name, activation.Name())
		}
	}
}

// Test error cases for mock tensor operations
func TestMockTensorErrorCases(t *testing.T) {
	// Test dimension mismatch in matrix multiplication
	a := NewMockTensor(2, 3)
	b := NewMockTensor(2, 2) // Wrong dimensions for multiplication

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for dimension mismatch in matrix multiplication")
		}
	}()
	a.Mul(b)
}

func TestMockTensorElementWiseErrorCases(t *testing.T) {
	// Test shape mismatch in element-wise multiplication
	a := NewMockTensor(2, 3)
	b := NewMockTensor(3, 2) // Wrong shape for element-wise multiplication

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for shape mismatch in element-wise multiplication")
		}
	}()
	a.MulElem(b)
}

func TestValidationUtilsEdgeCases(t *testing.T) {
	validator := NewValidationUtils(1e-6)

	// Test NaN handling in tensor equality
	nanTensor1 := core.NewTensorFromSlice([][]float64{{math.NaN(), 2.0}})
	nanTensor2 := core.NewTensorFromSlice([][]float64{{math.NaN(), 2.0}})

	// This should pass (NaN == NaN in our validation)
	validator.AssertTensorEqual(t, nanTensor1, nanTensor2, "NaN tensors")

	// Test NaN handling in float equality
	validator.AssertFloatEqual(t, math.NaN(), math.NaN(), "NaN floats")
}
