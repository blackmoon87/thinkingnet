package layers

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/activations"
	"github.com/blackmoon87/thinkingnet/pkg/core"
	"gonum.org/v1/gonum/mat"
)

// MockTensor provides a configurable mock tensor implementation for testing.
type MockTensor struct {
	rows, cols int
	data       []float64
	name       string

	// Configurable behaviors
	shouldFailAt    func(i, j int) bool
	shouldReturnNaN func(i, j int) bool
	shouldReturnInf func(i, j int) bool
	copyBehavior    func() core.Tensor
	mulBehavior     func(other core.Tensor) core.Tensor
	mulElemBehavior func(other core.Tensor) core.Tensor
}

// NewMockTensor creates a new mock tensor with default behaviors.
func NewMockTensor(rows, cols int) *MockTensor {
	return &MockTensor{
		rows:            rows,
		cols:            cols,
		data:            make([]float64, rows*cols),
		shouldFailAt:    func(i, j int) bool { return false },
		shouldReturnNaN: func(i, j int) bool { return false },
		shouldReturnInf: func(i, j int) bool { return false },
	}
}

// NewMockTensorFromData creates a mock tensor with specific data.
func NewMockTensorFromData(rows, cols int, data []float64) *MockTensor {
	mock := NewMockTensor(rows, cols)
	copy(mock.data, data)
	return mock
}

// Dims returns the tensor dimensions.
func (m *MockTensor) Dims() (int, int) {
	return m.rows, m.cols
}

// At returns the value at position (i, j).
func (m *MockTensor) At(i, j int) float64 {
	if m.shouldFailAt(i, j) {
		panic("Mock tensor configured to fail at this position")
	}
	if m.shouldReturnNaN(i, j) {
		return math.NaN()
	}
	if m.shouldReturnInf(i, j) {
		return math.Inf(1)
	}
	return m.data[i*m.cols+j]
}

// Set sets the value at position (i, j).
func (m *MockTensor) Set(i, j int, val float64) {
	if m.shouldFailAt(i, j) {
		panic("Mock tensor configured to fail at this position")
	}
	m.data[i*m.cols+j] = val
}

// Copy returns a copy of the tensor.
func (m *MockTensor) Copy() core.Tensor {
	if m.copyBehavior != nil {
		return m.copyBehavior()
	}
	newMock := NewMockTensor(m.rows, m.cols)
	copy(newMock.data, m.data)
	return newMock
}

// Mul performs matrix multiplication.
func (m *MockTensor) Mul(other core.Tensor) core.Tensor {
	if m.mulBehavior != nil {
		return m.mulBehavior(other)
	}
	// Default implementation
	otherRows, otherCols := other.Dims()
	if m.cols != otherRows {
		panic(fmt.Sprintf("Dimension mismatch: (%d,%d) * (%d,%d)", m.rows, m.cols, otherRows, otherCols))
	}

	result := NewMockTensor(m.rows, otherCols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < otherCols; j++ {
			var sum float64
			for k := 0; k < m.cols; k++ {
				sum += m.At(i, k) * other.At(k, j)
			}
			result.Set(i, j, sum)
		}
	}
	return result
}

// MulElem performs element-wise multiplication.
func (m *MockTensor) MulElem(other core.Tensor) core.Tensor {
	if m.mulElemBehavior != nil {
		return m.mulElemBehavior(other)
	}
	// Default implementation
	otherRows, otherCols := other.Dims()
	if m.rows != otherRows || m.cols != otherCols {
		panic(fmt.Sprintf("Shape mismatch: (%d,%d) vs (%d,%d)", m.rows, m.cols, otherRows, otherCols))
	}

	result := NewMockTensor(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Set(i, j, m.At(i, j)*other.At(i, j))
		}
	}
	return result
}

// T returns the transpose of the tensor.
func (m *MockTensor) T() core.Tensor {
	result := NewMockTensor(m.cols, m.rows)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Set(j, i, m.At(i, j))
		}
	}
	return result
}

// Equal checks if two tensors are equal.
func (m *MockTensor) Equal(other core.Tensor) bool {
	otherRows, otherCols := other.Dims()
	if m.rows != otherRows || m.cols != otherCols {
		return false
	}

	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			if math.Abs(m.At(i, j)-other.At(i, j)) > 1e-10 {
				return false
			}
		}
	}
	return true
}

// Name returns the tensor name.
func (m *MockTensor) Name() string {
	return m.name
}

// SetName sets the tensor name.
func (m *MockTensor) SetName(name string) {
	m.name = name
}

// ConfigureFailureAt configures the mock to fail at specific positions.
func (m *MockTensor) ConfigureFailureAt(shouldFail func(i, j int) bool) {
	m.shouldFailAt = shouldFail
}

// ConfigureNaNAt configures the mock to return NaN at specific positions.
func (m *MockTensor) ConfigureNaNAt(shouldReturnNaN func(i, j int) bool) {
	m.shouldReturnNaN = shouldReturnNaN
}

// ConfigureInfAt configures the mock to return Inf at specific positions.
func (m *MockTensor) ConfigureInfAt(shouldReturnInf func(i, j int) bool) {
	m.shouldReturnInf = shouldReturnInf
}

// TestDataGenerator provides utilities for generating test data.
type TestDataGenerator struct {
	seed int64
}

// NewTestDataGenerator creates a new test data generator.
func NewTestDataGenerator(seed int64) *TestDataGenerator {
	return &TestDataGenerator{seed: seed}
}

// GenerateRandomTensor generates a tensor with random values in the specified range.
func (g *TestDataGenerator) GenerateRandomTensor(rows, cols int, min, max float64) core.Tensor {
	rand.Seed(g.seed)
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = min + rand.Float64()*(max-min)
	}
	return core.NewTensorFromData(rows, cols, data)
}

// GenerateNormalTensor generates a tensor with normally distributed values.
func (g *TestDataGenerator) GenerateNormalTensor(rows, cols int, mean, std float64) core.Tensor {
	rand.Seed(g.seed)
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.NormFloat64()*std + mean
	}
	return core.NewTensorFromData(rows, cols, data)
}

// GenerateEdgeCaseTensor generates a tensor with edge case values.
func (g *TestDataGenerator) GenerateEdgeCaseTensor(rows, cols int, includeNaN, includeInf bool) core.Tensor {
	rand.Seed(g.seed)
	data := make([]float64, rows*cols)

	for i := range data {
		switch rand.Intn(10) {
		case 0:
			data[i] = 0.0
		case 1:
			data[i] = 1.0
		case 2:
			data[i] = -1.0
		case 3:
			data[i] = 1e10 // Very large
		case 4:
			data[i] = -1e10 // Very large negative
		case 5:
			data[i] = 1e-10 // Very small positive
		case 6:
			data[i] = -1e-10 // Very small negative
		case 7:
			if includeNaN {
				data[i] = math.NaN()
			} else {
				data[i] = rand.Float64()
			}
		case 8:
			if includeInf {
				data[i] = math.Inf(1)
			} else {
				data[i] = rand.Float64()
			}
		case 9:
			if includeInf {
				data[i] = math.Inf(-1)
			} else {
				data[i] = rand.Float64()
			}
		default:
			data[i] = rand.Float64()
		}
	}

	return core.NewTensorFromData(rows, cols, data)
}

// GenerateSequentialTensor generates a tensor with sequential values for predictable testing.
func (g *TestDataGenerator) GenerateSequentialTensor(rows, cols int, start float64) core.Tensor {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = start + float64(i)
	}
	return core.NewTensorFromData(rows, cols, data)
}

// ValidationUtils provides utilities for numerical validation and error checking.
type ValidationUtils struct {
	tolerance float64
}

// NewValidationUtils creates a new validation utilities instance.
func NewValidationUtils(tolerance float64) *ValidationUtils {
	return &ValidationUtils{tolerance: tolerance}
}

// AssertTensorEqual asserts that two tensors are equal within tolerance.
func (v *ValidationUtils) AssertTensorEqual(t *testing.T, expected, actual core.Tensor, message string) {
	t.Helper()

	expectedRows, expectedCols := expected.Dims()
	actualRows, actualCols := actual.Dims()

	if expectedRows != actualRows || expectedCols != actualCols {
		t.Errorf("%s: Shape mismatch - expected (%d,%d), got (%d,%d)",
			message, expectedRows, expectedCols, actualRows, actualCols)
		return
	}

	for i := 0; i < expectedRows; i++ {
		for j := 0; j < expectedCols; j++ {
			expectedVal := expected.At(i, j)
			actualVal := actual.At(i, j)

			if math.IsNaN(expectedVal) && math.IsNaN(actualVal) {
				continue // Both NaN is considered equal
			}

			if math.Abs(expectedVal-actualVal) > v.tolerance {
				t.Errorf("%s: Value mismatch at (%d,%d) - expected %f, got %f (tolerance: %e)",
					message, i, j, expectedVal, actualVal, v.tolerance)
			}
		}
	}
}

// AssertTensorFinite asserts that all values in a tensor are finite.
func (v *ValidationUtils) AssertTensorFinite(t *testing.T, tensor core.Tensor, message string) {
	t.Helper()

	rows, cols := tensor.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := tensor.At(i, j)
			if math.IsNaN(val) || math.IsInf(val, 0) {
				t.Errorf("%s: Non-finite value at (%d,%d): %f", message, i, j, val)
			}
		}
	}
}

// AssertTensorShape asserts that a tensor has the expected shape.
func (v *ValidationUtils) AssertTensorShape(t *testing.T, tensor core.Tensor, expectedRows, expectedCols int, message string) {
	t.Helper()

	actualRows, actualCols := tensor.Dims()
	if actualRows != expectedRows || actualCols != expectedCols {
		t.Errorf("%s: Expected shape (%d,%d), got (%d,%d)",
			message, expectedRows, expectedCols, actualRows, actualCols)
	}
}

// AssertFloatEqual asserts that two float values are equal within tolerance.
func (v *ValidationUtils) AssertFloatEqual(t *testing.T, expected, actual float64, message string) {
	t.Helper()

	if math.IsNaN(expected) && math.IsNaN(actual) {
		return // Both NaN is considered equal
	}

	if math.Abs(expected-actual) > v.tolerance {
		t.Errorf("%s: Expected %f, got %f (tolerance: %e)", message, expected, actual, v.tolerance)
	}
}

// ActivationTestHelper provides utilities for testing activation functions.
type ActivationTestHelper struct {
	validator *ValidationUtils
}

// NewActivationTestHelper creates a new activation test helper.
func NewActivationTestHelper(tolerance float64) *ActivationTestHelper {
	return &ActivationTestHelper{
		validator: NewValidationUtils(tolerance),
	}
}

// TestActivationFunction tests an activation function with various inputs.
func (h *ActivationTestHelper) TestActivationFunction(t *testing.T, activation core.Activation, testCases []ActivationTestCase) {
	t.Helper()

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			// Test forward pass
			actualForward := activation.Forward(tc.Input)
			h.validator.AssertFloatEqual(t, tc.ExpectedForward, actualForward,
				fmt.Sprintf("Forward pass for %s", tc.Name))

			// Test backward pass
			actualBackward := activation.Backward(tc.Input)
			h.validator.AssertFloatEqual(t, tc.ExpectedBackward, actualBackward,
				fmt.Sprintf("Backward pass for %s", tc.Name))
		})
	}
}

// ActivationTestCase represents a test case for activation functions.
type ActivationTestCase struct {
	Name             string
	Input            float64
	ExpectedForward  float64
	ExpectedBackward float64
}

// GetStandardActivationTestCases returns standard test cases for common activation functions.
func (h *ActivationTestHelper) GetStandardActivationTestCases(activationType string) []ActivationTestCase {
	switch activationType {
	case "relu":
		return []ActivationTestCase{
			{"Positive input", 2.0, 2.0, 1.0},
			{"Negative input", -1.0, 0.0, 0.0},
			{"Zero input", 0.0, 0.0, 0.0},
			{"Large positive", 100.0, 100.0, 1.0},
			{"Large negative", -100.0, 0.0, 0.0},
		}
	case "sigmoid":
		return []ActivationTestCase{
			{"Zero input", 0.0, 0.5, 0.25},
			{"Positive input", 2.0, 1.0 / (1.0 + math.Exp(-2.0)), 0.10499358540350662},
			{"Negative input", -2.0, 1.0 / (1.0 + math.Exp(2.0)), 0.10499358540350662},
			{"Large positive", 10.0, 0.9999546021312976, 4.539786870243442e-05},
			{"Large negative", -10.0, 4.5397868702434395e-05, 4.539786870243442e-05},
		}
	case "tanh":
		return []ActivationTestCase{
			{"Zero input", 0.0, 0.0, 1.0},
			{"Positive input", 1.0, math.Tanh(1.0), 1.0 - math.Tanh(1.0)*math.Tanh(1.0)},
			{"Negative input", -1.0, math.Tanh(-1.0), 1.0 - math.Tanh(-1.0)*math.Tanh(-1.0)},
		}
	case "linear":
		return []ActivationTestCase{
			{"Zero input", 0.0, 0.0, 1.0},
			{"Positive input", 5.0, 5.0, 1.0},
			{"Negative input", -3.0, -3.0, 1.0},
		}
	default:
		return []ActivationTestCase{}
	}
}

// GradientTestHelper provides utilities for gradient validation.
type GradientTestHelper struct {
	epsilon   float64
	validator *ValidationUtils
}

// NewGradientTestHelper creates a new gradient test helper.
func NewGradientTestHelper(epsilon, tolerance float64) *GradientTestHelper {
	return &GradientTestHelper{
		epsilon:   epsilon,
		validator: NewValidationUtils(tolerance),
	}
}

// NumericalGradientCheck performs numerical gradient checking for a layer.
func (h *GradientTestHelper) NumericalGradientCheck(t *testing.T, layer core.Layer, input core.Tensor, outputGrad core.Tensor) {
	t.Helper()

	// Forward pass to get baseline
	_, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// Backward pass to get analytical gradients
	inputGrad, err := layer.Backward(outputGrad)
	if err != nil {
		t.Fatalf("Backward pass failed: %v", err)
	}

	// Check input gradients numerically
	h.checkInputGradients(t, layer, input, outputGrad, inputGrad)

	// Check parameter gradients numerically if layer is trainable
	if layer.IsTrainable() {
		h.checkParameterGradients(t, layer, input, outputGrad)
	}
}

// checkInputGradients performs numerical gradient checking for input gradients.
func (h *GradientTestHelper) checkInputGradients(t *testing.T, layer core.Layer, input, outputGrad, analyticalGrad core.Tensor) {
	t.Helper()

	rows, cols := input.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			// Perturb input positively
			originalVal := input.At(i, j)
			input.Set(i, j, originalVal+h.epsilon)
			outputPlus, err := layer.Forward(input)
			if err != nil {
				t.Errorf("Forward pass failed with positive perturbation: %v", err)
				continue
			}

			// Perturb input negatively
			input.Set(i, j, originalVal-h.epsilon)
			outputMinus, err := layer.Forward(input)
			if err != nil {
				t.Errorf("Forward pass failed with negative perturbation: %v", err)
				continue
			}

			// Restore original value
			input.Set(i, j, originalVal)

			// Compute numerical gradient
			numericalGrad := h.computeNumericalGradient(outputPlus, outputMinus, outputGrad)
			analyticalVal := analyticalGrad.At(i, j)

			// Compare
			h.validator.AssertFloatEqual(t, numericalGrad, analyticalVal,
				fmt.Sprintf("Input gradient at (%d,%d)", i, j))
		}
	}
}

// checkParameterGradients performs numerical gradient checking for parameter gradients.
func (h *GradientTestHelper) checkParameterGradients(t *testing.T, layer core.Layer, input, outputGrad core.Tensor) {
	t.Helper()

	params := layer.Parameters()
	grads := layer.Gradients()

	if len(params) != len(grads) {
		t.Errorf("Parameter and gradient count mismatch: %d vs %d", len(params), len(grads))
		return
	}

	for paramIdx, param := range params {
		grad := grads[paramIdx]
		rows, cols := param.Dims()

		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				// Perturb parameter positively
				originalVal := param.At(i, j)
				param.Set(i, j, originalVal+h.epsilon)
				outputPlus, err := layer.Forward(input)
				if err != nil {
					t.Errorf("Forward pass failed with positive parameter perturbation: %v", err)
					continue
				}

				// Perturb parameter negatively
				param.Set(i, j, originalVal-h.epsilon)
				outputMinus, err := layer.Forward(input)
				if err != nil {
					t.Errorf("Forward pass failed with negative parameter perturbation: %v", err)
					continue
				}

				// Restore original value
				param.Set(i, j, originalVal)

				// Compute numerical gradient
				numericalGrad := h.computeNumericalGradient(outputPlus, outputMinus, outputGrad)
				analyticalVal := grad.At(i, j)

				// Compare
				h.validator.AssertFloatEqual(t, numericalGrad, analyticalVal,
					fmt.Sprintf("Parameter %d gradient at (%d,%d)", paramIdx, i, j))
			}
		}
	}
}

// computeNumericalGradient computes numerical gradient using finite differences.
func (h *GradientTestHelper) computeNumericalGradient(outputPlus, outputMinus, outputGrad core.Tensor) float64 {
	rows, cols := outputPlus.Dims()
	var gradient float64

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			diff := (outputPlus.At(i, j) - outputMinus.At(i, j)) / (2 * h.epsilon)
			gradient += diff * outputGrad.At(i, j)
		}
	}

	return gradient
}

// GetAllActivationFunctions returns all available activation functions for testing.
func GetAllActivationFunctions() map[string]core.Activation {
	return map[string]core.Activation{
		"relu":    activations.NewReLU(),
		"sigmoid": activations.NewSigmoid(),
		"tanh":    activations.NewTanh(),
		"linear":  activations.NewLinear(),
		"softmax": activations.NewSoftmax(),
	}
}

// Additional methods to implement core.Tensor interface

// Add performs element-wise addition.
func (m *MockTensor) Add(other core.Tensor) core.Tensor {
	otherRows, otherCols := other.Dims()
	if m.rows != otherRows || m.cols != otherCols {
		panic(fmt.Sprintf("Shape mismatch: (%d,%d) vs (%d,%d)", m.rows, m.cols, otherRows, otherCols))
	}

	result := NewMockTensor(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Set(i, j, m.At(i, j)+other.At(i, j))
		}
	}
	return result
}

// Sub performs element-wise subtraction.
func (m *MockTensor) Sub(other core.Tensor) core.Tensor {
	otherRows, otherCols := other.Dims()
	if m.rows != otherRows || m.cols != otherCols {
		panic(fmt.Sprintf("Shape mismatch: (%d,%d) vs (%d,%d)", m.rows, m.cols, otherRows, otherCols))
	}

	result := NewMockTensor(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Set(i, j, m.At(i, j)-other.At(i, j))
		}
	}
	return result
}

// Div performs element-wise division.
func (m *MockTensor) Div(other core.Tensor) core.Tensor {
	otherRows, otherCols := other.Dims()
	if m.rows != otherRows || m.cols != otherCols {
		panic(fmt.Sprintf("Shape mismatch: (%d,%d) vs (%d,%d)", m.rows, m.cols, otherRows, otherCols))
	}

	result := NewMockTensor(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Set(i, j, m.At(i, j)/other.At(i, j))
		}
	}
	return result
}

// Scale multiplies all elements by a scalar.
func (m *MockTensor) Scale(scalar float64) core.Tensor {
	result := NewMockTensor(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Set(i, j, m.At(i, j)*scalar)
		}
	}
	return result
}

// Pow raises all elements to a power.
func (m *MockTensor) Pow(power float64) core.Tensor {
	result := NewMockTensor(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Set(i, j, math.Pow(m.At(i, j), power))
		}
	}
	return result
}

// Sqrt computes square root of all elements.
func (m *MockTensor) Sqrt() core.Tensor {
	result := NewMockTensor(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Set(i, j, math.Sqrt(m.At(i, j)))
		}
	}
	return result
}

// Exp computes exponential of all elements.
func (m *MockTensor) Exp() core.Tensor {
	result := NewMockTensor(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Set(i, j, math.Exp(m.At(i, j)))
		}
	}
	return result
}

// Log computes natural logarithm of all elements.
func (m *MockTensor) Log() core.Tensor {
	result := NewMockTensor(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Set(i, j, math.Log(m.At(i, j)))
		}
	}
	return result
}

// Abs computes absolute value of all elements.
func (m *MockTensor) Abs() core.Tensor {
	result := NewMockTensor(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Set(i, j, math.Abs(m.At(i, j)))
		}
	}
	return result
}

// Sign computes sign of all elements.
func (m *MockTensor) Sign() core.Tensor {
	result := NewMockTensor(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			val := m.At(i, j)
			if val > 0 {
				result.Set(i, j, 1.0)
			} else if val < 0 {
				result.Set(i, j, -1.0)
			} else {
				result.Set(i, j, 0.0)
			}
		}
	}
	return result
}

// Clamp clamps all elements to the given range.
func (m *MockTensor) Clamp(min, max float64) core.Tensor {
	result := NewMockTensor(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			val := m.At(i, j)
			if val < min {
				val = min
			} else if val > max {
				val = max
			}
			result.Set(i, j, val)
		}
	}
	return result
}

// Sum computes the sum of all elements.
func (m *MockTensor) Sum() float64 {
	var sum float64
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			sum += m.At(i, j)
		}
	}
	return sum
}

// Mean computes the mean of all elements.
func (m *MockTensor) Mean() float64 {
	return m.Sum() / float64(m.rows*m.cols)
}

// Std computes the standard deviation of all elements.
func (m *MockTensor) Std() float64 {
	mean := m.Mean()
	var variance float64
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			diff := m.At(i, j) - mean
			variance += diff * diff
		}
	}
	variance /= float64(m.rows * m.cols)
	return math.Sqrt(variance)
}

// Max returns the maximum element.
func (m *MockTensor) Max() float64 {
	if m.rows == 0 || m.cols == 0 {
		return 0
	}
	max := m.At(0, 0)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			if val := m.At(i, j); val > max {
				max = val
			}
		}
	}
	return max
}

// Min returns the minimum element.
func (m *MockTensor) Min() float64 {
	if m.rows == 0 || m.cols == 0 {
		return 0
	}
	min := m.At(0, 0)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			if val := m.At(i, j); val < min {
				min = val
			}
		}
	}
	return min
}

// Norm computes the Frobenius norm.
func (m *MockTensor) Norm() float64 {
	var sum float64
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			val := m.At(i, j)
			sum += val * val
		}
	}
	return math.Sqrt(sum)
}

// Reshape changes the shape of the tensor.
func (m *MockTensor) Reshape(newRows, newCols int) core.Tensor {
	if newRows*newCols != m.rows*m.cols {
		panic(fmt.Sprintf("Cannot reshape (%d,%d) to (%d,%d): size mismatch", m.rows, m.cols, newRows, newCols))
	}

	result := NewMockTensor(newRows, newCols)
	for i := 0; i < len(m.data); i++ {
		newI := i / newCols
		newJ := i % newCols
		result.Set(newI, newJ, m.data[i])
	}
	return result
}

// Flatten returns a 1D tensor.
func (m *MockTensor) Flatten() core.Tensor {
	return m.Reshape(1, m.rows*m.cols)
}

// Shape returns the shape as a slice.
func (m *MockTensor) Shape() []int {
	return []int{m.rows, m.cols}
}

// Apply applies a function to each element.
func (m *MockTensor) Apply(fn func(i, j int, v float64) float64) core.Tensor {
	result := NewMockTensor(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Set(i, j, fn(i, j, m.At(i, j)))
		}
	}
	return result
}

// Fill sets all elements to a value.
func (m *MockTensor) Fill(value float64) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.Set(i, j, value)
		}
	}
}

// Zero sets all elements to zero.
func (m *MockTensor) Zero() {
	m.Fill(0.0)
}

// Release is a no-op for mock tensor.
func (m *MockTensor) Release() {
	// No-op for mock
}

// Row returns a specific row.
func (m *MockTensor) Row(i int) core.Tensor {
	result := NewMockTensor(1, m.cols)
	for j := 0; j < m.cols; j++ {
		result.Set(0, j, m.At(i, j))
	}
	return result
}

// Col returns a specific column.
func (m *MockTensor) Col(j int) core.Tensor {
	result := NewMockTensor(m.rows, 1)
	for i := 0; i < m.rows; i++ {
		result.Set(i, 0, m.At(i, j))
	}
	return result
}

// Slice returns a slice of the tensor.
func (m *MockTensor) Slice(r0, r1, c0, c1 int) core.Tensor {
	result := NewMockTensor(r1-r0, c1-c0)
	for i := r0; i < r1; i++ {
		for j := c0; j < c1; j++ {
			result.Set(i-r0, j-c0, m.At(i, j))
		}
	}
	return result
}

// SetRow sets a specific row.
func (m *MockTensor) SetRow(i int, data []float64) {
	if len(data) != m.cols {
		panic(fmt.Sprintf("Row data length %d doesn't match columns %d", len(data), m.cols))
	}
	for j, val := range data {
		m.Set(i, j, val)
	}
}

// SetCol sets a specific column.
func (m *MockTensor) SetCol(j int, data []float64) {
	if len(data) != m.rows {
		panic(fmt.Sprintf("Column data length %d doesn't match rows %d", len(data), m.rows))
	}
	for i, val := range data {
		m.Set(i, j, val)
	}
}

// IsEmpty returns true if the tensor is empty.
func (m *MockTensor) IsEmpty() bool {
	return m.rows == 0 || m.cols == 0
}

// IsSquare returns true if the tensor is square.
func (m *MockTensor) IsSquare() bool {
	return m.rows == m.cols
}

// IsVector returns true if the tensor is a vector.
func (m *MockTensor) IsVector() bool {
	return m.rows == 1 || m.cols == 1
}

// String returns a string representation.
func (m *MockTensor) String() string {
	return fmt.Sprintf("MockTensor(%d,%d)", m.rows, m.cols)
}

// Dot computes dot product with another tensor.
func (m *MockTensor) Dot(other core.Tensor) float64 {
	otherRows, otherCols := other.Dims()
	if m.rows != otherRows || m.cols != otherCols {
		panic(fmt.Sprintf("Shape mismatch for dot product: (%d,%d) vs (%d,%d)", m.rows, m.cols, otherRows, otherCols))
	}

	var sum float64
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			sum += m.At(i, j) * other.At(i, j)
		}
	}
	return sum
}

// AddScalar adds a scalar to all elements.
func (m *MockTensor) AddScalar(scalar float64) core.Tensor {
	result := NewMockTensor(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Set(i, j, m.At(i, j)+scalar)
		}
	}
	return result
}

// SubScalar subtracts a scalar from all elements.
func (m *MockTensor) SubScalar(scalar float64) core.Tensor {
	result := NewMockTensor(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Set(i, j, m.At(i, j)-scalar)
		}
	}
	return result
}

// DivScalar divides all elements by a scalar.
func (m *MockTensor) DivScalar(scalar float64) core.Tensor {
	result := NewMockTensor(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Set(i, j, m.At(i, j)/scalar)
		}
	}
	return result
}

// Trace computes the trace (sum of diagonal elements).
func (m *MockTensor) Trace() float64 {
	if !m.IsSquare() {
		panic("Trace is only defined for square matrices")
	}

	var trace float64
	for i := 0; i < m.rows; i++ {
		trace += m.At(i, i)
	}
	return trace
}

// Diagonal returns the diagonal elements.
func (m *MockTensor) Diagonal() core.Tensor {
	size := m.rows
	if m.cols < size {
		size = m.cols
	}

	result := NewMockTensor(size, 1)
	for i := 0; i < size; i++ {
		result.Set(i, 0, m.At(i, i))
	}
	return result
}

// Validate checks if the tensor is valid.
func (m *MockTensor) Validate() error {
	if m.rows < 0 || m.cols < 0 {
		return fmt.Errorf("invalid dimensions: (%d,%d)", m.rows, m.cols)
	}
	if len(m.data) != m.rows*m.cols {
		return fmt.Errorf("data length %d doesn't match dimensions %dx%d", len(m.data), m.rows, m.cols)
	}
	return nil
}

// HasNaN checks if the tensor contains NaN values.
func (m *MockTensor) HasNaN() bool {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			if math.IsNaN(m.At(i, j)) {
				return true
			}
		}
	}
	return false
}

// HasInf checks if the tensor contains infinite values.
func (m *MockTensor) HasInf() bool {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			if math.IsInf(m.At(i, j), 0) {
				return true
			}
		}
	}
	return false
}

// IsFinite checks if all values are finite.
func (m *MockTensor) IsFinite() bool {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			val := m.At(i, j)
			if math.IsNaN(val) || math.IsInf(val, 0) {
				return false
			}
		}
	}
	return true
}

// RawMatrix returns the underlying matrix (not implemented for mock).
func (m *MockTensor) RawMatrix() *mat.Dense {
	// Create a mat.Dense from our data for compatibility
	dense := mat.NewDense(m.rows, m.cols, m.data)
	return dense
}
