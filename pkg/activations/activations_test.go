package activations

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

const epsilon = 1e-7

// Helper function for numerical gradient checking
func numericalGradient(activation core.Activation, x float64, h float64) float64 {
	return (activation.Forward(x+h) - activation.Forward(x-h)) / (2 * h)
}

// Test ReLU activation function
func TestReLU(t *testing.T) {
	relu := NewReLU()

	if relu.Name() != "relu" {
		t.Errorf("Expected name 'relu', got '%s'", relu.Name())
	}

	// Test forward pass
	testCases := []struct {
		input    float64
		expected float64
	}{
		{-2.0, 0.0},
		{-0.1, 0.0},
		{0.0, 0.0},
		{0.1, 0.1},
		{2.0, 2.0},
		{10.0, 10.0},
	}

	for _, tc := range testCases {
		result := relu.Forward(tc.input)
		if result != tc.expected {
			t.Errorf("ReLU.Forward(%f): expected %f, got %f", tc.input, tc.expected, result)
		}
	}

	// Test backward pass
	backwardCases := []struct {
		input    float64
		expected float64
	}{
		{-2.0, 0.0},
		{-0.1, 0.0},
		{0.0, 0.0},
		{0.1, 1.0},
		{2.0, 1.0},
		{10.0, 1.0},
	}

	for _, tc := range backwardCases {
		result := relu.Backward(tc.input)
		if result != tc.expected {
			t.Errorf("ReLU.Backward(%f): expected %f, got %f", tc.input, tc.expected, result)
		}
	}
}

// Test LeakyReLU activation function
func TestLeakyReLU(t *testing.T) {
	alpha := 0.01
	leakyRelu := NewLeakyReLU(alpha)

	if leakyRelu.Name() != "leaky_relu" {
		t.Errorf("Expected name 'leaky_relu', got '%s'", leakyRelu.Name())
	}

	// Test forward pass
	testCases := []struct {
		input    float64
		expected float64
	}{
		{-2.0, -0.02},
		{-0.1, -0.001},
		{0.0, 0.0},
		{0.1, 0.1},
		{2.0, 2.0},
	}

	for _, tc := range testCases {
		result := leakyRelu.Forward(tc.input)
		if math.Abs(result-tc.expected) > epsilon {
			t.Errorf("LeakyReLU.Forward(%f): expected %f, got %f", tc.input, tc.expected, result)
		}
	}

	// Test backward pass
	backwardCases := []struct {
		input    float64
		expected float64
	}{
		{-2.0, alpha},
		{-0.1, alpha},
		{0.0, alpha},
		{0.1, 1.0},
		{2.0, 1.0},
	}

	for _, tc := range backwardCases {
		result := leakyRelu.Backward(tc.input)
		if math.Abs(result-tc.expected) > epsilon {
			t.Errorf("LeakyReLU.Backward(%f): expected %f, got %f", tc.input, tc.expected, result)
		}
	}
}

// Test Sigmoid activation function
func TestSigmoid(t *testing.T) {
	sigmoid := NewSigmoid()

	if sigmoid.Name() != "sigmoid" {
		t.Errorf("Expected name 'sigmoid', got '%s'", sigmoid.Name())
	}

	// Test forward pass
	testCases := []struct {
		input    float64
		expected float64
	}{
		{0.0, 0.5},
		{-10.0, 1.0 / (1.0 + math.Exp(10.0))},
		{10.0, 1.0 / (1.0 + math.Exp(-10.0))},
	}

	for _, tc := range testCases {
		result := sigmoid.Forward(tc.input)
		if math.Abs(result-tc.expected) > epsilon {
			t.Errorf("Sigmoid.Forward(%f): expected %f, got %f", tc.input, tc.expected, result)
		}
	}

	// Test gradient checking
	testPoints := []float64{-2.0, -0.5, 0.0, 0.5, 2.0}
	for _, x := range testPoints {
		analytical := sigmoid.Backward(x)
		numerical := numericalGradient(sigmoid, x, 1e-5)
		if math.Abs(analytical-numerical) > 1e-4 {
			t.Errorf("Sigmoid gradient check failed at x=%f: analytical=%f, numerical=%f", x, analytical, numerical)
		}
	}
}

// Test Tanh activation function
func TestTanh(t *testing.T) {
	tanh := NewTanh()

	if tanh.Name() != "tanh" {
		t.Errorf("Expected name 'tanh', got '%s'", tanh.Name())
	}

	// Test forward pass
	testCases := []struct {
		input    float64
		expected float64
	}{
		{0.0, 0.0},
		{1.0, math.Tanh(1.0)},
		{-1.0, math.Tanh(-1.0)},
	}

	for _, tc := range testCases {
		result := tanh.Forward(tc.input)
		if math.Abs(result-tc.expected) > epsilon {
			t.Errorf("Tanh.Forward(%f): expected %f, got %f", tc.input, tc.expected, result)
		}
	}

	// Test gradient checking
	testPoints := []float64{-2.0, -0.5, 0.0, 0.5, 2.0}
	for _, x := range testPoints {
		analytical := tanh.Backward(x)
		numerical := numericalGradient(tanh, x, 1e-5)
		if math.Abs(analytical-numerical) > 1e-4 {
			t.Errorf("Tanh gradient check failed at x=%f: analytical=%f, numerical=%f", x, analytical, numerical)
		}
	}
}

// Test Linear activation function
func TestLinear(t *testing.T) {
	linear := NewLinear()

	if linear.Name() != "linear" {
		t.Errorf("Expected name 'linear', got '%s'", linear.Name())
	}

	// Test forward pass
	testCases := []float64{-10.0, -1.0, 0.0, 1.0, 10.0}
	for _, input := range testCases {
		result := linear.Forward(input)
		if result != input {
			t.Errorf("Linear.Forward(%f): expected %f, got %f", input, input, result)
		}
	}

	// Test backward pass
	for _, input := range testCases {
		result := linear.Backward(input)
		if result != 1.0 {
			t.Errorf("Linear.Backward(%f): expected 1.0, got %f", input, result)
		}
	}
}

// Test ELU activation function
func TestELU(t *testing.T) {
	alpha := 1.0
	elu := NewELU(alpha)

	if elu.Name() != "elu" {
		t.Errorf("Expected name 'elu', got '%s'", elu.Name())
	}

	// Test forward pass for positive values
	positiveInputs := []float64{0.1, 1.0, 2.0}
	for _, input := range positiveInputs {
		result := elu.Forward(input)
		if result != input {
			t.Errorf("ELU.Forward(%f): expected %f, got %f", input, input, result)
		}
	}

	// Test forward pass for negative values
	negativeInputs := []struct {
		input    float64
		expected float64
	}{
		{-1.0, alpha * (math.Exp(-1.0) - 1)},
		{-0.5, alpha * (math.Exp(-0.5) - 1)},
	}

	for _, tc := range negativeInputs {
		result := elu.Forward(tc.input)
		if math.Abs(result-tc.expected) > epsilon {
			t.Errorf("ELU.Forward(%f): expected %f, got %f", tc.input, tc.expected, result)
		}
	}

	// Test gradient checking
	testPoints := []float64{-2.0, -0.5, 0.0, 0.5, 2.0}
	for _, x := range testPoints {
		analytical := elu.Backward(x)
		numerical := numericalGradient(elu, x, 1e-5)
		if math.Abs(analytical-numerical) > 1e-4 {
			t.Errorf("ELU gradient check failed at x=%f: analytical=%f, numerical=%f", x, analytical, numerical)
		}
	}
}

// Test Swish activation function
func TestSwish(t *testing.T) {
	swish := NewSwish()

	if swish.Name() != "swish" {
		t.Errorf("Expected name 'swish', got '%s'", swish.Name())
	}

	// Test gradient checking
	testPoints := []float64{-2.0, -0.5, 0.0, 0.5, 2.0}
	for _, x := range testPoints {
		analytical := swish.Backward(x)
		numerical := numericalGradient(swish, x, 1e-5)
		if math.Abs(analytical-numerical) > 1e-4 {
			t.Errorf("Swish gradient check failed at x=%f: analytical=%f, numerical=%f", x, analytical, numerical)
		}
	}
}

// Test GELU activation function
func TestGELU(t *testing.T) {
	gelu := NewGELU()

	if gelu.Name() != "gelu" {
		t.Errorf("Expected name 'gelu', got '%s'", gelu.Name())
	}

	// Test gradient checking
	testPoints := []float64{-2.0, -0.5, 0.0, 0.5, 2.0}
	for _, x := range testPoints {
		analytical := gelu.Backward(x)
		numerical := numericalGradient(gelu, x, 1e-5)
		if math.Abs(analytical-numerical) > 1e-3 { // Slightly higher tolerance for GELU approximation
			t.Errorf("GELU gradient check failed at x=%f: analytical=%f, numerical=%f", x, analytical, numerical)
		}
	}
}

// Test Softmax activation function
func TestSoftmax(t *testing.T) {
	softmax := NewSoftmax()

	if softmax.Name() != "softmax" {
		t.Errorf("Expected name 'softmax', got '%s'", softmax.Name())
	}

	// Test tensorwise application
	input := core.NewTensorFromSlice([][]float64{{1.0, 2.0, 3.0}, {0.0, 1.0, 0.0}})
	result := softmax.ApplyTensorwise(input)

	rows, cols := result.Dims()
	if rows != 2 || cols != 3 {
		t.Errorf("Softmax result has wrong dimensions: expected (2,3), got (%d,%d)", rows, cols)
	}

	// Check that each row sums to 1
	for i := 0; i < rows; i++ {
		sum := 0.0
		for j := 0; j < cols; j++ {
			val := result.At(i, j)
			if val < 0 || val > 1 {
				t.Errorf("Softmax output should be in [0,1], got %f at (%d,%d)", val, i, j)
			}
			sum += val
		}
		if math.Abs(sum-1.0) > epsilon {
			t.Errorf("Softmax row %d should sum to 1.0, got %f", i, sum)
		}
	}
}

// Test activation registry
func TestActivationRegistry(t *testing.T) {
	registry := NewActivationRegistry()

	// Test getting existing activation
	relu, err := registry.Get("relu")
	if err != nil {
		t.Errorf("Failed to get ReLU activation: %v", err)
	}
	if relu.Name() != "relu" {
		t.Errorf("Expected ReLU, got %s", relu.Name())
	}

	// Test getting non-existent activation
	_, err = registry.Get("nonexistent")
	if err == nil {
		t.Error("Expected error for non-existent activation")
	}

	// Test registering custom activation
	registry.Register("custom", func() core.Activation { return NewLinear() })
	custom, err := registry.Get("custom")
	if err != nil {
		t.Errorf("Failed to get custom activation: %v", err)
	}
	if custom.Name() != "linear" {
		t.Errorf("Expected linear (custom), got %s", custom.Name())
	}

	// Test listing activations
	names := registry.List()
	if len(names) < 9 { // Should have at least 9 default activations
		t.Errorf("Expected at least 9 activations, got %d", len(names))
	}
}

// Test global registry functions
func TestGlobalRegistry(t *testing.T) {
	// Test getting activation from global registry
	sigmoid, err := GetActivation("sigmoid")
	if err != nil {
		t.Errorf("Failed to get sigmoid from global registry: %v", err)
	}
	if sigmoid.Name() != "sigmoid" {
		t.Errorf("Expected sigmoid, got %s", sigmoid.Name())
	}

	// Test registering to global registry
	RegisterActivation("test_global", func() core.Activation { return NewReLU() })
	testGlobal, err := GetActivation("test_global")
	if err != nil {
		t.Errorf("Failed to get test_global activation: %v", err)
	}
	if testGlobal.Name() != "relu" {
		t.Errorf("Expected relu (test_global), got %s", testGlobal.Name())
	}

	// Test listing global activations
	names := ListActivations()
	found := false
	for _, name := range names {
		if name == "test_global" {
			found = true
			break
		}
	}
	if !found {
		t.Error("test_global activation not found in global list")
	}
}

// Test numerical stability
func TestNumericalStability(t *testing.T) {
	activations := []core.Activation{
		NewSigmoid(),
		NewTanh(),
		NewELU(1.0),
		NewSwish(),
		NewGELU(),
	}

	// Test with extreme values
	extremeValues := []float64{-1000.0, -100.0, 100.0, 1000.0}

	for _, activation := range activations {
		for _, x := range extremeValues {
			forward := activation.Forward(x)
			backward := activation.Backward(x)

			// Check for NaN or Inf
			if math.IsNaN(forward) || math.IsInf(forward, 0) {
				t.Errorf("%s.Forward(%f) returned invalid value: %f", activation.Name(), x, forward)
			}
			if math.IsNaN(backward) || math.IsInf(backward, 0) {
				t.Errorf("%s.Backward(%f) returned invalid value: %f", activation.Name(), x, backward)
			}
		}
	}
}

// Benchmark tests
func BenchmarkReLUForward(b *testing.B) {
	relu := NewReLU()
	x := 1.5
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		relu.Forward(x)
	}
}

func BenchmarkSigmoidForward(b *testing.B) {
	sigmoid := NewSigmoid()
	x := 1.5
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sigmoid.Forward(x)
	}
}

func BenchmarkSoftmaxTensorwise(b *testing.B) {
	softmax := NewSoftmax()
	input := core.NewTensorFromSlice([][]float64{{1.0, 2.0, 3.0, 4.0, 5.0}})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := softmax.ApplyTensorwise(input)
		result.Release()
	}
}
