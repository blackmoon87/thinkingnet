package activations

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestApplyActivation(t *testing.T) {
	relu := NewReLU()
	input := core.NewTensorFromSlice([][]float64{{-1.0, 0.0, 1.0}, {-2.0, 2.0, 3.0}})

	result := ApplyActivation(relu, input)

	expected := [][]float64{{0.0, 0.0, 1.0}, {0.0, 2.0, 3.0}}
	rows, cols := result.Dims()

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if result.At(i, j) != expected[i][j] {
				t.Errorf("ApplyActivation: expected %f at (%d,%d), got %f",
					expected[i][j], i, j, result.At(i, j))
			}
		}
	}
}

func TestApplyActivationBackward(t *testing.T) {
	sigmoid := NewSigmoid()
	input := core.NewTensorFromSlice([][]float64{{0.0, 1.0}, {-1.0, 2.0}})
	gradOutput := core.NewTensorFromSlice([][]float64{{1.0, 1.0}, {1.0, 1.0}})

	result := ApplyActivationBackward(sigmoid, input, gradOutput)

	rows, cols := result.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			expected := sigmoid.Backward(input.At(i, j)) * gradOutput.At(i, j)
			if math.Abs(result.At(i, j)-expected) > 1e-10 {
				t.Errorf("ApplyActivationBackward: expected %f at (%d,%d), got %f",
					expected, i, j, result.At(i, j))
			}
		}
	}
}

func TestApplyActivationInPlace(t *testing.T) {
	relu := NewReLU()
	tensor := core.NewTensorFromSlice([][]float64{{-1.0, 0.0, 1.0}, {-2.0, 2.0, 3.0}})

	ApplyActivationInPlace(relu, tensor)

	expected := [][]float64{{0.0, 0.0, 1.0}, {0.0, 2.0, 3.0}}
	rows, cols := tensor.Dims()

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if tensor.At(i, j) != expected[i][j] {
				t.Errorf("ApplyActivationInPlace: expected %f at (%d,%d), got %f",
					expected[i][j], i, j, tensor.At(i, j))
			}
		}
	}
}

func TestBatchApplyActivation(t *testing.T) {
	relu := NewReLU()
	inputs := []core.Tensor{
		core.NewTensorFromSlice([][]float64{{-1.0, 1.0}}),
		core.NewTensorFromSlice([][]float64{{-2.0, 2.0}}),
	}

	results := BatchApplyActivation(relu, inputs)

	if len(results) != len(inputs) {
		t.Errorf("BatchApplyActivation: expected %d results, got %d", len(inputs), len(results))
	}

	expected := [][]float64{{0.0, 1.0}, {0.0, 2.0}}
	for i, result := range results {
		rows, cols := result.Dims()
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				if result.At(r, c) != expected[i][c] {
					t.Errorf("BatchApplyActivation[%d]: expected %f at (%d,%d), got %f",
						i, expected[i][c], r, c, result.At(r, c))
				}
			}
		}
	}
}

func TestCompareActivations(t *testing.T) {
	activations := []core.Activation{
		NewReLU(),
		NewLinear(),
		NewSigmoid(),
	}

	input := core.NewTensorFromSlice([][]float64{{-1.0, 0.0, 1.0}})
	results := CompareActivations(activations, input)

	if len(results) != len(activations) {
		t.Errorf("CompareActivations: expected %d results, got %d", len(activations), len(results))
	}

	// Check that each activation is present
	for _, activation := range activations {
		if _, exists := results[activation.Name()]; !exists {
			t.Errorf("CompareActivations: missing result for %s", activation.Name())
		}
	}

	// Check ReLU result
	reluResult := results["relu"]
	if reluResult.At(0, 0) != 0.0 || reluResult.At(0, 1) != 0.0 || reluResult.At(0, 2) != 1.0 {
		t.Error("CompareActivations: ReLU result incorrect")
	}

	// Check Linear result
	linearResult := results["linear"]
	if linearResult.At(0, 0) != -1.0 || linearResult.At(0, 1) != 0.0 || linearResult.At(0, 2) != 1.0 {
		t.Error("CompareActivations: Linear result incorrect")
	}
}

func TestActivationLayer(t *testing.T) {
	relu := NewReLU()
	layer := NewActivationLayer(relu)

	// Test layer properties
	if layer.Name() != "activation_relu" {
		t.Errorf("Expected layer name 'activation_relu', got '%s'", layer.Name())
	}

	if layer.IsTrainable() {
		t.Error("Activation layer should not be trainable")
	}

	if layer.ParameterCount() != 0 {
		t.Errorf("Expected 0 parameters, got %d", layer.ParameterCount())
	}

	if len(layer.Parameters()) != 0 {
		t.Errorf("Expected 0 parameters, got %d", len(layer.Parameters()))
	}

	if len(layer.Gradients()) != 0 {
		t.Errorf("Expected 0 gradients, got %d", len(layer.Gradients()))
	}

	// Test output shape
	inputShape := []int{32, 10}
	outputShape := layer.OutputShape(inputShape)
	if len(outputShape) != len(inputShape) {
		t.Error("Output shape should match input shape")
	}
	for i, dim := range outputShape {
		if dim != inputShape[i] {
			t.Errorf("Output shape mismatch at dimension %d: expected %d, got %d", i, inputShape[i], dim)
		}
	}

	// Test forward pass
	input := core.NewTensorFromSlice([][]float64{{-1.0, 0.0, 1.0}, {-2.0, 2.0, 3.0}})
	output := layer.Forward(input)

	expected := [][]float64{{0.0, 0.0, 1.0}, {0.0, 2.0, 3.0}}
	rows, cols := output.Dims()

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if output.At(i, j) != expected[i][j] {
				t.Errorf("Forward: expected %f at (%d,%d), got %f",
					expected[i][j], i, j, output.At(i, j))
			}
		}
	}

	// Test backward pass
	gradOutput := core.NewTensorFromSlice([][]float64{{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}})
	gradInput := layer.Backward(gradOutput)

	expectedGrad := [][]float64{{0.0, 0.0, 1.0}, {0.0, 1.0, 1.0}}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if gradInput.At(i, j) != expectedGrad[i][j] {
				t.Errorf("Backward: expected %f at (%d,%d), got %f",
					expectedGrad[i][j], i, j, gradInput.At(i, j))
			}
		}
	}
}

func TestActivationLayerSoftmax(t *testing.T) {
	softmax := NewSoftmax()
	layer := NewActivationLayer(softmax)

	input := core.NewTensorFromSlice([][]float64{{1.0, 2.0, 3.0}})
	output := layer.Forward(input)

	// Check that output sums to 1
	sum := 0.0
	_, cols := output.Dims()
	for j := 0; j < cols; j++ {
		val := output.At(0, j)
		if val < 0 || val > 1 {
			t.Errorf("Softmax output should be in [0,1], got %f", val)
		}
		sum += val
	}

	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("Softmax output should sum to 1.0, got %f", sum)
	}

	// Test backward pass (simplified)
	gradOutput := core.NewTensorFromSlice([][]float64{{0.1, 0.2, 0.3}})
	gradInput := layer.Backward(gradOutput)

	// For softmax, this is a simplified test
	if gradInput == nil {
		t.Error("Backward pass should return a gradient")
	}
}

func TestActivationLayerSetName(t *testing.T) {
	relu := NewReLU()
	layer := NewActivationLayer(relu)

	newName := "custom_activation"
	layer.SetName(newName)

	if layer.Name() != newName {
		t.Errorf("Expected layer name '%s', got '%s'", newName, layer.Name())
	}
}

func TestActivationLayerBackwardWithoutForward(t *testing.T) {
	relu := NewReLU()
	layer := NewActivationLayer(relu)

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic when calling Backward without Forward")
		}
	}()

	gradOutput := core.NewTensorFromSlice([][]float64{{1.0}})
	layer.Backward(gradOutput)
}

// Benchmark tests
func BenchmarkApplyActivation(b *testing.B) {
	relu := NewReLU()
	input := core.NewTensorFromSlice([][]float64{{-1.0, 0.0, 1.0, 2.0, 3.0}})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := ApplyActivation(relu, input)
		result.Release()
	}
}

func BenchmarkApplyActivationInPlace(b *testing.B) {
	relu := NewReLU()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor := core.NewTensorFromSlice([][]float64{{-1.0, 0.0, 1.0, 2.0, 3.0}})
		ApplyActivationInPlace(relu, tensor)
		tensor.Release()
	}
}

func BenchmarkActivationLayerForward(b *testing.B) {
	relu := NewReLU()
	layer := NewActivationLayer(relu)
	input := core.NewTensorFromSlice([][]float64{{-1.0, 0.0, 1.0, 2.0, 3.0}})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := layer.Forward(input)
		result.Release()
	}
}
