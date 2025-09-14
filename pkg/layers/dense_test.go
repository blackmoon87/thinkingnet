package layers

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/activations"
	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestDenseLayerCreation(t *testing.T) {
	// Test basic creation
	layer := NewDense(10, nil)
	if layer.GetUnits() != 10 {
		t.Errorf("Expected 10 units, got %d", layer.GetUnits())
	}
	if layer.useBias != true {
		t.Error("Expected useBias to be true by default")
	}
	if layer.built {
		t.Error("Layer should not be built initially")
	}

	// Test with custom config
	config := &DenseConfig{
		Activation:  activations.NewReLU(),
		UseBias:     false,
		Initializer: HeNormal,
	}
	layer2 := NewDense(5, config)
	if layer2.GetUnits() != 5 {
		t.Errorf("Expected 5 units, got %d", layer2.GetUnits())
	}
	if layer2.useBias != false {
		t.Error("Expected useBias to be false")
	}
	if layer2.GetActivation().Name() != "relu" {
		t.Errorf("Expected ReLU activation, got %s", layer2.GetActivation().Name())
	}
}

func TestDenseBuild(t *testing.T) {
	layer := NewDense(3, nil)

	// Build with input dimension 4
	err := layer.Build(4)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	if !layer.built {
		t.Error("Layer should be built after Build() call")
	}
	if layer.inputDim != 4 {
		t.Errorf("Expected input dim 4, got %d", layer.inputDim)
	}

	// Check weight dimensions
	wRows, wCols := layer.weights.Dims()
	if wRows != 4 || wCols != 3 {
		t.Errorf("Expected weight shape (4,3), got (%d,%d)", wRows, wCols)
	}

	// Check bias dimensions (if enabled)
	if layer.useBias {
		bRows, bCols := layer.biases.Dims()
		if bRows != 1 || bCols != 3 {
			t.Errorf("Expected bias shape (1,3), got (%d,%d)", bRows, bCols)
		}
	}

	// Test IsTrainable
	if !layer.IsTrainable() {
		t.Error("Dense layer should be trainable")
	}

	// Test OutputShape
	outputShape, err := layer.OutputShape([]int{32, 4})
	if err != nil {
		t.Fatalf("OutputShape failed: %v", err)
	}
	expected := []int{32, 3}
	if len(outputShape) != len(expected) {
		t.Errorf("Expected output shape length %d, got %d", len(expected), len(outputShape))
	}
	for i, dim := range expected {
		if outputShape[i] != dim {
			t.Errorf("Expected output shape[%d] = %d, got %d", i, dim, outputShape[i])
		}
	}
}

func TestDenseForward(t *testing.T) {
	// Create layer with linear activation
	layer := NewDense(2, &DenseConfig{
		Activation:  activations.NewLinear(),
		UseBias:     true,
		Initializer: Zeros, // Use zeros for predictable results
	})

	// Create input: 2 samples, 3 features
	input := core.NewTensorFromSlice([][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
	})

	// Forward pass (will auto-build)
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// Check output shape
	outRows, outCols := output.Dims()
	if outRows != 2 || outCols != 2 {
		t.Errorf("Expected output shape (2,2), got (%d,%d)", outRows, outCols)
	}

	// With zero weights and biases, output should be all zeros
	for i := 0; i < outRows; i++ {
		for j := 0; j < outCols; j++ {
			if math.Abs(output.At(i, j)) > 1e-10 {
				t.Errorf("Expected zero output with zero weights, got %f at (%d,%d)",
					output.At(i, j), i, j)
			}
		}
	}
}

func TestDenseForwardWithReLU(t *testing.T) {
	// Create layer with ReLU activation
	layer := NewDense(2, &DenseConfig{
		Activation:  activations.NewReLU(),
		UseBias:     false,
		Initializer: Zeros,
	})

	// Manually set weights after building
	input := core.NewTensorFromSlice([][]float64{{1.0, -1.0}})
	_, err := layer.Forward(input) // Build the layer
	if err != nil {
		t.Fatalf("Initial forward pass failed: %v", err)
	}

	// Set specific weights for testing
	layer.weights.Set(0, 0, 1.0)  // Positive weight
	layer.weights.Set(1, 0, -1.0) // Negative weight
	layer.weights.Set(0, 1, -1.0) // Negative weight
	layer.weights.Set(1, 1, 1.0)  // Positive weight

	// Forward pass
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// Expected: [1*1 + (-1)*(-1), 1*(-1) + (-1)*1] = [2, -2]
	// After ReLU: [2, 0]
	expected := [][]float64{{2.0, 0.0}}

	for i := 0; i < 1; i++ {
		for j := 0; j < 2; j++ {
			if math.Abs(output.At(i, j)-expected[i][j]) > 1e-10 {
				t.Errorf("Expected %f, got %f at (%d,%d)",
					expected[i][j], output.At(i, j), i, j)
			}
		}
	}
}

func TestDenseBackward(t *testing.T) {
	// Create simple layer for gradient testing
	layer := NewDense(1, &DenseConfig{
		Activation:  activations.NewLinear(),
		UseBias:     false,
		Initializer: Zeros,
	})

	// Input: single sample, single feature
	input := core.NewTensorFromSlice([][]float64{{2.0}})

	// Forward pass to build the layer
	_, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// Set weight to 3.0
	layer.weights.Set(0, 0, 3.0)

	// Forward pass again to cache input for backward pass
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// Expected output: 2.0 * 3.0 = 6.0
	if math.Abs(output.At(0, 0)-6.0) > 1e-10 {
		t.Errorf("Expected output 6.0, got %f", output.At(0, 0))
	}

	// Backward pass with gradient 1.0
	grad := core.NewTensorFromSlice([][]float64{{1.0}})
	inputGrad, err := layer.Backward(grad)
	if err != nil {
		t.Fatalf("Backward pass failed: %v", err)
	}

	// Check weight gradient: should be input * grad = 2.0 * 1.0 = 2.0
	weightGrad := layer.Gradients()[0]
	if math.Abs(weightGrad.At(0, 0)-2.0) > 1e-10 {
		t.Errorf("Expected weight gradient 2.0, got %f", weightGrad.At(0, 0))
	}

	// Check input gradient: should be weight * grad = 3.0 * 1.0 = 3.0
	if math.Abs(inputGrad.At(0, 0)-3.0) > 1e-10 {
		t.Errorf("Expected input gradient 3.0, got %f", inputGrad.At(0, 0))
	}
}

func TestDenseGradientFlow(t *testing.T) {
	// Test gradient flow through ReLU activation
	layer := NewDense(2, &DenseConfig{
		Activation:  activations.NewReLU(),
		UseBias:     true,
		Initializer: Zeros,
	})

	// Input that will produce both positive and negative pre-activations
	input := core.NewTensorFromSlice([][]float64{{1.0, -1.0}})

	// Build and set weights
	_, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}
	layer.weights.Set(0, 0, 1.0)
	layer.weights.Set(1, 0, 1.0) // Sum = 1*1 + (-1)*1 = 0, ReLU -> 0
	layer.weights.Set(0, 1, 2.0)
	layer.weights.Set(1, 1, -1.0) // Sum = 1*2 + (-1)*(-1) = 3, ReLU -> 3

	// Set biases to zero for predictable results
	layer.biases.Set(0, 0, 0.0)
	layer.biases.Set(0, 1, 0.0)

	// Forward pass to cache input for backward pass
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// Expected pre-activation: [0, 3], after ReLU: [0, 3]
	if math.Abs(output.At(0, 0)-0.0) > 1e-10 {
		t.Errorf("Expected output[0] = 0.0, got %f", output.At(0, 0))
	}
	if math.Abs(output.At(0, 1)-3.0) > 1e-10 {
		t.Errorf("Expected output[1] = 3.0, got %f", output.At(0, 1))
	}

	// Backward pass
	grad := core.NewTensorFromSlice([][]float64{{1.0, 1.0}})
	_, err = layer.Backward(grad)
	if err != nil {
		t.Fatalf("Backward pass failed: %v", err)
	}

	// For ReLU: gradient flows through only where pre-activation > 0
	// First unit: pre-activation = 0, so gradient should be 0
	// Second unit: pre-activation = 1 > 0, so gradient flows through

	// Check that gradients are computed correctly
	weightGrads := layer.Gradients()[0]

	// Weight gradients for first unit should be 0 (ReLU derivative = 0)
	if math.Abs(weightGrads.At(0, 0)) > 1e-10 {
		t.Errorf("Expected weight grad[0,0] = 0, got %f", weightGrads.At(0, 0))
	}
	if math.Abs(weightGrads.At(1, 0)) > 1e-10 {
		t.Errorf("Expected weight grad[1,0] = 0, got %f", weightGrads.At(1, 0))
	}

	// Weight gradients for second unit should be input values (ReLU derivative = 1)
	if math.Abs(weightGrads.At(0, 1)-1.0) > 1e-10 {
		t.Errorf("Expected weight grad[0,1] = 1.0, got %f", weightGrads.At(0, 1))
	}
	if math.Abs(weightGrads.At(1, 1)-(-1.0)) > 1e-10 {
		t.Errorf("Expected weight grad[1,1] = -1.0, got %f", weightGrads.At(1, 1))
	}
}

func TestDenseParameterCount(t *testing.T) {
	// Test parameter count with bias
	layer1 := NewDense(5, &DenseConfig{UseBias: true})
	_ = layer1.Build(3)

	expected1 := 3*5 + 5 // weights + biases
	if layer1.ParameterCount() != expected1 {
		t.Errorf("Expected %d parameters, got %d", expected1, layer1.ParameterCount())
	}

	// Test parameter count without bias
	layer2 := NewDense(5, &DenseConfig{UseBias: false})
	_ = layer2.Build(3)

	expected2 := 3 * 5 // only weights
	if layer2.ParameterCount() != expected2 {
		t.Errorf("Expected %d parameters, got %d", expected2, layer2.ParameterCount())
	}
}

func TestDenseOutputShape(t *testing.T) {
	layer := NewDense(10, nil)

	inputShape := []int{32, 784} // batch_size=32, features=784
	outputShape, err := layer.OutputShape(inputShape)
	if err != nil {
		t.Fatalf("OutputShape failed: %v", err)
	}

	expected := []int{32, 10}
	if len(outputShape) != len(expected) {
		t.Errorf("Expected output shape length %d, got %d", len(expected), len(outputShape))
	}

	for i, dim := range expected {
		if outputShape[i] != dim {
			t.Errorf("Expected output shape[%d] = %d, got %d", i, dim, outputShape[i])
		}
	}
}

func TestWeightInitialization(t *testing.T) {
	testCases := []struct {
		name        string
		initializer WeightInitializer
		rows, cols  int
	}{
		{"XavierUniform", XavierUniform, 10, 5},
		{"XavierNormal", XavierNormal, 10, 5},
		{"HeUniform", HeUniform, 10, 5},
		{"HeNormal", HeNormal, 10, 5},
		{"RandomNormal", RandomNormal, 10, 5},
		{"Zeros", Zeros, 10, 5},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			weights := InitializeWeights(tc.rows, tc.cols, tc.initializer)

			rows, cols := weights.Dims()
			if rows != tc.rows || cols != tc.cols {
				t.Errorf("Expected shape (%d,%d), got (%d,%d)",
					tc.rows, tc.cols, rows, cols)
			}

			// Check that weights are initialized (not all zeros unless Zeros initializer)
			allZeros := true
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					if math.Abs(weights.At(i, j)) > 1e-10 {
						allZeros = false
						break
					}
				}
				if !allZeros {
					break
				}
			}

			if tc.initializer == Zeros {
				if !allZeros {
					t.Error("Zeros initializer should produce all zeros")
				}
			} else {
				if allZeros {
					t.Errorf("%s initializer should not produce all zeros", tc.name)
				}
			}
		})
	}
}

func TestDenseGettersAndSetters(t *testing.T) {
	// Test GetUnits
	layer := NewDense(5, nil)
	if layer.GetUnits() != 5 {
		t.Errorf("Expected 5 units, got %d", layer.GetUnits())
	}

	// Test GetActivation and SetActivation
	layer2 := NewDense(3, &DenseConfig{
		Activation: activations.NewReLU(),
	})

	activation := layer2.GetActivation()
	if activation == nil || activation.Name() != "relu" {
		t.Errorf("Expected ReLU activation, got %v", activation)
	}

	// Test SetActivation
	newActivation := activations.NewSigmoid()
	layer2.SetActivation(newActivation)

	updatedActivation := layer2.GetActivation()
	if updatedActivation == nil || updatedActivation.Name() != "sigmoid" {
		t.Errorf("Expected Sigmoid activation after setting, got %v", updatedActivation)
	}
}
func TestBaseLayerFunctionality(t *testing.T) {
	layer := NewDense(3, nil)

	// Test Name functionality
	originalName := layer.Name()
	if originalName == "" {
		t.Error("Layer should have a default name")
	}

	// Test SetName
	newName := "custom_dense_layer"
	layer.SetName(newName)
	if layer.Name() != newName {
		t.Errorf("Expected name %s, got %s", newName, layer.Name())
	}

	// Test training mode functionality
	if !layer.IsTraining() {
		t.Error("Layer should be in training mode by default")
	}

	layer.SetTraining(false)
	if layer.IsTraining() {
		t.Error("Layer should be in inference mode after SetTraining(false)")
	}

	layer.SetTraining(true)
	if !layer.IsTraining() {
		t.Error("Layer should be in training mode after SetTraining(true)")
	}
}
