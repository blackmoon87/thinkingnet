package layers

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/activations"
	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// TestLayerIntegration tests multiple layers working together.
func TestLayerIntegration(t *testing.T) {
	// Create a simple 2-layer network: input -> dense -> dropout -> dense -> output

	// Layer 1: Dense with ReLU activation
	layer1 := NewDense(4, &DenseConfig{
		Activation:  activations.NewReLU(),
		UseBias:     true,
		Initializer: XavierUniform,
	})

	// Layer 2: Dropout
	dropout := NewDropout(0.5, nil)

	// Layer 3: Dense with linear activation (output layer)
	layer2 := NewDense(2, &DenseConfig{
		Activation:  activations.NewLinear(),
		UseBias:     true,
		Initializer: XavierUniform,
	})

	// Input: batch of 3 samples, 5 features each
	input := core.NewTensorFromSlice([][]float64{
		{1.0, 2.0, 3.0, 4.0, 5.0},
		{-1.0, -2.0, 3.0, 4.0, -5.0},
		{0.5, 1.5, -2.5, 3.5, -4.5},
	})

	// Forward pass through the network

	// Layer 1 forward
	hidden1, err := layer1.Forward(input)
	if err != nil {
		t.Fatalf("Layer 1 forward failed: %v", err)
	}

	// Check output shape
	h1Rows, h1Cols := hidden1.Dims()
	if h1Rows != 3 || h1Cols != 4 {
		t.Errorf("Expected hidden1 shape (3,4), got (%d,%d)", h1Rows, h1Cols)
	}

	// Dropout forward (set to inference mode for predictable results)
	dropout.SetTraining(false)
	hidden2, err := dropout.Forward(hidden1)
	if err != nil {
		t.Fatalf("Dropout forward failed: %v", err)
	}

	// Check that dropout in inference mode doesn't change the output
	if !hidden2.Equal(hidden1) {
		t.Error("Dropout in inference mode should not change the output")
	}

	// Layer 2 forward
	output, err := layer2.Forward(hidden2)
	if err != nil {
		t.Fatalf("Layer 2 forward failed: %v", err)
	}

	// Check output shape
	outRows, outCols := output.Dims()
	if outRows != 3 || outCols != 2 {
		t.Errorf("Expected output shape (3,2), got (%d,%d)", outRows, outCols)
	}

	// Backward pass through the network

	// Simulate loss gradient (e.g., from mean squared error)
	lossGrad := core.NewTensorFromSlice([][]float64{
		{0.1, -0.2},
		{-0.3, 0.4},
		{0.5, -0.6},
	})

	// Layer 2 backward
	grad2, err := layer2.Backward(lossGrad)
	if err != nil {
		t.Fatalf("Layer 2 backward failed: %v", err)
	}

	// Check gradient shape
	g2Rows, g2Cols := grad2.Dims()
	if g2Rows != 3 || g2Cols != 4 {
		t.Errorf("Expected grad2 shape (3,4), got (%d,%d)", g2Rows, g2Cols)
	}

	// Dropout backward
	grad1, err := dropout.Backward(grad2)
	if err != nil {
		t.Fatalf("Dropout backward failed: %v", err)
	}

	// In inference mode, gradients should pass through unchanged
	if !grad1.Equal(grad2) {
		t.Error("Dropout in inference mode should pass gradients unchanged")
	}

	// Layer 1 backward
	inputGrad, err := layer1.Backward(grad1)
	if err != nil {
		t.Fatalf("Layer 1 backward failed: %v", err)
	}

	// Check input gradient shape
	igRows, igCols := inputGrad.Dims()
	if igRows != 3 || igCols != 5 {
		t.Errorf("Expected input grad shape (3,5), got (%d,%d)", igRows, igCols)
	}

	// Verify that all layers have computed gradients
	layer1Grads := layer1.Gradients()
	if len(layer1Grads) != 2 { // weights + biases
		t.Errorf("Expected 2 gradients for layer1, got %d", len(layer1Grads))
	}

	layer2Grads := layer2.Gradients()
	if len(layer2Grads) != 2 { // weights + biases
		t.Errorf("Expected 2 gradients for layer2, got %d", len(layer2Grads))
	}

	dropoutGrads := dropout.Gradients()
	if len(dropoutGrads) != 0 { // no parameters
		t.Errorf("Expected 0 gradients for dropout, got %d", len(dropoutGrads))
	}
}

// TestLayerWithTrainingMode tests layers in training vs inference mode.
func TestLayerWithTrainingMode(t *testing.T) {
	dropout := NewDropout(0.8, nil) // High dropout rate for clear effect

	input := core.NewTensorFromSlice([][]float64{
		{1.0, 1.0, 1.0, 1.0, 1.0},
	})

	// Training mode - should apply dropout
	dropout.SetTraining(true)

	var zeroCount, nonZeroCount int
	numTrials := 100

	for i := 0; i < numTrials; i++ {
		output, err := dropout.Forward(input)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		_, cols := output.Dims()
		for j := 0; j < cols; j++ {
			if math.Abs(output.At(0, j)) < 1e-10 {
				zeroCount++
			} else {
				nonZeroCount++
			}
		}
	}

	// With 80% dropout rate, most values should be zero
	totalElements := numTrials * 5
	zeroRatio := float64(zeroCount) / float64(totalElements)

	if zeroRatio < 0.7 || zeroRatio > 0.9 {
		t.Errorf("Expected zero ratio around 0.8, got %f", zeroRatio)
	}

	// Inference mode - should not apply dropout
	dropout.SetTraining(false)
	output, err := dropout.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// All values should be preserved
	_, cols := output.Dims()
	for j := 0; j < cols; j++ {
		if math.Abs(output.At(0, j)-1.0) > 1e-10 {
			t.Errorf("Expected preserved value 1.0, got %f", output.At(0, j))
		}
	}
}

// TestGradientNumericalCheck performs numerical gradient checking.
func TestGradientNumericalCheck(t *testing.T) {
	// Simple test with one dense layer
	layer := NewDense(1, &DenseConfig{
		Activation:  activations.NewLinear(),
		UseBias:     false,
		Initializer: Zeros,
	})

	// Single input, single output
	input := core.NewTensorFromSlice([][]float64{{2.0}})

	// Build layer and set known weight
	_, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}
	layer.weights.Set(0, 0, 3.0)

	// Forward pass to cache input for backward pass
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}
	expectedOutput := 2.0 * 3.0 // input * weight

	if math.Abs(output.At(0, 0)-expectedOutput) > 1e-10 {
		t.Errorf("Expected output %f, got %f", expectedOutput, output.At(0, 0))
	}

	// Backward pass
	grad := core.NewTensorFromSlice([][]float64{{1.0}})
	inputGrad, err := layer.Backward(grad)
	if err != nil {
		t.Fatalf("Backward pass failed: %v", err)
	}

	// Analytical gradient
	analyticalWeightGrad := layer.Gradients()[0].At(0, 0)
	analyticalInputGrad := inputGrad.At(0, 0)

	// Numerical gradient check for weight
	epsilon := 1e-5

	// Perturb weight positively
	layer.weights.Set(0, 0, 3.0+epsilon)
	outputPlus, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// Perturb weight negatively
	layer.weights.Set(0, 0, 3.0-epsilon)
	outputMinus, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// Numerical gradient
	numericalWeightGrad := (outputPlus.At(0, 0) - outputMinus.At(0, 0)) / (2 * epsilon)

	// Check if analytical and numerical gradients match
	if math.Abs(analyticalWeightGrad-numericalWeightGrad) > 1e-5 {
		t.Errorf("Weight gradient mismatch: analytical=%f, numerical=%f",
			analyticalWeightGrad, numericalWeightGrad)
	}

	// Expected gradients:
	// dL/dw = input * grad = 2.0 * 1.0 = 2.0
	// dL/dx = weight * grad = 3.0 * 1.0 = 3.0

	if math.Abs(analyticalWeightGrad-2.0) > 1e-10 {
		t.Errorf("Expected weight gradient 2.0, got %f", analyticalWeightGrad)
	}

	if math.Abs(analyticalInputGrad-3.0) > 1e-10 {
		t.Errorf("Expected input gradient 3.0, got %f", analyticalInputGrad)
	}
}

// TestLayerParameterManagement tests parameter and gradient management.
func TestLayerParameterManagement(t *testing.T) {
	// Dense layer with bias
	denseWithBias := NewDense(3, &DenseConfig{
		UseBias:     true,
		Initializer: Zeros,
	})

	// Dense layer without bias
	denseNoBias := NewDense(3, &DenseConfig{
		UseBias:     false,
		Initializer: Zeros,
	})

	// Dropout layer
	dropout := NewDropout(0.5, nil)

	// Build layers
	input := core.NewTensorFromSlice([][]float64{{1.0, 2.0}})
	_, _ = denseWithBias.Forward(input)
	_, _ = denseNoBias.Forward(input)

	// Test parameter counts
	if denseWithBias.ParameterCount() != 2*3+3 { // weights + biases
		t.Errorf("Expected 9 parameters for dense with bias, got %d", denseWithBias.ParameterCount())
	}

	if denseNoBias.ParameterCount() != 2*3 { // only weights
		t.Errorf("Expected 6 parameters for dense without bias, got %d", denseNoBias.ParameterCount())
	}

	if dropout.ParameterCount() != 0 {
		t.Errorf("Expected 0 parameters for dropout, got %d", dropout.ParameterCount())
	}

	// Test trainable flags
	if !denseWithBias.IsTrainable() {
		t.Error("Dense layer should be trainable")
	}

	if !denseNoBias.IsTrainable() {
		t.Error("Dense layer should be trainable")
	}

	if dropout.IsTrainable() {
		t.Error("Dropout layer should not be trainable")
	}

	// Test parameter access
	paramsWithBias := denseWithBias.Parameters()
	if len(paramsWithBias) != 2 {
		t.Errorf("Expected 2 parameter tensors (weights, biases), got %d", len(paramsWithBias))
	}

	paramsNoBias := denseNoBias.Parameters()
	if len(paramsNoBias) != 1 {
		t.Errorf("Expected 1 parameter tensor (weights only), got %d", len(paramsNoBias))
	}

	dropoutParams := dropout.Parameters()
	if len(dropoutParams) != 0 {
		t.Errorf("Expected 0 parameter tensors for dropout, got %d", len(dropoutParams))
	}
}
