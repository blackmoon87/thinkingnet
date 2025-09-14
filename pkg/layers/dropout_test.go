package layers

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestDropoutCreation(t *testing.T) {
	// Test valid dropout rate
	layer := NewDropout(0.5, nil)
	if layer.GetRate() != 0.5 {
		t.Errorf("Expected rate 0.5, got %f", layer.GetRate())
	}
	if !layer.built {
		t.Error("Dropout layer should be built immediately")
	}

	// Test edge cases
	layer0 := NewDropout(0.0, nil)
	if layer0.GetRate() != 0.0 {
		t.Errorf("Expected rate 0.0, got %f", layer0.GetRate())
	}

	layer99 := NewDropout(0.99, nil)
	if layer99.GetRate() != 0.99 {
		t.Errorf("Expected rate 0.99, got %f", layer99.GetRate())
	}

	// Test IsTrainable
	if layer.IsTrainable() {
		t.Error("Dropout layer should not be trainable")
	}

	// Test ParameterCount
	if layer.ParameterCount() != 0 {
		t.Errorf("Expected 0 parameters, got %d", layer.ParameterCount())
	}

	// Test Parameters and Gradients
	params := layer.Parameters()
	if len(params) != 0 {
		t.Errorf("Expected 0 parameter tensors, got %d", len(params))
	}

	grads := layer.Gradients()
	if len(grads) != 0 {
		t.Errorf("Expected 0 gradient tensors, got %d", len(grads))
	}

	// Test OutputShape
	inputShape := []int{32, 128}
	outputShape, err := layer.OutputShape(inputShape)
	if err != nil {
		t.Fatalf("OutputShape failed: %v", err)
	}
	if len(outputShape) != len(inputShape) {
		t.Errorf("Expected output shape length %d, got %d", len(inputShape), len(outputShape))
	}
	for i, dim := range inputShape {
		if outputShape[i] != dim {
			t.Errorf("Expected output shape[%d] = %d, got %d", i, dim, outputShape[i])
		}
	}
}

func TestDropoutInvalidRate(t *testing.T) {
	// Test invalid rates
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for negative rate")
		}
	}()
	NewDropout(-0.1, nil)
}

func TestDropoutInvalidRateHigh(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for rate >= 1.0")
		}
	}()
	NewDropout(1.0, nil)
}

func TestDropoutForwardInference(t *testing.T) {
	layer := NewDropout(0.5, nil)
	layer.SetTraining(false) // Set to inference mode

	input := core.NewTensorFromSlice([][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
	})

	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// In inference mode, output should be identical to input
	rows, cols := input.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if math.Abs(output.At(i, j)-input.At(i, j)) > 1e-10 {
				t.Errorf("Expected output[%d,%d] = %f, got %f",
					i, j, input.At(i, j), output.At(i, j))
			}
		}
	}
}

func TestDropoutForwardTraining(t *testing.T) {
	layer := NewDropout(0.5, nil)
	layer.SetTraining(true) // Set to training mode

	input := core.NewTensorFromSlice([][]float64{
		{2.0, 2.0, 2.0, 2.0, 2.0},
	})

	// Run multiple times to check randomness
	var zeroCount, nonZeroCount int
	numTrials := 1000

	for trial := 0; trial < numTrials; trial++ {
		output, err := layer.Forward(input)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		// Check each element
		_, cols := output.Dims()
		for j := 0; j < cols; j++ {
			val := output.At(0, j)
			if math.Abs(val) < 1e-10 {
				zeroCount++
			} else {
				nonZeroCount++
				// Non-zero values should be scaled by 1/(1-rate) = 2.0
				// So input 2.0 should become 4.0
				expected := 4.0
				if math.Abs(val-expected) > 1e-10 {
					t.Errorf("Expected scaled value %f, got %f", expected, val)
				}
			}
		}
	}

	// With rate=0.5, approximately half should be zero
	totalElements := numTrials * 5
	zeroRatio := float64(zeroCount) / float64(totalElements)

	// Allow some tolerance (should be around 0.5)
	if zeroRatio < 0.4 || zeroRatio > 0.6 {
		t.Errorf("Expected zero ratio around 0.5, got %f", zeroRatio)
	}
}

func TestDropoutZeroRate(t *testing.T) {
	layer := NewDropout(0.0, nil)
	layer.SetTraining(true)

	input := core.NewTensorFromSlice([][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
	})

	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// With rate=0.0, output should be identical to input even in training
	rows, cols := input.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if math.Abs(output.At(i, j)-input.At(i, j)) > 1e-10 {
				t.Errorf("Expected output[%d,%d] = %f, got %f",
					i, j, input.At(i, j), output.At(i, j))
			}
		}
	}
}

func TestDropoutBackwardInference(t *testing.T) {
	layer := NewDropout(0.5, nil)
	layer.SetTraining(false)

	input := core.NewTensorFromSlice([][]float64{
		{1.0, 2.0, 3.0},
	})

	// Forward pass
	_, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// Backward pass
	grad := core.NewTensorFromSlice([][]float64{
		{0.1, 0.2, 0.3},
	})

	inputGrad, err := layer.Backward(grad)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	// In inference mode, gradients should pass through unchanged
	rows, cols := grad.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if math.Abs(inputGrad.At(i, j)-grad.At(i, j)) > 1e-10 {
				t.Errorf("Expected input grad[%d,%d] = %f, got %f",
					i, j, grad.At(i, j), inputGrad.At(i, j))
			}
		}
	}
}

func TestDropoutBackwardTraining(t *testing.T) {
	layer := NewDropout(0.5, nil)
	layer.SetTraining(true)

	input := core.NewTensorFromSlice([][]float64{
		{1.0, 1.0, 1.0, 1.0},
	})

	// Forward pass to generate mask
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Backward pass
	grad := core.NewTensorFromSlice([][]float64{
		{1.0, 1.0, 1.0, 1.0},
	})

	inputGrad, err := layer.Backward(grad)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	// Check that gradients are masked consistently with forward pass
	_, cols := output.Dims()
	for j := 0; j < cols; j++ {
		outputVal := output.At(0, j)
		gradVal := inputGrad.At(0, j)

		if math.Abs(outputVal) < 1e-10 {
			// If output was dropped, gradient should also be zero
			if math.Abs(gradVal) > 1e-10 {
				t.Errorf("Expected zero gradient for dropped unit %d, got %f", j, gradVal)
			}
		} else {
			// If output was kept and scaled, gradient should be scaled too
			expected := 2.0 // 1/(1-0.5) = 2.0
			if math.Abs(gradVal-expected) > 1e-10 {
				t.Errorf("Expected scaled gradient %f for unit %d, got %f", expected, j, gradVal)
			}
		}
	}
}

func TestDropoutParameters(t *testing.T) {
	layer := NewDropout(0.3, nil)

	// Dropout should have no parameters
	params := layer.Parameters()
	if len(params) != 0 {
		t.Errorf("Expected 0 parameters, got %d", len(params))
	}

	grads := layer.Gradients()
	if len(grads) != 0 {
		t.Errorf("Expected 0 gradients, got %d", len(grads))
	}

	if layer.IsTrainable() {
		t.Error("Dropout layer should not be trainable")
	}

	if layer.ParameterCount() != 0 {
		t.Errorf("Expected 0 parameter count, got %d", layer.ParameterCount())
	}
}

func TestDropoutOutputShape(t *testing.T) {
	layer := NewDropout(0.2, nil)

	inputShape := []int{32, 128}
	outputShape, err := layer.OutputShape(inputShape)
	if err != nil {
		t.Fatalf("OutputShape failed: %v", err)
	}

	// Output shape should be identical to input shape
	if len(outputShape) != len(inputShape) {
		t.Errorf("Expected output shape length %d, got %d", len(inputShape), len(outputShape))
	}

	for i, dim := range inputShape {
		if outputShape[i] != dim {
			t.Errorf("Expected output shape[%d] = %d, got %d", i, dim, outputShape[i])
		}
	}
}

func TestDropoutSetRate(t *testing.T) {
	layer := NewDropout(0.3, nil)

	// Test valid rate change
	err := layer.SetRate(0.7)
	if err != nil {
		t.Errorf("Unexpected error setting valid rate: %v", err)
	}
	if layer.GetRate() != 0.7 {
		t.Errorf("Expected rate 0.7, got %f", layer.GetRate())
	}

	// Test invalid rate
	err = layer.SetRate(-0.1)
	if err == nil {
		t.Error("Expected error for negative rate")
	}

	err = layer.SetRate(1.0)
	if err == nil {
		t.Error("Expected error for rate >= 1.0")
	}

	// Rate should remain unchanged after invalid attempts
	if layer.GetRate() != 0.7 {
		t.Errorf("Rate should remain 0.7 after invalid attempts, got %f", layer.GetRate())
	}
}

func TestDropoutConsistency(t *testing.T) {
	// Test that the same mask is used for forward and backward passes
	layer := NewDropout(0.5, nil)
	layer.SetTraining(true)

	input := core.NewTensorFromSlice([][]float64{
		{1.0, 2.0, 3.0, 4.0, 5.0},
	})

	// Forward pass
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Create gradient with different values
	grad := core.NewTensorFromSlice([][]float64{
		{0.1, 0.2, 0.3, 0.4, 0.5},
	})

	// Backward pass
	inputGrad, err := layer.Backward(grad)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	// Check consistency: if output[i] == 0, then inputGrad[i] should be 0
	// if output[i] != 0, then inputGrad[i] should be scaled version of grad[i]
	_, cols := output.Dims()
	for j := 0; j < cols; j++ {
		outputVal := output.At(0, j)
		gradVal := inputGrad.At(0, j)
		originalGrad := grad.At(0, j)

		if math.Abs(outputVal) < 1e-10 {
			// Dropped unit
			if math.Abs(gradVal) > 1e-10 {
				t.Errorf("Inconsistent masking: output[%d]=0 but grad[%d]=%f", j, j, gradVal)
			}
		} else {
			// Kept unit - gradient should be scaled by same factor as output
			expectedGrad := originalGrad * 2.0 // 1/(1-0.5) = 2.0
			if math.Abs(gradVal-expectedGrad) > 1e-10 {
				t.Errorf("Inconsistent scaling: expected grad[%d]=%f, got %f",
					j, expectedGrad, gradVal)
			}
		}
	}
}
func TestDropoutGettersAndSetters(t *testing.T) {
	// Test GetRate
	layer := NewDropout(0.3, nil)
	if layer.GetRate() != 0.3 {
		t.Errorf("Expected rate 0.3, got %f", layer.GetRate())
	}

	// Test SetRate
	layer.SetRate(0.7)
	if layer.GetRate() != 0.7 {
		t.Errorf("Expected rate 0.7 after setting, got %f", layer.GetRate())
	}

	// Test OutputShape
	inputShape := []int{32, 128}
	outputShape, err := layer.OutputShape(inputShape)
	if err != nil {
		t.Fatalf("OutputShape failed: %v", err)
	}

	if len(outputShape) != len(inputShape) {
		t.Errorf("Expected output shape length %d, got %d", len(inputShape), len(outputShape))
	}

	for i, dim := range inputShape {
		if outputShape[i] != dim {
			t.Errorf("Expected output shape[%d] = %d, got %d", i, dim, outputShape[i])
		}
	}

	// Test ParameterCount
	if layer.ParameterCount() != 0 {
		t.Errorf("Expected 0 parameters for dropout, got %d", layer.ParameterCount())
	}

	// Test IsTrainable
	if layer.IsTrainable() {
		t.Error("Dropout layer should not be trainable")
	}

	// Test Parameters and Gradients
	params := layer.Parameters()
	if len(params) != 0 {
		t.Errorf("Expected 0 parameter tensors, got %d", len(params))
	}

	grads := layer.Gradients()
	if len(grads) != 0 {
		t.Errorf("Expected 0 gradient tensors, got %d", len(grads))
	}
}
