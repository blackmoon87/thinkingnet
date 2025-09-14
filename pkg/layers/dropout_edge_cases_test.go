package layers

import (
	"fmt"
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// TestDropoutLayerTrainingVsInferenceModes tests comprehensive training vs inference mode behavior.
func TestDropoutLayerTrainingVsInferenceModes(t *testing.T) {
	testCases := []struct {
		name string
		rate float64
	}{
		{"Low dropout rate", 0.1},
		{"Medium dropout rate", 0.5},
		{"High dropout rate", 0.8},
		{"Very high dropout rate", 0.95},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			layer := NewDropout(tc.rate, nil)
			validator := NewValidationUtils(1e-10)

			// Test input
			input := core.NewTensorFromSlice([][]float64{
				{1.0, 2.0, 3.0, 4.0, 5.0},
				{6.0, 7.0, 8.0, 9.0, 10.0},
			})

			// Test inference mode
			layer.SetTraining(false)
			inferenceOutput, err := layer.Forward(input)
			if err != nil {
				t.Fatalf("Inference forward failed: %v", err)
			}

			// In inference mode, output should be identical to input
			validator.AssertTensorEqual(t, input, inferenceOutput, "Inference mode output should equal input")

			// Test training mode
			layer.SetTraining(true)

			// Run multiple trials to test randomness
			numTrials := 100
			zeroCount := 0
			nonZeroCount := 0

			for trial := 0; trial < numTrials; trial++ {
				trainingOutput, err := layer.Forward(input)
				if err != nil {
					t.Fatalf("Training forward failed: %v", err)
				}

				// Count zeros and non-zeros
				rows, cols := trainingOutput.Dims()
				for i := 0; i < rows; i++ {
					for j := 0; j < cols; j++ {
						val := trainingOutput.At(i, j)
						if math.Abs(val) < 1e-10 {
							zeroCount++
						} else {
							nonZeroCount++
							// Non-zero values should be scaled by 1/(1-rate)
							expectedScale := 1.0 / (1.0 - tc.rate)
							originalVal := input.At(i, j)
							expectedVal := originalVal * expectedScale
							if math.Abs(val-expectedVal) > 1e-10 {
								t.Errorf("Expected scaled value %f, got %f", expectedVal, val)
							}
						}
					}
				}
			}

			// Check that dropout rate is approximately correct
			totalElements := numTrials * 2 * 5 // trials * rows * cols
			actualDropoutRate := float64(zeroCount) / float64(totalElements)
			tolerance := 0.1 // Allow 10% tolerance due to randomness

			if math.Abs(actualDropoutRate-tc.rate) > tolerance {
				t.Errorf("Expected dropout rate ~%f, got %f (tolerance: %f)",
					tc.rate, actualDropoutRate, tolerance)
			}
		})
	}
}

// TestDropoutLayerEdgeCaseRates tests dropout layer with edge case rates.
func TestDropoutLayerEdgeCaseRates(t *testing.T) {
	testCases := []struct {
		name        string
		rate        float64
		expectPanic bool
	}{
		{"Zero rate", 0.0, false},
		{"Very small rate", 0.001, false},
		{"Almost one rate", 0.99, false},
		{"Exactly one rate", 1.0, true},
		{"Negative rate", -0.1, true},
		{"Rate greater than one", 1.5, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.expectPanic {
				defer func() {
					if r := recover(); r == nil {
						t.Errorf("Expected panic for rate %f, but got none", tc.rate)
					}
				}()
			}

			layer := NewDropout(tc.rate, nil)

			if !tc.expectPanic {
				// Test that the layer works correctly
				input := core.NewTensorFromSlice([][]float64{{1.0, 2.0, 3.0}})

				// Test inference mode
				layer.SetTraining(false)
				output, err := layer.Forward(input)
				if err != nil {
					t.Fatalf("Forward failed: %v", err)
				}

				// Should be identical to input in inference mode
				validator := NewValidationUtils(1e-10)
				validator.AssertTensorEqual(t, input, output, "Inference output should equal input")

				// Test training mode
				layer.SetTraining(true)
				_, err = layer.Forward(input)
				if err != nil {
					t.Fatalf("Training forward failed: %v", err)
				}
			}
		})
	}
}

// TestDropoutLayerMaskConsistency tests mask consistency between forward and backward passes.
func TestDropoutLayerMaskConsistency(t *testing.T) {
	layer := NewDropout(0.5, nil)
	layer.SetTraining(true)

	input := core.NewTensorFromSlice([][]float64{
		{1.0, 2.0, 3.0, 4.0, 5.0},
		{6.0, 7.0, 8.0, 9.0, 10.0},
	})

	// Forward pass
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// Create gradient
	grad := core.NewTensorFromSlice([][]float64{
		{0.1, 0.2, 0.3, 0.4, 0.5},
		{0.6, 0.7, 0.8, 0.9, 1.0},
	})

	// Backward pass
	inputGrad, err := layer.Backward(grad)
	if err != nil {
		t.Fatalf("Backward pass failed: %v", err)
	}

	// Check consistency: if output[i,j] == 0, then inputGrad[i,j] should be 0
	// if output[i,j] != 0, then inputGrad[i,j] should be scaled version of grad[i,j]
	rows, cols := output.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			outputVal := output.At(i, j)
			gradVal := inputGrad.At(i, j)
			originalGrad := grad.At(i, j)

			if math.Abs(outputVal) < 1e-10 {
				// Dropped unit - gradient should also be zero
				if math.Abs(gradVal) > 1e-10 {
					t.Errorf("Inconsistent masking: output[%d,%d]=0 but grad[%d,%d]=%f",
						i, j, i, j, gradVal)
				}
			} else {
				// Kept unit - gradient should be scaled by same factor as output
				expectedGrad := originalGrad * 2.0 // 1/(1-0.5) = 2.0
				if math.Abs(gradVal-expectedGrad) > 1e-10 {
					t.Errorf("Inconsistent scaling: expected grad[%d,%d]=%f, got %f",
						i, j, expectedGrad, gradVal)
				}
			}
		}
	}
}

// TestDropoutLayerErrorHandling tests error handling and invalid input scenarios.
func TestDropoutLayerErrorHandling(t *testing.T) {
	layer := NewDropout(0.5, nil)

	// Test with small tensor (1x1) instead of empty tensor
	smallTensor := core.NewTensorFromSlice([][]float64{{1.0}})
	layer.SetTraining(true)

	output, err := layer.Forward(smallTensor)
	if err != nil {
		t.Fatalf("Forward with small tensor failed: %v", err)
	}

	// Output should have same shape
	rows, cols := output.Dims()
	if rows != 1 || cols != 1 {
		t.Errorf("Expected output shape (1,1), got (%d,%d)", rows, cols)
	}

	// Test backward with small gradient
	smallGrad := core.NewTensorFromSlice([][]float64{{0.5}})
	inputGrad, err := layer.Backward(smallGrad)
	if err != nil {
		t.Fatalf("Backward with small gradient failed: %v", err)
	}

	// Input gradient should have same shape
	gradRows, gradCols := inputGrad.Dims()
	if gradRows != 1 || gradCols != 1 {
		t.Errorf("Expected input gradient shape (1,1), got (%d,%d)", gradRows, gradCols)
	}
}

// TestDropoutLayerPerformanceWithLargeTensors tests performance with large tensors.
func TestDropoutLayerPerformanceWithLargeTensors(t *testing.T) {
	layer := NewDropout(0.5, nil)
	layer.SetTraining(true)

	// Create a large tensor
	gen := NewTestDataGenerator(42)
	largeTensor := gen.GenerateRandomTensor(1000, 1000, -1.0, 1.0)

	// Test forward pass performance
	output, err := layer.Forward(largeTensor)
	if err != nil {
		t.Fatalf("Forward pass with large tensor failed: %v", err)
	}

	// Validate output shape
	rows, cols := output.Dims()
	expectedRows, expectedCols := largeTensor.Dims()
	if rows != expectedRows || cols != expectedCols {
		t.Errorf("Expected output shape (%d,%d), got (%d,%d)",
			expectedRows, expectedCols, rows, cols)
	}

	// Test backward pass performance
	grad := gen.GenerateRandomTensor(1000, 1000, -0.1, 0.1)
	inputGrad, err := layer.Backward(grad)
	if err != nil {
		t.Fatalf("Backward pass with large tensor failed: %v", err)
	}

	// Validate input gradient shape
	gradRows, gradCols := inputGrad.Dims()
	if gradRows != expectedRows || gradCols != expectedCols {
		t.Errorf("Expected input gradient shape (%d,%d), got (%d,%d)",
			expectedRows, expectedCols, gradRows, gradCols)
	}

	// Basic sanity check - some values should be zero (dropped) and some non-zero (kept)
	zeroCount := 0
	nonZeroCount := 0

	for i := 0; i < rows && i < 100; i++ { // Sample first 100 rows for performance
		for j := 0; j < cols && j < 100; j++ { // Sample first 100 cols for performance
			val := output.At(i, j)
			if math.Abs(val) < 1e-10 {
				zeroCount++
			} else {
				nonZeroCount++
			}
		}
	}

	// Should have both zeros and non-zeros
	if zeroCount == 0 {
		t.Error("Expected some dropped values (zeros) in large tensor output")
	}
	if nonZeroCount == 0 {
		t.Error("Expected some kept values (non-zeros) in large tensor output")
	}
}

// TestDropoutLayerSetRate tests the SetRate functionality.
func TestDropoutLayerSetRate(t *testing.T) {
	layer := NewDropout(0.3, nil)

	// Test valid rate changes
	validRates := []float64{0.0, 0.1, 0.5, 0.9, 0.99}

	for _, rate := range validRates {
		err := layer.SetRate(rate)
		if err != nil {
			t.Errorf("Unexpected error setting valid rate %f: %v", rate, err)
		}

		if layer.GetRate() != rate {
			t.Errorf("Expected rate %f, got %f", rate, layer.GetRate())
		}
	}

	// Test invalid rate changes
	invalidRates := []float64{-0.1, 1.0, 1.5, -1.0}

	for _, rate := range invalidRates {
		originalRate := layer.GetRate()
		err := layer.SetRate(rate)
		if err == nil {
			t.Errorf("Expected error for invalid rate %f, but got none", rate)
		}

		// Rate should remain unchanged after invalid attempt
		if layer.GetRate() != originalRate {
			t.Errorf("Rate changed after invalid attempt: expected %f, got %f",
				originalRate, layer.GetRate())
		}
	}
}

// TestDropoutLayerMultipleForwardPasses tests multiple forward passes with same input.
func TestDropoutLayerMultipleForwardPasses(t *testing.T) {
	layer := NewDropout(0.5, nil)
	layer.SetTraining(true)

	input := core.NewTensorFromSlice([][]float64{
		{1.0, 2.0, 3.0, 4.0, 5.0},
	})

	// Perform multiple forward passes
	outputs := make([]core.Tensor, 10)
	for i := 0; i < 10; i++ {
		output, err := layer.Forward(input)
		if err != nil {
			t.Fatalf("Forward pass %d failed: %v", i, err)
		}
		outputs[i] = output
	}

	// Check that outputs are different (due to randomness)
	allSame := true
	for i := 1; i < len(outputs); i++ {
		if !outputs[0].Equal(outputs[i]) {
			allSame = false
			break
		}
	}

	if allSame {
		t.Error("All outputs are identical - dropout randomness not working")
	}

	// Check that each output has the correct properties
	validator := NewValidationUtils(1e-10)
	for i, output := range outputs {
		// Shape should be preserved
		validator.AssertTensorShape(t, output, 1, 5, fmt.Sprintf("Output %d shape", i))

		// Values should be either 0 or scaled versions of input
		_, cols := output.Dims()
		for j := 0; j < cols; j++ {
			val := output.At(0, j)
			originalVal := input.At(0, j)

			if math.Abs(val) < 1e-10 {
				// Dropped value - should be exactly zero
				continue
			} else {
				// Kept value - should be scaled by 1/(1-rate) = 2.0
				expectedVal := originalVal * 2.0
				if math.Abs(val-expectedVal) > 1e-10 {
					t.Errorf("Output %d: expected scaled value %f, got %f",
						i, expectedVal, val)
				}
			}
		}
	}
}

// TestDropoutLayerInferenceModeStability tests that inference mode is stable.
func TestDropoutLayerInferenceModeStability(t *testing.T) {
	layer := NewDropout(0.8, nil) // High dropout rate
	layer.SetTraining(false)

	input := core.NewTensorFromSlice([][]float64{
		{1.0, 2.0, 3.0, 4.0, 5.0},
		{6.0, 7.0, 8.0, 9.0, 10.0},
	})

	// Perform multiple forward passes in inference mode
	outputs := make([]core.Tensor, 10)
	for i := 0; i < 10; i++ {
		output, err := layer.Forward(input)
		if err != nil {
			t.Fatalf("Inference forward pass %d failed: %v", i, err)
		}
		outputs[i] = output
	}

	// All outputs should be identical to input and to each other
	validator := NewValidationUtils(1e-10)
	for i, output := range outputs {
		validator.AssertTensorEqual(t, input, output,
			fmt.Sprintf("Inference output %d should equal input", i))

		if i > 0 {
			validator.AssertTensorEqual(t, outputs[0], output,
				fmt.Sprintf("Inference output %d should equal output 0", i))
		}
	}
}

// TestDropoutLayerTrainingModeToggle tests toggling between training and inference modes.
func TestDropoutLayerTrainingModeToggle(t *testing.T) {
	layer := NewDropout(0.5, nil)

	input := core.NewTensorFromSlice([][]float64{
		{1.0, 2.0, 3.0, 4.0, 5.0},
	})

	validator := NewValidationUtils(1e-10)

	// Start in training mode
	layer.SetTraining(true)
	if !layer.IsTraining() {
		t.Error("Layer should be in training mode")
	}

	trainingOutput, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Training forward failed: %v", err)
	}

	// Switch to inference mode
	layer.SetTraining(false)
	if layer.IsTraining() {
		t.Error("Layer should be in inference mode")
	}

	inferenceOutput, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Inference forward failed: %v", err)
	}

	// Inference output should equal input
	validator.AssertTensorEqual(t, input, inferenceOutput, "Inference output should equal input")

	// Switch back to training mode
	layer.SetTraining(true)
	if !layer.IsTraining() {
		t.Error("Layer should be back in training mode")
	}

	trainingOutput2, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("Second training forward failed: %v", err)
	}

	// Training outputs might be different due to randomness
	// But they should both have the dropout properties
	checkDropoutProperties := func(output core.Tensor, name string) {
		_, cols := output.Dims()

		for j := 0; j < cols; j++ {
			val := output.At(0, j)
			if math.Abs(val) > 1e-10 {
				// Should be scaled version of input
				expectedVal := input.At(0, j) * 2.0 // 1/(1-0.5) = 2.0
				if math.Abs(val-expectedVal) > 1e-10 {
					t.Errorf("%s: expected scaled value %f, got %f", name, expectedVal, val)
				}
			}
		}
	}

	checkDropoutProperties(trainingOutput, "First training output")
	checkDropoutProperties(trainingOutput2, "Second training output")
}
