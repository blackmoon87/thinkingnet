package layers

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// TestWeightInitializerStrategiesComprehensive tests all WeightInitializer strategies comprehensively.
func TestWeightInitializerStrategiesComprehensive(t *testing.T) {
	testCases := []struct {
		name        string
		initializer WeightInitializer
		rows        int
		cols        int
	}{
		{"XavierUniform_small", XavierUniform, 5, 3},
		{"XavierUniform_medium", XavierUniform, 50, 30},
		{"XavierUniform_large", XavierUniform, 100, 200},
		{"XavierNormal_small", XavierNormal, 5, 3},
		{"XavierNormal_medium", XavierNormal, 50, 30},
		{"XavierNormal_large", XavierNormal, 100, 200},
		{"HeUniform_small", HeUniform, 5, 3},
		{"HeUniform_medium", HeUniform, 50, 30},
		{"HeUniform_large", HeUniform, 100, 200},
		{"HeNormal_small", HeNormal, 5, 3},
		{"HeNormal_medium", HeNormal, 50, 30},
		{"HeNormal_large", HeNormal, 100, 200},
		{"RandomNormal_small", RandomNormal, 5, 3},
		{"RandomNormal_medium", RandomNormal, 50, 30},
		{"RandomNormal_large", RandomNormal, 100, 200},
		{"Zeros_small", Zeros, 5, 3},
		{"Zeros_medium", Zeros, 50, 30},
		{"Zeros_large", Zeros, 100, 200},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			weights := InitializeWeights(tc.rows, tc.cols, tc.initializer)

			// Check dimensions
			rows, cols := weights.Dims()
			if rows != tc.rows || cols != tc.cols {
				t.Errorf("Expected shape (%d,%d), got (%d,%d)", tc.rows, tc.cols, rows, cols)
			}

			// Check that weights are initialized appropriately
			validateWeightInitialization(t, weights, tc.initializer, tc.rows, tc.cols)
		})
	}
}

// validateWeightInitialization validates that weights are initialized according to the strategy.
func validateWeightInitialization(t *testing.T, weights core.Tensor, initializer WeightInitializer, rows, cols int) {
	t.Helper()

	// Collect all weight values
	values := make([]float64, 0, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			values = append(values, weights.At(i, j))
		}
	}

	switch initializer {
	case Zeros:
		// All values should be exactly zero
		for i, val := range values {
			if val != 0.0 {
				t.Errorf("Zeros initializer: expected 0.0 at index %d, got %f", i, val)
			}
		}

	case XavierUniform:
		// Values should be in range [-limit, limit] where limit = sqrt(6/(fan_in + fan_out))
		limit := math.Sqrt(6.0 / float64(rows+cols))
		for i, val := range values {
			if val < -limit || val > limit {
				t.Errorf("XavierUniform: value %f at index %d outside range [%f, %f]", val, i, -limit, limit)
			}
		}
		// Should not be all zeros (unless very unlucky)
		allZeros := true
		for _, val := range values {
			if val != 0.0 {
				allZeros = false
				break
			}
		}
		if allZeros && len(values) > 1 {
			t.Error("XavierUniform: all values are zero (very unlikely)")
		}

	case XavierNormal:
		// Values should follow normal distribution with std = sqrt(2/(fan_in + fan_out))
		expectedStd := math.Sqrt(2.0 / float64(rows+cols))
		validateNormalDistribution(t, values, 0.0, expectedStd, "XavierNormal")

	case HeUniform:
		// Values should be in range [-limit, limit] where limit = sqrt(6/fan_in)
		limit := math.Sqrt(6.0 / float64(rows))
		for i, val := range values {
			if val < -limit || val > limit {
				t.Errorf("HeUniform: value %f at index %d outside range [%f, %f]", val, i, -limit, limit)
			}
		}
		// Should not be all zeros
		allZeros := true
		for _, val := range values {
			if val != 0.0 {
				allZeros = false
				break
			}
		}
		if allZeros && len(values) > 1 {
			t.Error("HeUniform: all values are zero (very unlikely)")
		}

	case HeNormal:
		// Values should follow normal distribution with std = sqrt(2/fan_in)
		expectedStd := math.Sqrt(2.0 / float64(rows))
		validateNormalDistribution(t, values, 0.0, expectedStd, "HeNormal")

	case RandomNormal:
		// Values should follow normal distribution with std = 0.01
		expectedStd := 0.01
		validateNormalDistribution(t, values, 0.0, expectedStd, "RandomNormal")

	default:
		t.Errorf("Unknown initializer: %s", initializer)
	}
}

// validateNormalDistribution validates that values approximately follow a normal distribution.
func validateNormalDistribution(t *testing.T, values []float64, expectedMean, expectedStd float64, name string) {
	t.Helper()

	if len(values) < 30 {
		// Skip statistical tests for very small samples (need at least 30 for reasonable statistics)
		return
	}

	// Calculate sample mean
	var sum float64
	for _, val := range values {
		sum += val
	}
	mean := sum / float64(len(values))

	// Calculate sample standard deviation
	var variance float64
	for _, val := range values {
		diff := val - mean
		variance += diff * diff
	}
	variance /= float64(len(values) - 1)
	std := math.Sqrt(variance)

	// Allow reasonable tolerance for statistical variation
	meanTolerance := expectedStd / math.Sqrt(float64(len(values))) * 3 // 3-sigma confidence
	stdTolerance := expectedStd * 0.3                                  // 30% tolerance for std

	if math.Abs(mean-expectedMean) > meanTolerance {
		t.Errorf("%s: sample mean %f differs from expected %f by more than tolerance %f",
			name, mean, expectedMean, meanTolerance)
	}

	if math.Abs(std-expectedStd) > stdTolerance {
		t.Errorf("%s: sample std %f differs from expected %f by more than tolerance %f",
			name, std, expectedStd, stdTolerance)
	}
}

// TestWeightInitializationEdgeCaseDimensions tests weight initialization with edge case dimensions.
func TestWeightInitializationEdgeCaseDimensions(t *testing.T) {
	testCases := []struct {
		name string
		rows int
		cols int
	}{
		{"1x1", 1, 1},
		{"1x10", 1, 10},
		{"10x1", 10, 1},
		{"2x2", 2, 2},
		{"1000x1", 1000, 1},
		{"1x1000", 1, 1000},
		{"100x100", 100, 100},
	}

	initializers := []WeightInitializer{
		XavierUniform, XavierNormal, HeUniform, HeNormal, RandomNormal, Zeros,
	}

	for _, tc := range testCases {
		for _, init := range initializers {
			t.Run(fmt.Sprintf("%s_%s", tc.name, init), func(t *testing.T) {
				weights := InitializeWeights(tc.rows, tc.cols, init)

				// Check dimensions
				rows, cols := weights.Dims()
				if rows != tc.rows || cols != tc.cols {
					t.Errorf("Expected shape (%d,%d), got (%d,%d)", tc.rows, tc.cols, rows, cols)
				}

				// Basic validation
				validateWeightInitialization(t, weights, init, tc.rows, tc.cols)
			})
		}
	}
}

// TestWeightInitializationStatisticalValidation tests statistical properties of weight initialization.
func TestWeightInitializationStatisticalValidation(t *testing.T) {
	// Use larger matrices for better statistical properties
	rows, cols := 100, 50

	testCases := []struct {
		name          string
		initializer   WeightInitializer
		expectedMean  float64
		expectedStdFn func(int, int) float64
		checkRange    bool
		rangeFn       func(int, int) (float64, float64)
	}{
		{
			name:         "XavierUniform",
			initializer:  XavierUniform,
			expectedMean: 0.0,
			expectedStdFn: func(r, c int) float64 {
				// For uniform distribution [-a, a], std = a/sqrt(3)
				limit := math.Sqrt(6.0 / float64(r+c))
				return limit / math.Sqrt(3)
			},
			checkRange: true,
			rangeFn: func(r, c int) (float64, float64) {
				limit := math.Sqrt(6.0 / float64(r+c))
				return -limit, limit
			},
		},
		{
			name:         "XavierNormal",
			initializer:  XavierNormal,
			expectedMean: 0.0,
			expectedStdFn: func(r, c int) float64 {
				return math.Sqrt(2.0 / float64(r+c))
			},
		},
		{
			name:         "HeUniform",
			initializer:  HeUniform,
			expectedMean: 0.0,
			expectedStdFn: func(r, c int) float64 {
				limit := math.Sqrt(6.0 / float64(r))
				return limit / math.Sqrt(3)
			},
			checkRange: true,
			rangeFn: func(r, c int) (float64, float64) {
				limit := math.Sqrt(6.0 / float64(r))
				return -limit, limit
			},
		},
		{
			name:         "HeNormal",
			initializer:  HeNormal,
			expectedMean: 0.0,
			expectedStdFn: func(r, c int) float64 {
				return math.Sqrt(2.0 / float64(r))
			},
		},
		{
			name:         "RandomNormal",
			initializer:  RandomNormal,
			expectedMean: 0.0,
			expectedStdFn: func(r, c int) float64 {
				return 0.01
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			weights := InitializeWeights(rows, cols, tc.initializer)

			// Collect values
			values := make([]float64, 0, rows*cols)
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					val := weights.At(i, j)
					values = append(values, val)

					// Check range if specified
					if tc.checkRange {
						minVal, maxVal := tc.rangeFn(rows, cols)
						if val < minVal || val > maxVal {
							t.Errorf("Value %f outside expected range [%f, %f]", val, minVal, maxVal)
						}
					}
				}
			}

			// Statistical validation
			expectedStd := tc.expectedStdFn(rows, cols)
			validateNormalDistribution(t, values, tc.expectedMean, expectedStd, tc.name)
		})
	}
}

// TestWeightInitializationErrorHandling tests error handling for invalid parameters.
func TestWeightInitializationErrorHandling(t *testing.T) {
	// Test with invalid dimensions
	invalidDimensions := []struct {
		name string
		rows int
		cols int
	}{
		{"negative_rows", -1, 5},
		{"negative_cols", 5, -1},
		{"zero_rows", 0, 5},
		{"zero_cols", 5, 0},
		{"both_negative", -1, -1},
		{"both_zero", 0, 0},
	}

	for _, tc := range invalidDimensions {
		t.Run(tc.name, func(t *testing.T) {
			// Most initializers should handle invalid dimensions gracefully
			// by returning a tensor with the requested dimensions (even if invalid)
			// The tensor creation itself might fail, but that's handled by the tensor implementation

			defer func() {
				if r := recover(); r != nil {
					// Panic is acceptable for invalid dimensions
					t.Logf("Expected panic for invalid dimensions (%d,%d): %v", tc.rows, tc.cols, r)
				}
			}()

			weights := InitializeWeights(tc.rows, tc.cols, XavierUniform)
			if weights != nil {
				rows, cols := weights.Dims()
				if rows != tc.rows || cols != tc.cols {
					t.Errorf("Expected dimensions (%d,%d), got (%d,%d)", tc.rows, tc.cols, rows, cols)
				}
			}
		})
	}

	// Test with unknown initializer (should default to XavierUniform)
	t.Run("unknown_initializer", func(t *testing.T) {
		weights := InitializeWeights(5, 3, WeightInitializer("unknown"))
		if weights == nil {
			t.Error("Expected weights to be initialized with default strategy")
		}

		rows, cols := weights.Dims()
		if rows != 5 || cols != 3 {
			t.Errorf("Expected shape (5,3), got (%d,%d)", rows, cols)
		}

		// Should behave like XavierUniform (default case)
		validateWeightInitialization(t, weights, XavierUniform, 5, 3)
	})
}

// TestWeightInitializationPerformanceBenchmarks tests performance of different initialization strategies.
func TestWeightInitializationPerformanceBenchmarks(t *testing.T) {
	sizes := []struct {
		name string
		rows int
		cols int
	}{
		{"small", 10, 10},
		{"medium", 100, 100},
		{"large", 500, 500},
	}

	initializers := []WeightInitializer{
		XavierUniform, XavierNormal, HeUniform, HeNormal, RandomNormal, Zeros,
	}

	for _, size := range sizes {
		for _, init := range initializers {
			t.Run(fmt.Sprintf("%s_%s", size.name, init), func(t *testing.T) {
				start := time.Now()

				// Perform initialization multiple times to get meaningful timing
				numIterations := 10
				for i := 0; i < numIterations; i++ {
					weights := InitializeWeights(size.rows, size.cols, init)
					if weights == nil {
						t.Errorf("Initialization failed for %s with %s", size.name, init)
					}
				}

				duration := time.Since(start)
				avgDuration := duration / time.Duration(numIterations)

				t.Logf("%s %s: avg time per initialization: %v", size.name, init, avgDuration)

				// Performance should be reasonable (less than 100ms for large matrices)
				maxDuration := 100 * time.Millisecond
				if avgDuration > maxDuration {
					t.Errorf("Initialization too slow: %v > %v", avgDuration, maxDuration)
				}
			})
		}
	}
}

// TestWeightInitializationReproducibility tests that initialization is deterministic when it should be.
func TestWeightInitializationReproducibility(t *testing.T) {
	// Note: The current implementation uses rand.Float64() and rand.NormFloat64()
	// which are not seeded consistently, so this test checks for non-deterministic behavior

	rows, cols := 10, 5

	// Test that multiple initializations produce different results (for random initializers)
	randomInitializers := []WeightInitializer{
		XavierUniform, XavierNormal, HeUniform, HeNormal, RandomNormal,
	}

	for _, init := range randomInitializers {
		t.Run(string(init), func(t *testing.T) {
			weights1 := InitializeWeights(rows, cols, init)
			weights2 := InitializeWeights(rows, cols, init)

			// Should be different (very high probability)
			identical := true
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					if weights1.At(i, j) != weights2.At(i, j) {
						identical = false
						break
					}
				}
				if !identical {
					break
				}
			}

			if identical {
				t.Errorf("Two random initializations produced identical results (very unlikely)")
			}
		})
	}

	// Test that Zeros initializer is deterministic
	t.Run("Zeros_deterministic", func(t *testing.T) {
		weights1 := InitializeWeights(rows, cols, Zeros)
		weights2 := InitializeWeights(rows, cols, Zeros)

		// Should be identical
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				if weights1.At(i, j) != weights2.At(i, j) {
					t.Errorf("Zeros initializer produced different results at (%d,%d): %f vs %f",
						i, j, weights1.At(i, j), weights2.At(i, j))
				}
			}
		}
	})
}

// TestWeightInitializationIntegrationWithDenseLayer tests integration with Dense layer.
func TestWeightInitializationIntegrationWithDenseLayer(t *testing.T) {
	initializers := []WeightInitializer{
		XavierUniform, XavierNormal, HeUniform, HeNormal, RandomNormal, Zeros,
	}

	for _, init := range initializers {
		t.Run(string(init), func(t *testing.T) {
			// Create Dense layer with specific initializer
			layer := NewDense(5, &DenseConfig{
				Activation:  nil, // Linear
				UseBias:     true,
				Initializer: init,
			})

			// Build the layer
			err := layer.Build(10)
			if err != nil {
				t.Fatalf("Failed to build layer: %v", err)
			}

			// Check that weights were initialized correctly
			params := layer.Parameters()
			if len(params) == 0 {
				t.Error("No parameters found")
				return
			}

			weights := params[0]
			rows, cols := weights.Dims()
			if rows != 10 || cols != 5 {
				t.Errorf("Expected weight shape (10,5), got (%d,%d)", rows, cols)
			}

			// Validate initialization
			validateWeightInitialization(t, weights, init, rows, cols)

			// Test that the layer works with initialized weights
			input := core.NewTensorFromSlice([][]float64{
				{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			})

			output, err := layer.Forward(input)
			if err != nil {
				t.Fatalf("Forward pass failed: %v", err)
			}

			// Check output shape
			outRows, outCols := output.Dims()
			if outRows != 1 || outCols != 5 {
				t.Errorf("Expected output shape (1,5), got (%d,%d)", outRows, outCols)
			}

			// For Zeros initializer, output should be all zeros (since no bias and zero weights)
			if init == Zeros {
				for j := 0; j < outCols; j++ {
					val := output.At(0, j)
					if val != 0.0 {
						t.Errorf("Expected zero output with zero weights, got %f at column %d", val, j)
					}
				}
			}
		})
	}
}
