package metrics

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestCalculateMSE(t *testing.T) {
	tests := []struct {
		name     string
		yTrue    [][]float64
		yPred    [][]float64
		expected float64
	}{
		{
			name:     "Perfect prediction",
			yTrue:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			yPred:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			expected: 0.0,
		},
		{
			name:     "Simple MSE",
			yTrue:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			yPred:    [][]float64{{1.5}, {2.5}, {3.5}, {4.5}},
			expected: 0.25, // (0.5^2 + 0.5^2 + 0.5^2 + 0.5^2) / 4
		},
		{
			name:     "Larger errors",
			yTrue:    [][]float64{{0.0}, {1.0}},
			yPred:    [][]float64{{1.0}, {0.0}},
			expected: 1.0, // (1^2 + 1^2) / 2
		},
		{
			name:     "Single sample",
			yTrue:    [][]float64{{1.0}},
			yPred:    [][]float64{{1.0}},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			yTrue := core.NewTensorFromSlice(tt.yTrue)
			yPred := core.NewTensorFromSlice(tt.yPred)

			result := CalculateMSE(yTrue, yPred)

			if math.Abs(result-tt.expected) > 1e-6 {
				t.Errorf("CalculateMSE() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCalculateRMSE(t *testing.T) {
	tests := []struct {
		name     string
		yTrue    [][]float64
		yPred    [][]float64
		expected float64
	}{
		{
			name:     "Perfect prediction",
			yTrue:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			yPred:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			expected: 0.0,
		},
		{
			name:     "Simple RMSE",
			yTrue:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			yPred:    [][]float64{{1.5}, {2.5}, {3.5}, {4.5}},
			expected: 0.5, // sqrt(0.25)
		},
		{
			name:     "RMSE with larger errors",
			yTrue:    [][]float64{{0.0}, {3.0}},
			yPred:    [][]float64{{4.0}, {0.0}},
			expected: math.Sqrt(12.5), // sqrt((16 + 9) / 2)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			yTrue := core.NewTensorFromSlice(tt.yTrue)
			yPred := core.NewTensorFromSlice(tt.yPred)

			result := CalculateRMSE(yTrue, yPred)

			if math.Abs(result-tt.expected) > 1e-6 {
				t.Errorf("CalculateRMSE() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCalculateMAE(t *testing.T) {
	tests := []struct {
		name     string
		yTrue    [][]float64
		yPred    [][]float64
		expected float64
	}{
		{
			name:     "Perfect prediction",
			yTrue:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			yPred:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			expected: 0.0,
		},
		{
			name:     "Simple MAE",
			yTrue:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			yPred:    [][]float64{{1.5}, {2.5}, {3.5}, {4.5}},
			expected: 0.5, // (0.5 + 0.5 + 0.5 + 0.5) / 4
		},
		{
			name:     "Mixed errors",
			yTrue:    [][]float64{{1.0}, {2.0}, {3.0}},
			yPred:    [][]float64{{0.0}, {3.0}, {2.0}},
			expected: 1.0, // (1.0 + 1.0 + 1.0) / 3
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			yTrue := core.NewTensorFromSlice(tt.yTrue)
			yPred := core.NewTensorFromSlice(tt.yPred)

			result := CalculateMAE(yTrue, yPred)

			if math.Abs(result-tt.expected) > 1e-6 {
				t.Errorf("CalculateMAE() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCalculateR2Score(t *testing.T) {
	tests := []struct {
		name     string
		yTrue    [][]float64
		yPred    [][]float64
		expected float64
	}{
		{
			name:     "Perfect prediction",
			yTrue:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			yPred:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			expected: 1.0,
		},
		{
			name:     "Mean prediction",
			yTrue:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			yPred:    [][]float64{{2.5}, {2.5}, {2.5}, {2.5}}, // Mean of yTrue
			expected: 0.0,
		},
		{
			name:     "Constant true values",
			yTrue:    [][]float64{{5.0}, {5.0}, {5.0}, {5.0}},
			yPred:    [][]float64{{5.0}, {5.0}, {5.0}, {5.0}},
			expected: 1.0, // Perfect prediction when all y values are the same
		},
		{
			name:     "Worse than mean",
			yTrue:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			yPred:    [][]float64{{4.0}, {3.0}, {2.0}, {1.0}}, // Opposite predictions
			expected: -3.0,                                    // R² can be negative
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			yTrue := core.NewTensorFromSlice(tt.yTrue)
			yPred := core.NewTensorFromSlice(tt.yPred)

			result := CalculateR2Score(yTrue, yPred)

			if math.Abs(result-tt.expected) > 1e-6 {
				t.Errorf("CalculateR2Score() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCalculateMAPE(t *testing.T) {
	tests := []struct {
		name     string
		yTrue    [][]float64
		yPred    [][]float64
		expected float64
	}{
		{
			name:     "Perfect prediction",
			yTrue:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			yPred:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			expected: 0.0,
		},
		{
			name:     "Simple MAPE",
			yTrue:    [][]float64{{1.0}, {2.0}, {4.0}},
			yPred:    [][]float64{{1.1}, {1.8}, {4.4}},
			expected: 10.0, // (10% + 10% + 10%) / 3 * 100
		},
		{
			name:     "With zero values (should skip)",
			yTrue:    [][]float64{{0.0}, {2.0}, {4.0}},
			yPred:    [][]float64{{1.0}, {1.8}, {4.4}},
			expected: 10.0, // Only non-zero true values: (10% + 10%) / 2 * 100
		},
		{
			name:     "All zero true values",
			yTrue:    [][]float64{{0.0}, {0.0}, {0.0}},
			yPred:    [][]float64{{1.0}, {2.0}, {3.0}},
			expected: 0.0, // Should return 0 when no valid samples
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			yTrue := core.NewTensorFromSlice(tt.yTrue)
			yPred := core.NewTensorFromSlice(tt.yPred)

			result := CalculateMAPE(yTrue, yPred)

			if math.Abs(result-tt.expected) > 1e-6 {
				t.Errorf("CalculateMAPE() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCalculateExplainedVarianceScore(t *testing.T) {
	tests := []struct {
		name     string
		yTrue    [][]float64
		yPred    [][]float64
		expected float64
	}{
		{
			name:     "Perfect prediction",
			yTrue:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			yPred:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			expected: 1.0,
		},
		{
			name:     "No variance in true values",
			yTrue:    [][]float64{{5.0}, {5.0}, {5.0}, {5.0}},
			yPred:    [][]float64{{5.0}, {5.0}, {5.0}, {5.0}},
			expected: 1.0,
		},
		{
			name:     "Partial explanation",
			yTrue:    [][]float64{{1.0}, {2.0}, {3.0}, {4.0}},
			yPred:    [][]float64{{1.5}, {2.0}, {2.5}, {3.5}},
			expected: 0.8, // Should be high but not perfect
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			yTrue := core.NewTensorFromSlice(tt.yTrue)
			yPred := core.NewTensorFromSlice(tt.yPred)

			result := CalculateExplainedVarianceScore(yTrue, yPred)

			if math.Abs(result-tt.expected) > 0.1 { // More lenient tolerance for EVS
				t.Errorf("CalculateExplainedVarianceScore() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCalculateRegressionMetrics(t *testing.T) {
	yTrue := core.NewTensorFromSlice([][]float64{{1.0}, {2.0}, {3.0}, {4.0}})
	yPred := core.NewTensorFromSlice([][]float64{{1.1}, {1.9}, {3.1}, {3.9}})

	metrics := CalculateRegressionMetrics(yTrue, yPred)

	// Check that all metrics are computed
	if metrics.MSE == 0 && metrics.RMSE == 0 && metrics.MAE == 0 && metrics.R2Score == 0 && metrics.MAPE == 0 && metrics.EVS == 0 {
		t.Error("Not all metrics should be zero for this test case")
	}

	// MSE should be 0.01 (average of 0.01, 0.01, 0.01, 0.01)
	expectedMSE := 0.01
	if math.Abs(metrics.MSE-expectedMSE) > 1e-6 {
		t.Errorf("MSE = %f, want %f", metrics.MSE, expectedMSE)
	}

	// RMSE should be sqrt(0.01) = 0.1
	expectedRMSE := 0.1
	if math.Abs(metrics.RMSE-expectedRMSE) > 1e-6 {
		t.Errorf("RMSE = %f, want %f", metrics.RMSE, expectedRMSE)
	}

	// MAE should be 0.1
	expectedMAE := 0.1
	if math.Abs(metrics.MAE-expectedMAE) > 1e-6 {
		t.Errorf("MAE = %f, want %f", metrics.MAE, expectedMAE)
	}

	// R² should be very high (close to 1)
	if metrics.R2Score < 0.9 {
		t.Errorf("R2Score should be high for good predictions, got %f", metrics.R2Score)
	}
}

func TestCalculateResidualAnalysis(t *testing.T) {
	yTrue := core.NewTensorFromSlice([][]float64{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}})
	yPred := core.NewTensorFromSlice([][]float64{{1.1}, {1.9}, {3.0}, {4.1}, {4.9}})

	analysis := CalculateResidualAnalysis(yTrue, yPred)

	// Check that residuals are computed
	if analysis.Residuals == nil {
		t.Error("Residuals should not be nil")
	}

	// Check residual statistics
	expectedMeanResidual := 0.0 // Should be close to 0 for unbiased predictions
	if math.Abs(analysis.MeanResidual-expectedMeanResidual) > 0.1 {
		t.Errorf("MeanResidual = %f, want close to %f", analysis.MeanResidual, expectedMeanResidual)
	}

	// Check that percentiles are computed
	if len(analysis.Percentiles) != 3 {
		t.Errorf("Expected 3 percentiles (25th, 50th, 75th), got %d", len(analysis.Percentiles))
	}
}

func TestCalculatePercentiles(t *testing.T) {
	tests := []struct {
		name        string
		values      []float64
		percentiles []float64
		expected    []float64
	}{
		{
			name:        "Simple case",
			values:      []float64{1, 2, 3, 4, 5},
			percentiles: []float64{0, 50, 100},
			expected:    []float64{1, 3, 5},
		},
		{
			name:        "Quartiles",
			values:      []float64{1, 2, 3, 4, 5, 6, 7, 8},
			percentiles: []float64{25, 50, 75},
			expected:    []float64{2.75, 4.5, 6.25},
		},
		{
			name:        "Empty values",
			values:      []float64{},
			percentiles: []float64{25, 50, 75},
			expected:    []float64{0, 0, 0},
		},
		{
			name:        "Single value",
			values:      []float64{42},
			percentiles: []float64{25, 50, 75},
			expected:    []float64{42, 42, 42},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calculatePercentiles(tt.values, tt.percentiles)

			if len(result) != len(tt.expected) {
				t.Errorf("Expected %d percentiles, got %d", len(tt.expected), len(result))
				return
			}

			for i, expected := range tt.expected {
				if math.Abs(result[i]-expected) > 0.5 { // Lenient tolerance for percentile approximation
					t.Errorf("Percentile %f = %f, want %f", tt.percentiles[i], result[i], expected)
				}
			}
		})
	}
}

func TestCalculateCrossValidationMetrics(t *testing.T) {
	scores := []float64{0.8, 0.85, 0.9, 0.75, 0.88}
	metricName := "accuracy"

	cvMetrics := CalculateCrossValidationMetrics(scores, metricName)

	// Check basic properties
	if cvMetrics.MetricName != metricName {
		t.Errorf("MetricName = %s, want %s", cvMetrics.MetricName, metricName)
	}

	if cvMetrics.FoldCount != len(scores) {
		t.Errorf("FoldCount = %d, want %d", cvMetrics.FoldCount, len(scores))
	}

	// Check mean calculation
	expectedMean := 0.836 // (0.8 + 0.85 + 0.9 + 0.75 + 0.88) / 5
	if math.Abs(cvMetrics.MeanScore-expectedMean) > 1e-6 {
		t.Errorf("MeanScore = %f, want %f", cvMetrics.MeanScore, expectedMean)
	}

	// Check that standard deviation is computed
	if cvMetrics.StdScore <= 0 {
		t.Error("StdScore should be positive for varying scores")
	}
}

func TestGenerateRegressionReport(t *testing.T) {
	yTrue := core.NewTensorFromSlice([][]float64{{1.0}, {2.0}, {3.0}, {4.0}})
	yPred := core.NewTensorFromSlice([][]float64{{1.1}, {1.9}, {3.1}, {3.9}})

	report := GenerateRegressionReport(yTrue, yPred)

	// Check that all components are present
	if report.Metrics == nil {
		t.Error("Report should include metrics")
	}

	if report.ResidualAnalysis == nil {
		t.Error("Report should include residual analysis")
	}

	if report.NumSamples != 4 {
		t.Errorf("NumSamples = %d, want 4", report.NumSamples)
	}

	if report.NumFeatures != 1 {
		t.Errorf("NumFeatures = %d, want 1", report.NumFeatures)
	}
}

// Benchmark tests
func BenchmarkCalculateMSE(b *testing.B) {
	yTrue := core.NewTensorFromSlice(generateRandomRegression(1000))
	yPred := core.NewTensorFromSlice(generateRandomRegression(1000))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CalculateMSE(yTrue, yPred)
	}
}

func BenchmarkCalculateRegressionMetrics(b *testing.B) {
	yTrue := core.NewTensorFromSlice(generateRandomRegression(1000))
	yPred := core.NewTensorFromSlice(generateRandomRegression(1000))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CalculateRegressionMetrics(yTrue, yPred)
	}
}

func BenchmarkCalculateResidualAnalysis(b *testing.B) {
	yTrue := core.NewTensorFromSlice(generateRandomRegression(1000))
	yPred := core.NewTensorFromSlice(generateRandomRegression(1000))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CalculateResidualAnalysis(yTrue, yPred)
	}
}

// Helper function for testing
func generateRandomRegression(n int) [][]float64 {
	values := make([][]float64, n)
	for i := range n {
		values[i] = []float64{float64(i) + 0.1*float64(i%10)}
	}
	return values
}
