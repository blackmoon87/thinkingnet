package metrics

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestCalculateAccuracy(t *testing.T) {
	tests := []struct {
		name     string
		yTrue    [][]float64
		yPred    [][]float64
		expected float64
	}{
		{
			name:     "Perfect accuracy",
			yTrue:    [][]float64{{1}, {0}, {1}, {0}},
			yPred:    [][]float64{{1}, {0}, {1}, {0}},
			expected: 1.0,
		},
		{
			name:     "Half accuracy",
			yTrue:    [][]float64{{1}, {0}, {1}, {0}},
			yPred:    [][]float64{{1}, {1}, {1}, {1}},
			expected: 0.5,
		},
		{
			name:     "Zero accuracy",
			yTrue:    [][]float64{{1}, {0}, {1}, {0}},
			yPred:    [][]float64{{0}, {1}, {0}, {1}},
			expected: 0.0,
		},
		{
			name:     "Single sample",
			yTrue:    [][]float64{{1}},
			yPred:    [][]float64{{1}},
			expected: 1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			yTrue := core.NewTensorFromSlice(tt.yTrue)
			yPred := core.NewTensorFromSlice(tt.yPred)

			result := CalculateAccuracy(yTrue, yPred)

			if math.Abs(result-tt.expected) > 1e-6 {
				t.Errorf("CalculateAccuracy() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCalculatePrecision(t *testing.T) {
	tests := []struct {
		name     string
		yTrue    [][]float64
		yPred    [][]float64
		expected float64
	}{
		{
			name:     "Perfect precision",
			yTrue:    [][]float64{{1}, {1}, {0}, {0}},
			yPred:    [][]float64{{1}, {1}, {0}, {0}},
			expected: 1.0,
		},
		{
			name:     "Half precision",
			yTrue:    [][]float64{{1}, {0}, {1}, {0}},
			yPred:    [][]float64{{1}, {1}, {1}, {0}},
			expected: 0.6666666666666666, // (1.0 + 0.333...) / 2
		},
		{
			name:     "No true positives",
			yTrue:    [][]float64{{0}, {0}, {0}, {0}},
			yPred:    [][]float64{{1}, {1}, {1}, {1}},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			yTrue := core.NewTensorFromSlice(tt.yTrue)
			yPred := core.NewTensorFromSlice(tt.yPred)

			result := CalculatePrecision(yTrue, yPred)

			if math.Abs(result-tt.expected) > 1e-6 {
				t.Errorf("CalculatePrecision() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCalculateRecall(t *testing.T) {
	tests := []struct {
		name     string
		yTrue    [][]float64
		yPred    [][]float64
		expected float64
	}{
		{
			name:     "Perfect recall",
			yTrue:    [][]float64{{1}, {1}, {0}, {0}},
			yPred:    [][]float64{{1}, {1}, {0}, {0}},
			expected: 1.0,
		},
		{
			name:     "Half recall",
			yTrue:    [][]float64{{1}, {1}, {0}, {0}},
			yPred:    [][]float64{{1}, {0}, {0}, {0}},
			expected: 0.5, // (0.5 + 1.0) / 2
		},
		{
			name:     "No predictions",
			yTrue:    [][]float64{{1}, {1}, {1}, {1}},
			yPred:    [][]float64{{0}, {0}, {0}, {0}},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			yTrue := core.NewTensorFromSlice(tt.yTrue)
			yPred := core.NewTensorFromSlice(tt.yPred)

			result := CalculateRecall(yTrue, yPred)

			if math.Abs(result-tt.expected) > 1e-6 {
				t.Errorf("CalculateRecall() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCalculateF1Score(t *testing.T) {
	tests := []struct {
		name     string
		yTrue    [][]float64
		yPred    [][]float64
		expected float64
	}{
		{
			name:     "Perfect F1",
			yTrue:    [][]float64{{1}, {1}, {0}, {0}},
			yPred:    [][]float64{{1}, {1}, {0}, {0}},
			expected: 1.0,
		},
		{
			name:     "Balanced F1",
			yTrue:    [][]float64{{1}, {0}, {1}, {0}},
			yPred:    [][]float64{{1}, {0}, {0}, {0}},
			expected: 0.6666666666666666, // F1 for balanced precision and recall
		},
		{
			name:     "Zero F1",
			yTrue:    [][]float64{{1}, {1}, {1}, {1}},
			yPred:    [][]float64{{0}, {0}, {0}, {0}},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			yTrue := core.NewTensorFromSlice(tt.yTrue)
			yPred := core.NewTensorFromSlice(tt.yPred)

			result := CalculateF1Score(yTrue, yPred)

			if math.Abs(result-tt.expected) > 1e-5 {
				t.Errorf("CalculateF1Score() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestNewConfusionMatrix(t *testing.T) {
	yTrue := core.NewTensorFromSlice([][]float64{{0}, {1}, {2}, {0}, {1}, {2}})
	yPred := core.NewTensorFromSlice([][]float64{{0}, {1}, {1}, {1}, {1}, {2}})

	cm := NewConfusionMatrix(yTrue, yPred)

	// Check basic properties
	if cm.NumClasses != 3 {
		t.Errorf("Expected 3 classes, got %d", cm.NumClasses)
	}

	if cm.NumSamples != 6 {
		t.Errorf("Expected 6 samples, got %d", cm.NumSamples)
	}

	// Check confusion matrix structure
	expectedMatrix := [][]int{
		{1, 1, 0}, // Class 0: 1 correct, 1 predicted as class 1
		{0, 2, 0}, // Class 1: 2 correct
		{0, 1, 1}, // Class 2: 1 predicted as class 1, 1 correct
	}

	for i := range cm.NumClasses {
		for j := range cm.NumClasses {
			if cm.Matrix[i][j] != expectedMatrix[i][j] {
				t.Errorf("Matrix[%d][%d] = %d, want %d", i, j, cm.Matrix[i][j], expectedMatrix[i][j])
			}
		}
	}
}

func TestCalculateROCCurve(t *testing.T) {
	// Simple binary classification case
	yTrue := core.NewTensorFromSlice([][]float64{{0}, {0}, {1}, {1}})
	yProba := core.NewTensorFromSlice([][]float64{{0.1}, {0.4}, {0.35}, {0.8}})

	roc := CalculateROCCurve(yTrue, yProba)

	// Check that AUC is reasonable (between 0 and 1)
	if roc.AUC < 0 || roc.AUC > 1 {
		t.Errorf("AUC should be between 0 and 1, got %f", roc.AUC)
	}

	// Check that we have FPR and TPR points
	if len(roc.FPR) == 0 || len(roc.TPR) == 0 {
		t.Error("ROC curve should have FPR and TPR points")
	}

	// Check that FPR and TPR have same length
	if len(roc.FPR) != len(roc.TPR) {
		t.Errorf("FPR and TPR should have same length: %d vs %d", len(roc.FPR), len(roc.TPR))
	}

	// First point should be (0, 0)
	if roc.FPR[0] != 0.0 || roc.TPR[0] != 0.0 {
		t.Errorf("First ROC point should be (0,0), got (%f,%f)", roc.FPR[0], roc.TPR[0])
	}
}

func TestCalculateClassificationMetrics(t *testing.T) {
	yTrue := core.NewTensorFromSlice([][]float64{{1}, {0}, {1}, {0}, {1}})
	yPred := core.NewTensorFromSlice([][]float64{{1}, {0}, {0}, {0}, {1}})

	metrics := CalculateClassificationMetrics(yTrue, yPred)

	// Check that all metrics are computed
	if metrics.Accuracy == 0 && metrics.Precision == 0 && metrics.Recall == 0 && metrics.F1Score == 0 {
		t.Error("All metrics should not be zero for this test case")
	}

	// Accuracy should be 4/5 = 0.8 (correct predictions: positions 0, 1, 3, 4)
	expectedAccuracy := 0.8
	if math.Abs(metrics.Accuracy-expectedAccuracy) > 1e-6 {
		t.Errorf("Accuracy = %f, want %f", metrics.Accuracy, expectedAccuracy)
	}
}

func TestConfusionMatrixMethods(t *testing.T) {
	yTrue := core.NewTensorFromSlice([][]float64{{0}, {1}, {0}, {1}})
	yPred := core.NewTensorFromSlice([][]float64{{0}, {1}, {1}, {1}})

	cm := NewConfusionMatrix(yTrue, yPred)

	// Test accuracy
	accuracy := cm.GetAccuracy()
	expectedAccuracy := 0.75 // 3 correct out of 4 (positions 0, 1, 3)
	if math.Abs(accuracy-expectedAccuracy) > 1e-6 {
		t.Errorf("GetAccuracy() = %f, want %f", accuracy, expectedAccuracy)
	}

	// Test precision for class 1
	precision := cm.GetPrecision(1)
	expectedPrecision := 2.0 / 3.0 // 2 TP, 1 FP
	if math.Abs(precision-expectedPrecision) > 1e-6 {
		t.Errorf("GetPrecision(1) = %f, want %f", precision, expectedPrecision)
	}

	// Test recall for class 1
	recall := cm.GetRecall(1)
	expectedRecall := 1.0 // 2 TP, 0 FN
	if math.Abs(recall-expectedRecall) > 1e-6 {
		t.Errorf("GetRecall(1) = %f, want %f", recall, expectedRecall)
	}
}

func TestIsBinaryClassification(t *testing.T) {
	tests := []struct {
		name     string
		yTrue    [][]float64
		expected bool
	}{
		{
			name:     "Binary classification",
			yTrue:    [][]float64{{0}, {1}, {0}, {1}},
			expected: true,
		},
		{
			name:     "Multiclass classification",
			yTrue:    [][]float64{{0}, {1}, {2}, {1}},
			expected: false,
		},
		{
			name:     "Single class",
			yTrue:    [][]float64{{1}, {1}, {1}, {1}},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			yTrue := core.NewTensorFromSlice(tt.yTrue)
			result := isBinaryClassification(yTrue)

			if result != tt.expected {
				t.Errorf("isBinaryClassification() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCalculateClassificationMetricsWithProba(t *testing.T) {
	yTrue := core.NewTensorFromSlice([][]float64{{0}, {0}, {1}, {1}})
	yPred := core.NewTensorFromSlice([][]float64{{0}, {0}, {1}, {1}})
	yProba := core.NewTensorFromSlice([][]float64{{0.1}, {0.2}, {0.8}, {0.9}})

	metrics := CalculateClassificationMetricsWithProba(yTrue, yPred, yProba)

	// Should have ROC-AUC for binary classification
	if metrics.ROCAUC == 0 {
		t.Error("ROC-AUC should be computed for binary classification")
	}

	// Perfect classification should have AUC = 1.0
	if math.Abs(metrics.ROCAUC-1.0) > 1e-6 {
		t.Errorf("Perfect classification should have AUC = 1.0, got %f", metrics.ROCAUC)
	}
}

// Benchmark tests
func BenchmarkCalculateAccuracy(b *testing.B) {
	yTrue := core.NewTensorFromSlice(generateRandomBinaryLabels(1000))
	yPred := core.NewTensorFromSlice(generateRandomBinaryLabels(1000))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CalculateAccuracy(yTrue, yPred)
	}
}

func BenchmarkNewConfusionMatrix(b *testing.B) {
	yTrue := core.NewTensorFromSlice(generateRandomMulticlassLabels(1000, 5))
	yPred := core.NewTensorFromSlice(generateRandomMulticlassLabels(1000, 5))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NewConfusionMatrix(yTrue, yPred)
	}
}

func BenchmarkCalculateROCCurve(b *testing.B) {
	yTrue := core.NewTensorFromSlice(generateRandomBinaryLabels(1000))
	yProba := core.NewTensorFromSlice(generateRandomProbabilities(1000))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CalculateROCCurve(yTrue, yProba)
	}
}

// Helper functions for testing
func generateRandomBinaryLabels(n int) [][]float64 {
	labels := make([][]float64, n)
	for i := range n {
		labels[i] = []float64{float64(i % 2)}
	}
	return labels
}

func generateRandomMulticlassLabels(n, numClasses int) [][]float64 {
	labels := make([][]float64, n)
	for i := range n {
		labels[i] = []float64{float64(i % numClasses)}
	}
	return labels
}

func generateRandomProbabilities(n int) [][]float64 {
	probs := make([][]float64, n)
	for i := range n {
		probs[i] = []float64{float64(i) / float64(n)}
	}
	return probs
}
