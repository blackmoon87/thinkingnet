package algorithms

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestLogisticRegression_BinaryClassification(t *testing.T) {
	// Create simple binary classification dataset
	// Class 0: points around (0, 0)
	// Class 1: points around (2, 2)
	XData := [][]float64{
		{0.1, 0.1}, {0.2, 0.0}, {0.0, 0.2}, {0.3, 0.1},
		{1.9, 1.8}, {2.1, 2.0}, {1.8, 2.1}, {2.0, 1.9},
	}
	yData := [][]float64{
		{0}, {0}, {0}, {0},
		{1}, {1}, {1}, {1},
	}

	X := core.NewTensorFromSlice(XData)
	y := core.NewTensorFromSlice(yData)

	// Create and train model
	model := NewLogisticRegression(
		WithLearningRate(0.1),
		WithLRMaxIters(1000),
		WithLRTolerance(1e-6),
	)

	err := model.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	if !model.fitted {
		t.Error("Model should be fitted after training")
	}

	// Test predictions
	predictions, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// Check prediction shape
	predRows, predCols := predictions.Dims()
	if predRows != 8 || predCols != 1 {
		t.Errorf("Expected predictions shape (8, 1), got (%d, %d)", predRows, predCols)
	}

	// Test probability predictions
	probabilities, err := model.PredictProba(X)
	if err != nil {
		t.Fatalf("Failed to predict probabilities: %v", err)
	}

	probRows, probCols := probabilities.Dims()
	if probRows != 8 || probCols != 1 {
		t.Errorf("Expected probabilities shape (8, 1), got (%d, %d)", probRows, probCols)
	}

	// Check that probabilities are between 0 and 1
	for i := 0; i < probRows; i++ {
		prob := probabilities.At(i, 0)
		if prob < 0 || prob > 1 {
			t.Errorf("Probability should be between 0 and 1, got %f", prob)
		}
	}

	// Test accuracy
	accuracy, err := model.Score(X, y)
	if err != nil {
		t.Fatalf("Failed to calculate score: %v", err)
	}

	if accuracy < 0.5 { // Should be better than random
		t.Errorf("Accuracy should be > 0.5, got %f", accuracy)
	}
}

func TestEasyLogisticRegression(t *testing.T) {
	// Create simple binary classification dataset
	XData := [][]float64{
		{0.1, 0.1}, {0.2, 0.0}, {0.0, 0.2}, {0.3, 0.1},
		{1.9, 1.8}, {2.1, 2.0}, {1.8, 2.1}, {2.0, 1.9},
	}
	yData := [][]float64{
		{0}, {0}, {0}, {0},
		{1}, {1}, {1}, {1},
	}

	X := core.NewTensorFromSlice(XData)
	y := core.NewTensorFromSlice(yData)

	// Test the easy constructor
	model := EasyLogisticRegression()

	// Verify default parameters are set correctly
	if model.learningRate != 0.01 {
		t.Errorf("Expected learning rate 0.01, got %f", model.learningRate)
	}
	if model.maxIters != 1000 {
		t.Errorf("Expected max iterations 1000, got %d", model.maxIters)
	}
	if model.tolerance != 1e-6 {
		t.Errorf("Expected tolerance 1e-6, got %f", model.tolerance)
	}
	if !model.fitIntercept {
		t.Error("Expected fitIntercept to be true")
	}

	// Test that it works for training and prediction
	err := model.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	predictions, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// Check prediction shape
	predRows, predCols := predictions.Dims()
	if predRows != 8 || predCols != 1 {
		t.Errorf("Expected predictions shape (8, 1), got (%d, %d)", predRows, predCols)
	}

	// Test accuracy
	accuracy, err := model.Score(X, y)
	if err != nil {
		t.Fatalf("Failed to calculate accuracy: %v", err)
	}

	if accuracy < 0.5 {
		t.Errorf("Accuracy should be > 0.5, got %f", accuracy)
	}
}

func TestLogisticRegression_MulticlassClassification(t *testing.T) {
	// Create simple 3-class dataset
	XData := [][]float64{
		{0, 0}, {0.1, 0.1}, {-0.1, 0.1}, // Class 0
		{2, 2}, {2.1, 1.9}, {1.9, 2.1}, // Class 1
		{-2, 2}, {-1.9, 2.1}, {-2.1, 1.9}, // Class 2
	}
	yData := [][]float64{
		{0}, {0}, {0},
		{1}, {1}, {1},
		{2}, {2}, {2},
	}

	X := core.NewTensorFromSlice(XData)
	y := core.NewTensorFromSlice(yData)

	// Create and train model
	model := NewLogisticRegression(
		WithLearningRate(0.1),
		WithLRMaxIters(1000),
	)

	err := model.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	// Check classes
	classes := model.Classes()
	if len(classes) != 3 {
		t.Errorf("Expected 3 classes, got %d", len(classes))
	}

	// Test predictions
	_, err = model.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// Test probability predictions
	probabilities, err := model.PredictProba(X)
	if err != nil {
		t.Fatalf("Failed to predict probabilities: %v", err)
	}

	probRows, probCols := probabilities.Dims()
	if probRows != 9 || probCols != 3 {
		t.Errorf("Expected probabilities shape (9, 3), got (%d, %d)", probRows, probCols)
	}

	// Check that probabilities sum to 1 for each sample
	for i := 0; i < probRows; i++ {
		sum := 0.0
		for j := 0; j < probCols; j++ {
			prob := probabilities.At(i, j)
			if prob < 0 || prob > 1 {
				t.Errorf("Probability should be between 0 and 1, got %f", prob)
			}
			sum += prob
		}
		if math.Abs(sum-1.0) > 1e-6 {
			t.Errorf("Probabilities should sum to 1, got %f", sum)
		}
	}
}

func TestLogisticRegression_Regularization(t *testing.T) {
	// Create dataset
	XData := [][]float64{
		{0, 0}, {0.1, 0.1}, {0.2, 0.0}, {0.0, 0.2},
		{2, 2}, {2.1, 1.9}, {1.9, 2.1}, {2.0, 2.0},
	}
	yData := [][]float64{
		{0}, {0}, {0}, {0},
		{1}, {1}, {1}, {1},
	}

	X := core.NewTensorFromSlice(XData)
	y := core.NewTensorFromSlice(yData)

	// Test L2 regularization
	modelL2 := NewLogisticRegression(
		WithLearningRate(0.1),
		WithLRMaxIters(500),
		WithRegularization("l2", 0.1),
	)

	err := modelL2.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit L2 model: %v", err)
	}

	// Test L1 regularization
	modelL1 := NewLogisticRegression(
		WithLearningRate(0.1),
		WithLRMaxIters(500),
		WithRegularization("l1", 0.1),
	)

	err = modelL1.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit L1 model: %v", err)
	}

	// Test elastic net
	modelElastic := NewLogisticRegression(
		WithLearningRate(0.1),
		WithLRMaxIters(500),
		WithElasticNet(0.1, 0.5),
	)

	err = modelElastic.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit elastic net model: %v", err)
	}

	// All models should be able to make predictions
	_, err = modelL2.Predict(X)
	if err != nil {
		t.Errorf("L2 model failed to predict: %v", err)
	}

	_, err = modelL1.Predict(X)
	if err != nil {
		t.Errorf("L1 model failed to predict: %v", err)
	}

	_, err = modelElastic.Predict(X)
	if err != nil {
		t.Errorf("Elastic net model failed to predict: %v", err)
	}
}

func TestLogisticRegression_ValidationErrors(t *testing.T) {
	model := NewLogisticRegression()

	// Test prediction before fitting
	X := core.NewTensorFromSlice([][]float64{{1, 2}})
	_, err := model.Predict(X)
	if err == nil {
		t.Error("Should fail when predicting before fitting")
	}

	// Test invalid input dimensions
	XInvalid := core.NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	yInvalid := core.NewTensorFromSlice([][]float64{{1}}) // Mismatched dimensions

	err = model.Fit(XInvalid, yInvalid)
	if err == nil {
		t.Error("Should fail with mismatched X and y dimensions")
	}

	// Test single class
	XSingle := core.NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	ySingle := core.NewTensorFromSlice([][]float64{{1}, {1}}) // Only one class

	err = model.Fit(XSingle, ySingle)
	if err == nil {
		t.Error("Should fail with only one class")
	}
}

func TestClassificationMetrics(t *testing.T) {
	// Create test data
	yTrue := core.NewTensorFromSlice([][]float64{{0}, {0}, {1}, {1}, {0}, {1}, {1}, {0}})
	yPred := core.NewTensorFromSlice([][]float64{{0}, {1}, {1}, {1}, {0}, {0}, {1}, {0}})

	// Test accuracy
	accuracy := CalculateAccuracy(yTrue, yPred)
	expected := 6.0 / 8.0 // 6 correct out of 8
	if math.Abs(accuracy-expected) > 1e-10 {
		t.Errorf("Expected accuracy %f, got %f", expected, accuracy)
	}

	// Test precision for class 1
	precision := CalculatePrecision(yTrue, yPred, 1.0)
	expectedPrecision := 3.0 / 4.0 // 3 TP, 1 FP
	if math.Abs(precision-expectedPrecision) > 1e-10 {
		t.Errorf("Expected precision %f, got %f", expectedPrecision, precision)
	}

	// Test recall for class 1
	recall := CalculateRecall(yTrue, yPred, 1.0)
	expectedRecall := 3.0 / 4.0 // 3 TP, 1 FN
	if math.Abs(recall-expectedRecall) > 1e-10 {
		t.Errorf("Expected recall %f, got %f", expectedRecall, recall)
	}

	// Test F1 score
	f1 := CalculateF1Score(yTrue, yPred, 1.0)
	expectedF1 := 2 * (precision * recall) / (precision + recall)
	if math.Abs(f1-expectedF1) > 1e-10 {
		t.Errorf("Expected F1 %f, got %f", expectedF1, f1)
	}

	// Test confusion matrix
	cm := CalculateConfusionMatrix(yTrue, yPred, 1.0)
	if cm.TruePositives != 3 {
		t.Errorf("Expected 3 true positives, got %d", cm.TruePositives)
	}
	if cm.TrueNegatives != 3 {
		t.Errorf("Expected 3 true negatives, got %d", cm.TrueNegatives)
	}
	if cm.FalsePositives != 1 {
		t.Errorf("Expected 1 false positive, got %d", cm.FalsePositives)
	}
	if cm.FalseNegatives != 1 {
		t.Errorf("Expected 1 false negative, got %d", cm.FalseNegatives)
	}

	// Test comprehensive metrics
	metrics := CalculateClassificationMetrics(yTrue, yPred, 1.0)
	if math.Abs(metrics.Accuracy-accuracy) > 1e-10 {
		t.Errorf("Metrics accuracy mismatch")
	}
	if math.Abs(metrics.Precision-precision) > 1e-10 {
		t.Errorf("Metrics precision mismatch")
	}
	if math.Abs(metrics.Recall-recall) > 1e-10 {
		t.Errorf("Metrics recall mismatch")
	}
	if math.Abs(metrics.F1Score-f1) > 1e-10 {
		t.Errorf("Metrics F1 mismatch")
	}
}

func TestLogisticRegression_Options(t *testing.T) {
	// Test all configuration options
	model := NewLogisticRegression(
		WithLearningRate(0.05),
		WithLRMaxIters(500),
		WithLRTolerance(1e-5),
		WithFitIntercept(false),
		WithLRRandomSeed(42),
	)

	if model.learningRate != 0.05 {
		t.Errorf("Expected learning rate 0.05, got %f", model.learningRate)
	}
	if model.maxIters != 500 {
		t.Errorf("Expected max iters 500, got %d", model.maxIters)
	}
	if model.tolerance != 1e-5 {
		t.Errorf("Expected tolerance 1e-5, got %f", model.tolerance)
	}
	if model.fitIntercept != false {
		t.Errorf("Expected fit intercept false, got %t", model.fitIntercept)
	}
	if model.randomSeed != 42 {
		t.Errorf("Expected random seed 42, got %d", model.randomSeed)
	}
}

func TestLogisticRegression_NoIntercept(t *testing.T) {
	// Create dataset that passes through origin
	XData := [][]float64{
		{-1, -1}, {-0.5, -0.5},
		{1, 1}, {0.5, 0.5},
	}
	yData := [][]float64{
		{0}, {0},
		{1}, {1},
	}

	X := core.NewTensorFromSlice(XData)
	y := core.NewTensorFromSlice(yData)

	// Train without intercept
	model := NewLogisticRegression(
		WithFitIntercept(false),
		WithLearningRate(0.1),
		WithLRMaxIters(1000),
	)

	err := model.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model without intercept: %v", err)
	}

	// Should still be able to make predictions
	predictions, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// Check prediction shape
	predRows, predCols := predictions.Dims()
	if predRows != 4 || predCols != 1 {
		t.Errorf("Expected predictions shape (4, 1), got (%d, %d)", predRows, predCols)
	}
}
