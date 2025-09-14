package main

import (
	"fmt"
	"log"

	"github.com/blackmoon87/thinkingnet/pkg/algorithms"
	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func main() {
	fmt.Println("Classification Algorithms Demo")
	fmt.Println("=============================")

	// Create a simple binary classification dataset
	// Class 0: points around (0, 0)
	// Class 1: points around (3, 3)
	XData := [][]float64{
		{0.1, 0.2}, {0.3, 0.1}, {0.2, 0.3}, {0.0, 0.1}, {0.1, 0.0},
		{2.8, 2.9}, {3.1, 2.8}, {2.9, 3.2}, {3.0, 3.0}, {2.7, 3.1},
		{0.2, 0.0}, {0.0, 0.3}, {0.1, 0.1},
		{3.2, 2.7}, {2.9, 2.8}, {3.1, 3.1},
	}
	yData := [][]float64{
		{0}, {0}, {0}, {0}, {0},
		{1}, {1}, {1}, {1}, {1},
		{0}, {0}, {0},
		{1}, {1}, {1},
	}

	X := core.NewTensorFromSlice(XData)
	y := core.NewTensorFromSlice(yData)

	fmt.Printf("Dataset: %d samples, %d features\n", len(XData), len(XData[0]))
	fmt.Printf("Classes: 0 and 1\n\n")

	// Test Logistic Regression
	fmt.Println("1. Logistic Regression")
	fmt.Println("----------------------")

	lr := algorithms.NewLogisticRegression(
		algorithms.WithLearningRate(0.1),
		algorithms.WithLRMaxIters(1000),
		algorithms.WithLRTolerance(1e-6),
		algorithms.WithRegularization("l2", 0.01),
	)

	err := lr.Fit(X, y)
	if err != nil {
		log.Fatalf("Failed to fit logistic regression: %v", err)
	}

	// Make predictions
	predictions, err := lr.Predict(X)
	if err != nil {
		log.Fatalf("Failed to predict: %v", err)
	}

	// Get probabilities
	probabilities, err := lr.PredictProba(X)
	if err != nil {
		log.Fatalf("Failed to predict probabilities: %v", err)
	}

	// Calculate accuracy
	accuracy, err := lr.Score(X, y)
	if err != nil {
		log.Fatalf("Failed to calculate accuracy: %v", err)
	}

	fmt.Printf("Accuracy: %.3f\n", accuracy)

	// Calculate detailed metrics
	metrics := algorithms.CalculateClassificationMetrics(y, predictions, 1.0)
	fmt.Printf("Precision: %.3f\n", metrics.Precision)
	fmt.Printf("Recall: %.3f\n", metrics.Recall)
	fmt.Printf("F1-Score: %.3f\n", metrics.F1Score)
	fmt.Printf("Confusion Matrix: TP=%d, TN=%d, FP=%d, FN=%d\n",
		metrics.ConfusionMatrix.TruePositives,
		metrics.ConfusionMatrix.TrueNegatives,
		metrics.ConfusionMatrix.FalsePositives,
		metrics.ConfusionMatrix.FalseNegatives)

	// Show some predictions
	fmt.Println("\nSample predictions:")
	for i := 0; i < 5; i++ {
		fmt.Printf("Sample %d: True=%d, Pred=%d, Prob=%.3f\n",
			i, int(y.At(i, 0)), int(predictions.At(i, 0)), probabilities.At(i, 0))
	}

	fmt.Println()

	// Test Random Forest
	fmt.Println("2. Random Forest")
	fmt.Println("----------------")

	rf := algorithms.NewRandomForest(
		algorithms.WithNEstimators(50),
		algorithms.WithMaxDepth(10),
		algorithms.WithMinSamplesSplit(2),
		algorithms.WithMinSamplesLeaf(1),
		algorithms.WithRFRandomSeed(42),
	)

	err = rf.Fit(X, y)
	if err != nil {
		log.Fatalf("Failed to fit random forest: %v", err)
	}

	// Make predictions
	rfPredictions, err := rf.Predict(X)
	if err != nil {
		log.Fatalf("Failed to predict: %v", err)
	}

	// Get probabilities
	rfProbabilities, err := rf.PredictProba(X)
	if err != nil {
		log.Fatalf("Failed to predict probabilities: %v", err)
	}

	// Calculate accuracy
	rfAccuracy, err := rf.Score(X, y)
	if err != nil {
		log.Fatalf("Failed to calculate accuracy: %v", err)
	}

	fmt.Printf("Accuracy: %.3f\n", rfAccuracy)

	// Calculate detailed metrics
	rfMetrics := algorithms.CalculateClassificationMetrics(y, rfPredictions, 1.0)
	fmt.Printf("Precision: %.3f\n", rfMetrics.Precision)
	fmt.Printf("Recall: %.3f\n", rfMetrics.Recall)
	fmt.Printf("F1-Score: %.3f\n", rfMetrics.F1Score)
	fmt.Printf("Confusion Matrix: TP=%d, TN=%d, FP=%d, FN=%d\n",
		rfMetrics.ConfusionMatrix.TruePositives,
		rfMetrics.ConfusionMatrix.TrueNegatives,
		rfMetrics.ConfusionMatrix.FalsePositives,
		rfMetrics.ConfusionMatrix.FalseNegatives)

	// Show some predictions
	fmt.Println("\nSample predictions:")
	for i := 0; i < 5; i++ {
		fmt.Printf("Sample %d: True=%d, Pred=%d, Prob_Class0=%.3f, Prob_Class1=%.3f\n",
			i, int(y.At(i, 0)), int(rfPredictions.At(i, 0)),
			rfProbabilities.At(i, 0), rfProbabilities.At(i, 1))
	}

	fmt.Println()

	// Test on new data
	fmt.Println("3. Testing on New Data")
	fmt.Println("----------------------")

	newXData := [][]float64{
		{0.05, 0.05}, // Should be class 0
		{2.95, 2.95}, // Should be class 1
		{1.5, 1.5},   // Boundary case
	}
	newX := core.NewTensorFromSlice(newXData)

	// Logistic Regression predictions
	lrNewPred, _ := lr.Predict(newX)
	lrNewProb, _ := lr.PredictProba(newX)

	// Random Forest predictions
	rfNewPred, _ := rf.Predict(newX)
	rfNewProb, _ := rf.PredictProba(newX)

	fmt.Println("Logistic Regression:")
	for i := 0; i < 3; i++ {
		fmt.Printf("  Point (%.2f, %.2f): Pred=%d, Prob=%.3f\n",
			newXData[i][0], newXData[i][1], int(lrNewPred.At(i, 0)), lrNewProb.At(i, 0))
	}

	fmt.Println("Random Forest:")
	for i := 0; i < 3; i++ {
		fmt.Printf("  Point (%.2f, %.2f): Pred=%d, Prob_Class0=%.3f, Prob_Class1=%.3f\n",
			newXData[i][0], newXData[i][1], int(rfNewPred.At(i, 0)),
			rfNewProb.At(i, 0), rfNewProb.At(i, 1))
	}

	fmt.Println("\nDemo completed successfully!")
}
