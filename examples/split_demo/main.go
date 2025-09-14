package main

import (
	"fmt"
	"log"

	"github.com/blackmoon87/thinkingnet/pkg/core"
	"github.com/blackmoon87/thinkingnet/pkg/preprocessing"
)

func main() {
	fmt.Println("ThinkingNet Data Splitting and Validation Demo")
	fmt.Println("==============================================")

	// Create sample dataset
	X := core.NewTensorFromSlice([][]float64{
		{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
		{5.0, 6.0}, {6.0, 7.0}, {7.0, 8.0}, {8.0, 9.0},
		{9.0, 10.0}, {10.0, 11.0}, {11.0, 12.0}, {12.0, 13.0},
	})

	y := core.NewTensorFromSlice([][]float64{
		{0}, {1}, {0}, {1}, {0}, {1}, {0}, {1}, {0}, {1}, {0}, {1},
	})

	fmt.Printf("Original dataset shape: X=%v, y=%v\n", X.Shape(), y.Shape())

	// 1. Basic train-test split
	fmt.Println("\n1. Basic Train-Test Split (70-30)")
	XTrain, XTest, yTrain, yTest, err := preprocessing.TrainTestSplit(X, y,
		preprocessing.WithTestSize(0.3),
		preprocessing.WithShuffle(true),
		preprocessing.WithRandomState(42))

	if err != nil {
		log.Fatalf("Error in train-test split: %v", err)
	}

	fmt.Printf("Train set: X=%v, y=%v\n", XTrain.Shape(), yTrain.Shape())
	fmt.Printf("Test set: X=%v, y=%v\n", XTest.Shape(), yTest.Shape())

	// 2. Stratified split
	fmt.Println("\n2. Stratified Train-Test Split")
	XTrainStrat, XTestStrat, yTrainStrat, yTestStrat, err := preprocessing.TrainTestSplit(X, y,
		preprocessing.WithTestSize(0.3),
		preprocessing.WithStratify(true),
		preprocessing.WithShuffle(true),
		preprocessing.WithRandomState(42))

	if err != nil {
		log.Fatalf("Error in stratified split: %v", err)
	}

	fmt.Printf("Stratified Train set: X=%v, y=%v\n", XTrainStrat.Shape(), yTrainStrat.Shape())
	fmt.Printf("Stratified Test set: X=%v, y=%v\n", XTestStrat.Shape(), yTestStrat.Shape())

	// Print class distributions
	trainLabels := extractLabels(yTrainStrat)
	testLabels := extractLabels(yTestStrat)
	fmt.Printf("Train class distribution: %v\n", countClasses(trainLabels))
	fmt.Printf("Test class distribution: %v\n", countClasses(testLabels))

	// 3. Train-Validation-Test split
	fmt.Println("\n3. Train-Validation-Test Split (60-20-20)")
	XTrainFull, XVal, XTestFull, yTrainFull, yVal, yTestFull, err := preprocessing.TrainValTestSplit(X, y, 0.2,
		preprocessing.WithTestSize(0.2),
		preprocessing.WithShuffle(true),
		preprocessing.WithRandomState(42))

	if err != nil {
		log.Fatalf("Error in train-val-test split: %v", err)
	}

	fmt.Printf("Train set: X=%v, y=%v\n", XTrainFull.Shape(), yTrainFull.Shape())
	fmt.Printf("Validation set: X=%v, y=%v\n", XVal.Shape(), yVal.Shape())
	fmt.Printf("Test set: X=%v, y=%v\n", XTestFull.Shape(), yTestFull.Shape())

	// 4. Data validation
	fmt.Println("\n4. Data Quality Assessment")
	result := preprocessing.CheckDataQuality(X, y)
	fmt.Printf("Data is valid: %v\n", result.IsValid)
	if len(result.Errors) > 0 {
		fmt.Printf("Errors: %v\n", result.Errors)
	}
	if len(result.Warnings) > 0 {
		fmt.Printf("Warnings: %v\n", result.Warnings)
	}

	// Print statistics
	if stats, ok := result.Statistics["X"].(map[string]interface{}); ok {
		fmt.Printf("X statistics: shape=%v, mean=%.3f, std=%.3f\n",
			stats["shape"], stats["mean"], stats["std"])
	}

	// 5. Feature quality assessment
	fmt.Println("\n5. Feature Quality Assessment")
	featureResult := preprocessing.CheckFeatureQuality(X)
	fmt.Printf("Features are valid: %v\n", featureResult.IsValid)
	if len(featureResult.Warnings) > 0 {
		fmt.Printf("Feature warnings: %v\n", featureResult.Warnings)
	}

	// 6. Label quality assessment
	fmt.Println("\n6. Label Quality Assessment")
	labelResult := preprocessing.CheckLabelQuality(y)
	fmt.Printf("Labels are valid: %v\n", labelResult.IsValid)
	if dist, ok := labelResult.Statistics["label_distribution"].(map[int]int); ok {
		fmt.Printf("Label distribution: %v\n", dist)
	}

	// 7. Test with problematic data
	fmt.Println("\n7. Testing with Imbalanced Data")
	yImbalanced := core.NewTensorFromSlice([][]float64{
		{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {1}, // 11:1 ratio
	})

	imbalancedResult := preprocessing.CheckLabelQuality(yImbalanced)
	fmt.Printf("Imbalanced labels valid: %v\n", imbalancedResult.IsValid)
	if len(imbalancedResult.Warnings) > 0 {
		fmt.Printf("Imbalance warnings: %v\n", imbalancedResult.Warnings)
	}

	fmt.Println("\nDemo completed successfully!")
}

// Helper functions
func extractLabels(y core.Tensor) []int {
	rows, _ := y.Dims()
	labels := make([]int, rows)
	for i := range rows {
		labels[i] = int(y.At(i, 0))
	}
	return labels
}

func countClasses(labels []int) map[int]int {
	counts := make(map[int]int)
	for _, label := range labels {
		counts[label]++
	}
	return counts
}
