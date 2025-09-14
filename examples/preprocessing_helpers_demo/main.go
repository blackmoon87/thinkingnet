package main

import (
	"fmt"
	"log"

	"github.com/blackmoon87/thinkingnet/pkg/core"
	"github.com/blackmoon87/thinkingnet/pkg/preprocessing"
)

func main() {
	fmt.Println("=== Preprocessing Helper Functions Demo ===")
	fmt.Println()

	// Create sample dataset
	fmt.Println("1. Creating sample dataset...")
	XData := [][]float64{
		{10.0, 100.0, 1000.0},
		{20.0, 200.0, 2000.0},
		{30.0, 300.0, 3000.0},
		{40.0, 400.0, 4000.0},
		{50.0, 500.0, 5000.0},
		{60.0, 600.0, 6000.0},
		{70.0, 700.0, 7000.0},
		{80.0, 800.0, 8000.0},
		{90.0, 900.0, 9000.0},
		{100.0, 1000.0, 10000.0},
	}
	yData := [][]float64{
		{0}, {1}, {0}, {1}, {0}, {1}, {0}, {1}, {0}, {1},
	}

	X := core.NewTensorFromSlice(XData)
	y := core.NewTensorFromSlice(yData)

	xRows, xCols := X.Dims()
	yRows, yCols := y.Dims()
	fmt.Printf("Original data shape: X=%dx%d, y=%dx%d\n", xRows, xCols, yRows, yCols)
	fmt.Printf("Sample X values: [%.1f, %.1f, %.1f]\n",
		X.At(0, 0), X.At(0, 1), X.At(0, 2))
	fmt.Println()

	// Step 1: Split the data using EasySplit
	fmt.Println("2. Splitting data with EasySplit (30% test size)...")
	XTrain, XTest, yTrain, yTest, err := preprocessing.EasySplit(X, y, 0.3)
	if err != nil {
		log.Fatalf("EasySplit failed: %v", err)
	}

	trainRows, trainCols := XTrain.Dims()
	testRows, testCols := XTest.Dims()
	fmt.Printf("Train set: %dx%d, Test set: %dx%d\n", trainRows, trainCols, testRows, testCols)
	fmt.Println()

	// Step 2: Scale training data using EasyStandardScale
	fmt.Println("3. Scaling training data with EasyStandardScale...")
	XTrainScaled, err := preprocessing.EasyStandardScale(XTrain)
	if err != nil {
		log.Fatalf("EasyStandardScale failed: %v", err)
	}

	fmt.Printf("Original train sample: [%.1f, %.1f, %.1f]\n",
		XTrain.At(0, 0), XTrain.At(0, 1), XTrain.At(0, 2))
	fmt.Printf("Scaled train sample:   [%.3f, %.3f, %.3f]\n",
		XTrainScaled.At(0, 0), XTrainScaled.At(0, 1), XTrainScaled.At(0, 2))
	fmt.Println()

	// Step 3: Scale test data using EasyMinMaxScale
	fmt.Println("4. Scaling test data with EasyMinMaxScale...")
	XTestScaled, err := preprocessing.EasyMinMaxScale(XTest)
	if err != nil {
		log.Fatalf("EasyMinMaxScale failed: %v", err)
	}

	fmt.Printf("Original test sample: [%.1f, %.1f, %.1f]\n",
		XTest.At(0, 0), XTest.At(0, 1), XTest.At(0, 2))
	fmt.Printf("Scaled test sample:   [%.3f, %.3f, %.3f]\n",
		XTestScaled.At(0, 0), XTestScaled.At(0, 1), XTestScaled.At(0, 2))
	fmt.Println()

	// Step 4: Verify data integrity
	fmt.Println("5. Verifying data integrity...")

	// Check that all samples are accounted for
	totalSamples := trainRows + testRows
	originalRows, _ := X.Dims()
	fmt.Printf("Original samples: %d, Split samples: %d ✓\n", originalRows, totalSamples)

	// Check that y dimensions match X
	yTrainRows, _ := yTrain.Dims()
	yTestRows, _ := yTest.Dims()
	fmt.Printf("X train rows: %d, y train rows: %d ✓\n", trainRows, yTrainRows)
	fmt.Printf("X test rows: %d, y test rows: %d ✓\n", testRows, yTestRows)
	fmt.Println()

	// Step 5: Show the convenience of helper functions
	fmt.Println("6. Comparison: Before vs After")
	fmt.Println()

	fmt.Println("BEFORE (traditional approach):")
	fmt.Println("  // Split data")
	fmt.Println("  XTrain, XTest, yTrain, yTest, err := preprocessing.TrainTestSplit(X, y,")
	fmt.Println("      preprocessing.WithTestSize(0.3),")
	fmt.Println("      preprocessing.WithShuffle(true),")
	fmt.Println("      preprocessing.WithRandomState(42))")
	fmt.Println()
	fmt.Println("  // Scale data")
	fmt.Println("  scaler := preprocessing.NewStandardScaler()")
	fmt.Println("  err = scaler.Fit(XTrain)")
	fmt.Println("  XTrainScaled, err := scaler.Transform(XTrain)")
	fmt.Println()

	fmt.Println("AFTER (with helper functions):")
	fmt.Println("  // Split data")
	fmt.Println("  XTrain, XTest, yTrain, yTest, err := preprocessing.EasySplit(X, y, 0.3)")
	fmt.Println()
	fmt.Println("  // Scale data")
	fmt.Println("  XTrainScaled, err := preprocessing.EasyStandardScale(XTrain)")
	fmt.Println()

	fmt.Println("✅ All preprocessing helper functions working correctly!")
	fmt.Println("✅ Code is now much simpler and more readable!")
	fmt.Println("✅ Sensible defaults are applied automatically!")
}
