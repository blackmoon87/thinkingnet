package preprocessing

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// TestEasyStandardScale tests the EasyStandardScale helper function
func TestEasyStandardScale(t *testing.T) {
	// Create sample data
	data := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
		{10.0, 11.0, 12.0},
	}
	X := core.NewTensorFromSlice(data)

	// Test EasyStandardScale
	scaled, err := EasyStandardScale(X)
	if err != nil {
		t.Fatalf("EasyStandardScale failed: %v", err)
	}

	// Verify dimensions are preserved
	rows, cols := scaled.Dims()
	if rows != 4 || cols != 3 {
		t.Errorf("Expected dimensions (4, 3), got (%d, %d)", rows, cols)
	}

	// Verify that each column has approximately zero mean and unit variance
	tolerance := 1e-10
	for j := 0; j < cols; j++ {
		// Calculate mean
		sum := 0.0
		for i := 0; i < rows; i++ {
			sum += scaled.At(i, j)
		}
		mean := sum / float64(rows)

		if math.Abs(mean) > tolerance {
			t.Errorf("Column %d mean should be ~0, got %f", j, mean)
		}

		// Calculate variance
		sumSq := 0.0
		for i := 0; i < rows; i++ {
			diff := scaled.At(i, j) - mean
			sumSq += diff * diff
		}
		variance := sumSq / float64(rows)

		if math.Abs(variance-1.0) > tolerance {
			t.Errorf("Column %d variance should be ~1, got %f", j, variance)
		}
	}
}

// TestEasyMinMaxScale tests the EasyMinMaxScale helper function
func TestEasyMinMaxScale(t *testing.T) {
	// Create sample data
	data := [][]float64{
		{1.0, 10.0, 100.0},
		{2.0, 20.0, 200.0},
		{3.0, 30.0, 300.0},
		{4.0, 40.0, 400.0},
	}
	X := core.NewTensorFromSlice(data)

	// Test EasyMinMaxScale
	scaled, err := EasyMinMaxScale(X)
	if err != nil {
		t.Fatalf("EasyMinMaxScale failed: %v", err)
	}

	// Verify dimensions are preserved
	rows, cols := scaled.Dims()
	if rows != 4 || cols != 3 {
		t.Errorf("Expected dimensions (4, 3), got (%d, %d)", rows, cols)
	}

	// Verify that each column is scaled to [0, 1] range
	tolerance := 1e-10
	for j := 0; j < cols; j++ {
		min := scaled.At(0, j)
		max := scaled.At(0, j)

		for i := 0; i < rows; i++ {
			val := scaled.At(i, j)
			if val < min {
				min = val
			}
			if val > max {
				max = val
			}
		}

		if math.Abs(min) > tolerance {
			t.Errorf("Column %d min should be ~0, got %f", j, min)
		}

		if math.Abs(max-1.0) > tolerance {
			t.Errorf("Column %d max should be ~1, got %f", j, max)
		}
	}
}

// TestEasySplit tests the EasySplit helper function
func TestEasySplit(t *testing.T) {
	// Create sample data
	XData := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
		{7.0, 8.0},
		{9.0, 10.0},
		{11.0, 12.0},
		{13.0, 14.0},
		{15.0, 16.0},
		{17.0, 18.0},
		{19.0, 20.0},
	}
	yData := [][]float64{
		{0}, {1}, {0}, {1}, {0}, {1}, {0}, {1}, {0}, {1},
	}

	X := core.NewTensorFromSlice(XData)
	y := core.NewTensorFromSlice(yData)

	// Test EasySplit with 30% test size
	XTrain, XTest, yTrain, yTest, err := EasySplit(X, y, 0.3)
	if err != nil {
		t.Fatalf("EasySplit failed: %v", err)
	}

	// Verify split sizes
	trainRows, _ := XTrain.Dims()
	testRows, _ := XTest.Dims()
	totalRows := trainRows + testRows

	if totalRows != 10 {
		t.Errorf("Total samples should be 10, got %d", totalRows)
	}

	// Test size should be approximately 30% (3 samples)
	expectedTestSize := 3
	if testRows != expectedTestSize {
		t.Errorf("Expected test size %d, got %d", expectedTestSize, testRows)
	}

	// Verify dimensions consistency
	_, XTrainCols := XTrain.Dims()
	_, XTestCols := XTest.Dims()
	yTrainRows, yTrainCols := yTrain.Dims()
	yTestRows, yTestCols := yTest.Dims()

	if XTrainCols != 2 || XTestCols != 2 {
		t.Errorf("X should have 2 columns, got train: %d, test: %d", XTrainCols, XTestCols)
	}

	if yTrainCols != 1 || yTestCols != 1 {
		t.Errorf("y should have 1 column, got train: %d, test: %d", yTrainCols, yTestCols)
	}

	if yTrainRows != trainRows || yTestRows != testRows {
		t.Errorf("X and y row counts should match. X: train=%d, test=%d; y: train=%d, test=%d",
			trainRows, testRows, yTrainRows, yTestRows)
	}
}

// TestEasyStandardScaleWithNilData tests error handling
func TestEasyStandardScaleWithNilData(t *testing.T) {
	// Test with nil tensor
	_, err := EasyStandardScale(nil)
	if err == nil {
		t.Error("Expected error for nil tensor")
	}
}

// TestEasyMinMaxScaleWithNilData tests error handling
func TestEasyMinMaxScaleWithNilData(t *testing.T) {
	// Test with nil tensor
	_, err := EasyMinMaxScale(nil)
	if err == nil {
		t.Error("Expected error for nil tensor")
	}
}

// TestEasySplitWithInvalidData tests error handling
func TestEasySplitWithInvalidData(t *testing.T) {
	// Create valid data
	XData := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
	yData := [][]float64{{0}, {1}}
	X := core.NewTensorFromSlice(XData)
	y := core.NewTensorFromSlice(yData)

	// Test with invalid test size
	_, _, _, _, err := EasySplit(X, y, 1.5)
	if err == nil {
		t.Error("Expected error for test_size > 1.0")
	}

	_, _, _, _, err = EasySplit(X, y, -0.1)
	if err == nil {
		t.Error("Expected error for negative test_size")
	}

	// Test with mismatched X and y
	yMismatched := core.NewTensorFromSlice([][]float64{{0}, {1}, {0}}) // 3 rows vs 2 in X
	_, _, _, _, err = EasySplit(X, yMismatched, 0.3)
	if err == nil {
		t.Error("Expected error for mismatched X and y dimensions")
	}
}

// TestHelperFunctionsIntegration tests using all helper functions together
func TestHelperFunctionsIntegration(t *testing.T) {
	// Create sample dataset
	XData := [][]float64{
		{10.0, 100.0, 1000.0},
		{20.0, 200.0, 2000.0},
		{30.0, 300.0, 3000.0},
		{40.0, 400.0, 4000.0},
		{50.0, 500.0, 5000.0},
		{60.0, 600.0, 6000.0},
		{70.0, 700.0, 7000.0},
		{80.0, 800.0, 8000.0},
	}
	yData := [][]float64{
		{0}, {1}, {0}, {1}, {0}, {1}, {0}, {1},
	}

	X := core.NewTensorFromSlice(XData)
	y := core.NewTensorFromSlice(yData)

	// Step 1: Split the data
	XTrain, XTest, yTrain, yTest, err := EasySplit(X, y, 0.25)
	if err != nil {
		t.Fatalf("EasySplit failed: %v", err)
	}

	// Step 2: Scale the training data
	XTrainScaled, err := EasyStandardScale(XTrain)
	if err != nil {
		t.Fatalf("EasyStandardScale failed: %v", err)
	}

	// Step 3: Scale the test data using MinMax scaling
	XTestScaled, err := EasyMinMaxScale(XTest)
	if err != nil {
		t.Fatalf("EasyMinMaxScale failed: %v", err)
	}

	// Verify all operations completed successfully
	trainRows, trainCols := XTrainScaled.Dims()
	testRows, testCols := XTestScaled.Dims()

	if trainCols != 3 || testCols != 3 {
		t.Errorf("Expected 3 columns, got train: %d, test: %d", trainCols, testCols)
	}

	if trainRows+testRows != 8 {
		t.Errorf("Expected total 8 rows, got %d", trainRows+testRows)
	}

	// Verify y dimensions are preserved
	yTrainRows, yTrainCols := yTrain.Dims()
	yTestRows, yTestCols := yTest.Dims()

	if yTrainCols != 1 || yTestCols != 1 {
		t.Errorf("y should have 1 column, got train: %d, test: %d", yTrainCols, yTestCols)
	}

	if yTrainRows != trainRows || yTestRows != testRows {
		t.Errorf("X and y row counts should match")
	}

	t.Logf("Integration test passed: Train=%dx%d, Test=%dx%d", trainRows, trainCols, testRows, testCols)
}

// TestEasyStandardScaleBackwardCompatibility tests that EasyStandardScale produces same results as regular StandardScaler
func TestEasyStandardScaleBackwardCompatibility(t *testing.T) {
	data := [][]float64{
		{1.0, 10.0, 100.0},
		{2.0, 20.0, 200.0},
		{3.0, 30.0, 300.0},
		{4.0, 40.0, 400.0},
	}
	X := core.NewTensorFromSlice(data)

	// Use EasyStandardScale
	easyResult, err := EasyStandardScale(X)
	if err != nil {
		t.Fatalf("EasyStandardScale failed: %v", err)
	}

	// Use regular StandardScaler
	scaler := NewStandardScaler()
	regularResult, err := scaler.FitTransform(X)
	if err != nil {
		t.Fatalf("Regular StandardScaler failed: %v", err)
	}

	// Results should be identical
	rows, cols := easyResult.Dims()
	tolerance := 1e-10

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			easyVal := easyResult.At(i, j)
			regularVal := regularResult.At(i, j)
			if math.Abs(easyVal-regularVal) > tolerance {
				t.Errorf("Values differ at (%d,%d): easy=%f, regular=%f", i, j, easyVal, regularVal)
			}
		}
	}
}

// TestEasyMinMaxScaleBackwardCompatibility tests that EasyMinMaxScale produces same results as regular MinMaxScaler
func TestEasyMinMaxScaleBackwardCompatibility(t *testing.T) {
	data := [][]float64{
		{1.0, 10.0, 100.0},
		{2.0, 20.0, 200.0},
		{3.0, 30.0, 300.0},
		{4.0, 40.0, 400.0},
	}
	X := core.NewTensorFromSlice(data)

	// Use EasyMinMaxScale
	easyResult, err := EasyMinMaxScale(X)
	if err != nil {
		t.Fatalf("EasyMinMaxScale failed: %v", err)
	}

	// Use regular MinMaxScaler
	scaler := NewMinMaxScaler()
	regularResult, err := scaler.FitTransform(X)
	if err != nil {
		t.Fatalf("Regular MinMaxScaler failed: %v", err)
	}

	// Results should be identical
	rows, cols := easyResult.Dims()
	tolerance := 1e-10

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			easyVal := easyResult.At(i, j)
			regularVal := regularResult.At(i, j)
			if math.Abs(easyVal-regularVal) > tolerance {
				t.Errorf("Values differ at (%d,%d): easy=%f, regular=%f", i, j, easyVal, regularVal)
			}
		}
	}
}

// TestEasySplitBackwardCompatibility tests that EasySplit produces same results as regular TrainTestSplit
func TestEasySplitBackwardCompatibility(t *testing.T) {
	XData := [][]float64{
		{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0},
		{9.0, 10.0}, {11.0, 12.0}, {13.0, 14.0}, {15.0, 16.0},
	}
	yData := [][]float64{
		{0}, {1}, {0}, {1}, {0}, {1}, {0}, {1},
	}

	X := core.NewTensorFromSlice(XData)
	y := core.NewTensorFromSlice(yData)

	testSize := 0.25

	// Use EasySplit
	easyXTrain, easyXTest, easyYTrain, easyYTest, err := EasySplit(X, y, testSize)
	if err != nil {
		t.Fatalf("EasySplit failed: %v", err)
	}

	// Use regular TrainTestSplit with same parameters
	regularXTrain, regularXTest, regularYTrain, regularYTest, err := TrainTestSplit(X, y,
		WithTestSize(testSize),
		WithShuffle(true),
		WithRandomState(42))
	if err != nil {
		t.Fatalf("Regular TrainTestSplit failed: %v", err)
	}

	// Dimensions should match
	easyTrainRows, easyTrainCols := easyXTrain.Dims()
	regularTrainRows, regularTrainCols := regularXTrain.Dims()

	if easyTrainRows != regularTrainRows || easyTrainCols != regularTrainCols {
		t.Errorf("Training set dimensions differ: easy=(%d,%d), regular=(%d,%d)",
			easyTrainRows, easyTrainCols, regularTrainRows, regularTrainCols)
	}

	easyTestRows, easyTestCols := easyXTest.Dims()
	regularTestRows, regularTestCols := regularXTest.Dims()

	if easyTestRows != regularTestRows || easyTestCols != regularTestCols {
		t.Errorf("Test set dimensions differ: easy=(%d,%d), regular=(%d,%d)",
			easyTestRows, easyTestCols, regularTestRows, regularTestCols)
	}

	// Y dimensions should also match
	easyYTrainRows, _ := easyYTrain.Dims()
	regularYTrainRows, _ := regularYTrain.Dims()

	if easyYTrainRows != regularYTrainRows {
		t.Errorf("Y training set rows differ: easy=%d, regular=%d", easyYTrainRows, regularYTrainRows)
	}

	easyYTestRows, _ := easyYTest.Dims()
	regularYTestRows, _ := regularYTest.Dims()

	if easyYTestRows != regularYTestRows {
		t.Errorf("Y test set rows differ: easy=%d, regular=%d", easyYTestRows, regularYTestRows)
	}
}

// TestPreprocessingHelperFunctionsWithEdgeCases tests edge cases
func TestPreprocessingHelperFunctionsWithEdgeCases(t *testing.T) {
	// Test with single row
	t.Run("SingleRow", func(t *testing.T) {
		data := [][]float64{{1.0, 2.0, 3.0}}
		X := core.NewTensorFromSlice(data)

		// Standard scaling with single row should handle division by zero
		scaled, err := EasyStandardScale(X)
		if err != nil {
			t.Errorf("EasyStandardScale failed with single row: %v", err)
		} else {
			// Should return zeros or handle gracefully
			rows, cols := scaled.Dims()
			if rows != 1 || cols != 3 {
				t.Errorf("Dimensions changed: expected (1,3), got (%d,%d)", rows, cols)
			}
		}

		// MinMax scaling with single row should work
		scaled, err = EasyMinMaxScale(X)
		if err != nil {
			t.Errorf("EasyMinMaxScale failed with single row: %v", err)
		} else {
			rows, cols := scaled.Dims()
			if rows != 1 || cols != 3 {
				t.Errorf("Dimensions changed: expected (1,3), got (%d,%d)", rows, cols)
			}
		}
	})

	// Test with constant features
	t.Run("ConstantFeatures", func(t *testing.T) {
		data := [][]float64{
			{5.0, 1.0, 2.0},
			{5.0, 3.0, 4.0},
			{5.0, 5.0, 6.0},
		}
		X := core.NewTensorFromSlice(data)

		// Standard scaling should handle constant features
		scaled, err := EasyStandardScale(X)
		if err != nil {
			t.Errorf("EasyStandardScale failed with constant features: %v", err)
		} else {
			// First column should be handled gracefully (constant feature)
			rows, cols := scaled.Dims()
			if rows != 3 || cols != 3 {
				t.Errorf("Dimensions changed: expected (3,3), got (%d,%d)", rows, cols)
			}
		}

		// MinMax scaling should handle constant features
		scaled, err = EasyMinMaxScale(X)
		if err != nil {
			t.Errorf("EasyMinMaxScale failed with constant features: %v", err)
		} else {
			rows, cols := scaled.Dims()
			if rows != 3 || cols != 3 {
				t.Errorf("Dimensions changed: expected (3,3), got (%d,%d)", rows, cols)
			}
		}
	})

	// Test with empty tensor
	t.Run("EmptyTensor", func(t *testing.T) {
		X := core.NewZerosTensor(0, 0)

		_, err := EasyStandardScale(X)
		if err == nil {
			t.Error("Expected error for empty tensor in EasyStandardScale")
		}

		_, err = EasyMinMaxScale(X)
		if err == nil {
			t.Error("Expected error for empty tensor in EasyMinMaxScale")
		}
	})
}

// TestPreprocessingHelperFunctionsPerformance tests that helper functions are reasonably fast
func TestPreprocessingHelperFunctionsPerformance(t *testing.T) {
	// Create larger dataset for performance testing
	rows, cols := 1000, 50
	data := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		data[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			data[i][j] = float64(i*cols + j)
		}
	}

	X := core.NewTensorFromSlice(data)

	// Test EasyStandardScale performance
	t.Run("EasyStandardScalePerformance", func(t *testing.T) {
		_, err := EasyStandardScale(X)
		if err != nil {
			t.Errorf("EasyStandardScale failed on large dataset: %v", err)
		}
	})

	// Test EasyMinMaxScale performance
	t.Run("EasyMinMaxScalePerformance", func(t *testing.T) {
		_, err := EasyMinMaxScale(X)
		if err != nil {
			t.Errorf("EasyMinMaxScale failed on large dataset: %v", err)
		}
	})
}
