package pkg

import (
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/algorithms"
	"github.com/blackmoon87/thinkingnet/pkg/core"
	"github.com/blackmoon87/thinkingnet/pkg/preprocessing"
)

// TestHelperFunctionsIntegration tests all helper functions working together
func TestHelperFunctionsIntegration(t *testing.T) {
	// Create sample dataset for testing
	XData := [][]float64{
		{1.0, 10.0, 100.0},
		{2.0, 20.0, 200.0},
		{3.0, 30.0, 300.0},
		{4.0, 40.0, 400.0},
		{5.0, 50.0, 500.0},
		{6.0, 60.0, 600.0},
		{7.0, 70.0, 700.0},
		{8.0, 80.0, 800.0},
		{9.0, 90.0, 900.0},
		{10.0, 100.0, 1000.0},
	}

	yData := [][]float64{
		{1}, {0}, {1}, {0}, {1}, {0}, {1}, {0}, {1}, {0},
	}

	// Step 1: Create tensors using EasyTensor
	X := core.EasyTensor(XData)
	y := core.EasyTensor(yData)

	// Verify tensor creation
	rows, cols := X.Dims()
	if rows != 10 || cols != 3 {
		t.Errorf("Expected X dimensions (10, 3), got (%d, %d)", rows, cols)
	}

	yRows, yCols := y.Dims()
	if yRows != 10 || yCols != 1 {
		t.Errorf("Expected y dimensions (10, 1), got (%d, %d)", yRows, yCols)
	}

	// Step 2: Split data using EasySplit
	XTrain, XTest, yTrain, yTest, err := preprocessing.EasySplit(X, y, 0.3)
	if err != nil {
		t.Fatalf("EasySplit failed: %v", err)
	}

	trainRows, _ := XTrain.Dims()
	testRows, _ := XTest.Dims()

	if trainRows+testRows != 10 {
		t.Errorf("Split sample count mismatch: expected 10, got %d", trainRows+testRows)
	}

	// Verify y split dimensions
	yTrainRows, _ := yTrain.Dims()
	yTestRows, _ := yTest.Dims()
	if yTrainRows != trainRows || yTestRows != testRows {
		t.Errorf("Y split dimensions don't match X split: X=(%d,%d), y=(%d,%d)", trainRows, testRows, yTrainRows, yTestRows)
	}

	// Step 3: Scale training data using EasyStandardScale
	XTrainScaled, err := preprocessing.EasyStandardScale(XTrain)
	if err != nil {
		t.Fatalf("EasyStandardScale failed: %v", err)
	}

	// Step 4: Scale test data using EasyMinMaxScale
	XTestScaled, err := preprocessing.EasyMinMaxScale(XTest)
	if err != nil {
		t.Fatalf("EasyMinMaxScale failed: %v", err)
	}

	// Verify scaling preserved dimensions
	scaledTrainRows, scaledTrainCols := XTrainScaled.Dims()
	scaledTestRows, scaledTestCols := XTestScaled.Dims()

	if scaledTrainRows != trainRows || scaledTrainCols != 3 {
		t.Errorf("Training scaling changed dimensions: expected (%d, 3), got (%d, %d)", trainRows, scaledTrainRows, scaledTrainCols)
	}

	if scaledTestRows != testRows || scaledTestCols != 3 {
		t.Errorf("Test scaling changed dimensions: expected (%d, 3), got (%d, %d)", testRows, scaledTestRows, scaledTestCols)
	}

	// Step 5: Test algorithm helper functions
	t.Run("LinearRegression", func(t *testing.T) {
		// Create regression data (use first column as target)
		yRegression := core.EasyTensor([][]float64{
			{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0},
		})

		model := algorithms.EasyLinearRegression()
		err := model.Fit(XTrainScaled, yRegression)
		if err != nil {
			t.Errorf("Linear regression fit failed: %v", err)
		}

		predictions, err := model.Predict(XTestScaled)
		if err != nil {
			t.Errorf("Linear regression predict failed: %v", err)
		}

		predRows, predCols := predictions.Dims()
		if predRows != testRows || predCols != 1 {
			t.Errorf("Prediction dimensions incorrect: expected (%d, 1), got (%d, %d)", testRows, predRows, predCols)
		}
	})

	t.Run("LogisticRegression", func(t *testing.T) {
		model := algorithms.EasyLogisticRegression()
		err := model.Fit(XTrainScaled, yTrain)
		if err != nil {
			t.Errorf("Logistic regression fit failed: %v", err)
		}

		predictions, err := model.Predict(XTestScaled)
		if err != nil {
			t.Errorf("Logistic regression predict failed: %v", err)
		}

		predRows, predCols := predictions.Dims()
		if predRows != testRows || predCols != 1 {
			t.Errorf("Prediction dimensions incorrect: expected (%d, 1), got (%d, %d)", testRows, predRows, predCols)
		}
	})

	t.Run("KMeans", func(t *testing.T) {
		model := algorithms.EasyKMeans(2)
		err := model.Fit(XTrainScaled)
		if err != nil {
			t.Errorf("K-means fit failed: %v", err)
		}

		labels, err := model.Predict(XTestScaled)
		if err != nil {
			t.Errorf("K-means predict failed: %v", err)
		}

		if len(labels) != testRows {
			t.Errorf("K-means labels count incorrect: expected %d, got %d", testRows, len(labels))
		}

		// Verify labels are valid (0 or 1 for k=2)
		for i, label := range labels {
			if label < 0 || label >= 2 {
				t.Errorf("Invalid cluster label at index %d: %d", i, label)
			}
		}
	})

	t.Logf("Integration test passed successfully!")
	t.Logf("Data: %d samples, %d features", rows, cols)
	t.Logf("Split: %d train, %d test", trainRows, testRows)
	t.Logf("All helper functions work correctly together")
}

// TestHelperFunctionsErrorHandling tests error handling across all helper functions
func TestHelperFunctionsErrorHandling(t *testing.T) {
	// Test EasyTensor error handling
	t.Run("EasyTensorErrors", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for empty data")
			}
		}()
		core.EasyTensor([][]float64{})
	})

	// Test EasySplit error handling
	t.Run("EasySplitErrors", func(t *testing.T) {
		X := core.EasyTensor([][]float64{{1.0, 2.0}, {3.0, 4.0}})
		y := core.EasyTensor([][]float64{{1.0}, {0.0}})

		_, _, _, _, err := preprocessing.EasySplit(X, y, 1.5)
		if err == nil {
			t.Error("Expected error for invalid test size")
		}
	})

	// Test EasyStandardScale error handling
	t.Run("EasyStandardScaleErrors", func(t *testing.T) {
		_, err := preprocessing.EasyStandardScale(nil)
		if err == nil {
			t.Error("Expected error for nil tensor")
		}
	})

	// Test EasyMinMaxScale error handling
	t.Run("EasyMinMaxScaleErrors", func(t *testing.T) {
		_, err := preprocessing.EasyMinMaxScale(nil)
		if err == nil {
			t.Error("Expected error for nil tensor")
		}
	})

	// Test algorithm error handling
	t.Run("AlgorithmErrors", func(t *testing.T) {
		linearModel := algorithms.EasyLinearRegression()
		err := linearModel.Fit(nil, nil)
		if err == nil {
			t.Error("Expected error for nil data in linear regression")
		}

		logisticModel := algorithms.EasyLogisticRegression()
		err = logisticModel.Fit(nil, nil)
		if err == nil {
			t.Error("Expected error for nil data in logistic regression")
		}

		kmeansModel := algorithms.EasyKMeans(2)
		err = kmeansModel.Fit(nil)
		if err == nil {
			t.Error("Expected error for nil data in k-means")
		}
	})
}

// TestHelperFunctionsBackwardCompatibility tests that helper functions don't break existing functionality
func TestHelperFunctionsBackwardCompatibility(t *testing.T) {
	// Create test data
	XData := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
		{10.0, 11.0, 12.0},
	}
	yData := [][]float64{
		{1}, {0}, {1}, {0},
	}
	y := core.NewTensorFromSlice(yData)

	// Test that EasyTensor produces same results as NewTensorFromSlice
	easyTensor := core.EasyTensor(XData)
	regularTensor := core.NewTensorFromSlice(XData)

	if !easyTensor.Equal(regularTensor) {
		t.Error("EasyTensor produces different results than NewTensorFromSlice")
	}

	// Test that EasyStandardScale produces same results as regular StandardScaler
	X := core.NewTensorFromSlice(XData)

	easyScaled, err := preprocessing.EasyStandardScale(X)
	if err != nil {
		t.Fatalf("EasyStandardScale failed: %v", err)
	}

	scaler := preprocessing.NewStandardScaler()
	regularScaled, err := scaler.FitTransform(X)
	if err != nil {
		t.Fatalf("Regular StandardScaler failed: %v", err)
	}

	if !easyScaled.Equal(regularScaled) {
		t.Error("EasyStandardScale produces different results than regular StandardScaler")
	}

	// Test that EasyMinMaxScale produces same results as regular MinMaxScaler
	easyMinMax, err := preprocessing.EasyMinMaxScale(X)
	if err != nil {
		t.Fatalf("EasyMinMaxScale failed: %v", err)
	}

	minMaxScaler := preprocessing.NewMinMaxScaler()
	regularMinMax, err := minMaxScaler.FitTransform(X)
	if err != nil {
		t.Fatalf("Regular MinMaxScaler failed: %v", err)
	}

	if !easyMinMax.Equal(regularMinMax) {
		t.Error("EasyMinMaxScale produces different results than regular MinMaxScaler")
	}

	// Use y to avoid unused variable warning
	_ = y

	t.Log("All helper functions maintain backward compatibility")
}
