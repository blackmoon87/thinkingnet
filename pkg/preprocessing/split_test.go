package preprocessing

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestTrainTestSplit(t *testing.T) {
	// Create test data
	X := core.NewTensorFromSlice([][]float64{
		{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10},
		{11, 12}, {13, 14}, {15, 16}, {17, 18}, {19, 20},
	})
	y := core.NewTensorFromSlice([][]float64{
		{0}, {1}, {0}, {1}, {0}, {1}, {0}, {1}, {0}, {1},
	})

	tests := []struct {
		name     string
		testSize float64
		shuffle  bool
		wantErr  bool
	}{
		{"basic split", 0.3, false, false},
		{"with shuffle", 0.2, true, false},
		{"half split", 0.5, false, false},
		{"invalid size", 1.5, false, true},
		{"zero size", 0.0, false, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			XTrain, XTest, yTrain, yTest, err := TrainTestSplit(X, y,
				WithTestSize(tt.testSize),
				WithShuffle(tt.shuffle))

			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			// Check dimensions
			xTrainRows, _ := XTrain.Dims()
			xTestRows, _ := XTest.Dims()
			yTrainRows, _ := yTrain.Dims()
			yTestRows, _ := yTest.Dims()

			totalRows, _ := X.Dims()
			if xTrainRows+xTestRows != totalRows {
				t.Errorf("split sizes don't add up: %d + %d != %d", xTrainRows, xTestRows, totalRows)
			}

			if xTrainRows != yTrainRows {
				t.Errorf("X_train and y_train have different number of rows: %d vs %d", xTrainRows, yTrainRows)
			}

			if xTestRows != yTestRows {
				t.Errorf("X_test and y_test have different number of rows: %d vs %d", xTestRows, yTestRows)
			}

			// Check test size proportion
			expectedTestSize := int(math.Round(float64(totalRows) * tt.testSize))
			if math.Abs(float64(xTestRows-expectedTestSize)) > 1 {
				t.Errorf("test size not as expected: got %d, expected ~%d", xTestRows, expectedTestSize)
			}
		})
	}
}

func TestStratifiedSplit(t *testing.T) {
	// Create balanced test data
	X := core.NewTensorFromSlice([][]float64{
		{1, 2}, {3, 4}, {5, 6}, {7, 8}, // class 0
		{9, 10}, {11, 12}, {13, 14}, {15, 16}, // class 1
		{17, 18}, {19, 20}, {21, 22}, {23, 24}, // class 2
	})
	y := core.NewTensorFromSlice([][]float64{
		{0}, {0}, {0}, {0}, {1}, {1}, {1}, {1}, {2}, {2}, {2}, {2},
	})

	_, _, yTrain, yTest, err := TrainTestSplit(X, y,
		WithTestSize(0.25),
		WithStratify(true),
		WithShuffle(true))

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Check that each class is represented in both train and test sets
	trainLabels := extractLabels(yTrain)
	testLabels := extractLabels(yTest)

	trainClasses := countClasses(trainLabels)
	testClasses := countClasses(testLabels)

	if len(trainClasses) != 3 || len(testClasses) != 3 {
		t.Errorf("not all classes represented in splits. Train: %v, Test: %v", trainClasses, testClasses)
	}

	// Check approximate proportions
	for class := 0; class < 3; class++ {
		trainCount := trainClasses[class]
		testCount := testClasses[class]
		totalCount := trainCount + testCount

		if totalCount != 4 {
			t.Errorf("class %d should have 4 samples total, got %d", class, totalCount)
		}
	}
}

func TestTrainValTestSplit(t *testing.T) {
	X := core.NewTensorFromSlice([][]float64{
		{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10},
		{11, 12}, {13, 14}, {15, 16}, {17, 18}, {19, 20},
	})
	y := core.NewTensorFromSlice([][]float64{
		{0}, {1}, {0}, {1}, {0}, {1}, {0}, {1}, {0}, {1},
	})

	XTrain, XVal, XTest, _, _, _, err := TrainValTestSplit(X, y, 0.2,
		WithTestSize(0.2),
		WithShuffle(true))

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Check dimensions
	trainRows, _ := XTrain.Dims()
	valRows, _ := XVal.Dims()
	testRows, _ := XTest.Dims()
	totalRows, _ := X.Dims()

	if trainRows+valRows+testRows != totalRows {
		t.Errorf("split sizes don't add up: %d + %d + %d != %d", trainRows, valRows, testRows, totalRows)
	}

	// Check approximate proportions
	expectedTrain := int(0.6 * float64(totalRows)) // 1 - 0.2 - 0.2
	expectedVal := int(0.2 * float64(totalRows))
	expectedTest := int(0.2 * float64(totalRows))

	if math.Abs(float64(trainRows-expectedTrain)) > 1 {
		t.Errorf("train size not as expected: got %d, expected ~%d", trainRows, expectedTrain)
	}
	if math.Abs(float64(valRows-expectedVal)) > 1 {
		t.Errorf("val size not as expected: got %d, expected ~%d", valRows, expectedVal)
	}
	if math.Abs(float64(testRows-expectedTest)) > 1 {
		t.Errorf("test size not as expected: got %d, expected ~%d", testRows, expectedTest)
	}
}

func TestDataValidator(t *testing.T) {
	validator := NewDataValidator()

	t.Run("valid data", func(t *testing.T) {
		X := core.NewTensorFromSlice([][]float64{
			{1, 2, 3}, {4, 5, 6}, {7, 8, 9},
		})
		y := core.NewTensorFromSlice([][]float64{
			{0}, {1}, {0},
		})

		result := validator.ValidateData(X, y)
		if !result.IsValid {
			t.Errorf("expected valid data, got errors: %v", result.Errors)
		}
	})

	t.Run("nil tensors", func(t *testing.T) {
		result := validator.ValidateData(nil, nil)
		if result.IsValid {
			t.Errorf("expected invalid data for nil tensors")
		}
		if len(result.Errors) == 0 {
			t.Errorf("expected errors for nil tensors")
		}
	})

	t.Run("dimension mismatch", func(t *testing.T) {
		X := core.NewTensorFromSlice([][]float64{
			{1, 2}, {3, 4}, {5, 6},
		})
		y := core.NewTensorFromSlice([][]float64{
			{0}, {1},
		})

		result := validator.ValidateData(X, y)
		if result.IsValid {
			t.Errorf("expected invalid data for dimension mismatch")
		}
	})
}

func TestValidateFeatures(t *testing.T) {
	validator := NewDataValidator()

	t.Run("constant features", func(t *testing.T) {
		X := core.NewTensorFromSlice([][]float64{
			{1, 5, 3}, {1, 6, 4}, {1, 7, 5}, // first column is constant
		})

		result := validator.ValidateFeatures(X)
		if !result.IsValid {
			t.Errorf("data should be valid despite constant features")
		}
		if len(result.Warnings) == 0 {
			t.Errorf("expected warning about constant features")
		}
	})

	t.Run("normal features", func(t *testing.T) {
		X := core.NewTensorFromSlice([][]float64{
			{1, 2, 3}, {4, 5, 6}, {7, 8, 9},
		})

		result := validator.ValidateFeatures(X)
		if !result.IsValid {
			t.Errorf("expected valid features, got errors: %v", result.Errors)
		}
	})
}

func TestValidateLabels(t *testing.T) {
	validator := NewDataValidator()

	t.Run("balanced labels", func(t *testing.T) {
		y := core.NewTensorFromSlice([][]float64{
			{0}, {1}, {0}, {1}, {0}, {1},
		})

		result := validator.ValidateLabels(y)
		if !result.IsValid {
			t.Errorf("expected valid labels, got errors: %v", result.Errors)
		}
	})

	t.Run("single class", func(t *testing.T) {
		y := core.NewTensorFromSlice([][]float64{
			{0}, {0}, {0}, {0},
		})

		result := validator.ValidateLabels(y)
		if result.IsValid {
			t.Errorf("expected invalid labels for single class")
		}
	})

	t.Run("imbalanced labels", func(t *testing.T) {
		y := core.NewTensorFromSlice([][]float64{
			{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {1}, // 11:1 ratio
		})

		result := validator.ValidateLabels(y)
		if !result.IsValid {
			t.Errorf("imbalanced labels should still be valid")
		}

		if len(result.Warnings) == 0 {
			t.Errorf("expected warning about class imbalance")
		}
	})
}

func TestSplitConfigValidation(t *testing.T) {
	X := core.NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	y := core.NewTensorFromSlice([][]float64{{0}, {1}})

	tests := []struct {
		name     string
		testSize float64
		wantErr  bool
	}{
		{"valid size", 0.5, false},
		{"too large", 1.1, true},
		{"negative", -0.1, true},
		{"zero", 0.0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, _, err := TrainTestSplit(X, y, WithTestSize(tt.testSize))
			if (err != nil) != tt.wantErr {
				t.Errorf("expected error: %v, got: %v", tt.wantErr, err)
			}
		})
	}
}

func TestStratifiedSplitEdgeCases(t *testing.T) {
	t.Run("insufficient samples per class", func(t *testing.T) {
		X := core.NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
		y := core.NewTensorFromSlice([][]float64{{0}, {1}}) // only 1 sample per class

		_, _, _, _, err := TrainTestSplit(X, y,
			WithTestSize(0.5),
			WithStratify(true))

		if err == nil {
			t.Errorf("expected error for insufficient samples per class")
		}
	})

	t.Run("multi-column y", func(t *testing.T) {
		X := core.NewTensorFromSlice([][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}})
		y := core.NewTensorFromSlice([][]float64{{0, 1}, {1, 0}, {0, 1}, {1, 0}}) // 2 columns

		_, _, _, _, err := TrainTestSplit(X, y,
			WithTestSize(0.5),
			WithStratify(true))

		if err == nil {
			t.Errorf("expected error for multi-column y in stratified split")
		}
	})
}

// Helper functions

func extractLabels(y core.Tensor) []int {
	rows, _ := y.Dims()
	labels := make([]int, rows)
	for i := range rows {
		labels[i] = int(math.Round(y.At(i, 0)))
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

func TestCheckDataQuality(t *testing.T) {
	X := core.NewTensorFromSlice([][]float64{
		{1, 2, 3}, {4, 5, 6}, {7, 8, 9},
	})
	y := core.NewTensorFromSlice([][]float64{
		{0}, {1}, {0},
	})

	result := CheckDataQuality(X, y)
	if !result.IsValid {
		t.Errorf("expected valid data quality, got errors: %v", result.Errors)
	}

	if result.Statistics == nil {
		t.Errorf("expected statistics to be computed")
	}
}

func TestCheckFeatureQuality(t *testing.T) {
	X := core.NewTensorFromSlice([][]float64{
		{1, 2, 3}, {4, 5, 6}, {7, 8, 9},
	})

	result := CheckFeatureQuality(X)
	if !result.IsValid {
		t.Errorf("expected valid feature quality, got errors: %v", result.Errors)
	}
}

func TestCheckLabelQuality(t *testing.T) {
	y := core.NewTensorFromSlice([][]float64{
		{0}, {1}, {0}, {1},
	})

	result := CheckLabelQuality(y)
	if !result.IsValid {
		t.Errorf("expected valid label quality, got errors: %v", result.Errors)
	}
}
