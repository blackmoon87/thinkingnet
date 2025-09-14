package core

import (
	"math"
	"strings"
	"testing"
)

func TestEasyTensor(t *testing.T) {
	tests := []struct {
		name        string
		data        [][]float64
		expectPanic bool
		expectRows  int
		expectCols  int
	}{
		{
			name: "valid 2x3 tensor",
			data: [][]float64{
				{1.0, 2.0, 3.0},
				{4.0, 5.0, 6.0},
			},
			expectPanic: false,
			expectRows:  2,
			expectCols:  3,
		},
		{
			name: "valid 1x1 tensor",
			data: [][]float64{
				{42.0},
			},
			expectPanic: false,
			expectRows:  1,
			expectCols:  1,
		},
		{
			name:        "empty data",
			data:        [][]float64{},
			expectPanic: true,
		},
		{
			name: "empty rows",
			data: [][]float64{
				{},
			},
			expectPanic: true,
		},
		{
			name: "inconsistent row lengths",
			data: [][]float64{
				{1.0, 2.0},
				{3.0, 4.0, 5.0},
			},
			expectPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if tt.expectPanic && r == nil {
					t.Errorf("Expected panic but none occurred")
				} else if !tt.expectPanic && r != nil {
					t.Errorf("Unexpected panic: %v", r)
				}
			}()

			tensor := EasyTensor(tt.data)

			if !tt.expectPanic {
				rows, cols := tensor.Dims()
				if rows != tt.expectRows {
					t.Errorf("Expected %d rows, got %d", tt.expectRows, rows)
				}
				if cols != tt.expectCols {
					t.Errorf("Expected %d cols, got %d", tt.expectCols, cols)
				}

				// Verify data integrity
				for i := 0; i < rows; i++ {
					for j := 0; j < cols; j++ {
						expected := tt.data[i][j]
						actual := tensor.At(i, j)
						if math.Abs(expected-actual) > 1e-10 {
							t.Errorf("Data mismatch at (%d,%d): expected %f, got %f", i, j, expected, actual)
						}
					}
				}
			}
		})
	}
}

func TestEasyTensorErrorMessages(t *testing.T) {
	tests := []struct {
		name          string
		data          [][]float64
		expectArabic  bool
		expectEnglish bool
		arabicSubstr  string
		englishSubstr string
	}{
		{
			name:          "empty data",
			data:          [][]float64{},
			expectArabic:  true,
			expectEnglish: true,
			arabicSubstr:  "لا يمكن إنشاء tensor من بيانات فارغة",
			englishSubstr: "Cannot create tensor from empty data",
		},
		{
			name: "empty rows",
			data: [][]float64{
				{},
			},
			expectArabic:  true,
			expectEnglish: true,
			arabicSubstr:  "لا يمكن إنشاء tensor من صفوف فارغة",
			englishSubstr: "Cannot create tensor from empty rows",
		},
		{
			name: "inconsistent row lengths",
			data: [][]float64{
				{1.0, 2.0},
				{3.0, 4.0, 5.0},
			},
			expectArabic:  true,
			expectEnglish: true,
			arabicSubstr:  "جميع الصفوف يجب أن تكون بنفس الطول",
			englishSubstr: "All rows must have the same length",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Errorf("Expected panic but none occurred")
					return
				}

				errorMsg := ""
				if err, ok := r.(error); ok {
					errorMsg = err.Error()
				} else {
					errorMsg = r.(string)
				}

				if tt.expectArabic && !strings.Contains(errorMsg, tt.arabicSubstr) {
					t.Errorf("Expected Arabic error message containing '%s', got: %s", tt.arabicSubstr, errorMsg)
				}

				if tt.expectEnglish && !strings.Contains(errorMsg, tt.englishSubstr) {
					t.Errorf("Expected English error message containing '%s', got: %s", tt.englishSubstr, errorMsg)
				}
			}()

			EasyTensor(tt.data)
		})
	}
}

func TestEasySplit(t *testing.T) {
	// Create sample data
	X := EasyTensor([][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
		{7.0, 8.0},
		{9.0, 10.0},
		{11.0, 12.0},
	})

	y := EasyTensor([][]float64{
		{1.0},
		{0.0},
		{1.0},
		{0.0},
		{1.0},
		{0.0},
	})

	tests := []struct {
		name        string
		X           Tensor
		y           Tensor
		testSize    float64
		expectPanic bool
	}{
		{
			name:        "valid split 0.3",
			X:           X,
			y:           y,
			testSize:    0.3,
			expectPanic: false,
		},
		{
			name:        "valid split 0.5",
			X:           X,
			y:           y,
			testSize:    0.5,
			expectPanic: false,
		},
		{
			name:        "invalid test size 0",
			X:           X,
			y:           y,
			testSize:    0.0,
			expectPanic: true,
		},
		{
			name:        "invalid test size 1",
			X:           X,
			y:           y,
			testSize:    1.0,
			expectPanic: true,
		},
		{
			name:        "invalid test size > 1",
			X:           X,
			y:           y,
			testSize:    1.5,
			expectPanic: true,
		},
		{
			name:        "nil X tensor",
			X:           nil,
			y:           y,
			testSize:    0.3,
			expectPanic: true,
		},
		{
			name:        "nil y tensor",
			X:           X,
			y:           nil,
			testSize:    0.3,
			expectPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if tt.expectPanic && r == nil {
					t.Errorf("Expected panic but none occurred")
				} else if !tt.expectPanic && r != nil {
					t.Errorf("Unexpected panic: %v", r)
				}
			}()

			XTrain, XTest, yTrain, yTest := EasySplit(tt.X, tt.y, tt.testSize)

			if !tt.expectPanic {
				// Verify dimensions
				xRows, xCols := tt.X.Dims()
				_, yCols := tt.y.Dims()

				trainRows, _ := XTrain.Dims()
				testRows, _ := XTest.Dims()

				// Check that total samples are preserved
				if trainRows+testRows != xRows {
					t.Errorf("Sample count mismatch: expected %d, got %d", xRows, trainRows+testRows)
				}

				// Check dimensions consistency
				if trainRowsY, _ := yTrain.Dims(); trainRowsY != trainRows {
					t.Errorf("yTrain rows mismatch: expected %d, got %d", trainRows, trainRowsY)
				}

				if testRowsY, _ := yTest.Dims(); testRowsY != testRows {
					t.Errorf("yTest rows mismatch: expected %d, got %d", testRows, testRowsY)
				}

				// Check column consistency
				_, trainCols := XTrain.Dims()
				if trainCols != xCols {
					t.Errorf("XTrain cols mismatch: expected %d, got %d", xCols, trainCols)
				}

				_, testCols := XTest.Dims()
				if testCols != xCols {
					t.Errorf("XTest cols mismatch: expected %d, got %d", xCols, testCols)
				}

				_, trainColsY := yTrain.Dims()
				if trainColsY != yCols {
					t.Errorf("yTrain cols mismatch: expected %d, got %d", yCols, trainColsY)
				}

				_, testColsY := yTest.Dims()
				if testColsY != yCols {
					t.Errorf("yTest cols mismatch: expected %d, got %d", yCols, testColsY)
				}

				// Verify test size is approximately correct
				// The actual implementation might adjust the test size to ensure at least 1 train and 1 test sample
				// So we just verify that we have reasonable splits
				if testRows < 1 || trainRows < 1 {
					t.Errorf("Invalid split: trainRows=%d, testRows=%d", trainRows, testRows)
				}

				if testRows+trainRows != xRows {
					t.Errorf("Sample count mismatch: expected %d total, got %d", xRows, trainRows+testRows)
				}
			}
		})
	}
}

func TestEasySplitErrorMessages(t *testing.T) {
	X := EasyTensor([][]float64{{1.0, 2.0}, {3.0, 4.0}})
	y := EasyTensor([][]float64{{1.0}, {0.0}})

	tests := []struct {
		name          string
		X             Tensor
		y             Tensor
		testSize      float64
		expectArabic  bool
		expectEnglish bool
		arabicSubstr  string
		englishSubstr string
	}{
		{
			name:          "invalid test size",
			X:             X,
			y:             y,
			testSize:      1.5,
			expectArabic:  true,
			expectEnglish: true,
			arabicSubstr:  "حجم بيانات الاختبار يجب أن يكون بين 0 و 1",
			englishSubstr: "Test size must be between 0 and 1",
		},
		{
			name:          "nil input data",
			X:             nil,
			y:             y,
			testSize:      0.3,
			expectArabic:  true,
			expectEnglish: true,
			arabicSubstr:  "بيانات الإدخال غير صحيحة",
			englishSubstr: "Invalid input data",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Errorf("Expected panic but none occurred")
					return
				}

				errorMsg := ""
				if err, ok := r.(error); ok {
					errorMsg = err.Error()
				} else {
					errorMsg = r.(string)
				}

				if tt.expectArabic && !strings.Contains(errorMsg, tt.arabicSubstr) {
					t.Errorf("Expected Arabic error message containing '%s', got: %s", tt.arabicSubstr, errorMsg)
				}

				if tt.expectEnglish && !strings.Contains(errorMsg, tt.englishSubstr) {
					t.Errorf("Expected English error message containing '%s', got: %s", tt.englishSubstr, errorMsg)
				}
			}()

			EasySplit(tt.X, tt.y, tt.testSize)
		})
	}
}

func TestEasySplitWithSmallDataset(t *testing.T) {
	// Test with minimum viable dataset (2 samples)
	X := EasyTensor([][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	})

	y := EasyTensor([][]float64{
		{1.0},
		{0.0},
	})

	XTrain, XTest, yTrain, yTest := EasySplit(X, y, 0.5)

	// Should have 1 sample each
	trainRows, _ := XTrain.Dims()
	testRows, _ := XTest.Dims()

	if trainRows != 1 {
		t.Errorf("Expected 1 training sample, got %d", trainRows)
	}

	if testRows != 1 {
		t.Errorf("Expected 1 test sample, got %d", testRows)
	}

	// Verify y dimensions match
	yTrainRows, _ := yTrain.Dims()
	yTestRows, _ := yTest.Dims()

	if yTrainRows != 1 {
		t.Errorf("Expected 1 y training sample, got %d", yTrainRows)
	}

	if yTestRows != 1 {
		t.Errorf("Expected 1 y test sample, got %d", yTestRows)
	}
}

func TestEasySplitWithSingleSample(t *testing.T) {
	// Test with single sample (should panic)
	X := EasyTensor([][]float64{
		{1.0, 2.0},
	})

	y := EasyTensor([][]float64{
		{1.0},
	})

	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("Expected panic with single sample but none occurred")
		}
	}()

	EasySplit(X, y, 0.5)
}

func TestEasySplitDataIntegrity(t *testing.T) {
	// Create identifiable data to verify no data corruption
	X := EasyTensor([][]float64{
		{1.0, 10.0},
		{2.0, 20.0},
		{3.0, 30.0},
		{4.0, 40.0},
	})

	y := EasyTensor([][]float64{
		{100.0},
		{200.0},
		{300.0},
		{400.0},
	})

	XTrain, XTest, yTrain, yTest := EasySplit(X, y, 0.5)

	// Verify that training data comes from first part of original data
	trainRows, _ := XTrain.Dims()
	for i := 0; i < trainRows; i++ {
		expectedX1 := float64(i + 1)
		expectedX2 := float64((i + 1) * 10)
		expectedY := float64((i + 1) * 100)

		if math.Abs(XTrain.At(i, 0)-expectedX1) > 1e-10 {
			t.Errorf("XTrain data corruption at (%d,0): expected %f, got %f", i, expectedX1, XTrain.At(i, 0))
		}

		if math.Abs(XTrain.At(i, 1)-expectedX2) > 1e-10 {
			t.Errorf("XTrain data corruption at (%d,1): expected %f, got %f", i, expectedX2, XTrain.At(i, 1))
		}

		if math.Abs(yTrain.At(i, 0)-expectedY) > 1e-10 {
			t.Errorf("yTrain data corruption at (%d,0): expected %f, got %f", i, expectedY, yTrain.At(i, 0))
		}
	}

	// Verify that test data comes from second part of original data
	testRows, _ := XTest.Dims()
	for i := 0; i < testRows; i++ {
		originalIdx := trainRows + i
		expectedX1 := float64(originalIdx + 1)
		expectedX2 := float64((originalIdx + 1) * 10)
		expectedY := float64((originalIdx + 1) * 100)

		if math.Abs(XTest.At(i, 0)-expectedX1) > 1e-10 {
			t.Errorf("XTest data corruption at (%d,0): expected %f, got %f", i, expectedX1, XTest.At(i, 0))
		}

		if math.Abs(XTest.At(i, 1)-expectedX2) > 1e-10 {
			t.Errorf("XTest data corruption at (%d,1): expected %f, got %f", i, expectedX2, XTest.At(i, 1))
		}

		if math.Abs(yTest.At(i, 0)-expectedY) > 1e-10 {
			t.Errorf("yTest data corruption at (%d,0): expected %f, got %f", i, expectedY, yTest.At(i, 0))
		}
	}
}

// TestTensorHelperFunctionsBackwardCompatibility tests that helper functions don't break existing functionality
func TestTensorHelperFunctionsBackwardCompatibility(t *testing.T) {
	// Test EasyTensor vs NewTensorFromSlice
	data := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
	}

	easyTensor := EasyTensor(data)
	regularTensor := NewTensorFromSlice(data)

	// Should have same dimensions
	easyRows, easyCols := easyTensor.Dims()
	regularRows, regularCols := regularTensor.Dims()

	if easyRows != regularRows || easyCols != regularCols {
		t.Errorf("Dimensions differ: easy=(%d,%d), regular=(%d,%d)", easyRows, easyCols, regularRows, regularCols)
	}

	// Should have same data
	for i := 0; i < easyRows; i++ {
		for j := 0; j < easyCols; j++ {
			easyVal := easyTensor.At(i, j)
			regularVal := regularTensor.At(i, j)
			if math.Abs(easyVal-regularVal) > 1e-10 {
				t.Errorf("Data differs at (%d,%d): easy=%f, regular=%f", i, j, easyVal, regularVal)
			}
		}
	}
}

// TestTensorHelperFunctionsWithLargeData tests performance with larger datasets
func TestTensorHelperFunctionsWithLargeData(t *testing.T) {
	// Create larger dataset
	rows, cols := 1000, 10
	data := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		data[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			data[i][j] = float64(i*cols + j)
		}
	}

	// Test EasyTensor with large data
	tensor := EasyTensor(data)
	actualRows, actualCols := tensor.Dims()

	if actualRows != rows || actualCols != cols {
		t.Errorf("Large tensor dimensions incorrect: expected (%d,%d), got (%d,%d)", rows, cols, actualRows, actualCols)
	}

	// Create y data for splitting
	yData := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		yData[i] = []float64{float64(i % 2)} // Binary labels
	}
	yTensor := EasyTensor(yData)

	// Test EasySplit with large data
	XTrain, XTest, yTrain, yTest := EasySplit(tensor, yTensor, 0.2)

	trainRows, _ := XTrain.Dims()
	testRows, _ := XTest.Dims()

	if trainRows+testRows != rows {
		t.Errorf("Split sample count mismatch: expected %d, got %d", rows, trainRows+testRows)
	}

	// Verify approximate split ratio
	expectedTestRows := int(float64(rows) * 0.2)
	if testRows < expectedTestRows-1 || testRows > expectedTestRows+1 {
		t.Errorf("Test split size not approximately correct: expected ~%d, got %d", expectedTestRows, testRows)
	}

	// Verify y split matches X split
	yTrainRows, _ := yTrain.Dims()
	yTestRows, _ := yTest.Dims()

	if yTrainRows != trainRows || yTestRows != testRows {
		t.Errorf("Y split doesn't match X split: X=(%d,%d), y=(%d,%d)", trainRows, testRows, yTrainRows, yTestRows)
	}
}

// TestTensorHelperFunctionsEdgeCases tests various edge cases
func TestTensorHelperFunctionsEdgeCases(t *testing.T) {
	// Test with very small test size
	t.Run("VerySmallTestSize", func(t *testing.T) {
		X := EasyTensor([][]float64{
			{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}, {9.0}, {10.0},
		})
		y := EasyTensor([][]float64{
			{1}, {0}, {1}, {0}, {1}, {0}, {1}, {0}, {1}, {0},
		})

		// Very small test size should still work
		XTrain, XTest, yTrain, yTest := EasySplit(X, y, 0.05) // 5%

		trainRows, _ := XTrain.Dims()
		testRows, _ := XTest.Dims()

		// Should have at least 1 sample in each split
		if trainRows < 1 || testRows < 1 {
			t.Errorf("Invalid split with very small test size: train=%d, test=%d", trainRows, testRows)
		}

		// Total should be preserved
		if trainRows+testRows != 10 {
			t.Errorf("Sample count not preserved: expected 10, got %d", trainRows+testRows)
		}

		// Y should match X
		yTrainRows, _ := yTrain.Dims()
		yTestRows, _ := yTest.Dims()
		if yTrainRows != trainRows || yTestRows != testRows {
			t.Errorf("Y split doesn't match X split")
		}
	})

	// Test with single column data
	t.Run("SingleColumn", func(t *testing.T) {
		X := EasyTensor([][]float64{
			{1.0}, {2.0}, {3.0}, {4.0},
		})
		y := EasyTensor([][]float64{
			{1}, {0}, {1}, {0},
		})

		XTrain, XTest, yTrain, yTest := EasySplit(X, y, 0.5)

		// Verify dimensions
		_, trainCols := XTrain.Dims()
		_, testCols := XTest.Dims()

		if trainCols != 1 || testCols != 1 {
			t.Errorf("Column count should be preserved: train=%d, test=%d", trainCols, testCols)
		}

		// Verify y dimensions
		_, yTrainCols := yTrain.Dims()
		_, yTestCols := yTest.Dims()

		if yTrainCols != 1 || yTestCols != 1 {
			t.Errorf("Y column count should be preserved: train=%d, test=%d", yTrainCols, yTestCols)
		}
	})

	// Test with wide data (many columns)
	t.Run("WideData", func(t *testing.T) {
		cols := 100
		data := make([][]float64, 4)
		for i := 0; i < 4; i++ {
			data[i] = make([]float64, cols)
			for j := 0; j < cols; j++ {
				data[i][j] = float64(i*cols + j)
			}
		}

		X := EasyTensor(data)
		y := EasyTensor([][]float64{{1}, {0}, {1}, {0}})

		XTrain, XTest, yTrain, yTest := EasySplit(X, y, 0.5)

		// Verify column count is preserved
		_, trainCols := XTrain.Dims()
		_, testCols := XTest.Dims()

		if trainCols != cols || testCols != cols {
			t.Errorf("Wide data column count not preserved: expected %d, got train=%d, test=%d", cols, trainCols, testCols)
		}

		// Verify y dimensions
		_, yTrainCols := yTrain.Dims()
		_, yTestCols := yTest.Dims()

		if yTrainCols != 1 || yTestCols != 1 {
			t.Errorf("Y column count should be preserved: train=%d, test=%d", yTrainCols, yTestCols)
		}

		// Verify data integrity for a few columns
		trainRows, _ := XTrain.Dims()
		for i := 0; i < trainRows; i++ {
			for j := 0; j < 5; j++ { // Check first 5 columns
				expected := float64(i*cols + j)
				actual := XTrain.At(i, j)
				if math.Abs(expected-actual) > 1e-10 {
					t.Errorf("Wide data corruption at (%d,%d): expected %f, got %f", i, j, expected, actual)
				}
			}
		}
	})
}
