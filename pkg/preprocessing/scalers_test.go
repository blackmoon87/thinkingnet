package preprocessing

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// Test data for scalers
var testData = [][]float64{
	{1.0, 2.0, 3.0},
	{4.0, 5.0, 6.0},
	{7.0, 8.0, 9.0},
	{10.0, 11.0, 12.0},
}

func createTestTensor() core.Tensor {
	return core.NewTensorFromSlice(testData)
}

func assertTensorEqual(t *testing.T, expected, actual core.Tensor, tolerance float64) {
	t.Helper()

	if expected == nil && actual == nil {
		return
	}

	if expected == nil || actual == nil {
		t.Fatalf("One tensor is nil: expected=%v, actual=%v", expected, actual)
	}

	eRows, eCols := expected.Dims()
	aRows, aCols := actual.Dims()

	if eRows != aRows || eCols != aCols {
		t.Fatalf("Tensor dimensions mismatch: expected=(%d,%d), actual=(%d,%d)", eRows, eCols, aRows, aCols)
	}

	for i := 0; i < eRows; i++ {
		for j := 0; j < eCols; j++ {
			expectedVal := expected.At(i, j)
			actualVal := actual.At(i, j)
			if math.Abs(expectedVal-actualVal) > tolerance {
				t.Errorf("Values differ at (%d,%d): expected=%f, actual=%f, diff=%f",
					i, j, expectedVal, actualVal, math.Abs(expectedVal-actualVal))
			}
		}
	}
}

func TestStandardScaler_Basic(t *testing.T) {
	scaler := NewStandardScaler()
	data := createTestTensor()

	// Test initial state
	if scaler.IsFitted() {
		t.Error("Scaler should not be fitted initially")
	}

	if scaler.Name() != "StandardScaler" {
		t.Errorf("Expected name 'StandardScaler', got '%s'", scaler.Name())
	}

	// Test fit
	err := scaler.Fit(data)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	if !scaler.IsFitted() {
		t.Error("Scaler should be fitted after Fit()")
	}

	// Check learned parameters
	mean := scaler.GetMean()
	if mean == nil {
		t.Fatal("Mean should not be nil after fitting")
	}

	expectedMean := []float64{5.5, 6.5, 7.5} // Mean of each column
	for j := 0; j < 3; j++ {
		if math.Abs(mean.At(0, j)-expectedMean[j]) > 1e-10 {
			t.Errorf("Mean[%d]: expected=%f, got=%f", j, expectedMean[j], mean.At(0, j))
		}
	}

	std := scaler.GetStd()
	if std == nil {
		t.Fatal("Std should not be nil after fitting")
	}

	// Standard deviation should be approximately 3.354 for each column (population std)
	expectedStd := 3.354101966249685
	for j := 0; j < 3; j++ {
		if math.Abs(std.At(0, j)-expectedStd) > 1e-10 {
			t.Errorf("Std[%d]: expected=%f, got=%f", j, expectedStd, std.At(0, j))
		}
	}
}

func TestStandardScaler_Transform(t *testing.T) {
	scaler := NewStandardScaler()
	data := createTestTensor()

	// Fit and transform
	transformed, err := scaler.FitTransform(data)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	// Check that transformed data has zero mean and unit variance
	rows, cols := transformed.Dims()

	// Check mean is approximately zero
	for j := 0; j < cols; j++ {
		var sum float64
		for i := 0; i < rows; i++ {
			sum += transformed.At(i, j)
		}
		mean := sum / float64(rows)
		if math.Abs(mean) > 1e-10 {
			t.Errorf("Transformed mean[%d] should be ~0, got %f", j, mean)
		}
	}

	// Check standard deviation is approximately 1
	for j := 0; j < cols; j++ {
		var sumSq float64
		for i := 0; i < rows; i++ {
			val := transformed.At(i, j)
			sumSq += val * val
		}
		variance := sumSq / float64(rows)
		std := math.Sqrt(variance)
		if math.Abs(std-1.0) > 1e-10 {
			t.Errorf("Transformed std[%d] should be ~1, got %f", j, std)
		}
	}
}

func TestStandardScaler_InverseTransform(t *testing.T) {
	scaler := NewStandardScaler()
	data := createTestTensor()

	// Fit, transform, and inverse transform
	transformed, err := scaler.FitTransform(data)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	recovered, err := scaler.InverseTransform(transformed)
	if err != nil {
		t.Fatalf("InverseTransform failed: %v", err)
	}

	// Should recover original data
	assertTensorEqual(t, data, recovered, 1e-10)
}

func TestStandardScaler_Options(t *testing.T) {
	// Test with mean disabled
	scaler := NewStandardScaler(WithMean(false))
	data := createTestTensor()

	err := scaler.Fit(data)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	if scaler.GetMean() != nil {
		t.Error("Mean should be nil when WithMean(false)")
	}

	// Test with std disabled
	scaler2 := NewStandardScaler(WithStd(false))
	err = scaler2.Fit(data)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	if scaler2.GetStd() != nil {
		t.Error("Std should be nil when WithStd(false)")
	}
}

func TestMinMaxScaler_Basic(t *testing.T) {
	scaler := NewMinMaxScaler()
	data := createTestTensor()

	// Test initial state
	if scaler.IsFitted() {
		t.Error("Scaler should not be fitted initially")
	}

	if scaler.Name() != "MinMaxScaler" {
		t.Errorf("Expected name 'MinMaxScaler', got '%s'", scaler.Name())
	}

	// Test default range
	min, max := scaler.GetFeatureRange()
	if min != 0.0 || max != 1.0 {
		t.Errorf("Expected default range [0, 1], got [%f, %f]", min, max)
	}

	// Test fit
	err := scaler.Fit(data)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	if !scaler.IsFitted() {
		t.Error("Scaler should be fitted after Fit()")
	}

	// Check learned parameters
	dataMin := scaler.GetDataMin()
	dataMax := scaler.GetDataMax()

	expectedMin := []float64{1.0, 2.0, 3.0}
	expectedMax := []float64{10.0, 11.0, 12.0}

	for j := 0; j < 3; j++ {
		if math.Abs(dataMin.At(0, j)-expectedMin[j]) > 1e-10 {
			t.Errorf("DataMin[%d]: expected=%f, got=%f", j, expectedMin[j], dataMin.At(0, j))
		}
		if math.Abs(dataMax.At(0, j)-expectedMax[j]) > 1e-10 {
			t.Errorf("DataMax[%d]: expected=%f, got=%f", j, expectedMax[j], dataMax.At(0, j))
		}
	}
}

func TestMinMaxScaler_Transform(t *testing.T) {
	scaler := NewMinMaxScaler()
	data := createTestTensor()

	// Fit and transform
	transformed, err := scaler.FitTransform(data)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	// Check that all values are in [0, 1]
	rows, cols := transformed.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := transformed.At(i, j)
			if val < 0.0 || val > 1.0 {
				t.Errorf("Transformed value at (%d,%d) = %f is outside [0,1]", i, j, val)
			}
		}
	}

	// Check specific values
	// First row should be all zeros (minimum values)
	for j := 0; j < cols; j++ {
		if math.Abs(transformed.At(0, j)) > 1e-10 {
			t.Errorf("First row should be zeros, got %f at column %d", transformed.At(0, j), j)
		}
	}

	// Last row should be all ones (maximum values)
	for j := 0; j < cols; j++ {
		if math.Abs(transformed.At(rows-1, j)-1.0) > 1e-10 {
			t.Errorf("Last row should be ones, got %f at column %d", transformed.At(rows-1, j), j)
		}
	}
}

func TestMinMaxScaler_CustomRange(t *testing.T) {
	scaler := NewMinMaxScaler(WithFeatureRange(-1.0, 1.0))
	data := createTestTensor()

	min, max := scaler.GetFeatureRange()
	if min != -1.0 || max != 1.0 {
		t.Errorf("Expected range [-1, 1], got [%f, %f]", min, max)
	}

	transformed, err := scaler.FitTransform(data)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	// Check that all values are in [-1, 1]
	rows, cols := transformed.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := transformed.At(i, j)
			if val < -1.0 || val > 1.0 {
				t.Errorf("Transformed value at (%d,%d) = %f is outside [-1,1]", i, j, val)
			}
		}
	}
}

func TestMinMaxScaler_InverseTransform(t *testing.T) {
	scaler := NewMinMaxScaler()
	data := createTestTensor()

	// Fit, transform, and inverse transform
	transformed, err := scaler.FitTransform(data)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	recovered, err := scaler.InverseTransform(transformed)
	if err != nil {
		t.Fatalf("InverseTransform failed: %v", err)
	}

	// Should recover original data
	assertTensorEqual(t, data, recovered, 1e-10)
}

func TestOneHotEncoder_Basic(t *testing.T) {
	encoder := NewOneHotEncoder()
	data := []string{"cat", "dog", "bird", "cat", "dog"}

	// Test initial state
	if encoder.IsFitted() {
		t.Error("Encoder should not be fitted initially")
	}

	if encoder.Name() != "OneHotEncoder" {
		t.Errorf("Expected name 'OneHotEncoder', got '%s'", encoder.Name())
	}

	// Test fit
	err := encoder.Fit(data)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	if !encoder.IsFitted() {
		t.Error("Encoder should be fitted after Fit()")
	}

	// Check learned classes (should be sorted)
	classes := encoder.Classes()
	expectedClasses := []string{"bird", "cat", "dog"}

	if len(classes) != len(expectedClasses) {
		t.Fatalf("Expected %d classes, got %d", len(expectedClasses), len(classes))
	}

	for i, expected := range expectedClasses {
		if classes[i] != expected {
			t.Errorf("Classes[%d]: expected='%s', got='%s'", i, expected, classes[i])
		}
	}

	if encoder.GetNumClasses() != 3 {
		t.Errorf("Expected 3 classes, got %d", encoder.GetNumClasses())
	}
}

func TestOneHotEncoder_Transform(t *testing.T) {
	encoder := NewOneHotEncoder()
	data := []string{"cat", "dog", "bird"}

	// Fit and transform
	encoded, err := encoder.FitTransform(data)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	rows, cols := encoded.Dims()
	if rows != 3 || cols != 3 {
		t.Fatalf("Expected shape (3,3), got (%d,%d)", rows, cols)
	}

	// Check encoding: bird=0, cat=1, dog=2
	expected := [][]float64{
		{0, 1, 0}, // cat
		{0, 0, 1}, // dog
		{1, 0, 0}, // bird
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if math.Abs(encoded.At(i, j)-expected[i][j]) > 1e-10 {
				t.Errorf("Encoded[%d,%d]: expected=%f, got=%f", i, j, expected[i][j], encoded.At(i, j))
			}
		}
	}
}

func TestOneHotEncoder_UnknownCategory(t *testing.T) {
	encoder := NewOneHotEncoder()
	trainData := []string{"cat", "dog"}
	testData := []string{"cat", "bird"} // "bird" is unknown

	err := encoder.Fit(trainData)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	_, err = encoder.Transform(testData)
	if err == nil {
		t.Error("Transform should fail with unknown category")
	}
}

func TestLabelEncoder_Basic(t *testing.T) {
	encoder := NewLabelEncoder()
	data := []string{"cat", "dog", "bird", "cat", "dog"}

	// Test initial state
	if encoder.IsFitted() {
		t.Error("Encoder should not be fitted initially")
	}

	if encoder.Name() != "LabelEncoder" {
		t.Errorf("Expected name 'LabelEncoder', got '%s'", encoder.Name())
	}

	// Test fit
	err := encoder.Fit(data)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	if !encoder.IsFitted() {
		t.Error("Encoder should be fitted after Fit()")
	}

	// Check learned classes (should be sorted)
	classes := encoder.Classes()
	expectedClasses := []string{"bird", "cat", "dog"}

	if len(classes) != len(expectedClasses) {
		t.Fatalf("Expected %d classes, got %d", len(expectedClasses), len(classes))
	}

	for i, expected := range expectedClasses {
		if classes[i] != expected {
			t.Errorf("Classes[%d]: expected='%s', got='%s'", i, expected, classes[i])
		}
	}

	if encoder.GetNumClasses() != 3 {
		t.Errorf("Expected 3 classes, got %d", encoder.GetNumClasses())
	}
}

func TestLabelEncoder_Transform(t *testing.T) {
	encoder := NewLabelEncoder()
	data := []string{"cat", "dog", "bird", "cat"}

	// Fit and transform
	encoded, err := encoder.FitTransform(data)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	rows, cols := encoded.Dims()
	if rows != 4 || cols != 1 {
		t.Fatalf("Expected shape (4,1), got (%d,%d)", rows, cols)
	}

	// Check encoding: bird=0, cat=1, dog=2
	expected := []float64{1, 2, 0, 1} // cat, dog, bird, cat

	for i := 0; i < rows; i++ {
		if math.Abs(encoded.At(i, 0)-expected[i]) > 1e-10 {
			t.Errorf("Encoded[%d]: expected=%f, got=%f", i, expected[i], encoded.At(i, 0))
		}
	}
}

func TestLabelEncoder_InverseTransform(t *testing.T) {
	encoder := NewLabelEncoder()
	data := []string{"cat", "dog", "bird"}

	// Fit and transform
	encoded, err := encoder.FitTransform(data)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	// Inverse transform
	decoded, err := encoder.InverseTransform(encoded)
	if err != nil {
		t.Fatalf("InverseTransform failed: %v", err)
	}

	// Should recover original data
	if len(decoded) != len(data) {
		t.Fatalf("Expected %d items, got %d", len(data), len(decoded))
	}

	for i, expected := range data {
		if decoded[i] != expected {
			t.Errorf("Decoded[%d]: expected='%s', got='%s'", i, expected, decoded[i])
		}
	}
}

func TestLabelEncoder_UnknownCategory(t *testing.T) {
	encoder := NewLabelEncoder()
	trainData := []string{"cat", "dog"}
	testData := []string{"cat", "bird"} // "bird" is unknown

	err := encoder.Fit(trainData)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	_, err = encoder.Transform(testData)
	if err == nil {
		t.Error("Transform should fail with unknown category")
	}
}

// Test error conditions
func TestScalers_ErrorConditions(t *testing.T) {
	// Test empty data - create a nil tensor instead of zero-sized tensor
	scaler := NewStandardScaler()

	err := scaler.Fit(nil)
	if err == nil {
		t.Error("Fit should fail with nil tensor")
	}

	// Test transform before fit
	data := createTestTensor()
	_, err = scaler.Transform(data)
	if err == nil {
		t.Error("Transform should fail when not fitted")
	}

	// Test dimension mismatch
	scaler.Fit(data)
	wrongData := core.NewZerosTensor(2, 5) // Different number of features
	_, err = scaler.Transform(wrongData)
	if err == nil {
		t.Error("Transform should fail with dimension mismatch")
	}
}

func TestEncoders_ErrorConditions(t *testing.T) {
	// Test empty data
	encoder := NewOneHotEncoder()

	err := encoder.Fit([]string{})
	if err == nil {
		t.Error("Fit should fail with empty data")
	}

	// Test transform before fit
	_, err = encoder.Transform([]string{"cat"})
	if err == nil {
		t.Error("Transform should fail when not fitted")
	}

	// Test invalid feature range for MinMaxScaler
	scaler := NewMinMaxScaler(WithFeatureRange(1.0, 0.0)) // max < min
	data := createTestTensor()

	err = scaler.Fit(data)
	if err == nil {
		t.Error("Fit should fail with invalid feature range")
	}
}
