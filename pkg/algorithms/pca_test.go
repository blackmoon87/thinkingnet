package algorithms

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestNewPCA(t *testing.T) {
	pca := NewPCA(2)

	if pca.nComponents != 2 {
		t.Errorf("Expected nComponents=2, got %d", pca.nComponents)
	}

	if pca.whiten {
		t.Error("Expected whiten=false by default")
	}

	if pca.fitted {
		t.Error("Expected fitted=false initially")
	}
}

func TestPCAOptions(t *testing.T) {
	pca := NewPCA(3, WithWhiten(true), WithPCARandomSeed(123))

	if !pca.whiten {
		t.Error("Expected whiten=true")
	}

	if pca.randomSeed != 123 {
		t.Errorf("Expected randomSeed=123, got %d", pca.randomSeed)
	}
}

func TestPCAFitValidation(t *testing.T) {
	pca := NewPCA(2)

	// Test nil input
	err := pca.Fit(nil)
	if err == nil {
		t.Error("Expected error for nil input")
	}

	// Test empty data - skip this test as NewZerosTensor(0,0) panics
	// This is expected behavior as empty tensors are not valid

	// Test insufficient samples
	insufficientData := core.NewZerosTensor(1, 3)
	err = pca.Fit(insufficientData)
	if err == nil {
		t.Error("Expected error for insufficient samples")
	}
}

func TestPCAFitBasic(t *testing.T) {
	// Create simple 2D data
	data := [][]float64{
		{1.0, 2.0},
		{2.0, 3.0},
		{3.0, 4.0},
		{4.0, 5.0},
		{5.0, 6.0},
	}
	X := core.NewTensorFromSlice(data)

	pca := NewPCA(2)
	err := pca.Fit(X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	if !pca.fitted {
		t.Error("Expected fitted=true after successful fit")
	}

	// Check components shape
	components := pca.Components()
	if components == nil {
		t.Fatal("Components should not be nil after fitting")
	}

	rows, cols := components.Dims()
	if rows != 2 || cols != 2 {
		t.Errorf("Expected components shape (2,2), got (%d,%d)", rows, cols)
	}

	// Check explained variance
	explainedVar := pca.ExplainedVariance()
	if len(explainedVar) != 2 {
		t.Errorf("Expected 2 explained variance values, got %d", len(explainedVar))
	}

	// Check explained variance ratio
	explainedVarRatio := pca.ExplainedVarianceRatio()
	if len(explainedVarRatio) != 2 {
		t.Errorf("Expected 2 explained variance ratio values, got %d", len(explainedVarRatio))
	}

	// Sum of explained variance ratios should be close to 1
	sum := 0.0
	for _, ratio := range explainedVarRatio {
		sum += ratio
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("Sum of explained variance ratios should be 1.0, got %f", sum)
	}
}

func TestPCATransform(t *testing.T) {
	// Create test data
	data := [][]float64{
		{1.0, 2.0, 3.0},
		{2.0, 3.0, 4.0},
		{3.0, 4.0, 5.0},
		{4.0, 5.0, 6.0},
		{5.0, 6.0, 7.0},
	}
	X := core.NewTensorFromSlice(data)

	pca := NewPCA(2)
	err := pca.Fit(X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Transform the data
	transformed, err := pca.Transform(X)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	// Check transformed shape
	rows, cols := transformed.Dims()
	expectedRows, _ := X.Dims()
	if rows != expectedRows || cols != 2 {
		t.Errorf("Expected transformed shape (%d,2), got (%d,%d)", expectedRows, rows, cols)
	}
}

func TestPCATransformValidation(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0},
		{2.0, 3.0},
		{3.0, 4.0},
	}
	X := core.NewTensorFromSlice(data)

	pca := NewPCA(1)

	// Test transform before fit
	_, err := pca.Transform(X)
	if err == nil {
		t.Error("Expected error when transforming before fitting")
	}

	// Fit the model
	err = pca.Fit(X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Test transform with wrong number of features
	wrongData := core.NewZerosTensor(3, 3) // 3 features instead of 2
	_, err = pca.Transform(wrongData)
	if err == nil {
		t.Error("Expected error for wrong number of features")
	}
}

func TestPCAFitTransform(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0},
		{2.0, 3.0},
		{3.0, 4.0},
		{4.0, 5.0},
	}
	X := core.NewTensorFromSlice(data)

	pca := NewPCA(1)
	transformed, err := pca.FitTransform(X)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	// Check that model is fitted
	if !pca.fitted {
		t.Error("Expected model to be fitted after FitTransform")
	}

	// Check transformed shape
	rows, cols := transformed.Dims()
	expectedRows, _ := X.Dims()
	if rows != expectedRows || cols != 1 {
		t.Errorf("Expected transformed shape (%d,1), got (%d,%d)", expectedRows, rows, cols)
	}
}

func TestPCAInverseTransform(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0, 3.0},
		{2.0, 3.0, 4.0},
		{3.0, 4.0, 5.0},
		{4.0, 5.0, 6.0},
	}
	X := core.NewTensorFromSlice(data)

	pca := NewPCA(2)

	// Test inverse transform before fit
	testData := core.NewZerosTensor(4, 2)
	_, err := pca.InverseTransform(testData)
	if err == nil {
		t.Error("Expected error when inverse transforming before fitting")
	}

	// Fit and transform
	transformed, err := pca.FitTransform(X)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	// Inverse transform
	reconstructed, err := pca.InverseTransform(transformed)
	if err != nil {
		t.Fatalf("InverseTransform failed: %v", err)
	}

	// Check reconstructed shape
	rows, cols := reconstructed.Dims()
	expectedRows, expectedCols := X.Dims()
	if rows != expectedRows || cols != expectedCols {
		t.Errorf("Expected reconstructed shape (%d,%d), got (%d,%d)",
			expectedRows, expectedCols, rows, cols)
	}

	// Test with wrong number of components
	wrongData := core.NewZerosTensor(4, 3) // 3 components instead of 2
	_, err = pca.InverseTransform(wrongData)
	if err == nil {
		t.Error("Expected error for wrong number of components")
	}
}

func TestPCAWhitening(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0},
		{2.0, 4.0},
		{3.0, 6.0},
		{4.0, 8.0},
		{5.0, 10.0},
	}
	X := core.NewTensorFromSlice(data)

	// Test with whitening
	pcaWhiten := NewPCA(2, WithWhiten(true))
	transformedWhiten, err := pcaWhiten.FitTransform(X)
	if err != nil {
		t.Fatalf("FitTransform with whitening failed: %v", err)
	}

	// Test without whitening
	pcaNoWhiten := NewPCA(2, WithWhiten(false))
	transformedNoWhiten, err := pcaNoWhiten.FitTransform(X)
	if err != nil {
		t.Fatalf("FitTransform without whitening failed: %v", err)
	}

	// Results should be different
	if transformedWhiten.Equal(transformedNoWhiten) {
		t.Error("Whitened and non-whitened results should be different")
	}

	// Test inverse transform with whitening
	reconstructed, err := pcaWhiten.InverseTransform(transformedWhiten)
	if err != nil {
		t.Fatalf("InverseTransform with whitening failed: %v", err)
	}

	// Check shape
	rows, cols := reconstructed.Dims()
	expectedRows, expectedCols := X.Dims()
	if rows != expectedRows || cols != expectedCols {
		t.Errorf("Expected reconstructed shape (%d,%d), got (%d,%d)",
			expectedRows, expectedCols, rows, cols)
	}
}

func TestPCAComponentsAutoSelection(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0, 3.0},
		{2.0, 3.0, 4.0},
		{3.0, 4.0, 5.0},
	}
	X := core.NewTensorFromSlice(data)

	// Test with nComponents = 0 (should auto-select)
	pca := NewPCA(0)
	err := pca.Fit(X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Should select min(n_samples, n_features) = min(3, 3) = 3
	if pca.nComponents != 3 {
		t.Errorf("Expected auto-selected nComponents=3, got %d", pca.nComponents)
	}

	// Test with nComponents > min(n_samples, n_features)
	pcaLarge := NewPCA(5)
	err = pcaLarge.Fit(X)
	if err == nil {
		t.Error("Expected error for nComponents > min(n_samples, n_features)")
	}
}

func TestPCAGetters(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0},
		{2.0, 3.0},
		{3.0, 4.0},
		{4.0, 5.0},
	}
	X := core.NewTensorFromSlice(data)

	pca := NewPCA(2)

	// Test getters before fitting
	if pca.Components() != nil {
		t.Error("Components should be nil before fitting")
	}
	if pca.ExplainedVariance() != nil {
		t.Error("ExplainedVariance should be nil before fitting")
	}
	if pca.ExplainedVarianceRatio() != nil {
		t.Error("ExplainedVarianceRatio should be nil before fitting")
	}
	if pca.SingularValues() != nil {
		t.Error("SingularValues should be nil before fitting")
	}
	if pca.Mean() != nil {
		t.Error("Mean should be nil before fitting")
	}

	// Fit the model
	err := pca.Fit(X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Test getters after fitting
	if pca.Components() == nil {
		t.Error("Components should not be nil after fitting")
	}
	if pca.ExplainedVariance() == nil {
		t.Error("ExplainedVariance should not be nil after fitting")
	}
	if pca.ExplainedVarianceRatio() == nil {
		t.Error("ExplainedVarianceRatio should not be nil after fitting")
	}
	if pca.SingularValues() == nil {
		t.Error("SingularValues should not be nil after fitting")
	}
	if pca.Mean() == nil {
		t.Error("Mean should not be nil after fitting")
	}

	// Test Name and NComponents
	if pca.Name() != "PCA" {
		t.Errorf("Expected name 'PCA', got '%s'", pca.Name())
	}
	if pca.NComponents() != 2 {
		t.Errorf("Expected NComponents=2, got %d", pca.NComponents())
	}
}

func TestPCANumericalStability(t *testing.T) {
	// Create data with very small values
	data := [][]float64{
		{1e-10, 2e-10},
		{2e-10, 3e-10},
		{3e-10, 4e-10},
		{4e-10, 5e-10},
	}
	X := core.NewTensorFromSlice(data)

	pca := NewPCA(2)
	err := pca.Fit(X)
	if err != nil {
		t.Fatalf("Fit failed with small values: %v", err)
	}

	// Transform should work
	_, err = pca.Transform(X)
	if err != nil {
		t.Fatalf("Transform failed with small values: %v", err)
	}
}

func TestPCAReproducibility(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0, 3.0},
		{2.0, 3.0, 4.0},
		{3.0, 4.0, 5.0},
		{4.0, 5.0, 6.0},
		{5.0, 6.0, 7.0},
	}
	X := core.NewTensorFromSlice(data)

	// Fit two PCA models with same seed
	pca1 := NewPCA(2, WithPCARandomSeed(42))
	pca2 := NewPCA(2, WithPCARandomSeed(42))

	err1 := pca1.Fit(X)
	err2 := pca2.Fit(X)

	if err1 != nil || err2 != nil {
		t.Fatalf("Fit failed: %v, %v", err1, err2)
	}

	// Results should be identical (within numerical precision)
	components1 := pca1.Components()
	components2 := pca2.Components()

	if !components1.Equal(components2) {
		t.Error("PCA results should be reproducible with same seed")
	}
}

// Benchmark tests
func BenchmarkPCAFit(b *testing.B) {
	// Create larger dataset for benchmarking
	nSamples, nFeatures := 1000, 50
	data := make([][]float64, nSamples)
	for i := range nSamples {
		data[i] = make([]float64, nFeatures)
		for j := range nFeatures {
			data[i][j] = float64(i*nFeatures + j)
		}
	}
	X := core.NewTensorFromSlice(data)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pca := NewPCA(10)
		pca.Fit(X)
	}
}

func BenchmarkPCATransform(b *testing.B) {
	// Setup
	nSamples, nFeatures := 1000, 50
	data := make([][]float64, nSamples)
	for i := range nSamples {
		data[i] = make([]float64, nFeatures)
		for j := range nFeatures {
			data[i][j] = float64(i*nFeatures + j)
		}
	}
	X := core.NewTensorFromSlice(data)

	pca := NewPCA(10)
	pca.Fit(X)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pca.Transform(X)
	}
}
