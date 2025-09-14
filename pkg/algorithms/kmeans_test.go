package algorithms

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestKMeans_Basic(t *testing.T) {
	// Create simple 2D test data with clear clusters
	data := [][]float64{
		{1.0, 1.0}, {1.5, 2.0}, {3.0, 4.0},
		{5.0, 7.0}, {3.5, 5.0}, {4.5, 5.0},
		{3.5, 4.5}, {2.0, 2.0}, {1.0, 2.0},
	}
	X := core.NewTensorFromSlice(data)

	// Test K-means with k=2
	kmeans := NewKMeans(2, WithRandomSeed(42), WithMaxIters(100))

	err := kmeans.Fit(X)
	if err != nil {
		t.Fatalf("Failed to fit K-means: %v", err)
	}

	if !kmeans.fitted {
		t.Error("K-means should be marked as fitted after Fit()")
	}

	if kmeans.k != 2 {
		t.Errorf("Expected 2 clusters, got %d", kmeans.k)
	}

	// Test prediction
	labels, err := kmeans.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	if len(labels) != 9 {
		t.Errorf("Expected 9 labels, got %d", len(labels))
	}

	// Check that all labels are valid (0 or 1)
	for i, label := range labels {
		if label < 0 || label >= 2 {
			t.Errorf("Invalid label %d at index %d", label, i)
		}
	}

	// Test cluster centers
	centers := kmeans.ClusterCenters()
	if centers == nil {
		t.Error("Cluster centers should not be nil")
	}

	rows, cols := centers.Dims()
	if rows != 2 || cols != 2 {
		t.Errorf("Expected cluster centers shape (2,2), got (%d,%d)", rows, cols)
	}

	// Test inertia (should be positive)
	if kmeans.Inertia() <= 0 {
		t.Error("Inertia should be positive")
	}
}

func TestEasyKMeans(t *testing.T) {
	// Create simple 2D test data with clear clusters
	data := [][]float64{
		{1.0, 1.0}, {1.5, 2.0}, {3.0, 4.0},
		{5.0, 7.0}, {3.5, 5.0}, {4.5, 5.0},
		{3.5, 4.5}, {2.0, 2.0}, {1.0, 2.0},
	}
	X := core.NewTensorFromSlice(data)

	// Test the easy constructor
	kmeans := EasyKMeans(2)

	// Verify default parameters are set correctly
	if kmeans.k != 2 {
		t.Errorf("Expected k=2, got %d", kmeans.k)
	}
	if kmeans.maxIters != 300 {
		t.Errorf("Expected max iterations 300, got %d", kmeans.maxIters)
	}
	if kmeans.tolerance != 1e-4 {
		t.Errorf("Expected tolerance 1e-4, got %f", kmeans.tolerance)
	}
	if kmeans.initMethod != "kmeans++" {
		t.Errorf("Expected init method 'kmeans++', got %s", kmeans.initMethod)
	}
	if kmeans.randomSeed != 42 {
		t.Errorf("Expected random seed 42, got %d", kmeans.randomSeed)
	}

	// Test that it works for training and prediction
	err := kmeans.Fit(X)
	if err != nil {
		t.Fatalf("Failed to fit K-means: %v", err)
	}

	if !kmeans.fitted {
		t.Error("K-means should be marked as fitted after Fit()")
	}

	// Test prediction
	labels, err := kmeans.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	if len(labels) != 9 {
		t.Errorf("Expected 9 labels, got %d", len(labels))
	}

	// Check that all labels are valid (0 or 1)
	for i, label := range labels {
		if label < 0 || label >= 2 {
			t.Errorf("Invalid label %d at index %d", label, i)
		}
	}

	// Test cluster centers
	centers := kmeans.ClusterCenters()
	if centers == nil {
		t.Error("Cluster centers should not be nil")
	}

	centerRows, centerCols := centers.Dims()
	if centerRows != 2 || centerCols != 2 {
		t.Errorf("Expected centers shape (2, 2), got (%d, %d)", centerRows, centerCols)
	}

	// Test inertia
	if kmeans.Inertia() <= 0 {
		t.Error("Inertia should be positive")
	}
}

func TestKMeans_FitPredict(t *testing.T) {
	data := [][]float64{
		{0.0, 0.0}, {1.0, 1.0}, {10.0, 10.0}, {11.0, 11.0},
	}
	X := core.NewTensorFromSlice(data)

	kmeans := NewKMeans(2, WithRandomSeed(42))

	labels, err := kmeans.FitPredict(X)
	if err != nil {
		t.Fatalf("Failed to fit and predict: %v", err)
	}

	if len(labels) != 4 {
		t.Errorf("Expected 4 labels, got %d", len(labels))
	}

	// Check that we have two distinct clusters
	uniqueLabels := make(map[int]bool)
	for _, label := range labels {
		uniqueLabels[label] = true
	}

	if len(uniqueLabels) != 2 {
		t.Errorf("Expected 2 unique labels, got %d", len(uniqueLabels))
	}
}

func TestKMeans_InitializationMethods(t *testing.T) {
	data := [][]float64{
		{1.0, 1.0}, {2.0, 2.0}, {10.0, 10.0}, {11.0, 11.0},
	}
	X := core.NewTensorFromSlice(data)

	// Test random initialization
	kmeans1 := NewKMeans(2, WithInitMethod("random"), WithRandomSeed(42))
	err := kmeans1.Fit(X)
	if err != nil {
		t.Fatalf("Failed to fit with random initialization: %v", err)
	}

	// Test k-means++ initialization
	kmeans2 := NewKMeans(2, WithInitMethod("kmeans++"), WithRandomSeed(42))
	err = kmeans2.Fit(X)
	if err != nil {
		t.Fatalf("Failed to fit with k-means++ initialization: %v", err)
	}

	// Both should converge successfully
	if kmeans1.NIters() == 0 || kmeans2.NIters() == 0 {
		t.Error("Both initialization methods should perform at least one iteration")
	}
}

func TestKMeans_EdgeCases(t *testing.T) {
	// Test with nil input
	kmeans := NewKMeans(2)
	err := kmeans.Fit(nil)
	if err == nil {
		t.Error("Expected error for nil input")
	}

	// Test with empty data - skip this test as NewZerosTensor(0,0) panics
	// This is expected behavior as empty tensors are not valid

	// Test with k > number of samples
	data := [][]float64{{1.0, 1.0}, {2.0, 2.0}}
	X := core.NewTensorFromSlice(data)
	kmeans = NewKMeans(5)
	err = kmeans.Fit(X)
	if err == nil {
		t.Error("Expected error when k > number of samples")
	}

	// Test prediction before fitting
	kmeans = NewKMeans(2)
	_, err = kmeans.Predict(X)
	if err == nil {
		t.Error("Expected error when predicting before fitting")
	}
}

func TestKMeans_SingleCluster(t *testing.T) {
	// Test with k=1
	data := [][]float64{
		{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0},
	}
	X := core.NewTensorFromSlice(data)

	kmeans := NewKMeans(1, WithRandomSeed(42))
	err := kmeans.Fit(X)
	if err != nil {
		t.Fatalf("Failed to fit with k=1: %v", err)
	}

	labels, err := kmeans.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// All points should be in cluster 0
	for i, label := range labels {
		if label != 0 {
			t.Errorf("Expected label 0 at index %d, got %d", i, label)
		}
	}
}

func TestKMeans_Convergence(t *testing.T) {
	// Create data that should converge quickly
	data := [][]float64{
		{0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},
		{10.0, 10.0}, {10.1, 10.1}, {10.2, 10.2},
	}
	X := core.NewTensorFromSlice(data)

	kmeans := NewKMeans(2, WithRandomSeed(42), WithTolerance(1e-6))
	err := kmeans.Fit(X)
	if err != nil {
		t.Fatalf("Failed to fit: %v", err)
	}

	// Should converge in reasonable number of iterations
	if kmeans.NIters() > 50 {
		t.Errorf("Expected convergence in <= 50 iterations, got %d", kmeans.NIters())
	}

	// Inertia should be reasonable
	if kmeans.Inertia() > 1.0 {
		t.Errorf("Expected low inertia for well-separated clusters, got %f", kmeans.Inertia())
	}
}

func TestKMeans_Options(t *testing.T) {
	data := [][]float64{
		{1.0, 1.0}, {2.0, 2.0}, {10.0, 10.0}, {11.0, 11.0},
	}
	X := core.NewTensorFromSlice(data)

	// Test with custom options
	kmeans := NewKMeans(2,
		WithMaxIters(50),
		WithTolerance(1e-3),
		WithInitMethod("random"),
		WithRandomSeed(123),
	)

	if kmeans.maxIters != 50 {
		t.Errorf("Expected maxIters=50, got %d", kmeans.maxIters)
	}

	if kmeans.tolerance != 1e-3 {
		t.Errorf("Expected tolerance=1e-3, got %f", kmeans.tolerance)
	}

	if kmeans.initMethod != "random" {
		t.Errorf("Expected initMethod='random', got %s", kmeans.initMethod)
	}

	if kmeans.randomSeed != 123 {
		t.Errorf("Expected randomSeed=123, got %d", kmeans.randomSeed)
	}

	err := kmeans.Fit(X)
	if err != nil {
		t.Fatalf("Failed to fit with custom options: %v", err)
	}
}

func TestKMeans_Name(t *testing.T) {
	kmeans := NewKMeans(2)
	if kmeans.Name() != "KMeans" {
		t.Errorf("Expected name 'KMeans', got '%s'", kmeans.Name())
	}
}

func TestKMeans_InvalidInitMethod(t *testing.T) {
	data := [][]float64{{1.0, 1.0}, {2.0, 2.0}}
	X := core.NewTensorFromSlice(data)

	kmeans := NewKMeans(2, WithInitMethod("invalid"))
	err := kmeans.Fit(X)
	if err == nil {
		t.Error("Expected error for invalid initialization method")
	}
}

func TestKMeans_NumericalStability(t *testing.T) {
	// Test with data containing very large values
	data := [][]float64{
		{1e6, 1e6}, {1e6 + 1, 1e6 + 1},
		{2e6, 2e6}, {2e6 + 1, 2e6 + 1},
	}
	X := core.NewTensorFromSlice(data)

	kmeans := NewKMeans(2, WithRandomSeed(42))
	err := kmeans.Fit(X)
	if err != nil {
		t.Fatalf("Failed to fit with large values: %v", err)
	}

	// Should still produce valid results
	labels, err := kmeans.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	if len(labels) != 4 {
		t.Errorf("Expected 4 labels, got %d", len(labels))
	}

	// Check that inertia is finite
	inertia := kmeans.Inertia()
	if math.IsNaN(inertia) || math.IsInf(inertia, 0) {
		t.Error("Inertia should be finite")
	}
}

func BenchmarkKMeans_Fit(b *testing.B) {
	// Create larger dataset for benchmarking
	data := make([][]float64, 1000)
	for i := 0; i < 1000; i++ {
		data[i] = []float64{float64(i % 10), float64((i * 2) % 10)}
	}
	X := core.NewTensorFromSlice(data)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kmeans := NewKMeans(5, WithRandomSeed(42))
		err := kmeans.Fit(X)
		if err != nil {
			b.Fatalf("Failed to fit: %v", err)
		}
	}
}

func BenchmarkKMeans_Predict(b *testing.B) {
	// Setup
	data := make([][]float64, 100)
	for i := 0; i < 100; i++ {
		data[i] = []float64{float64(i % 10), float64((i * 2) % 10)}
	}
	X := core.NewTensorFromSlice(data)

	kmeans := NewKMeans(5, WithRandomSeed(42))
	err := kmeans.Fit(X)
	if err != nil {
		b.Fatalf("Failed to fit: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := kmeans.Predict(X)
		if err != nil {
			b.Fatalf("Failed to predict: %v", err)
		}
	}
}
