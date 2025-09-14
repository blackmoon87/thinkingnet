package main

import (
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
	"github.com/blackmoon87/thinkingnet/pkg/preprocessing"
)

// TestBasicIntegration tests basic library functionality
func TestBasicIntegration(t *testing.T) {
	t.Run("TensorOperations", func(t *testing.T) {
		// Test tensor creation and operations
		tensor1 := core.NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
		tensor2 := core.NewTensorFromSlice([][]float64{{5, 6}, {7, 8}})

		result := tensor1.Add(tensor2)
		if result.At(0, 0) != 6.0 {
			t.Errorf("Expected 6.0, got %f", result.At(0, 0))
		}

		// Test matrix pooling
		core.SetMatrixPoolEnabled(true)
		matrix := core.GetMatrix(10, 10)
		if matrix == nil {
			t.Error("Failed to get matrix from pool")
		}
		core.PutMatrix(matrix)
	})

	t.Run("PreprocessingBasics", func(t *testing.T) {
		// Create sample data
		data := core.NewTensorFromSlice([][]float64{
			{1, 10, 100},
			{2, 20, 200},
			{3, 30, 300},
			{4, 40, 400},
		})

		// Test standard scaling
		scaler := preprocessing.NewStandardScaler()
		err := scaler.Fit(data)
		if err != nil {
			t.Errorf("StandardScaler fit failed: %v", err)
		}

		scaledData, err := scaler.Transform(data)
		if err != nil {
			t.Errorf("StandardScaler transform failed: %v", err)
		}
		if scaledData == nil {
			t.Error("Scaled data should not be nil")
		}
	})
}

// TestPerformanceBasics runs basic performance tests
func TestPerformanceBasics(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance tests in short mode")
	}

	t.Run("LargeDatasetProcessing", func(t *testing.T) {
		// Create larger dataset
		size := 100
		data := make([][]float64, size)
		for i := 0; i < size; i++ {
			data[i] = []float64{float64(i), float64(i * 2), float64(i * 3)}
		}

		tensor := core.NewTensorFromSlice(data)
		if tensor == nil {
			t.Fatal("Failed to create large tensor")
		}

		// Test preprocessing on large data
		scaler := preprocessing.NewStandardScaler()
		err := scaler.Fit(tensor)
		if err != nil {
			t.Errorf("Failed to fit scaler on large data: %v", err)
		}

		_, err = scaler.Transform(tensor)
		if err != nil {
			t.Error("Failed to transform large data")
		}
	})
}

// BenchmarkBasicComponents benchmarks key library components
func BenchmarkBasicComponents(b *testing.B) {
	// Benchmark tensor operations
	b.Run("TensorOperations", func(b *testing.B) {
		data1 := make([]float64, 100*100)
		data2 := make([]float64, 100*100)
		for i := range data1 {
			data1[i] = float64(i % 10)
			data2[i] = float64((i + 1) % 10)
		}

		tensor1 := core.NewTensorFromData(100, 100, data1)
		tensor2 := core.NewTensorFromData(100, 100, data2)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = tensor1.Add(tensor2)
		}
	})
}
