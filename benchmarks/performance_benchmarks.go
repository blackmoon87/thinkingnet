package benchmarks_test

import (
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
	"gonum.org/v1/gonum/mat"
)

// BenchmarkTensorOperationsOptimized benchmarks optimized tensor operations.
func BenchmarkTensorOperationsOptimized(b *testing.B) {
	// Test different sizes
	sizes := []struct {
		name string
		rows int
		cols int
	}{
		{"Small_10x10", 10, 10},
		{"Medium_100x100", 100, 100},
		{"Large_500x500", 500, 500},
		{"XLarge_1000x1000", 1000, 1000},
	}

	for _, size := range sizes {
		// Create test tensors
		a := createRandomTensor(size.rows, size.cols)
		c := createRandomTensor(size.rows, size.cols)

		b.Run(size.name+"_Add_Standard", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = a.Add(c)
			}
		})

		b.Run(size.name+"_Add_Optimized", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = core.OptimizedTensorAdd(a, c)
			}
		})

		// Matrix multiplication benchmarks
		if size.rows <= 500 { // Limit size for matrix multiplication
			b.Run(size.name+"_MatMul_Standard", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = a.Mul(c)
				}
			})

			b.Run(size.name+"_MatMul_Optimized", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = core.OptimizedMatMul(a, c)
				}
			})
		}
	}
}

// BenchmarkParallelOperations benchmarks parallel vs sequential operations.
func BenchmarkParallelOperations(b *testing.B) {
	sizes := []int{100, 500, 1000, 2000}

	for _, size := range sizes {
		a := createRandomTensor(size, size)
		c := createRandomTensor(size, size)

		b.Run("Sequential_"+string(rune(size)), func(b *testing.B) {
			// Disable parallel processing
			originalConfig := core.GetParallelConfig()
			core.SetParallelConfig(core.ParallelConfig{Enabled: false})
			defer core.SetParallelConfig(originalConfig)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = a.Add(c)
			}
		})

		b.Run("Parallel_"+string(rune(size)), func(b *testing.B) {
			// Enable parallel processing
			config := core.DefaultParallelConfig()
			config.MinSize = 100 // Lower threshold for testing
			core.SetParallelConfig(config)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = a.Add(c)
			}
		})
	}
}

// BenchmarkMatrixPooling benchmarks matrix pooling performance.
func BenchmarkMatrixPooling(b *testing.B) {
	sizes := []struct {
		rows, cols int
	}{
		{10, 10},
		{100, 100},
		{500, 500},
	}

	for _, size := range sizes {
		b.Run("WithoutPooling", func(b *testing.B) {
			core.SetMatrixPoolEnabled(false)
			defer core.SetMatrixPoolEnabled(true)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matrix := mat.NewDense(size.rows, size.cols, nil)
				_ = matrix
			}
		})

		b.Run("WithPooling", func(b *testing.B) {
			core.SetMatrixPoolEnabled(true)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				matrix := core.GetMatrix(size.rows, size.cols)
				core.PutMatrix(matrix)
			}
		})
	}
}

// BenchmarkInPlaceOperations benchmarks in-place vs copy operations.
func BenchmarkInPlaceOperations(b *testing.B) {
	sizes := []int{100, 500, 1000}

	for _, size := range sizes {
		a := createRandomTensor(size, size)
		c := createRandomTensor(size, size)

		b.Run("Copy_Operations", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				aCopy := a.Copy()
				_ = aCopy.Add(c)
			}
		})

		b.Run("InPlace_Operations", func(b *testing.B) {
			inPlaceOps := core.GetInPlaceOperations()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				aCopy := a.Copy()
				_ = inPlaceOps.AddInPlace(aCopy, c)
			}
		})
	}
}

// BenchmarkCacheOptimization benchmarks cache-optimized operations.
func BenchmarkCacheOptimization(b *testing.B) {
	sizes := []int{64, 128, 256, 512}

	for _, size := range sizes {
		a := createRandomTensor(size, size)
		c := createRandomTensor(size, size)

		b.Run("Standard_MatMul", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = a.Mul(c)
			}
		})

		b.Run("CacheOptimized_MatMul", func(b *testing.B) {
			fastOps := core.GetFastTensorOperations()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = fastOps.FastMatMul(a, c)
			}
		})
	}
}

// BenchmarkMemoryAllocation benchmarks memory allocation patterns.
func BenchmarkMemoryAllocation(b *testing.B) {
	b.Run("Frequent_Small_Allocations", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for j := 0; j < 100; j++ {
				_ = core.NewZerosTensor(10, 10)
			}
		}
	})

	b.Run("Pooled_Allocations", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for j := 0; j < 100; j++ {
				matrix := core.GetMatrix(10, 10)
				tensor := core.NewTensor(matrix)
				tensor.Release() // Return to pool
			}
		}
	})

	b.Run("Large_Single_Allocation", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = core.NewZerosTensor(1000, 1000)
		}
	})
}

// BenchmarkBatchProcessing benchmarks batch processing performance.
func BenchmarkBatchProcessing(b *testing.B) {
	batchSizes := []int{1, 10, 50, 100}
	tensorSize := 100

	for _, batchSize := range batchSizes {
		batches := make([]core.Tensor, batchSize)
		for i := range batches {
			batches[i] = createRandomTensor(tensorSize, tensorSize)
		}

		processor := func(tensor core.Tensor) (core.Tensor, error) {
			// Simple operation: add 1 to all elements
			return tensor.AddScalar(1.0), nil
		}

		b.Run("Sequential_Batch_Processing", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				results := make([]core.Tensor, len(batches))
				for j, batch := range batches {
					result, _ := processor(batch)
					results[j] = result
				}
			}
		})

		b.Run("Parallel_Batch_Processing", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = core.ParallelBatchProcess(batches, processor)
			}
		})
	}
}

// BenchmarkPerformanceProfiler benchmarks the profiler overhead.
func BenchmarkPerformanceProfiler(b *testing.B) {
	profiler := core.GetPerformanceProfiler()
	a := createRandomTensor(100, 100)
	c := createRandomTensor(100, 100)

	b.Run("Without_Profiling", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = a.Add(c)
		}
	})

	b.Run("With_Profiling", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			session := profiler.StartProfile("tensor_add")
			_ = a.Add(c)
			session.End()
		}
	})
}

// Helper function to create random tensors for benchmarking.
func createRandomTensor(rows, cols int) core.Tensor {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = float64(i%10) + 0.5 // Simple pattern to avoid random overhead
	}
	return core.NewTensorFromData(rows, cols, data)
}

// BenchmarkOptimizationConfigs benchmarks different optimization configurations.
func BenchmarkOptimizationConfigs(b *testing.B) {
	a := createRandomTensor(500, 500)
	c := createRandomTensor(500, 500)

	configs := []struct {
		name   string
		config core.OptimizationConfig
	}{
		{
			"Default",
			core.DefaultOptimizationConfig(),
		},
		{
			"NoSIMD",
			core.OptimizationConfig{
				EnableSIMD:          false,
				EnableVectorization: true,
				EnableCaching:       true,
				EnableInPlace:       true,
				MinParallelSize:     1000,
			},
		},
		{
			"NoVectorization",
			core.OptimizationConfig{
				EnableSIMD:          true,
				EnableVectorization: false,
				EnableCaching:       true,
				EnableInPlace:       true,
				MinParallelSize:     1000,
			},
		},
		{
			"NoCaching",
			core.OptimizationConfig{
				EnableSIMD:          true,
				EnableVectorization: true,
				EnableCaching:       false,
				EnableInPlace:       true,
				MinParallelSize:     1000,
			},
		},
	}

	for _, config := range configs {
		b.Run(config.name, func(b *testing.B) {
			core.SetOptimizationConfig(config.config)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = core.OptimizedTensorAdd(a, c)
			}
		})
	}

	// Restore default config
	core.SetOptimizationConfig(core.DefaultOptimizationConfig())
}
