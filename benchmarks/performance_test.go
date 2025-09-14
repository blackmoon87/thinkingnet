package benchmarks_test

import (
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// Run all performance benchmarks - removed since functions are in different file

// BenchmarkBaseline provides baseline performance measurements
func BenchmarkBaseline(b *testing.B) {
	// Disable all optimizations for baseline
	originalParallelConfig := core.GetParallelConfig()
	originalOptConfig := core.GetOptimizationConfig()

	defer func() {
		core.SetParallelConfig(originalParallelConfig)
		core.SetOptimizationConfig(originalOptConfig)
	}()

	// Disable optimizations
	core.SetParallelConfig(core.ParallelConfig{Enabled: false})
	core.SetOptimizationConfig(core.OptimizationConfig{
		EnableSIMD:          false,
		EnableVectorization: false,
		EnableCaching:       false,
		EnableInPlace:       false,
		MinParallelSize:     999999,
	})
	core.SetMatrixPoolEnabled(false)

	a := createRandomTensor(100, 100)
	c := createRandomTensor(100, 100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Add(c)
	}
}

// BenchmarkOptimized provides optimized performance measurements
func BenchmarkOptimized(b *testing.B) {
	// Enable all optimizations
	core.SetParallelConfig(core.DefaultParallelConfig())
	core.SetOptimizationConfig(core.DefaultOptimizationConfig())
	core.SetMatrixPoolEnabled(true)

	a := createRandomTensor(100, 100)
	c := createRandomTensor(100, 100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = core.OptimizedTensorAdd(a, c)
	}
}
