package core

import (
	"fmt"
	"runtime"
	"strings"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

// BenchmarkSuite provides comprehensive performance benchmarking for ThinkingNet operations.
type BenchmarkSuite struct {
	results map[string]*BenchmarkResult
	mutex   sync.RWMutex
}

// BenchmarkResult holds the results of a performance benchmark.
type BenchmarkResult struct {
	Name            string
	Duration        time.Duration
	OperationsCount int64
	OpsPerSecond    float64
	MemoryUsed      int64
	Allocations     int64
	Speedup         float64 // Compared to baseline
}

// NewBenchmarkSuite creates a new benchmark suite.
func NewBenchmarkSuite() *BenchmarkSuite {
	return &BenchmarkSuite{
		results: make(map[string]*BenchmarkResult),
	}
}

// BenchmarkHighPerformanceOperations benchmarks the high-performance processor.
func (bs *BenchmarkSuite) BenchmarkHighPerformanceOperations(numOperations int64) *BenchmarkResult {
	processor := GetHighPerformanceProcessor()

	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	opsPerSecond := processor.PerformOperations(numOperations)
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	result := &BenchmarkResult{
		Name:            "HighPerformanceOperations",
		Duration:        duration,
		OperationsCount: numOperations,
		OpsPerSecond:    opsPerSecond,
		MemoryUsed:      int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:     int64(m2.Mallocs - m1.Mallocs),
	}

	bs.mutex.Lock()
	bs.results[result.Name] = result
	bs.mutex.Unlock()

	return result
}

// BenchmarkMatrixOperations benchmarks matrix multiplication performance.
func (bs *BenchmarkSuite) BenchmarkMatrixOperations(size int, iterations int) *BenchmarkResult {
	// Create test matrices
	a := NewTensor(mat.NewDense(size, size, nil))
	b := NewTensor(mat.NewDense(size, size, nil))

	// Fill with test data
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			a.Set(i, j, float64(i+j+1))
			b.Set(i, j, float64(i*j+1))
		}
	}

	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	for i := 0; i < iterations; i++ {
		_ = OptimizedMatMul(a, b)
	}
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	result := &BenchmarkResult{
		Name:            fmt.Sprintf("MatrixMul_%dx%d", size, size),
		Duration:        duration,
		OperationsCount: int64(iterations),
		OpsPerSecond:    float64(iterations) / duration.Seconds(),
		MemoryUsed:      int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:     int64(m2.Mallocs - m1.Mallocs),
	}

	bs.mutex.Lock()
	bs.results[result.Name] = result
	bs.mutex.Unlock()

	return result
}

// BenchmarkActivationFunctions benchmarks activation function performance.
func (bs *BenchmarkSuite) BenchmarkActivationFunctions(size int, iterations int) map[string]*BenchmarkResult {
	results := make(map[string]*BenchmarkResult)

	// Create test data
	data := make([]float64, size)
	output := make([]float64, size)

	for i := range data {
		data[i] = float64(i-size/2) * 0.001 // Range from -size/2 to size/2
	}

	activationProcessor := GetParallelActivationProcessor()

	// Benchmark ReLU
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	for i := 0; i < iterations; i++ {
		activationProcessor.ProcessReLU(data, output)
	}
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	reluResult := &BenchmarkResult{
		Name:            "ReLU_Parallel",
		Duration:        duration,
		OperationsCount: int64(iterations * size),
		OpsPerSecond:    float64(iterations*size) / duration.Seconds(),
		MemoryUsed:      int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:     int64(m2.Mallocs - m1.Mallocs),
	}
	results["relu"] = reluResult

	// Benchmark Sigmoid
	runtime.GC()
	runtime.ReadMemStats(&m1)

	start = time.Now()
	for i := 0; i < iterations; i++ {
		activationProcessor.ProcessSigmoid(data, output)
	}
	duration = time.Since(start)

	runtime.GC()
	runtime.ReadMemStats(&m2)

	sigmoidResult := &BenchmarkResult{
		Name:            "Sigmoid_Parallel",
		Duration:        duration,
		OperationsCount: int64(iterations * size),
		OpsPerSecond:    float64(iterations*size) / duration.Seconds(),
		MemoryUsed:      int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:     int64(m2.Mallocs - m1.Mallocs),
	}
	results["sigmoid"] = sigmoidResult

	// Benchmark Tanh
	runtime.GC()
	runtime.ReadMemStats(&m1)

	start = time.Now()
	for i := 0; i < iterations; i++ {
		activationProcessor.ProcessTanh(data, output)
	}
	duration = time.Since(start)

	runtime.GC()
	runtime.ReadMemStats(&m2)

	tanhResult := &BenchmarkResult{
		Name:            "Tanh_Parallel",
		Duration:        duration,
		OperationsCount: int64(iterations * size),
		OpsPerSecond:    float64(iterations*size) / duration.Seconds(),
		MemoryUsed:      int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:     int64(m2.Mallocs - m1.Mallocs),
	}
	results["tanh"] = tanhResult

	// Store results
	bs.mutex.Lock()
	for name, result := range results {
		bs.results[fmt.Sprintf("Activation_%s", name)] = result
	}
	bs.mutex.Unlock()

	return results
}

// BenchmarkMemoryPooling benchmarks memory pool performance.
func (bs *BenchmarkSuite) BenchmarkMemoryPooling(numMatrices int, matrixSize int) *BenchmarkResult {
	// Test with pooling
	SetMatrixPoolEnabled(true)

	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	for i := 0; i < numMatrices; i++ {
		matrix := GetMatrix(matrixSize, matrixSize)
		// Simulate some work
		for j := 0; j < matrixSize; j++ {
			for k := 0; k < matrixSize; k++ {
				matrix.Set(j, k, float64(i+j+k))
			}
		}
		PutMatrix(matrix)
	}
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	result := &BenchmarkResult{
		Name:            "MemoryPooling",
		Duration:        duration,
		OperationsCount: int64(numMatrices),
		OpsPerSecond:    float64(numMatrices) / duration.Seconds(),
		MemoryUsed:      int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:     int64(m2.Mallocs - m1.Mallocs),
	}

	bs.mutex.Lock()
	bs.results[result.Name] = result
	bs.mutex.Unlock()

	return result
}

// BenchmarkBatchProcessing benchmarks batch processing performance.
func (bs *BenchmarkSuite) BenchmarkBatchProcessing(batchSize int, tensorSize int) *BenchmarkResult {
	// Create batch of tensors
	inputs := make([]Tensor, batchSize)
	for i := range inputs {
		tensor := NewTensor(mat.NewDense(tensorSize, tensorSize, nil))
		// Fill with test data
		for j := 0; j < tensorSize; j++ {
			for k := 0; k < tensorSize; k++ {
				tensor.Set(j, k, float64(i+j+k))
			}
		}
		inputs[i] = tensor
	}

	// Simple processing function
	processor := func(input Tensor) Tensor {
		return input.Scale(2.0) // Simple scaling operation
	}

	batchProcessor := GetBatchProcessor()

	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	results := batchProcessor.ProcessBatches(inputs, processor)
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	result := &BenchmarkResult{
		Name:            fmt.Sprintf("BatchProcessing_%d", batchSize),
		Duration:        duration,
		OperationsCount: int64(batchSize),
		OpsPerSecond:    float64(batchSize) / duration.Seconds(),
		MemoryUsed:      int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:     int64(m2.Mallocs - m1.Mallocs),
	}

	// Ensure results are used
	_ = results

	bs.mutex.Lock()
	bs.results[result.Name] = result
	bs.mutex.Unlock()

	return result
}

// RunComprehensiveBenchmark runs all benchmarks and returns a summary.
func (bs *BenchmarkSuite) RunComprehensiveBenchmark() map[string]*BenchmarkResult {
	fmt.Println("🚀 Running ThinkingNet-Go Comprehensive Performance Benchmark")
	fmt.Println("=============================================================")

	// System info
	fmt.Printf("System: %d CPU cores, %s/%s\n", runtime.NumCPU(), runtime.GOOS, runtime.GOARCH)
	fmt.Printf("Go Version: %s\n\n", runtime.Version())

	// High-performance operations
	fmt.Println("1. High-Performance Operations (100M ops)...")
	bs.BenchmarkHighPerformanceOperations(100_000_000)

	// Matrix operations
	fmt.Println("2. Matrix Operations...")
	sizes := []int{128, 256, 512}
	for _, size := range sizes {
		fmt.Printf("   Testing %dx%d matrices...\n", size, size)
		bs.BenchmarkMatrixOperations(size, 10)
	}

	// Activation functions
	fmt.Println("3. Activation Functions...")
	bs.BenchmarkActivationFunctions(1_000_000, 100)

	// Memory pooling
	fmt.Println("4. Memory Pooling...")
	bs.BenchmarkMemoryPooling(1000, 100)

	// Batch processing
	fmt.Println("5. Batch Processing...")
	bs.BenchmarkBatchProcessing(100, 50)

	fmt.Println("\n✅ Benchmark completed!")

	return bs.GetResults()
}

// GetResults returns all benchmark results.
func (bs *BenchmarkSuite) GetResults() map[string]*BenchmarkResult {
	bs.mutex.RLock()
	defer bs.mutex.RUnlock()

	results := make(map[string]*BenchmarkResult)
	for name, result := range bs.results {
		results[name] = result
	}
	return results
}

// PrintResults prints benchmark results in a formatted table.
func (bs *BenchmarkSuite) PrintResults() {
	results := bs.GetResults()

	fmt.Println("\n📊 Benchmark Results Summary")
	fmt.Println("============================")
	fmt.Printf("%-30s %-12s %-15s %-12s %-10s\n", "Benchmark", "Duration", "Ops/Second", "Memory(KB)", "Allocs")
	fmt.Println(strings.Repeat("-", 85))

	for name, result := range results {
		fmt.Printf("%-30s %-12v %-15.0f %-12d %-10d\n",
			name,
			result.Duration.Truncate(time.Microsecond),
			result.OpsPerSecond,
			result.MemoryUsed/1024,
			result.Allocations)
	}
}

// CompareWithBaseline compares current results with baseline performance.
func (bs *BenchmarkSuite) CompareWithBaseline(baseline map[string]*BenchmarkResult) {
	current := bs.GetResults()

	fmt.Println("\n📈 Performance Comparison")
	fmt.Println("=========================")
	fmt.Printf("%-30s %-12s %-12s %-10s\n", "Benchmark", "Current", "Baseline", "Speedup")
	fmt.Println(strings.Repeat("-", 70))

	for name, currentResult := range current {
		if baselineResult, exists := baseline[name]; exists {
			speedup := currentResult.OpsPerSecond / baselineResult.OpsPerSecond
			fmt.Printf("%-30s %-12.0f %-12.0f %-10.2fx\n",
				name,
				currentResult.OpsPerSecond,
				baselineResult.OpsPerSecond,
				speedup)
		}
	}
}

// Global benchmark suite
var globalBenchmarkSuite = NewBenchmarkSuite()

// GetBenchmarkSuite returns the global benchmark suite.
func GetBenchmarkSuite() *BenchmarkSuite {
	return globalBenchmarkSuite
}

// RunQuickBenchmark runs a quick performance benchmark.
func RunQuickBenchmark() {
	suite := GetBenchmarkSuite()

	fmt.Println("🏃 Quick Performance Benchmark")
	fmt.Println("==============================")

	// Quick tests with smaller parameters
	suite.BenchmarkHighPerformanceOperations(10_000_000) // 10M ops
	suite.BenchmarkMatrixOperations(256, 5)              // 256x256, 5 iterations
	suite.BenchmarkActivationFunctions(100_000, 50)      // 100K elements, 50 iterations
	suite.BenchmarkMemoryPooling(100, 50)                // 100 matrices, 50x50

	suite.PrintResults()
}

// BenchmarkUltraFastOperations benchmarks the ultra-fast processor (maximum speed).
func (bs *BenchmarkSuite) BenchmarkUltraFastOperations(numOperations int64) *BenchmarkResult {
	processor := GetUltraFastProcessor()

	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	opsPerSecond := processor.PerformUltraFastOperations(numOperations)
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	result := &BenchmarkResult{
		Name:            "UltraFastOperations",
		Duration:        duration,
		OperationsCount: numOperations,
		OpsPerSecond:    opsPerSecond,
		MemoryUsed:      int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:     int64(m2.Mallocs - m1.Mallocs),
	}

	bs.mutex.Lock()
	bs.results[result.Name] = result
	bs.mutex.Unlock()

	return result
}

// BenchmarkUltraFastActivations benchmarks ultra-fast activation functions.
func (bs *BenchmarkSuite) BenchmarkUltraFastActivations(size int, iterations int) map[string]*BenchmarkResult {
	results := make(map[string]*BenchmarkResult)

	// Create test data
	data := make([]float64, size)
	output := make([]float64, size)

	for i := range data {
		data[i] = float64(i-size/2) * 0.001
	}

	processor := GetUltraFastActivationProcessor()

	// Benchmark Ultra-Fast ReLU
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	for i := 0; i < iterations; i++ {
		processor.UltraFastReLU(data, output)
	}
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	reluResult := &BenchmarkResult{
		Name:            "UltraFast_ReLU",
		Duration:        duration,
		OperationsCount: int64(iterations * size),
		OpsPerSecond:    float64(iterations*size) / duration.Seconds(),
		MemoryUsed:      int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:     int64(m2.Mallocs - m1.Mallocs),
	}
	results["ultra_relu"] = reluResult

	// Benchmark Ultra-Fast Sigmoid
	runtime.GC()
	runtime.ReadMemStats(&m1)

	start = time.Now()
	for i := 0; i < iterations; i++ {
		processor.UltraFastSigmoid(data, output)
	}
	duration = time.Since(start)

	runtime.GC()
	runtime.ReadMemStats(&m2)

	sigmoidResult := &BenchmarkResult{
		Name:            "UltraFast_Sigmoid",
		Duration:        duration,
		OperationsCount: int64(iterations * size),
		OpsPerSecond:    float64(iterations*size) / duration.Seconds(),
		MemoryUsed:      int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:     int64(m2.Mallocs - m1.Mallocs),
	}
	results["ultra_sigmoid"] = sigmoidResult

	// Store results
	bs.mutex.Lock()
	for name, result := range results {
		bs.results[fmt.Sprintf("UltraFast_%s", name)] = result
	}
	bs.mutex.Unlock()

	return results
}

// RunUltraFastBenchmark runs maximum speed benchmarks only.
func RunUltraFastBenchmark() {
	suite := GetBenchmarkSuite()

	fmt.Println("⚡ Ultra-Fast Maximum Speed Benchmark")
	fmt.Println("====================================")

	// Ultra-fast tests for maximum performance
	suite.BenchmarkUltraFastOperations(100_000_000)     // 100M ultra-fast ops
	suite.BenchmarkUltraFastActivations(1_000_000, 100) // 1M elements, 100 iterations

	suite.PrintResults()
}
