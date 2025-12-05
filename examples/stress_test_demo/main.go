// Package main provides a comprehensive heavy stress test for ThinkingNet-Go library.
// This example runs extensive benchmarks and generates a full detailed report.
package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/activations"
	"github.com/blackmoon87/thinkingnet/pkg/algorithms"
	"github.com/blackmoon87/thinkingnet/pkg/core"
	"github.com/blackmoon87/thinkingnet/pkg/layers"
	"github.com/blackmoon87/thinkingnet/pkg/losses"
	"github.com/blackmoon87/thinkingnet/pkg/models"
	"github.com/blackmoon87/thinkingnet/pkg/optimizers"
	"github.com/blackmoon87/thinkingnet/pkg/preprocessing"
	"gonum.org/v1/gonum/mat"
)

// ═══════════════════════════════════════════════════════════════════════════════
// STRESS TEST CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

type StressTestConfig struct {
	// Tensor Operations
	TensorSizes      []int // Matrix sizes to test
	TensorIterations int   // Iterations per size

	// Neural Network
	NNLayerSizes   []int // Network architecture
	NNTrainingSamples int
	NNEpochs       int
	NNBatchSize    int

	// Algorithms
	ClusteringSamples  int
	ClusteringFeatures int
	ClusteringClusters int

	// Memory Stress
	MemoryPoolMatrices int
	MemoryPoolSize     int

	// Activation Functions
	ActivationSize       int
	ActivationIterations int

	// Parallel Processing
	ParallelWorkers int

	// High Performance
	HighPerfOperations int64
}

// DefaultHeavyConfig returns configuration for heavy stress testing
func DefaultHeavyConfig() StressTestConfig {
	return StressTestConfig{
		TensorSizes:        []int{64, 128, 256, 512, 1024},
		TensorIterations:   100,
		NNLayerSizes:       []int{128, 256, 128, 64, 10},
		NNTrainingSamples:  10000,
		NNEpochs:           50,
		NNBatchSize:        64,
		ClusteringSamples:  5000,
		ClusteringFeatures: 50,
		ClusteringClusters: 10,
		MemoryPoolMatrices: 1000,
		MemoryPoolSize:     256,
		ActivationSize:     1000000,
		ActivationIterations: 100,
		ParallelWorkers:    runtime.NumCPU(),
		HighPerfOperations: 100_000_000,
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// STRESS TEST RESULT STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

type TestResult struct {
	Name         string
	Category     string
	Duration     time.Duration
	Operations   int64
	OpsPerSecond float64
	MemoryUsed   int64
	MemoryPeak   int64
	Allocations  int64
	Success      bool
	ErrorMsg     string
	Details      map[string]interface{}
}

type StressTestReport struct {
	StartTime      time.Time
	EndTime        time.Time
	TotalDuration  time.Duration
	SystemInfo     SystemInfo
	Results        []TestResult
	Summary        TestSummary
	Config         StressTestConfig
}

type SystemInfo struct {
	GoVersion    string
	NumCPU       int
	GOMAXPROCS   int
	OS           string
	Arch         string
	TotalMemory  uint64
	InitialAlloc uint64
	FinalAlloc   uint64
}

type TestSummary struct {
	TotalTests      int
	PassedTests     int
	FailedTests     int
	TotalOperations int64
	TotalDuration   time.Duration
	AvgOpsPerSecond float64
	PeakMemoryUsed  int64
	TotalAllocations int64
	Categories      map[string]CategorySummary
}

type CategorySummary struct {
	TestCount    int
	TotalOps     int64
	AvgOpsPerSec float64
	TotalTime    time.Duration
	MemoryUsed   int64
}

// ═══════════════════════════════════════════════════════════════════════════════
// STRESS TEST RUNNER
// ═══════════════════════════════════════════════════════════════════════════════

type StressTestRunner struct {
	config  StressTestConfig
	results []TestResult
	mutex   sync.Mutex
}

func NewStressTestRunner(config StressTestConfig) *StressTestRunner {
	return &StressTestRunner{
		config:  config,
		results: make([]TestResult, 0),
	}
}

func (r *StressTestRunner) addResult(result TestResult) {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	r.results = append(r.results, result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TENSOR OPERATIONS STRESS TEST
// ═══════════════════════════════════════════════════════════════════════════════

func (r *StressTestRunner) RunTensorStressTest() {
	fmt.Println("\n📐 TENSOR OPERATIONS STRESS TEST")
	fmt.Println(strings.Repeat("─", 60))

	for _, size := range r.config.TensorSizes {
		// Create test tensors
		a := createRandomTensor(size, size)
		b := createRandomTensor(size, size)

		// Test Addition
		r.benchmarkTensorOp("Addition", "Tensor", size, func() {
			_ = a.Add(b)
		})

		// Test Subtraction
		r.benchmarkTensorOp("Subtraction", "Tensor", size, func() {
			_ = a.Sub(b)
		})

		// Test Element-wise Multiplication
		r.benchmarkTensorOp("ElementMul", "Tensor", size, func() {
			_ = a.MulElem(b)
		})

		// Test Matrix Multiplication
		if size <= 512 { // Limit for heavy matmul
			r.benchmarkTensorOp("MatrixMul", "Tensor", size, func() {
				_ = a.Mul(b)
			})
		}

		// Test Transpose
		r.benchmarkTensorOp("Transpose", "Tensor", size, func() {
			_ = a.T()
		})

		// Test Scale
		r.benchmarkTensorOp("Scale", "Tensor", size, func() {
			_ = a.Scale(2.5)
		})

		// Test Optimized Operations
		r.benchmarkTensorOp("OptimizedAdd", "Tensor", size, func() {
			_ = core.OptimizedTensorAdd(a, b)
		})

		if size <= 512 {
			r.benchmarkTensorOp("OptimizedMatMul", "Tensor", size, func() {
				_ = core.OptimizedMatMul(a, b)
			})
		}
	}
}

func (r *StressTestRunner) benchmarkTensorOp(name, category string, size int, op func()) {
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	for i := 0; i < r.config.TensorIterations; i++ {
		op()
	}
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	ops := int64(r.config.TensorIterations)
	opsPerSec := safeOpsPerSecond(ops, duration)

	result := TestResult{
		Name:         fmt.Sprintf("%s_%dx%d", name, size, size),
		Category:     category,
		Duration:     duration,
		Operations:   ops,
		OpsPerSecond: opsPerSec,
		MemoryUsed:   int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:  int64(m2.Mallocs - m1.Mallocs),
		Success:      true,
		Details: map[string]interface{}{
			"size":       size,
			"iterations": r.config.TensorIterations,
		},
	}

	r.addResult(result)
	fmt.Printf("  %-30s %10.2f ops/sec  %10s  %8s\n",
		result.Name, opsPerSec, formatDuration(duration), formatBytes(result.MemoryUsed))
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACTIVATION FUNCTIONS STRESS TEST
// ═══════════════════════════════════════════════════════════════════════════════

func (r *StressTestRunner) RunActivationStressTest() {
	fmt.Println("\n⚡ ACTIVATION FUNCTIONS STRESS TEST")
	fmt.Println(strings.Repeat("─", 60))

	size := r.config.ActivationSize
	iterations := r.config.ActivationIterations

	// Create test data
	data := make([]float64, size)
	output := make([]float64, size)
	for i := range data {
		data[i] = (rand.Float64() - 0.5) * 10 // Range -5 to 5
	}

	activationProcessor := core.GetParallelActivationProcessor()
	ultraFastProcessor := core.GetUltraFastActivationProcessor()

	// Test ReLU variants
	r.benchmarkActivation("ReLU_Parallel", "Activation", size, iterations, func() {
		activationProcessor.ProcessReLU(data, output)
	})

	r.benchmarkActivation("ReLU_UltraFast", "Activation", size, iterations, func() {
		ultraFastProcessor.UltraFastReLU(data, output)
	})

	// Test Sigmoid variants
	r.benchmarkActivation("Sigmoid_Parallel", "Activation", size, iterations, func() {
		activationProcessor.ProcessSigmoid(data, output)
	})

	r.benchmarkActivation("Sigmoid_UltraFast", "Activation", size, iterations, func() {
		ultraFastProcessor.UltraFastSigmoid(data, output)
	})

	// Test Tanh
	r.benchmarkActivation("Tanh_Parallel", "Activation", size, iterations, func() {
		activationProcessor.ProcessTanh(data, output)
	})

	// Test activation objects
	activationFuncs := []struct {
		name string
		fn   func(float64) float64
	}{
		{"ReLU_Direct", activations.NewReLU().Forward},
		{"Sigmoid_Direct", activations.NewSigmoid().Forward},
		{"Tanh_Direct", activations.NewTanh().Forward},
		{"LeakyReLU_Direct", activations.NewLeakyReLU(0.01).Forward},
		{"ELU_Direct", activations.NewELU(1.0).Forward},
		{"Swish_Direct", activations.NewSwish().Forward},
		{"GELU_Direct", activations.NewGELU().Forward},
	}

	for _, af := range activationFuncs {
		r.benchmarkActivation(af.name, "Activation", size, iterations/10, func() {
			for i := range data {
				output[i] = af.fn(data[i])
			}
		})
	}
}

func (r *StressTestRunner) benchmarkActivation(name, category string, size, iterations int, op func()) {
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	for i := 0; i < iterations; i++ {
		op()
	}
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	ops := int64(iterations) * int64(size)
	opsPerSec := safeOpsPerSecond(ops, duration)

	result := TestResult{
		Name:         name,
		Category:     category,
		Duration:     duration,
		Operations:   ops,
		OpsPerSecond: opsPerSec,
		MemoryUsed:   int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:  int64(m2.Mallocs - m1.Mallocs),
		Success:      true,
		Details: map[string]interface{}{
			"size":       size,
			"iterations": iterations,
		},
	}

	r.addResult(result)
	fmt.Printf("  %-30s %12.2f M ops/sec  %10s  %8s\n",
		name, opsPerSec/1e6, formatDuration(duration), formatBytes(result.MemoryUsed))
}

// ═══════════════════════════════════════════════════════════════════════════════
// NEURAL NETWORK STRESS TEST
// ═══════════════════════════════════════════════════════════════════════════════

func (r *StressTestRunner) RunNeuralNetworkStressTest() {
	fmt.Println("\n🧠 NEURAL NETWORK STRESS TEST")
	fmt.Println(strings.Repeat("─", 60))

	// Create synthetic training data
	numSamples := r.config.NNTrainingSamples
	inputSize := r.config.NNLayerSizes[0]
	outputSize := r.config.NNLayerSizes[len(r.config.NNLayerSizes)-1]

	fmt.Printf("  Creating %d training samples...\n", numSamples)
	X, y := createSyntheticData(numSamples, inputSize, outputSize)

	// Build model
	fmt.Printf("  Building model with architecture: %v\n", r.config.NNLayerSizes)
	model := models.NewSequential()

	for i := 1; i < len(r.config.NNLayerSizes); i++ {
		var activation core.Activation
		if i < len(r.config.NNLayerSizes)-1 {
			activation = activations.NewReLU()
		} else {
			activation = activations.NewSoftmax()
		}
		model.AddLayer(layers.NewDense(r.config.NNLayerSizes[i], &layers.DenseConfig{
			Activation: activation,
		}))
	}

	// Compile model
	optimizer, _ := optimizers.NewAdamWithDefaults(0.001)
	loss := losses.NewCategoricalCrossEntropy()
	model.Compile(optimizer, loss)

	// Benchmark forward pass
	r.benchmarkNN("ForwardPass", X, func() error {
		_, err := model.Forward(X)
		return err
	})

	// Benchmark training
	fmt.Printf("  Training for %d epochs with batch size %d...\n", r.config.NNEpochs, r.config.NNBatchSize)

	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	config := core.TrainingConfig{
		Epochs:    r.config.NNEpochs,
		BatchSize: r.config.NNBatchSize,
		Verbose:   0,
	}
	history, err := model.Fit(X, y, config)
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	success := err == nil
	errorMsg := ""
	if err != nil {
		errorMsg = err.Error()
	}

	ops := int64(r.config.NNEpochs * numSamples)
	result := TestResult{
		Name:         "FullTraining",
		Category:     "NeuralNetwork",
		Duration:     duration,
		Operations:   ops,
		OpsPerSecond: safeOpsPerSecond(ops, duration),
		MemoryUsed:   int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:  int64(m2.Mallocs - m1.Mallocs),
		Success:      success,
		ErrorMsg:     errorMsg,
		Details: map[string]interface{}{
			"epochs":      r.config.NNEpochs,
			"samples":     numSamples,
			"batch_size":  r.config.NNBatchSize,
			"architecture": r.config.NNLayerSizes,
		},
	}

	if history != nil && len(history.Loss) > 0 {
		result.Details["final_loss"] = history.Loss[len(history.Loss)-1]
		result.Details["initial_loss"] = history.Loss[0]
	}

	r.addResult(result)

	status := "✓"
	if !success {
		status = "✗"
	}
	fmt.Printf("  %s %-28s %10.2f samples/sec  %10s  %8s\n",
		status, "FullTraining", result.OpsPerSecond, formatDuration(duration), formatBytes(result.MemoryUsed))
}

func (r *StressTestRunner) benchmarkNN(name string, X core.Tensor, op func() error) {
	iterations := 100

	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	var lastErr error
	for i := 0; i < iterations; i++ {
		if err := op(); err != nil {
			lastErr = err
		}
	}
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	samples, _ := X.Dims()
	ops := int64(iterations * samples)

	result := TestResult{
		Name:         name,
		Category:     "NeuralNetwork",
		Duration:     duration,
		Operations:   ops,
		OpsPerSecond: safeOpsPerSecond(ops, duration),
		MemoryUsed:   int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:  int64(m2.Mallocs - m1.Mallocs),
		Success:      lastErr == nil,
		Details: map[string]interface{}{
			"iterations": iterations,
			"samples":    samples,
		},
	}

	r.addResult(result)
	fmt.Printf("  %-30s %10.2f samples/sec  %10s  %8s\n",
		name, result.OpsPerSecond, formatDuration(duration), formatBytes(result.MemoryUsed))
}

// ═══════════════════════════════════════════════════════════════════════════════
// ML ALGORITHMS STRESS TEST
// ═══════════════════════════════════════════════════════════════════════════════

func (r *StressTestRunner) RunAlgorithmsStressTest() {
	fmt.Println("\n🔬 ML ALGORITHMS STRESS TEST")
	fmt.Println(strings.Repeat("─", 60))

	// Create test data
	samples := r.config.ClusteringSamples
	features := r.config.ClusteringFeatures
	clusters := r.config.ClusteringClusters

	fmt.Printf("  Creating %d samples with %d features...\n", samples, features)
	X := createClusteringData(samples, features, clusters)

	// K-Means Clustering
	fmt.Println("  Testing K-Means...")
	r.benchmarkAlgorithm("KMeans", "Algorithms", func() error {
		kmeans := algorithms.NewKMeans(clusters,
			algorithms.WithMaxIters(100),
			algorithms.WithTolerance(1e-4))
		return kmeans.Fit(X)
	})

	// DBSCAN Clustering
	fmt.Println("  Testing DBSCAN...")
	r.benchmarkAlgorithm("DBSCAN", "Algorithms", func() error {
		dbscan := algorithms.NewDBSCAN(0.5, 5)
		_, err := dbscan.FitPredict(X)
		return err
	})

	// PCA
	fmt.Println("  Testing PCA...")
	r.benchmarkAlgorithm("PCA", "Algorithms", func() error {
		pca := algorithms.NewPCA(10)
		return pca.Fit(X)
	})

	// Linear Regression (create regression data)
	fmt.Println("  Testing Linear Regression...")
	XReg, yReg := createRegressionData(samples, features)
	r.benchmarkAlgorithm("LinearRegression", "Algorithms", func() error {
		lr := algorithms.EasyLinearRegression()
		return lr.Fit(XReg, yReg)
	})

	// Logistic Regression
	fmt.Println("  Testing Logistic Regression...")
	yClass := createClassificationLabels(samples)
	r.benchmarkAlgorithm("LogisticRegression", "Algorithms", func() error {
		logReg := algorithms.EasyLogisticRegression()
		return logReg.Fit(X, yClass)
	})
}

func (r *StressTestRunner) benchmarkAlgorithm(name, category string, op func() error) {
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	err := op()
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	success := err == nil
	errorMsg := ""
	if err != nil {
		errorMsg = err.Error()
	}

	result := TestResult{
		Name:        name,
		Category:    category,
		Duration:    duration,
		Operations:  1,
		MemoryUsed:  int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations: int64(m2.Mallocs - m1.Mallocs),
		Success:     success,
		ErrorMsg:    errorMsg,
	}

	r.addResult(result)

	status := "✓"
	if !success {
		status = "✗"
	}
	fmt.Printf("  %s %-28s %10s  %8s\n",
		status, name, formatDuration(duration), formatBytes(result.MemoryUsed))
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEMORY STRESS TEST
// ═══════════════════════════════════════════════════════════════════════════════

func (r *StressTestRunner) RunMemoryStressTest() {
	fmt.Println("\n💾 MEMORY STRESS TEST")
	fmt.Println(strings.Repeat("─", 60))

	numMatrices := r.config.MemoryPoolMatrices
	size := r.config.MemoryPoolSize

	// Test without pooling
	fmt.Println("  Testing without memory pooling...")
	core.SetMatrixPoolEnabled(false)
	r.benchmarkMemory("WithoutPooling", numMatrices, size)

	// Test with pooling
	fmt.Println("  Testing with memory pooling...")
	core.SetMatrixPoolEnabled(true)
	r.benchmarkMemory("WithPooling", numMatrices, size)

	// Heavy allocation stress test
	fmt.Println("  Running heavy allocation stress test...")
	r.benchmarkHeavyAllocation("HeavyAllocation", numMatrices*10, size/2)

	// Concurrent memory stress
	fmt.Println("  Running concurrent memory stress test...")
	r.benchmarkConcurrentMemory("ConcurrentMemory", numMatrices, size, r.config.ParallelWorkers)
}

func (r *StressTestRunner) benchmarkMemory(name string, numMatrices, size int) {
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	for i := 0; i < numMatrices; i++ {
		matrix := core.GetMatrix(size, size)
		// Simulate work
		for j := 0; j < size; j++ {
			matrix.Set(j, j, float64(i+j))
		}
		core.PutMatrix(matrix)
	}
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	ops := int64(numMatrices)
	result := TestResult{
		Name:         name,
		Category:     "Memory",
		Duration:     duration,
		Operations:   ops,
		OpsPerSecond: safeOpsPerSecond(ops, duration),
		MemoryUsed:   int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:  int64(m2.Mallocs - m1.Mallocs),
		Success:      true,
		Details: map[string]interface{}{
			"matrices": numMatrices,
			"size":     size,
		},
	}

	r.addResult(result)
	fmt.Printf("  %-30s %10.2f ops/sec  %10s  %8s  %d allocs\n",
		name, result.OpsPerSecond, formatDuration(duration), formatBytes(result.MemoryUsed), result.Allocations)
}

func (r *StressTestRunner) benchmarkHeavyAllocation(name string, count, size int) {
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	tensors := make([]core.Tensor, 0, count)
	for i := 0; i < count; i++ {
		tensors = append(tensors, core.NewZerosTensor(size, size))
	}
	// Force usage to prevent optimization
	sum := 0.0
	for _, t := range tensors {
		if t != nil {
			rows, _ := t.Dims()
			if rows > 0 {
				sum += t.At(0, 0)
			}
		}
	}
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	ops := int64(count)
	result := TestResult{
		Name:         name,
		Category:     "Memory",
		Duration:     duration,
		Operations:   ops,
		OpsPerSecond: safeOpsPerSecond(ops, duration),
		MemoryUsed:   int64(m2.TotalAlloc - m1.TotalAlloc),
		MemoryPeak:   int64(m2.HeapAlloc),
		Allocations:  int64(m2.Mallocs - m1.Mallocs),
		Success:      true,
	}

	r.addResult(result)
	fmt.Printf("  %-30s %10.2f ops/sec  %10s  %8s\n",
		name, result.OpsPerSecond, formatDuration(duration), formatBytes(result.MemoryUsed))
}

func (r *StressTestRunner) benchmarkConcurrentMemory(name string, count, size, workers int) {
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	var wg sync.WaitGroup
	perWorker := count / workers

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < perWorker; i++ {
				matrix := core.GetMatrix(size, size)
				matrix.Set(0, 0, float64(i))
				core.PutMatrix(matrix)
			}
		}()
	}
	wg.Wait()
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	ops := int64(count)
	result := TestResult{
		Name:         name,
		Category:     "Memory",
		Duration:     duration,
		Operations:   ops,
		OpsPerSecond: safeOpsPerSecond(ops, duration),
		MemoryUsed:   int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:  int64(m2.Mallocs - m1.Mallocs),
		Success:      true,
		Details: map[string]interface{}{
			"workers": workers,
		},
	}

	r.addResult(result)
	fmt.Printf("  %-30s %10.2f ops/sec  %10s  %8s  (%d workers)\n",
		name, result.OpsPerSecond, formatDuration(duration), formatBytes(result.MemoryUsed), workers)
}

// ═══════════════════════════════════════════════════════════════════════════════
// HIGH PERFORMANCE STRESS TEST
// ═══════════════════════════════════════════════════════════════════════════════

func (r *StressTestRunner) RunHighPerformanceStressTest() {
	fmt.Println("\n🚀 HIGH PERFORMANCE STRESS TEST")
	fmt.Println(strings.Repeat("─", 60))

	ops := r.config.HighPerfOperations

	// Standard high-performance operations
	fmt.Printf("  Testing %s high-performance operations...\n", formatNumber(ops))
	r.benchmarkHighPerf("HighPerformance", ops, false)

	// Ultra-fast operations
	fmt.Printf("  Testing %s ultra-fast operations...\n", formatNumber(ops))
	r.benchmarkHighPerf("UltraFast", ops, true)

	// Batch processing stress
	fmt.Println("  Testing batch processing...")
	r.benchmarkBatchProcessing("BatchProcessing", 1000, 256)
}

func (r *StressTestRunner) benchmarkHighPerf(name string, ops int64, ultraFast bool) {
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	var opsPerSecond float64
	start := time.Now()

	if ultraFast {
		processor := core.GetUltraFastProcessor()
		opsPerSecond = processor.PerformUltraFastOperations(ops)
	} else {
		processor := core.GetHighPerformanceProcessor()
		opsPerSecond = processor.PerformOperations(ops)
	}

	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	result := TestResult{
		Name:         name,
		Category:     "HighPerformance",
		Duration:     duration,
		Operations:   ops,
		OpsPerSecond: opsPerSecond,
		MemoryUsed:   int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:  int64(m2.Mallocs - m1.Mallocs),
		Success:      true,
	}

	r.addResult(result)
	fmt.Printf("  %-30s %12.2f M ops/sec  %10s  %8s\n",
		name, opsPerSecond/1e6, formatDuration(duration), formatBytes(result.MemoryUsed))
}

func (r *StressTestRunner) benchmarkBatchProcessing(name string, batches, batchSize int) {
	// Create batch data
	inputs := make([]core.Tensor, batches)
	for i := range inputs {
		inputs[i] = createRandomTensor(batchSize, 64)
	}

	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	processor := core.GetBatchProcessor()
	results := processor.ProcessBatches(inputs, func(t core.Tensor) core.Tensor {
		return t.Scale(2.0)
	})
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	ops := int64(batches * batchSize)

	result := TestResult{
		Name:         name,
		Category:     "HighPerformance",
		Duration:     duration,
		Operations:   ops,
		OpsPerSecond: safeOpsPerSecond(ops, duration),
		MemoryUsed:   int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:  int64(m2.Mallocs - m1.Mallocs),
		Success:      len(results) == batches,
		Details: map[string]interface{}{
			"batches":    batches,
			"batch_size": batchSize,
		},
	}

	r.addResult(result)
	fmt.Printf("  %-30s %10.2f samples/sec  %10s  %8s\n",
		name, result.OpsPerSecond, formatDuration(duration), formatBytes(result.MemoryUsed))
}

// ═══════════════════════════════════════════════════════════════════════════════
// PREPROCESSING STRESS TEST
// ═══════════════════════════════════════════════════════════════════════════════

func (r *StressTestRunner) RunPreprocessingStressTest() {
	fmt.Println("\n📊 PREPROCESSING STRESS TEST")
	fmt.Println(strings.Repeat("─", 60))

	samples := r.config.ClusteringSamples
	features := r.config.ClusteringFeatures

	X := createRandomTensor(samples, features)

	// Standard Scaling
	fmt.Println("  Testing Standard Scaling...")
	r.benchmarkPreprocessing("StandardScale", func() error {
		_, err := preprocessing.EasyStandardScale(X)
		return err
	})

	// MinMax Scaling
	fmt.Println("  Testing MinMax Scaling...")
	r.benchmarkPreprocessing("MinMaxScale", func() error {
		_, err := preprocessing.EasyMinMaxScale(X)
		return err
	})

	// Train-Test Split
	fmt.Println("  Testing Train-Test Split...")
	y := createRandomTensor(samples, 1)
	r.benchmarkPreprocessing("TrainTestSplit", func() error {
		_, _, _, _, err := preprocessing.EasySplit(X, y, 0.2)
		return err
	})
}

func (r *StressTestRunner) benchmarkPreprocessing(name string, op func() error) {
	iterations := 10

	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	start := time.Now()
	var lastErr error
	for i := 0; i < iterations; i++ {
		if err := op(); err != nil {
			lastErr = err
		}
	}
	duration := time.Since(start)

	runtime.GC()
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)

	ops := int64(iterations)
	result := TestResult{
		Name:         name,
		Category:     "Preprocessing",
		Duration:     duration,
		Operations:   ops,
		OpsPerSecond: safeOpsPerSecond(ops, duration),
		MemoryUsed:   int64(m2.TotalAlloc - m1.TotalAlloc),
		Allocations:  int64(m2.Mallocs - m1.Mallocs),
		Success:      lastErr == nil,
	}

	r.addResult(result)

	status := "✓"
	if !result.Success {
		status = "✗"
	}
	fmt.Printf("  %s %-28s %10.2f ops/sec  %10s  %8s\n",
		status, name, result.OpsPerSecond, formatDuration(duration), formatBytes(result.MemoryUsed))
}

// ═══════════════════════════════════════════════════════════════════════════════
// REPORT GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

func (r *StressTestRunner) GenerateReport(startTime time.Time) StressTestReport {
	endTime := time.Now()

	var finalMem runtime.MemStats
	runtime.ReadMemStats(&finalMem)

	report := StressTestReport{
		StartTime:     startTime,
		EndTime:       endTime,
		TotalDuration: endTime.Sub(startTime),
		Config:        r.config,
		Results:       r.results,
		SystemInfo: SystemInfo{
			GoVersion:   runtime.Version(),
			NumCPU:      runtime.NumCPU(),
			GOMAXPROCS:  runtime.GOMAXPROCS(0),
			OS:          runtime.GOOS,
			Arch:        runtime.GOARCH,
			FinalAlloc:  finalMem.Alloc,
			TotalMemory: finalMem.Sys,
		},
	}

	// Calculate summary
	summary := TestSummary{
		Categories: make(map[string]CategorySummary),
	}

	for _, result := range r.results {
		summary.TotalTests++
		if result.Success {
			summary.PassedTests++
		} else {
			summary.FailedTests++
		}
		summary.TotalOperations += result.Operations
		summary.TotalDuration += result.Duration
		summary.TotalAllocations += result.Allocations

		if result.MemoryUsed > summary.PeakMemoryUsed {
			summary.PeakMemoryUsed = result.MemoryUsed
		}

		cat := summary.Categories[result.Category]
		cat.TestCount++
		cat.TotalOps += result.Operations
		cat.TotalTime += result.Duration
		cat.MemoryUsed += result.MemoryUsed
		summary.Categories[result.Category] = cat
	}

	if summary.TotalDuration > 0 {
		summary.AvgOpsPerSecond = float64(summary.TotalOperations) / summary.TotalDuration.Seconds()
	}

	// Calculate category averages
	for name, cat := range summary.Categories {
		if cat.TotalTime > 0 {
			cat.AvgOpsPerSec = float64(cat.TotalOps) / cat.TotalTime.Seconds()
		}
		summary.Categories[name] = cat
	}

	report.Summary = summary
	return report
}

func PrintFullReport(report StressTestReport) {
	fmt.Println("\n")
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                    THINKINGNET-GO STRESS TEST REPORT                        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════╝")

	// System Information
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                           SYSTEM INFORMATION                                 │")
	fmt.Println("├──────────────────────────────────────────────────────────────────────────────┤")
	fmt.Printf("│  Go Version:      %-58s │\n", report.SystemInfo.GoVersion)
	fmt.Printf("│  OS/Arch:         %-58s │\n", fmt.Sprintf("%s/%s", report.SystemInfo.OS, report.SystemInfo.Arch))
	fmt.Printf("│  CPU Cores:       %-58d │\n", report.SystemInfo.NumCPU)
	fmt.Printf("│  GOMAXPROCS:      %-58d │\n", report.SystemInfo.GOMAXPROCS)
	fmt.Printf("│  System Memory:   %-58s │\n", formatBytes(int64(report.SystemInfo.TotalMemory)))
	fmt.Println("└──────────────────────────────────────────────────────────────────────────────┘")

	// Test Configuration
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                          TEST CONFIGURATION                                  │")
	fmt.Println("├──────────────────────────────────────────────────────────────────────────────┤")
	fmt.Printf("│  Tensor Sizes:           %-50v │\n", report.Config.TensorSizes)
	fmt.Printf("│  NN Architecture:        %-50v │\n", report.Config.NNLayerSizes)
	fmt.Printf("│  Training Samples:       %-50d │\n", report.Config.NNTrainingSamples)
	fmt.Printf("│  Clustering Samples:     %-50d │\n", report.Config.ClusteringSamples)
	fmt.Printf("│  High-Perf Operations:   %-50s │\n", formatNumber(report.Config.HighPerfOperations))
	fmt.Println("└──────────────────────────────────────────────────────────────────────────────┘")

	// Execution Summary
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                          EXECUTION SUMMARY                                   │")
	fmt.Println("├──────────────────────────────────────────────────────────────────────────────┤")
	fmt.Printf("│  Start Time:      %-58s │\n", report.StartTime.Format("2006-01-02 15:04:05"))
	fmt.Printf("│  End Time:        %-58s │\n", report.EndTime.Format("2006-01-02 15:04:05"))
	fmt.Printf("│  Total Duration:  %-58s │\n", formatDuration(report.TotalDuration))
	fmt.Printf("│  Total Tests:     %-58d │\n", report.Summary.TotalTests)
	fmt.Printf("│  Passed:          %-58s │\n", fmt.Sprintf("%d (%.1f%%)", report.Summary.PassedTests, float64(report.Summary.PassedTests)/float64(report.Summary.TotalTests)*100))
	fmt.Printf("│  Failed:          %-58d │\n", report.Summary.FailedTests)
	fmt.Println("└──────────────────────────────────────────────────────────────────────────────┘")

	// Performance Metrics
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                          PERFORMANCE METRICS                                 │")
	fmt.Println("├──────────────────────────────────────────────────────────────────────────────┤")
	fmt.Printf("│  Total Operations:       %-50s │\n", formatNumber(report.Summary.TotalOperations))
	fmt.Printf("│  Average Ops/Second:     %-50s │\n", fmt.Sprintf("%.2f M", report.Summary.AvgOpsPerSecond/1e6))
	fmt.Printf("│  Peak Memory Used:       %-50s │\n", formatBytes(report.Summary.PeakMemoryUsed))
	fmt.Printf("│  Total Allocations:      %-50s │\n", formatNumber(report.Summary.TotalAllocations))
	fmt.Println("└──────────────────────────────────────────────────────────────────────────────┘")

	// Category Breakdown
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                          CATEGORY BREAKDOWN                                  │")
	fmt.Println("├────────────────────┬───────────┬───────────────┬─────────────┬───────────────┤")
	fmt.Println("│ Category           │ Tests     │ Operations    │ Avg Ops/Sec │ Memory Used   │")
	fmt.Println("├────────────────────┼───────────┼───────────────┼─────────────┼───────────────┤")

	// Sort categories for consistent output
	categories := make([]string, 0, len(report.Summary.Categories))
	for cat := range report.Summary.Categories {
		categories = append(categories, cat)
	}
	sort.Strings(categories)

	for _, cat := range categories {
		summary := report.Summary.Categories[cat]
		fmt.Printf("│ %-18s │ %9d │ %13s │ %11s │ %13s │\n",
			cat,
			summary.TestCount,
			formatNumber(summary.TotalOps),
			fmt.Sprintf("%.2fM", summary.AvgOpsPerSec/1e6),
			formatBytes(summary.MemoryUsed))
	}
	fmt.Println("└────────────────────┴───────────┴───────────────┴─────────────┴───────────────┘")

	// Detailed Results
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                          DETAILED RESULTS                                    │")
	fmt.Println("├────────────────────────────────────┬────────────┬────────────────┬───────────┤")
	fmt.Println("│ Test Name                          │ Duration   │ Ops/Sec        │ Status    │")
	fmt.Println("├────────────────────────────────────┼────────────┼────────────────┼───────────┤")

	for _, result := range report.Results {
		status := "✓ PASS"
		if !result.Success {
			status = "✗ FAIL"
		}
		name := result.Name
		if len(name) > 34 {
			name = name[:31] + "..."
		}
		fmt.Printf("│ %-34s │ %10s │ %14s │ %-9s │\n",
			name,
			formatDuration(result.Duration),
			fmt.Sprintf("%.2fM", result.OpsPerSecond/1e6),
			status)
	}
	fmt.Println("└────────────────────────────────────┴────────────┴────────────────┴───────────┘")

	// Top Performers
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                          TOP 10 FASTEST OPERATIONS                           │")
	fmt.Println("├──────────────────────────────────────────────────────────────────────────────┤")

	// Sort by ops/sec
	sortedResults := make([]TestResult, len(report.Results))
	copy(sortedResults, report.Results)
	sort.Slice(sortedResults, func(i, j int) bool {
		return sortedResults[i].OpsPerSecond > sortedResults[j].OpsPerSecond
	})

	for i := 0; i < min(10, len(sortedResults)); i++ {
		r := sortedResults[i]
		fmt.Printf("│  %2d. %-40s %20s ops/sec │\n", i+1, r.Name, formatNumber(int64(r.OpsPerSecond)))
	}
	fmt.Println("└──────────────────────────────────────────────────────────────────────────────┘")

	// Failed Tests (if any)
	if report.Summary.FailedTests > 0 {
		fmt.Println("\n┌──────────────────────────────────────────────────────────────────────────────┐")
		fmt.Println("│                          FAILED TESTS                                        │")
		fmt.Println("├──────────────────────────────────────────────────────────────────────────────┤")
		for _, result := range report.Results {
			if !result.Success {
				fmt.Printf("│  ✗ %-73s │\n", result.Name)
				if result.ErrorMsg != "" {
					errMsg := result.ErrorMsg
					if len(errMsg) > 70 {
						errMsg = errMsg[:67] + "..."
					}
					fmt.Printf("│    Error: %-66s │\n", errMsg)
				}
			}
		}
		fmt.Println("└──────────────────────────────────────────────────────────────────────────────┘")
	}

	// Final Status
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════╗")
	if report.Summary.FailedTests == 0 {
		fmt.Println("║                    ✓ ALL STRESS TESTS PASSED SUCCESSFULLY                   ║")
	} else {
		fmt.Printf("║               ⚠ %d TESTS FAILED - REVIEW RESULTS ABOVE                      ║\n", report.Summary.FailedTests)
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════╝")
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

// safeOpsPerSecond calculates ops/sec avoiding division by zero
func safeOpsPerSecond(ops int64, duration time.Duration) float64 {
	if duration <= 0 {
		return 0 // Return 0 for instantaneous operations to avoid +Inf
	}
	return float64(ops) / duration.Seconds()
}

func createRandomTensor(rows, cols int) core.Tensor {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.Float64()*2 - 1 // -1 to 1
	}
	return core.NewTensor(mat.NewDense(rows, cols, data))
}

func createSyntheticData(samples, inputSize, outputSize int) (core.Tensor, core.Tensor) {
	XData := make([]float64, samples*inputSize)
	yData := make([]float64, samples*outputSize)

	for i := 0; i < samples; i++ {
		// Create input
		for j := 0; j < inputSize; j++ {
			XData[i*inputSize+j] = rand.Float64()*2 - 1
		}
		// Create one-hot encoded output
		classIdx := rand.Intn(outputSize)
		for j := 0; j < outputSize; j++ {
			if j == classIdx {
				yData[i*outputSize+j] = 1.0
			} else {
				yData[i*outputSize+j] = 0.0
			}
		}
	}

	X := core.NewTensor(mat.NewDense(samples, inputSize, XData))
	y := core.NewTensor(mat.NewDense(samples, outputSize, yData))
	return X, y
}

func createClusteringData(samples, features, clusters int) core.Tensor {
	data := make([]float64, samples*features)
	samplesPerCluster := samples / clusters

	for c := 0; c < clusters; c++ {
		// Create cluster center
		center := make([]float64, features)
		for j := range center {
			center[j] = float64(c*10) + rand.Float64()*5
		}

		// Create samples around center
		for i := 0; i < samplesPerCluster; i++ {
			sampleIdx := c*samplesPerCluster + i
			if sampleIdx >= samples {
				break
			}
			for j := 0; j < features; j++ {
				data[sampleIdx*features+j] = center[j] + rand.NormFloat64()*2
			}
		}
	}

	return core.NewTensor(mat.NewDense(samples, features, data))
}

func createRegressionData(samples, features int) (core.Tensor, core.Tensor) {
	XData := make([]float64, samples*features)
	yData := make([]float64, samples)

	for i := 0; i < samples; i++ {
		sum := 0.0
		for j := 0; j < features; j++ {
			val := rand.Float64() * 10
			XData[i*features+j] = val
			sum += val * float64(j+1) // weighted sum
		}
		yData[i] = sum + rand.NormFloat64()*0.1 // add noise
	}

	X := core.NewTensor(mat.NewDense(samples, features, XData))
	y := core.NewTensor(mat.NewDense(samples, 1, yData))
	return X, y
}

func createClassificationLabels(samples int) core.Tensor {
	data := make([]float64, samples)
	for i := range data {
		data[i] = float64(rand.Intn(2)) // binary classification
	}
	return core.NewTensor(mat.NewDense(samples, 1, data))
}

func formatDuration(d time.Duration) string {
	if d < time.Microsecond {
		return fmt.Sprintf("%dns", d.Nanoseconds())
	} else if d < time.Millisecond {
		return fmt.Sprintf("%.2fµs", float64(d.Nanoseconds())/1000)
	} else if d < time.Second {
		return fmt.Sprintf("%.2fms", float64(d.Nanoseconds())/1e6)
	} else if d < time.Minute {
		return fmt.Sprintf("%.2fs", d.Seconds())
	}
	return fmt.Sprintf("%.2fm", d.Minutes())
}

func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

func formatNumber(n int64) string {
	if n < 1000 {
		return fmt.Sprintf("%d", n)
	} else if n < 1000000 {
		return fmt.Sprintf("%.1fK", float64(n)/1000)
	} else if n < 1000000000 {
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	}
	return fmt.Sprintf("%.1fB", float64(n)/1e9)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN FUNCTION
// ═══════════════════════════════════════════════════════════════════════════════

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║          THINKINGNET-GO HEAVY STRESS TEST - اختبار الإجهاد الشامل          ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("This comprehensive stress test will evaluate all components of the library.")
	fmt.Println("سيقوم هذا الاختبار الشامل بتقييم جميع مكونات المكتبة.")
	fmt.Println()

	config := DefaultHeavyConfig()
	runner := NewStressTestRunner(config)

	fmt.Printf("System: %s/%s with %d CPU cores\n", runtime.GOOS, runtime.GOARCH, runtime.NumCPU())
	fmt.Printf("Go Version: %s\n", runtime.Version())
	fmt.Println()

	startTime := time.Now()

	// Run all stress tests
	runner.RunTensorStressTest()
	runner.RunActivationStressTest()
	runner.RunNeuralNetworkStressTest()
	runner.RunAlgorithmsStressTest()
	runner.RunMemoryStressTest()
	runner.RunHighPerformanceStressTest()
	runner.RunPreprocessingStressTest()

	// Generate and print report
	report := runner.GenerateReport(startTime)
	PrintFullReport(report)
}
