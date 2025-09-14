package main

import (
	"fmt"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/core"
	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("🚀 ThinkingNet-Go Performance Optimizations Demo")
	fmt.Println("================================================")

	// Demo 1: High-Performance Operations
	demoHighPerformanceOperations()

	// Demo 2: Parallel Activation Functions
	demoParallelActivations()

	// Demo 3: Memory Pooling
	demoMemoryPooling()

	// Demo 4: Batch Processing
	demoBatchProcessing()

	// Demo 5: Performance Benchmarking
	demoBenchmarking()

	fmt.Println("\n✅ All optimization demos completed successfully!")
}

func demoHighPerformanceOperations() {
	fmt.Println("\n1. High-Performance Operations Demo")
	fmt.Println("===================================")

	// Standard high-performance processor
	processor := core.GetHighPerformanceProcessor()
	fmt.Println("Executing 10 million high-performance operations...")
	opsPerSecond := processor.PerformOperations(10_000_000)
	fmt.Printf("✅ Standard: %.0f operations per second\n", opsPerSecond)

	// Ultra-fast processor (maximum speed)
	ultraProcessor := core.GetUltraFastProcessor()
	fmt.Println("Executing 10 million ULTRA-FAST operations...")
	ultraOpsPerSecond := ultraProcessor.PerformUltraFastOperations(10_000_000)
	fmt.Printf("⚡ Ultra-Fast: %.0f operations per second\n", ultraOpsPerSecond)

	speedup := ultraOpsPerSecond / opsPerSecond
	fmt.Printf("🚀 Ultra-Fast Speedup: %.2fx faster!\n", speedup)
}

func demoParallelActivations() {
	fmt.Println("\n2. Parallel Activation Functions Demo")
	fmt.Println("====================================")

	// Create test data
	size := 1_000_000
	input := make([]float64, size)
	output := make([]float64, size)

	for i := range input {
		input[i] = float64(i-size/2) * 0.001 // Range from -500 to 500
	}

	// Standard parallel processor
	processor := core.GetParallelActivationProcessor()
	fmt.Printf("Processing %d elements with parallel activations...\n", size)

	start := time.Now()
	processor.ProcessReLU(input, output)
	standardReLUTime := time.Since(start)
	fmt.Printf("✅ Standard ReLU: %v\n", standardReLUTime)

	start = time.Now()
	processor.ProcessSigmoid(input, output)
	standardSigmoidTime := time.Since(start)
	fmt.Printf("✅ Standard Sigmoid: %v\n", standardSigmoidTime)

	// Ultra-fast processor with lookup tables
	ultraProcessor := core.GetUltraFastActivationProcessor()
	fmt.Printf("Processing %d elements with ULTRA-FAST activations...\n", size)

	start = time.Now()
	ultraProcessor.UltraFastReLU(input, output)
	ultraReLUTime := time.Since(start)
	fmt.Printf("⚡ Ultra-Fast ReLU: %v\n", ultraReLUTime)

	start = time.Now()
	ultraProcessor.UltraFastSigmoid(input, output)
	ultraSigmoidTime := time.Since(start)
	fmt.Printf("⚡ Ultra-Fast Sigmoid: %v\n", ultraSigmoidTime)

	reluSpeedup := float64(standardReLUTime) / float64(ultraReLUTime)
	sigmoidSpeedup := float64(standardSigmoidTime) / float64(ultraSigmoidTime)
	fmt.Printf("🚀 ReLU Speedup: %.2fx, Sigmoid Speedup: %.2fx\n", reluSpeedup, sigmoidSpeedup)
}

func demoMemoryPooling() {
	fmt.Println("\n3. Memory Pooling Demo")
	fmt.Println("======================")

	fmt.Println("Creating and reusing matrices with memory pooling...")

	// Create and use matrices
	for i := 0; i < 100; i++ {
		matrix := core.GetMatrix(100, 100)

		// Simulate some work
		for j := 0; j < 100; j++ {
			for k := 0; k < 100; k++ {
				matrix.Set(j, k, float64(i+j+k))
			}
		}

		core.PutMatrix(matrix) // Return to pool
	}

	// Check pool statistics
	stats := core.MatrixPoolStats()
	if poolStat, exists := stats["100x100"]; exists {
		fmt.Printf("✅ Pool stats - Gets: %d, Puts: %d, Hits: %d\n",
			poolStat.Gets, poolStat.Puts, poolStat.Hits)
	}
}

func demoBatchProcessing() {
	fmt.Println("\n4. Batch Processing Demo")
	fmt.Println("========================")

	// Create batch of tensors
	batchSize := 10
	inputs := make([]core.Tensor, batchSize)

	for i := range inputs {
		data := mat.NewDense(50, 50, nil)
		for j := 0; j < 50; j++ {
			for k := 0; k < 50; k++ {
				data.Set(j, k, float64(i+j+k))
			}
		}
		inputs[i] = core.NewTensor(data)
	}

	fmt.Printf("Processing batch of %d tensors...\n", batchSize)

	// Process batch in parallel
	batchProcessor := core.GetBatchProcessor()
	results := batchProcessor.ProcessBatches(inputs, func(input core.Tensor) core.Tensor {
		return input.Scale(2.0) // Simple scaling operation
	})

	fmt.Printf("✅ Processed %d tensors in parallel\n", len(results))
}

func demoBenchmarking() {
	fmt.Println("\n5. Performance Benchmarking Demo")
	fmt.Println("================================")

	fmt.Println("Running quick performance benchmark...")
	core.RunQuickBenchmark()

	fmt.Println("\n⚡ Running ULTRA-FAST maximum speed benchmark...")
	core.RunUltraFastBenchmark()
}
