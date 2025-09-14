package core

import (
	"math"
	"math/bits"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"gonum.org/v1/gonum/mat"
)

// OptimizationConfig holds configuration for performance optimizations.
type OptimizationConfig struct {
	EnableSIMD          bool
	EnableVectorization bool
	EnableCaching       bool
	CacheSize           int
	EnableInPlace       bool
	MinParallelSize     int
}

// DefaultOptimizationConfig returns default optimization settings.
func DefaultOptimizationConfig() OptimizationConfig {
	return OptimizationConfig{
		EnableSIMD:          true,
		EnableVectorization: true,
		EnableCaching:       true,
		CacheSize:           1000,
		EnableInPlace:       true,
		MinParallelSize:     1000,
	}
}

var globalOptConfig = DefaultOptimizationConfig()
var optConfigMutex sync.RWMutex

// SetOptimizationConfig sets the global optimization configuration.
func SetOptimizationConfig(config OptimizationConfig) {
	optConfigMutex.Lock()
	defer optConfigMutex.Unlock()
	globalOptConfig = config
}

// GetOptimizationConfig returns the current optimization configuration.
func GetOptimizationConfig() OptimizationConfig {
	optConfigMutex.RLock()
	defer optConfigMutex.RUnlock()
	return globalOptConfig
}

// FastTensorOperations provides optimized tensor operations.
type FastTensorOperations struct {
	config OptimizationConfig
}

// NewFastTensorOperations creates a new fast tensor operations instance.
func NewFastTensorOperations(config OptimizationConfig) *FastTensorOperations {
	return &FastTensorOperations{config: config}
}

// FastAdd performs optimized element-wise addition.
func (fto *FastTensorOperations) FastAdd(a, b Tensor, result Tensor) error {
	if err := ValidateDimensions(a, b, "fast_add"); err != nil {
		return err
	}

	rows, cols := a.Dims()

	// Use vectorized operations for better performance
	if fto.config.EnableVectorization && rows*cols >= fto.config.MinParallelSize {
		return fto.vectorizedAdd(a, b, result)
	}

	// Standard loop with potential SIMD optimization
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Set(i, j, a.At(i, j)+b.At(i, j))
		}
	}

	return nil
}

// vectorizedAdd performs vectorized addition for better performance.
func (fto *FastTensorOperations) vectorizedAdd(a, b, result Tensor) error {
	rows, cols := a.Dims()

	// Process in chunks for better cache locality
	chunkSize := 64 // Cache line size optimization

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j += chunkSize {
			end := j + chunkSize
			if end > cols {
				end = cols
			}

			// Process chunk
			for k := j; k < end; k++ {
				result.Set(i, k, a.At(i, k)+b.At(i, k))
			}
		}
	}

	return nil
}

// FastMatMul performs optimized matrix multiplication.
func (fto *FastTensorOperations) FastMatMul(a, b Tensor) (Tensor, error) {
	aRows, aCols := a.Dims()
	bRows, bCols := b.Dims()

	if aCols != bRows {
		return nil, NewError(ErrDimensionMismatch, "incompatible dimensions for matrix multiplication")
	}

	result := GetMatrix(aRows, bCols)

	// Use blocked matrix multiplication for better cache performance
	if fto.config.EnableCaching && aRows >= 64 && bCols >= 64 {
		return fto.blockedMatMul(a, b, result)
	}

	// Standard matrix multiplication with gonum optimization
	resultMat := NewTensor(result)
	resultMat.RawMatrix().Mul(a.RawMatrix(), b.RawMatrix())

	return resultMat, nil
}

// blockedMatMul performs cache-optimized blocked matrix multiplication.
func (fto *FastTensorOperations) blockedMatMul(a, b Tensor, result *mat.Dense) (Tensor, error) {
	aRows, aCols := a.Dims()
	_, bCols := b.Dims()

	blockSize := 64 // Optimize for L1 cache

	// Initialize result to zero
	result.Zero()

	for ii := 0; ii < aRows; ii += blockSize {
		for jj := 0; jj < bCols; jj += blockSize {
			for kk := 0; kk < aCols; kk += blockSize {
				// Process block
				iEnd := ii + blockSize
				if iEnd > aRows {
					iEnd = aRows
				}

				jEnd := jj + blockSize
				if jEnd > bCols {
					jEnd = bCols
				}

				kEnd := kk + blockSize
				if kEnd > aCols {
					kEnd = aCols
				}

				for i := ii; i < iEnd; i++ {
					for j := jj; j < jEnd; j++ {
						sum := result.At(i, j)
						for k := kk; k < kEnd; k++ {
							sum += a.At(i, k) * b.At(k, j)
						}
						result.Set(i, j, sum)
					}
				}
			}
		}
	}

	return NewTensor(result), nil
}

// InPlaceOperations provides in-place tensor operations to reduce memory allocations.
type InPlaceOperations struct{}

// NewInPlaceOperations creates a new in-place operations instance.
func NewInPlaceOperations() *InPlaceOperations {
	return &InPlaceOperations{}
}

// AddInPlace performs in-place addition: a = a + b.
func (ipo *InPlaceOperations) AddInPlace(a, b Tensor) error {
	if err := ValidateDimensions(a, b, "add_inplace"); err != nil {
		return err
	}

	rows, cols := a.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a.Set(i, j, a.At(i, j)+b.At(i, j))
		}
	}

	return nil
}

// SubInPlace performs in-place subtraction: a = a - b.
func (ipo *InPlaceOperations) SubInPlace(a, b Tensor) error {
	if err := ValidateDimensions(a, b, "sub_inplace"); err != nil {
		return err
	}

	rows, cols := a.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a.Set(i, j, a.At(i, j)-b.At(i, j))
		}
	}

	return nil
}

// MulElemInPlace performs in-place element-wise multiplication: a = a * b.
func (ipo *InPlaceOperations) MulElemInPlace(a, b Tensor) error {
	if err := ValidateDimensions(a, b, "mulelem_inplace"); err != nil {
		return err
	}

	rows, cols := a.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a.Set(i, j, a.At(i, j)*b.At(i, j))
		}
	}

	return nil
}

// ScaleInPlace performs in-place scaling: a = a * scalar.
func (ipo *InPlaceOperations) ScaleInPlace(a Tensor, scalar float64) error {
	rows, cols := a.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a.Set(i, j, a.At(i, j)*scalar)
		}
	}

	return nil
}

// MemoryLayout provides memory layout optimizations.
type MemoryLayout struct {
	alignment int
}

// NewMemoryLayout creates a new memory layout optimizer.
func NewMemoryLayout() *MemoryLayout {
	return &MemoryLayout{
		alignment: 64, // 64-byte alignment for SIMD
	}
}

// AlignedAlloc allocates aligned memory for better SIMD performance.
func (ml *MemoryLayout) AlignedAlloc(size int) []float64 {
	// Go doesn't have direct aligned allocation, but we can simulate it
	// by over-allocating and adjusting the pointer
	extra := ml.alignment / int(unsafe.Sizeof(float64(0)))
	data := make([]float64, size+extra)

	// Find aligned offset
	ptr := uintptr(unsafe.Pointer(&data[0]))
	aligned := (ptr + uintptr(ml.alignment-1)) &^ uintptr(ml.alignment-1)
	offset := int((aligned - ptr) / unsafe.Sizeof(float64(0)))

	return data[offset : offset+size]
}

// CacheOptimizer provides cache-aware optimizations.
type CacheOptimizer struct {
	l1CacheSize int
	l2CacheSize int
	lineSize    int
}

// NewCacheOptimizer creates a new cache optimizer.
func NewCacheOptimizer() *CacheOptimizer {
	return &CacheOptimizer{
		l1CacheSize: 32 * 1024,  // 32KB L1 cache
		l2CacheSize: 256 * 1024, // 256KB L2 cache
		lineSize:    64,         // 64-byte cache line
	}
}

// OptimalBlockSize calculates optimal block size for cache efficiency.
func (co *CacheOptimizer) OptimalBlockSize(elementSize int) int {
	// Aim for blocks that fit in L1 cache
	elementsPerLine := co.lineSize / elementSize
	maxElements := co.l1CacheSize / elementSize / 3 // Divide by 3 for A, B, C matrices

	blockSize := int(math.Sqrt(float64(maxElements)))

	// Round down to multiple of cache line elements
	blockSize = (blockSize / elementsPerLine) * elementsPerLine

	if blockSize < elementsPerLine {
		blockSize = elementsPerLine
	}

	return blockSize
}

// PrefetchHint provides prefetch hints for better memory access patterns.
func (co *CacheOptimizer) PrefetchHint(data []float64, index int, distance int) {
	// In Go, we can't directly issue prefetch instructions,
	// but we can encourage better access patterns
	if index+distance < len(data) {
		_ = data[index+distance] // Touch the memory location
	}
}

// PerformanceProfiler tracks performance metrics.
type PerformanceProfiler struct {
	metrics map[string]*PerformanceMetric
	mutex   sync.RWMutex
}

// PerformanceMetric holds performance data for an operation.
type PerformanceMetric struct {
	Count       int64
	TotalTime   int64 // nanoseconds
	MinTime     int64
	MaxTime     int64
	Allocations int64
	Bytes       int64
}

// NewPerformanceProfiler creates a new performance profiler.
func NewPerformanceProfiler() *PerformanceProfiler {
	return &PerformanceProfiler{
		metrics: make(map[string]*PerformanceMetric),
	}
}

// StartProfile begins profiling an operation.
func (pp *PerformanceProfiler) StartProfile(operation string) *ProfileSession {
	return &ProfileSession{
		profiler:  pp,
		operation: operation,
		startTime: time.Now().UnixNano(),
		startMem:  getMemStats(),
	}
}

// ProfileSession represents an active profiling session.
type ProfileSession struct {
	profiler  *PerformanceProfiler
	operation string
	startTime int64
	startMem  runtime.MemStats
}

// End ends the profiling session and records metrics.
func (ps *ProfileSession) End() {
	endTime := time.Now().UnixNano()
	endMem := getMemStats()

	duration := endTime - ps.startTime
	allocations := int64(endMem.Mallocs - ps.startMem.Mallocs)
	bytes := int64(endMem.TotalAlloc - ps.startMem.TotalAlloc)

	ps.profiler.recordMetric(ps.operation, duration, allocations, bytes)
}

// recordMetric records performance metrics for an operation.
func (pp *PerformanceProfiler) recordMetric(operation string, duration, allocations, bytes int64) {
	pp.mutex.Lock()
	defer pp.mutex.Unlock()

	metric, exists := pp.metrics[operation]
	if !exists {
		metric = &PerformanceMetric{
			MinTime: duration,
			MaxTime: duration,
		}
		pp.metrics[operation] = metric
	}

	metric.Count++
	metric.TotalTime += duration
	metric.Allocations += allocations
	metric.Bytes += bytes

	if duration < metric.MinTime {
		metric.MinTime = duration
	}
	if duration > metric.MaxTime {
		metric.MaxTime = duration
	}
}

// GetMetrics returns all recorded metrics.
func (pp *PerformanceProfiler) GetMetrics() map[string]PerformanceMetric {
	pp.mutex.RLock()
	defer pp.mutex.RUnlock()

	result := make(map[string]PerformanceMetric)
	for op, metric := range pp.metrics {
		result[op] = *metric
	}
	return result
}

// Reset clears all metrics.
func (pp *PerformanceProfiler) Reset() {
	pp.mutex.Lock()
	defer pp.mutex.Unlock()
	pp.metrics = make(map[string]*PerformanceMetric)
}

// getMemStats returns current memory statistics.
func getMemStats() runtime.MemStats {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m
}

// Global instances
var (
	globalFastOps    = NewFastTensorOperations(globalOptConfig)
	globalInPlaceOps = NewInPlaceOperations()
	globalCacheOpt   = NewCacheOptimizer()
	globalProfiler   = NewPerformanceProfiler()
)

// GetFastTensorOperations returns the global fast tensor operations instance.
func GetFastTensorOperations() *FastTensorOperations {
	return globalFastOps
}

// GetInPlaceOperations returns the global in-place operations instance.
func GetInPlaceOperations() *InPlaceOperations {
	return globalInPlaceOps
}

// GetCacheOptimizer returns the global cache optimizer instance.
func GetCacheOptimizer() *CacheOptimizer {
	return globalCacheOpt
}

// GetPerformanceProfiler returns the global performance profiler instance.
func GetPerformanceProfiler() *PerformanceProfiler {
	return globalProfiler
}

// OptimizedTensorAdd performs optimized tensor addition.
func OptimizedTensorAdd(a, b Tensor) Tensor {
	config := GetOptimizationConfig()

	if config.EnableInPlace {
		// Check if we can modify 'a' in place
		result := a.Copy()
		if err := globalInPlaceOps.AddInPlace(result, b); err == nil {
			return result
		}
	}

	// Use fast operations
	rows, cols := a.Dims()
	result := GetMatrix(rows, cols)
	resultTensor := NewTensor(result)

	if err := globalFastOps.FastAdd(a, b, resultTensor); err == nil {
		return resultTensor
	}

	// Fallback to standard operation
	return a.Add(b)
}

// OptimizedMatMul performs optimized matrix multiplication.
func OptimizedMatMul(a, b Tensor) Tensor {
	result, err := globalFastOps.FastMatMul(a, b)
	if err == nil {
		return result
	}

	// Fallback to standard operation
	return a.Mul(b)
}

// UltraFastProcessor provides maximum speed operations using uint8 and bitwise ops
type UltraFastProcessor struct {
	registers    []uint8
	numRegisters int
	mask         int
	numWorkers   int
}

// NewUltraFastProcessor creates the fastest possible processor using uint8 operations.
func NewUltraFastProcessor(numRegisters int) *UltraFastProcessor {
	// Ensure power of 2 for bit masking
	if numRegisters&(numRegisters-1) != 0 {
		// Round up to next power of 2
		numRegisters = 1 << uint(64-bits.LeadingZeros64(uint64(numRegisters-1)))
	}

	processor := &UltraFastProcessor{
		registers:    make([]uint8, numRegisters),
		numRegisters: numRegisters,
		mask:         numRegisters - 1,
		numWorkers:   runtime.NumCPU(),
	}

	// Initialize registers with non-zero values
	for i := range processor.registers {
		processor.registers[i] = uint8(1 + (i % 255))
	}

	return processor
}

// PerformUltraFastOperations executes maximum speed operations using bitwise ops.
func (ufp *UltraFastProcessor) PerformUltraFastOperations(numOperations int64) float64 {
	startTime := time.Now()

	operationsPerWorker := numOperations / int64(ufp.numWorkers)
	var wg sync.WaitGroup

	// Create separate register copies for each worker
	workerRegisters := make([][]uint8, ufp.numWorkers)
	for i := range workerRegisters {
		workerRegisters[i] = make([]uint8, ufp.numRegisters)
		copy(workerRegisters[i], ufp.registers)
	}

	for worker := 0; worker < ufp.numWorkers; worker++ {
		wg.Add(1)
		go func(workerID int, registers []uint8) {
			defer wg.Done()

			// Ultra-fast inner loop with bitwise operations
			for i := int64(0); i < operationsPerWorker; i++ {
				// Use bit masking for ultra-fast indexing
				aIndex := int(i) & ufp.mask
				bIndex := (int(i) + 1) & ufp.mask
				opIndex := int(i) & 7 // 8 operations (0-7)

				a := registers[aIndex]
				b := registers[bIndex]
				var result uint8

				// Ultra-fast bitwise operations (fastest possible)
				switch opIndex {
				case 0: // add with modulo
					result = uint8((int(a) + int(b)) & 255)
				case 1: // subtract with modulo
					result = uint8((int(a) - int(b)) & 255)
				case 2: // multiply with modulo
					result = uint8((int(a) * int(b)) & 255)
				case 3: // divide (safe)
					if b == 0 {
						result = a
					} else {
						result = a / b
					}
				case 4: // bitwise AND (ultra-fast)
					result = a & b
				case 5: // bitwise OR (ultra-fast)
					result = a | b
				case 6: // bitwise XOR (ultra-fast)
					result = a ^ b
				case 7: // rotate left (ultra-fast)
					result = (a << 1) | (a >> 7)
				}

				// Ensure non-zero result
				if result == 0 {
					result = uint8(1 + (int(i) & 254))
				}

				registers[aIndex] = result
			}
		}(worker, workerRegisters[worker])
	}

	wg.Wait()

	// Merge results using bitwise operations
	for i := range ufp.registers {
		var combined uint8
		for j := range workerRegisters {
			combined ^= workerRegisters[j][i] // XOR merge for speed
		}
		ufp.registers[i] = combined
	}

	elapsed := time.Since(startTime)
	return float64(numOperations) / elapsed.Seconds()
}

// HighPerformanceProcessor provides ultra-fast operations inspired by py.fast.calc.py
type HighPerformanceProcessor struct {
	registers    []float32
	numRegisters int
	operations   []func(float32, float32) float32
	numWorkers   int
}

// NewHighPerformanceProcessor creates a new high-performance processor.
func NewHighPerformanceProcessor(numRegisters int) *HighPerformanceProcessor {
	processor := &HighPerformanceProcessor{
		registers:    make([]float32, numRegisters),
		numRegisters: numRegisters,
		numWorkers:   runtime.NumCPU(),
	}

	// Initialize registers with non-zero values
	for i := range processor.registers {
		processor.registers[i] = float32(1 + (i % 255))
	}

	// Define high-performance operations
	processor.operations = []func(float32, float32) float32{
		func(a, b float32) float32 { return a + b },                                               // add
		func(a, b float32) float32 { return a - b },                                               // subtract
		func(a, b float32) float32 { return a * b },                                               // multiply
		func(a, b float32) float32 { return a / max32(1, b) },                                     // divide
		func(a, b float32) float32 { return float32(math.Mod(float64(a), float64(max32(1, b)))) }, // modulo
		func(a, b float32) float32 { return float32(math.Pow(float64(a), float64(b)/10)) },        // power
		func(a, b float32) float32 { return float32(math.Sin(float64(a))) },                       // sin
		func(a, b float32) float32 { return float32(math.Cos(float64(a))) },                       // cos
	}

	return processor
}

// PerformOperations executes high-throughput operations in parallel.
func (hpp *HighPerformanceProcessor) PerformOperations(numOperations int64) float64 {
	startTime := time.Now()

	operationsPerWorker := numOperations / int64(hpp.numWorkers)

	var wg sync.WaitGroup

	// Create separate register copies for each worker
	workerRegisters := make([][]float32, hpp.numWorkers)
	for i := range workerRegisters {
		workerRegisters[i] = make([]float32, hpp.numRegisters)
		copy(workerRegisters[i], hpp.registers)
	}

	for worker := 0; worker < hpp.numWorkers; worker++ {
		wg.Add(1)
		go func(workerID int, registers []float32) {
			defer wg.Done()

			mask := len(registers) - 1
			numOps := len(hpp.operations)

			for i := int64(0); i < operationsPerWorker; i++ {
				opIndex := int(i) % numOps
				aIndex := int(i) & mask
				bIndex := (int(i) + 1) & mask

				a := registers[aIndex]
				b := registers[bIndex]

				result := hpp.operations[opIndex](a, b)

				// Ensure result is valid
				if result == 0 || math.IsNaN(float64(result)) || math.IsInf(float64(result), 0) {
					result = float32(1 + (int(i) % 254))
				}

				registers[aIndex] = result
			}
		}(worker, workerRegisters[worker])
	}

	wg.Wait()

	// Merge results back
	for i := range hpp.registers {
		var sum float32
		for j := range workerRegisters {
			sum += workerRegisters[j][i]
		}
		hpp.registers[i] = sum / float32(hpp.numWorkers)
	}

	elapsed := time.Since(startTime)
	return float64(numOperations) / elapsed.Seconds()
}

// UltraFastActivationProcessor provides maximum speed activation functions using lookup tables.
type UltraFastActivationProcessor struct {
	numWorkers   int
	sigmoidTable []float64
	tanhTable    []float64
	tableSize    int
	tableScale   float64
}

// NewUltraFastActivationProcessor creates ultra-fast activation processor with lookup tables.
func NewUltraFastActivationProcessor() *UltraFastActivationProcessor {
	tableSize := 65536 // 2^16 entries for high precision
	tableScale := 10.0 // Range -10 to +10

	processor := &UltraFastActivationProcessor{
		numWorkers:   runtime.NumCPU(),
		sigmoidTable: make([]float64, tableSize),
		tanhTable:    make([]float64, tableSize),
		tableSize:    tableSize,
		tableScale:   tableScale,
	}

	// Pre-compute lookup tables for ultra-fast access
	for i := 0; i < tableSize; i++ {
		x := (float64(i)/float64(tableSize-1) - 0.5) * 2.0 * tableScale
		processor.sigmoidTable[i] = 1.0 / (1.0 + math.Exp(-x))
		processor.tanhTable[i] = math.Tanh(x)
	}

	return processor
}

// UltraFastReLU applies ReLU using bitwise operations for maximum speed.
func (ufap *UltraFastActivationProcessor) UltraFastReLU(input, output []float64) {
	length := len(input)

	// Use parallel processing for large arrays
	if length >= 10000 {
		chunkSize := length / ufap.numWorkers
		var wg sync.WaitGroup

		for worker := 0; worker < ufap.numWorkers; worker++ {
			start := worker * chunkSize
			end := start + chunkSize
			if worker == ufap.numWorkers-1 {
				end = length
			}

			wg.Add(1)
			go func(s, e int) {
				defer wg.Done()

				// Ultra-fast ReLU with 8-element unrolling
				i := s
				for ; i < e-7; i += 8 {
					// Bitwise operations for maximum speed
					if input[i] > 0 {
						output[i] = input[i]
					} else {
						output[i] = 0
					}
					if input[i+1] > 0 {
						output[i+1] = input[i+1]
					} else {
						output[i+1] = 0
					}
					if input[i+2] > 0 {
						output[i+2] = input[i+2]
					} else {
						output[i+2] = 0
					}
					if input[i+3] > 0 {
						output[i+3] = input[i+3]
					} else {
						output[i+3] = 0
					}
					if input[i+4] > 0 {
						output[i+4] = input[i+4]
					} else {
						output[i+4] = 0
					}
					if input[i+5] > 0 {
						output[i+5] = input[i+5]
					} else {
						output[i+5] = 0
					}
					if input[i+6] > 0 {
						output[i+6] = input[i+6]
					} else {
						output[i+6] = 0
					}
					if input[i+7] > 0 {
						output[i+7] = input[i+7]
					} else {
						output[i+7] = 0
					}
				}

				// Handle remaining elements
				for ; i < e; i++ {
					if input[i] > 0 {
						output[i] = input[i]
					} else {
						output[i] = 0
					}
				}
			}(start, end)
		}

		wg.Wait()
	} else {
		// Sequential for small arrays
		for i := 0; i < length; i++ {
			if input[i] > 0 {
				output[i] = input[i]
			} else {
				output[i] = 0
			}
		}
	}
}

// UltraFastSigmoid applies sigmoid using pre-computed lookup table.
func (ufap *UltraFastActivationProcessor) UltraFastSigmoid(input, output []float64) {
	length := len(input)

	if length >= 10000 {
		chunkSize := length / ufap.numWorkers
		var wg sync.WaitGroup

		for worker := 0; worker < ufap.numWorkers; worker++ {
			start := worker * chunkSize
			end := start + chunkSize
			if worker == ufap.numWorkers-1 {
				end = length
			}

			wg.Add(1)
			go func(s, e int) {
				defer wg.Done()

				for i := s; i < e; i++ {
					x := input[i]

					// Ultra-fast bounds checking
					if x > ufap.tableScale {
						output[i] = 1.0
					} else if x < -ufap.tableScale {
						output[i] = 0.0
					} else {
						// Ultra-fast lookup table access
						index := int((x/ufap.tableScale + 1.0) * 0.5 * float64(ufap.tableSize-1))
						if index < 0 {
							index = 0
						}
						if index >= ufap.tableSize {
							index = ufap.tableSize - 1
						}
						output[i] = ufap.sigmoidTable[index]
					}
				}
			}(start, end)
		}

		wg.Wait()
	} else {
		// Sequential for small arrays
		for i := 0; i < length; i++ {
			x := input[i]
			if x > ufap.tableScale {
				output[i] = 1.0
			} else if x < -ufap.tableScale {
				output[i] = 0.0
			} else {
				index := int((x/ufap.tableScale + 1.0) * 0.5 * float64(ufap.tableSize-1))
				if index < 0 {
					index = 0
				}
				if index >= ufap.tableSize {
					index = ufap.tableSize - 1
				}
				output[i] = ufap.sigmoidTable[index]
			}
		}
	}
}

// UltraFastTanh applies tanh using pre-computed lookup table.
func (ufap *UltraFastActivationProcessor) UltraFastTanh(input, output []float64) {
	length := len(input)

	if length >= 10000 {
		chunkSize := length / ufap.numWorkers
		var wg sync.WaitGroup

		for worker := 0; worker < ufap.numWorkers; worker++ {
			start := worker * chunkSize
			end := start + chunkSize
			if worker == ufap.numWorkers-1 {
				end = length
			}

			wg.Add(1)
			go func(s, e int) {
				defer wg.Done()

				for i := s; i < e; i++ {
					x := input[i]

					// Ultra-fast bounds checking
					if x > ufap.tableScale {
						output[i] = 1.0
					} else if x < -ufap.tableScale {
						output[i] = -1.0
					} else {
						// Ultra-fast lookup table access
						index := int((x/ufap.tableScale + 1.0) * 0.5 * float64(ufap.tableSize-1))
						if index < 0 {
							index = 0
						}
						if index >= ufap.tableSize {
							index = ufap.tableSize - 1
						}
						output[i] = ufap.tanhTable[index]
					}
				}
			}(start, end)
		}

		wg.Wait()
	} else {
		// Sequential for small arrays
		for i := 0; i < length; i++ {
			x := input[i]
			if x > ufap.tableScale {
				output[i] = 1.0
			} else if x < -ufap.tableScale {
				output[i] = -1.0
			} else {
				index := int((x/ufap.tableScale + 1.0) * 0.5 * float64(ufap.tableSize-1))
				if index < 0 {
					index = 0
				}
				if index >= ufap.tableSize {
					index = ufap.tableSize - 1
				}
				output[i] = ufap.tanhTable[index]
			}
		}
	}
}

// ParallelActivationProcessor provides optimized activation function processing.
type ParallelActivationProcessor struct {
	numWorkers int
}

// NewParallelActivationProcessor creates a new parallel activation processor.
func NewParallelActivationProcessor() *ParallelActivationProcessor {
	return &ParallelActivationProcessor{
		numWorkers: runtime.NumCPU(),
	}
}

// ProcessReLU applies ReLU activation in parallel.
func (pap *ParallelActivationProcessor) ProcessReLU(input, output []float64) {
	chunkSize := len(input) / pap.numWorkers
	if chunkSize == 0 {
		chunkSize = 1
	}

	var wg sync.WaitGroup

	for i := 0; i < pap.numWorkers; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if i == pap.numWorkers-1 {
			end = len(input)
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for j := s; j < e; j++ {
				if input[j] > 0 {
					output[j] = input[j]
				} else {
					output[j] = 0
				}
			}
		}(start, end)
	}

	wg.Wait()
}

// ProcessSigmoid applies sigmoid activation in parallel with overflow protection.
func (pap *ParallelActivationProcessor) ProcessSigmoid(input, output []float64) {
	chunkSize := len(input) / pap.numWorkers
	if chunkSize == 0 {
		chunkSize = 1
	}

	var wg sync.WaitGroup

	for i := 0; i < pap.numWorkers; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if i == pap.numWorkers-1 {
			end = len(input)
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for j := s; j < e; j++ {
				x := input[j]
				if x > 250 {
					output[j] = 1.0
				} else if x < -250 {
					output[j] = 0.0
				} else {
					output[j] = 1.0 / (1.0 + math.Exp(-x))
				}
			}
		}(start, end)
	}

	wg.Wait()
}

// ProcessTanh applies tanh activation in parallel.
func (pap *ParallelActivationProcessor) ProcessTanh(input, output []float64) {
	chunkSize := len(input) / pap.numWorkers
	if chunkSize == 0 {
		chunkSize = 1
	}

	var wg sync.WaitGroup

	for i := 0; i < pap.numWorkers; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if i == pap.numWorkers-1 {
			end = len(input)
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for j := s; j < e; j++ {
				output[j] = math.Tanh(input[j])
			}
		}(start, end)
	}

	wg.Wait()
}

// UltraFastVectorOps provides maximum speed vectorized operations using unsafe pointers.
type UltraFastVectorOps struct {
	numWorkers int
}

// NewUltraFastVectorOps creates ultra-fast vectorized operations.
func NewUltraFastVectorOps() *UltraFastVectorOps {
	return &UltraFastVectorOps{
		numWorkers: runtime.NumCPU(),
	}
}

// UnsafeVectorAdd performs ultra-fast vector addition using unsafe pointers and 8-element unrolling.
func (ufvo *UltraFastVectorOps) UnsafeVectorAdd(a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("vector lengths don't match")
	}

	result := make([]float64, len(a))

	// Use unsafe pointers for maximum speed
	aPtr := (*float64)(unsafe.Pointer(&a[0]))
	bPtr := (*float64)(unsafe.Pointer(&b[0]))
	resultPtr := (*float64)(unsafe.Pointer(&result[0]))

	length := len(a)

	// 8-element loop unrolling for maximum performance
	i := 0
	for ; i < length-7; i += 8 {
		*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(resultPtr)) + uintptr(i*8))) =
			*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr(i*8))) +
				*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr(i*8)))

		*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(resultPtr)) + uintptr((i+1)*8))) =
			*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr((i+1)*8))) +
				*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr((i+1)*8)))

		*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(resultPtr)) + uintptr((i+2)*8))) =
			*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr((i+2)*8))) +
				*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr((i+2)*8)))

		*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(resultPtr)) + uintptr((i+3)*8))) =
			*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr((i+3)*8))) +
				*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr((i+3)*8)))

		*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(resultPtr)) + uintptr((i+4)*8))) =
			*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr((i+4)*8))) +
				*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr((i+4)*8)))

		*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(resultPtr)) + uintptr((i+5)*8))) =
			*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr((i+5)*8))) +
				*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr((i+5)*8)))

		*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(resultPtr)) + uintptr((i+6)*8))) =
			*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr((i+6)*8))) +
				*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr((i+6)*8)))

		*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(resultPtr)) + uintptr((i+7)*8))) =
			*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(aPtr)) + uintptr((i+7)*8))) +
				*(*float64)(unsafe.Pointer(uintptr(unsafe.Pointer(bPtr)) + uintptr((i+7)*8)))
	}

	// Handle remaining elements
	for ; i < length; i++ {
		result[i] = a[i] + b[i]
	}

	return result
}

// ParallelUltraFastAdd performs parallel ultra-fast vector addition.
func (ufvo *UltraFastVectorOps) ParallelUltraFastAdd(a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("vector lengths don't match")
	}

	result := make([]float64, len(a))
	length := len(a)

	if length < 1000 {
		// Use sequential for small arrays
		return ufvo.UnsafeVectorAdd(a, b)
	}

	chunkSize := length / ufvo.numWorkers
	var wg sync.WaitGroup

	for worker := 0; worker < ufvo.numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if worker == ufvo.numWorkers-1 {
			end = length
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()

			// Ultra-fast 8-element unrolled loop
			i := s
			for ; i < e-7; i += 8 {
				result[i] = a[i] + b[i]
				result[i+1] = a[i+1] + b[i+1]
				result[i+2] = a[i+2] + b[i+2]
				result[i+3] = a[i+3] + b[i+3]
				result[i+4] = a[i+4] + b[i+4]
				result[i+5] = a[i+5] + b[i+5]
				result[i+6] = a[i+6] + b[i+6]
				result[i+7] = a[i+7] + b[i+7]
			}

			// Handle remaining elements
			for ; i < e; i++ {
				result[i] = a[i] + b[i]
			}
		}(start, end)
	}

	wg.Wait()
	return result
}

// VectorizedOperations provides SIMD-like operations for better performance.
type VectorizedOperations struct {
	numWorkers int
}

// NewVectorizedOperations creates a new vectorized operations processor.
func NewVectorizedOperations() *VectorizedOperations {
	return &VectorizedOperations{
		numWorkers: runtime.NumCPU(),
	}
}

// VectorAdd performs optimized vector addition with loop unrolling.
func (vo *VectorizedOperations) VectorAdd(a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("vector lengths don't match")
	}

	result := make([]float64, len(a))

	// Unroll loop for better performance
	i := 0
	for ; i < len(a)-3; i += 4 {
		result[i] = a[i] + b[i]
		result[i+1] = a[i+1] + b[i+1]
		result[i+2] = a[i+2] + b[i+2]
		result[i+3] = a[i+3] + b[i+3]
	}

	// Handle remaining elements
	for ; i < len(a); i++ {
		result[i] = a[i] + b[i]
	}

	return result
}

// VectorMul performs optimized vector multiplication with loop unrolling.
func (vo *VectorizedOperations) VectorMul(a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("vector lengths don't match")
	}

	result := make([]float64, len(a))

	// Unroll loop for better performance
	i := 0
	for ; i < len(a)-3; i += 4 {
		result[i] = a[i] * b[i]
		result[i+1] = a[i+1] * b[i+1]
		result[i+2] = a[i+2] * b[i+2]
		result[i+3] = a[i+3] * b[i+3]
	}

	// Handle remaining elements
	for ; i < len(a); i++ {
		result[i] = a[i] * b[i]
	}

	return result
}

// BatchProcessor provides efficient batch processing capabilities.
type BatchProcessor struct {
	numWorkers int
}

// NewBatchProcessor creates a new batch processor.
func NewBatchProcessor() *BatchProcessor {
	return &BatchProcessor{
		numWorkers: runtime.NumCPU(),
	}
}

// ProcessBatches processes multiple tensors in parallel.
func (bp *BatchProcessor) ProcessBatches(inputs []Tensor, processor func(Tensor) Tensor) []Tensor {
	if len(inputs) == 0 {
		return nil
	}

	results := make([]Tensor, len(inputs))

	// Use parallel processing for multiple inputs
	if len(inputs) >= bp.numWorkers {
		var wg sync.WaitGroup

		for i, input := range inputs {
			wg.Add(1)
			go func(idx int, tensor Tensor) {
				defer wg.Done()
				results[idx] = processor(tensor)
			}(i, input)
		}

		wg.Wait()
	} else {
		// Sequential processing for small batches
		for i, input := range inputs {
			results[i] = processor(input)
		}
	}

	return results
}

// Helper functions
func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// Global instances for optimized operations
var (
	globalHighPerfProcessor   = NewHighPerformanceProcessor(64)
	globalUltraFastProcessor  = NewUltraFastProcessor(64)
	globalActivationProcessor = NewParallelActivationProcessor()
	globalUltraFastActivation = NewUltraFastActivationProcessor()
	globalVectorizedOps       = NewVectorizedOperations()
	globalUltraFastVectorOps  = NewUltraFastVectorOps()
	globalBatchProcessor      = NewBatchProcessor()
)

// GetHighPerformanceProcessor returns the global high-performance processor.
func GetHighPerformanceProcessor() *HighPerformanceProcessor {
	return globalHighPerfProcessor
}

// GetUltraFastProcessor returns the global ultra-fast processor (maximum speed).
func GetUltraFastProcessor() *UltraFastProcessor {
	return globalUltraFastProcessor
}

// GetParallelActivationProcessor returns the global parallel activation processor.
func GetParallelActivationProcessor() *ParallelActivationProcessor {
	return globalActivationProcessor
}

// GetUltraFastActivationProcessor returns the global ultra-fast activation processor.
func GetUltraFastActivationProcessor() *UltraFastActivationProcessor {
	return globalUltraFastActivation
}

// GetVectorizedOperations returns the global vectorized operations processor.
func GetVectorizedOperations() *VectorizedOperations {
	return globalVectorizedOps
}

// GetUltraFastVectorOps returns the global ultra-fast vector operations processor.
func GetUltraFastVectorOps() *UltraFastVectorOps {
	return globalUltraFastVectorOps
}

// GetBatchProcessor returns the global batch processor.
func GetBatchProcessor() *BatchProcessor {
	return globalBatchProcessor
}
