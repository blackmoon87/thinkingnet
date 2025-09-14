package core

import (
	"context"
	"runtime"
	"sync"
)

// WorkerPool manages a pool of workers for parallel processing.
type WorkerPool struct {
	numWorkers int
	jobQueue   chan Job
	wg         sync.WaitGroup
	ctx        context.Context
	cancel     context.CancelFunc
	started    bool
	mutex      sync.RWMutex
}

// Job represents a unit of work to be processed.
type Job interface {
	Execute() error
}

// TensorJob represents a tensor operation job.
type TensorJob struct {
	Operation func() error
	Result    chan error
}

// Execute runs the tensor operation.
func (tj *TensorJob) Execute() error {
	err := tj.Operation()
	if tj.Result != nil {
		tj.Result <- err
	}
	return err
}

// NewWorkerPool creates a new worker pool.
func NewWorkerPool(numWorkers int) *WorkerPool {
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &WorkerPool{
		numWorkers: numWorkers,
		jobQueue:   make(chan Job, numWorkers*2), // Buffer for jobs
		ctx:        ctx,
		cancel:     cancel,
	}
}

// Start starts the worker pool.
func (wp *WorkerPool) Start() {
	wp.mutex.Lock()
	defer wp.mutex.Unlock()

	if wp.started {
		return
	}

	wp.started = true

	for i := 0; i < wp.numWorkers; i++ {
		wp.wg.Add(1)
		go wp.worker(i)
	}
}

// Stop stops the worker pool.
func (wp *WorkerPool) Stop() {
	wp.mutex.Lock()
	defer wp.mutex.Unlock()

	if !wp.started {
		return
	}

	wp.cancel()
	close(wp.jobQueue)
	wp.wg.Wait()
	wp.started = false
}

// Submit submits a job to the worker pool.
func (wp *WorkerPool) Submit(job Job) error {
	wp.mutex.RLock()
	defer wp.mutex.RUnlock()

	if !wp.started {
		return NewError(ErrInvalidInput, "worker pool not started")
	}

	select {
	case wp.jobQueue <- job:
		return nil
	case <-wp.ctx.Done():
		return NewError(ErrInvalidInput, "worker pool stopped")
	}
}

// worker processes jobs from the queue.
func (wp *WorkerPool) worker(id int) {
	defer wp.wg.Done()

	for {
		select {
		case job, ok := <-wp.jobQueue:
			if !ok {
				return // Channel closed
			}
			_ = job.Execute() // Execute job, ignore error for now
		case <-wp.ctx.Done():
			return
		}
	}
}

// ParallelConfig holds configuration for parallel operations.
type ParallelConfig struct {
	Enabled    bool
	NumWorkers int
	ChunkSize  int
	MinSize    int // Minimum size to use parallel processing
}

// DefaultParallelConfig returns default parallel configuration.
func DefaultParallelConfig() ParallelConfig {
	return ParallelConfig{
		Enabled:    true,
		NumWorkers: runtime.NumCPU(),
		ChunkSize:  1000,
		MinSize:    100,
	}
}

// Global worker pool
var (
	globalWorkerPool     *WorkerPool
	globalParallelConfig ParallelConfig
	poolMutex            sync.RWMutex
)

func init() {
	globalParallelConfig = DefaultParallelConfig()
	globalWorkerPool = NewWorkerPool(globalParallelConfig.NumWorkers)
	globalWorkerPool.Start()
}

// SetParallelConfig sets the global parallel configuration.
func SetParallelConfig(config ParallelConfig) {
	poolMutex.Lock()
	defer poolMutex.Unlock()

	globalParallelConfig = config

	// Restart worker pool with new configuration
	if globalWorkerPool != nil {
		globalWorkerPool.Stop()
	}

	if config.Enabled {
		globalWorkerPool = NewWorkerPool(config.NumWorkers)
		globalWorkerPool.Start()
	}
}

// GetParallelConfig returns the current parallel configuration.
func GetParallelConfig() ParallelConfig {
	poolMutex.RLock()
	defer poolMutex.RUnlock()
	return globalParallelConfig
}

// ParallelExecute executes operations in parallel if conditions are met.
func ParallelExecute(operations []func() error) error {
	config := GetParallelConfig()

	if !config.Enabled || len(operations) < 2 || globalWorkerPool == nil {
		// Execute sequentially
		for _, op := range operations {
			if err := op(); err != nil {
				return err
			}
		}
		return nil
	}

	// Execute in parallel
	errChan := make(chan error, len(operations))

	for _, op := range operations {
		job := &TensorJob{
			Operation: op,
			Result:    errChan,
		}

		if err := globalWorkerPool.Submit(job); err != nil {
			// Fallback to sequential execution
			for _, fallbackOp := range operations {
				if err := fallbackOp(); err != nil {
					return err
				}
			}
			return nil
		}
	}

	// Wait for all operations to complete
	for i := 0; i < len(operations); i++ {
		if err := <-errChan; err != nil {
			return err
		}
	}

	return nil
}

// ParallelTensorOperation applies an operation to tensor chunks in parallel.
func ParallelTensorOperation(tensor Tensor, operation func(startRow, endRow int) error) error {
	config := GetParallelConfig()
	rows, _ := tensor.Dims()

	if !config.Enabled || rows < config.MinSize || globalWorkerPool == nil {
		// Execute sequentially
		return operation(0, rows)
	}

	// Calculate chunk size
	chunkSize := config.ChunkSize
	if chunkSize > rows/config.NumWorkers {
		chunkSize = rows / config.NumWorkers
	}
	if chunkSize < 1 {
		chunkSize = 1
	}

	// Create operations for each chunk
	var operations []func() error
	for start := 0; start < rows; start += chunkSize {
		end := start + chunkSize
		if end > rows {
			end = rows
		}

		// Capture variables for closure
		startRow, endRow := start, end
		operations = append(operations, func() error {
			return operation(startRow, endRow)
		})
	}

	return ParallelExecute(operations)
}

// ParallelMatrixMultiply performs parallel matrix multiplication for large matrices.
func ParallelMatrixMultiply(a, b Tensor) (Tensor, error) {
	aRows, aCols := a.Dims()
	bRows, bCols := b.Dims()

	if aCols != bRows {
		return nil, NewError(ErrDimensionMismatch,
			"matrix dimensions incompatible for multiplication")
	}

	config := GetParallelConfig()

	// Use parallel processing for large matrices
	if !config.Enabled || aRows < config.MinSize {
		return a.Mul(b), nil
	}

	result := GetMatrix(aRows, bCols)
	defer func() {
		// Don't put result matrix back to pool as it's returned
	}()

	err := ParallelTensorOperation(a, func(startRow, endRow int) error {
		for i := startRow; i < endRow; i++ {
			for j := 0; j < bCols; j++ {
				var sum float64
				for k := 0; k < aCols; k++ {
					sum += a.At(i, k) * b.At(k, j)
				}
				result.Set(i, j, sum)
			}
		}
		return nil
	})

	if err != nil {
		PutMatrix(result)
		return nil, err
	}

	return NewTensor(result), nil
}

// ParallelElementWiseOperation performs element-wise operations in parallel.
func ParallelElementWiseOperation(a, b Tensor, operation func(float64, float64) float64) (Tensor, error) {
	if err := ValidateDimensions(a, b, "parallel_elementwise"); err != nil {
		return nil, err
	}

	rows, cols := a.Dims()
	config := GetParallelConfig()

	result := GetMatrix(rows, cols)

	if !config.Enabled || rows*cols < config.MinSize {
		// Sequential execution
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				result.Set(i, j, operation(a.At(i, j), b.At(i, j)))
			}
		}
		return NewTensor(result), nil
	}

	// Parallel execution
	err := ParallelTensorOperation(a, func(startRow, endRow int) error {
		for i := startRow; i < endRow; i++ {
			for j := 0; j < cols; j++ {
				result.Set(i, j, operation(a.At(i, j), b.At(i, j)))
			}
		}
		return nil
	})

	if err != nil {
		PutMatrix(result)
		return nil, err
	}

	return NewTensor(result), nil
}

// ParallelBatchProcess processes batches in parallel.
func ParallelBatchProcess(batches []Tensor, processor func(Tensor) (Tensor, error)) ([]Tensor, error) {
	config := GetParallelConfig()

	if !config.Enabled || len(batches) < 2 {
		// Sequential processing
		results := make([]Tensor, len(batches))
		for i, batch := range batches {
			result, err := processor(batch)
			if err != nil {
				return nil, err
			}
			results[i] = result
		}
		return results, nil
	}

	// Parallel processing
	results := make([]Tensor, len(batches))
	errChan := make(chan error, len(batches))

	for i, batch := range batches {
		idx := i
		batchCopy := batch

		job := &TensorJob{
			Operation: func() error {
				result, err := processor(batchCopy)
				if err != nil {
					return err
				}
				results[idx] = result
				return nil
			},
			Result: errChan,
		}

		if err := globalWorkerPool.Submit(job); err != nil {
			// Fallback to sequential
			for j := i; j < len(batches); j++ {
				result, err := processor(batches[j])
				if err != nil {
					return nil, err
				}
				results[j] = result
			}
			break
		}
	}

	// Wait for completion
	for i := 0; i < len(batches); i++ {
		if err := <-errChan; err != nil {
			return nil, err
		}
	}

	return results, nil
}

// Cleanup shuts down the global worker pool.
func Cleanup() {
	poolMutex.Lock()
	defer poolMutex.Unlock()

	if globalWorkerPool != nil {
		globalWorkerPool.Stop()
		globalWorkerPool = nil
	}
}
