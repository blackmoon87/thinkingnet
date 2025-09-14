package models

import (
	"fmt"
	"math/rand"
	"sync"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// BatchProcessor handles batch processing for training and inference.
type BatchProcessor struct {
	batchSize int
	shuffle   bool
	seed      int64
}

// NewBatchProcessor creates a new batch processor.
func NewBatchProcessor(batchSize int, shuffle bool, seed int64) *BatchProcessor {
	return &BatchProcessor{
		batchSize: batchSize,
		shuffle:   shuffle,
		seed:      seed,
	}
}

// BatchIterator provides iteration over batches of data.
type BatchIterator struct {
	X, y       core.Tensor
	batchSize  int
	indices    []int
	position   int
	numBatches int
}

// NewBatchIterator creates a new batch iterator.
func NewBatchIterator(X, y core.Tensor, batchSize int, shuffle bool, seed int64) *BatchIterator {
	numSamples, _ := X.Dims()
	numBatches := (numSamples + batchSize - 1) / batchSize

	indices := make([]int, numSamples)
	for i := range indices {
		indices[i] = i
	}

	if shuffle {
		rand.Seed(seed)
		rand.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}

	return &BatchIterator{
		X:          X,
		y:          y,
		batchSize:  batchSize,
		indices:    indices,
		position:   0,
		numBatches: numBatches,
	}
}

// HasNext returns true if there are more batches.
func (bi *BatchIterator) HasNext() bool {
	return bi.position < len(bi.indices)
}

// Next returns the next batch of data.
func (bi *BatchIterator) Next() (core.Tensor, core.Tensor, error) {
	if !bi.HasNext() {
		return nil, nil, fmt.Errorf("no more batches available")
	}

	// Calculate batch boundaries
	start := bi.position
	end := start + bi.batchSize
	if end > len(bi.indices) {
		end = len(bi.indices)
	}

	// Get batch indices
	batchIndices := bi.indices[start:end]
	batchSize := len(batchIndices)

	// Create batch tensors
	_, xCols := bi.X.Dims()
	_, yCols := bi.y.Dims()

	XBatch := bi.X.Copy().Reshape(batchSize, xCols)
	yBatch := bi.y.Copy().Reshape(batchSize, yCols)

	// Fill batch data
	for i, idx := range batchIndices {
		for j := 0; j < xCols; j++ {
			XBatch.Set(i, j, bi.X.At(idx, j))
		}
		for j := 0; j < yCols; j++ {
			yBatch.Set(i, j, bi.y.At(idx, j))
		}
	}

	bi.position = end
	return XBatch, yBatch, nil
}

// Reset resets the iterator to the beginning.
func (bi *BatchIterator) Reset() {
	bi.position = 0
}

// NumBatches returns the total number of batches.
func (bi *BatchIterator) NumBatches() int {
	return bi.numBatches
}

// BatchSize returns the batch size.
func (bi *BatchIterator) BatchSize() int {
	return bi.batchSize
}

// DataLoader provides efficient data loading with batching and shuffling.
type DataLoader struct {
	X, y       core.Tensor
	batchSize  int
	shuffle    bool
	seed       int64
	iterator   *BatchIterator
	numSamples int
	numBatches int
}

// NewDataLoader creates a new data loader.
func NewDataLoader(X, y core.Tensor, batchSize int, shuffle bool, seed int64) (*DataLoader, error) {
	if X == nil || y == nil {
		return nil, fmt.Errorf("X and y cannot be nil")
	}

	xRows, _ := X.Dims()
	yRows, _ := y.Dims()
	if xRows != yRows {
		return nil, fmt.Errorf("X and y must have the same number of samples")
	}

	numBatches := (xRows + batchSize - 1) / batchSize

	dl := &DataLoader{
		X:          X,
		y:          y,
		batchSize:  batchSize,
		shuffle:    shuffle,
		seed:       seed,
		numSamples: xRows,
		numBatches: numBatches,
	}

	dl.Reset()
	return dl, nil
}

// Reset creates a new iterator for the data loader.
func (dl *DataLoader) Reset() {
	dl.iterator = NewBatchIterator(dl.X, dl.y, dl.batchSize, dl.shuffle, dl.seed)
}

// HasNext returns true if there are more batches.
func (dl *DataLoader) HasNext() bool {
	return dl.iterator.HasNext()
}

// Next returns the next batch.
func (dl *DataLoader) Next() (core.Tensor, core.Tensor, error) {
	return dl.iterator.Next()
}

// NumBatches returns the total number of batches.
func (dl *DataLoader) NumBatches() int {
	return dl.numBatches
}

// NumSamples returns the total number of samples.
func (dl *DataLoader) NumSamples() int {
	return dl.numSamples
}

// BatchSize returns the batch size.
func (dl *DataLoader) BatchSize() int {
	return dl.batchSize
}

// MemoryPool manages tensor memory for efficient reuse.
type MemoryPool struct {
	tensors map[string][]core.Tensor
	maxSize int
}

// NewMemoryPool creates a new memory pool.
func NewMemoryPool(maxSize int) *MemoryPool {
	return &MemoryPool{
		tensors: make(map[string][]core.Tensor),
		maxSize: maxSize,
	}
}

// Get retrieves a tensor from the pool or creates a new one.
func (mp *MemoryPool) Get(rows, cols int) core.Tensor {
	key := fmt.Sprintf("%dx%d", rows, cols)

	if tensors, exists := mp.tensors[key]; exists && len(tensors) > 0 {
		// Reuse existing tensor
		tensor := tensors[len(tensors)-1]
		mp.tensors[key] = tensors[:len(tensors)-1]
		tensor.Zero() // Clear the tensor
		return tensor
	}

	// Create new tensor if pool is empty
	// Note: This would need to be implemented based on the actual Tensor implementation
	// For now, returning nil as placeholder
	return nil
}

// Put returns a tensor to the pool for reuse.
func (mp *MemoryPool) Put(tensor core.Tensor) {
	if tensor == nil {
		return
	}

	rows, cols := tensor.Dims()
	key := fmt.Sprintf("%dx%d", rows, cols)

	if tensors, exists := mp.tensors[key]; exists {
		if len(tensors) < mp.maxSize {
			mp.tensors[key] = append(tensors, tensor)
		}
	} else {
		mp.tensors[key] = []core.Tensor{tensor}
	}
}

// Clear clears all tensors from the pool.
func (mp *MemoryPool) Clear() {
	for key := range mp.tensors {
		delete(mp.tensors, key)
	}
}

// Size returns the current size of the pool.
func (mp *MemoryPool) Size() int {
	total := 0
	for _, tensors := range mp.tensors {
		total += len(tensors)
	}
	return total
}

// BatchTrainer handles batch-based training with memory management.
type BatchTrainer struct {
	model      *Sequential
	memoryPool *MemoryPool
	config     core.TrainingConfig
	parallel   bool
}

// NewBatchTrainer creates a new batch trainer.
func NewBatchTrainer(model *Sequential, config core.TrainingConfig) *BatchTrainer {
	return &BatchTrainer{
		model:      model,
		memoryPool: NewMemoryPool(100), // Pool size of 100 tensors
		config:     config,
		parallel:   true, // Enable parallel processing by default
	}
}

// NewBatchTrainerWithOptions creates a new batch trainer with options.
func NewBatchTrainerWithOptions(model *Sequential, config core.TrainingConfig, parallel bool, poolSize int) *BatchTrainer {
	return &BatchTrainer{
		model:      model,
		memoryPool: NewMemoryPool(poolSize),
		config:     config,
		parallel:   parallel,
	}
}

// TrainBatch trains the model on a single batch.
func (bt *BatchTrainer) TrainBatch(XBatch, yBatch core.Tensor) (float64, map[string]float64, error) {
	if bt.model == nil || !bt.model.IsCompiled() {
		return 0, nil, fmt.Errorf("model must be compiled before training")
	}

	// Forward pass
	yPred, err := bt.model.Forward(XBatch)
	if err != nil {
		return 0, nil, err
	}

	// Compute loss
	loss := bt.model.loss.Compute(yBatch, yPred)

	// Compute metrics
	metrics := bt.model.computeMetrics(yBatch, yPred, bt.config.Metrics)

	// Backward pass
	bt.model.Backward(bt.model.loss, yBatch, yPred)

	// Update parameters
	bt.model.updateParameters()

	return loss, metrics, nil
}

// ValidateBatch validates the model on a single batch.
func (bt *BatchTrainer) ValidateBatch(XBatch, yBatch core.Tensor) (float64, map[string]float64, error) {
	if bt.model == nil || !bt.model.IsCompiled() {
		return 0, nil, fmt.Errorf("model must be compiled before validation")
	}

	// Forward pass (no gradient computation)
	yPred, err := bt.model.Forward(XBatch)
	if err != nil {
		return 0, nil, err
	}

	// Compute loss
	loss := bt.model.loss.Compute(yBatch, yPred)

	// Compute metrics
	metrics := bt.model.computeMetrics(yBatch, yPred, bt.config.Metrics)

	return loss, metrics, nil
}

// GetMemoryPool returns the memory pool.
func (bt *BatchTrainer) GetMemoryPool() *MemoryPool {
	return bt.memoryPool
}

// ParallelBatchTrainer handles parallel batch processing for improved performance.
type ParallelBatchTrainer struct {
	*BatchTrainer
	numWorkers int
}

// NewParallelBatchTrainer creates a new parallel batch trainer.
func NewParallelBatchTrainer(model *Sequential, config core.TrainingConfig, numWorkers int) *ParallelBatchTrainer {
	if numWorkers <= 0 {
		numWorkers = 4 // Default to 4 workers
	}

	return &ParallelBatchTrainer{
		BatchTrainer: NewBatchTrainer(model, config),
		numWorkers:   numWorkers,
	}
}

// TrainBatchesParallel trains multiple batches in parallel.
func (pbt *ParallelBatchTrainer) TrainBatchesParallel(batches []core.Tensor, targets []core.Tensor) ([]float64, []map[string]float64, error) {
	if len(batches) != len(targets) {
		return nil, nil, fmt.Errorf("number of batches and targets must match")
	}

	if !pbt.parallel || len(batches) < 2 {
		// Sequential processing
		losses := make([]float64, len(batches))
		metrics := make([]map[string]float64, len(batches))

		for i := range batches {
			loss, metric, err := pbt.TrainBatch(batches[i], targets[i])
			if err != nil {
				return nil, nil, err
			}
			losses[i] = loss
			metrics[i] = metric
		}

		return losses, metrics, nil
	}

	// Parallel processing
	losses := make([]float64, len(batches))
	metrics := make([]map[string]float64, len(batches))

	// Process batches in parallel using core.ParallelBatchProcess
	processor := func(batch core.Tensor) (core.Tensor, error) {
		// This is a simplified version - in practice, you'd need to handle
		// the training logic more carefully for parallel execution
		return batch, nil
	}

	_, err := core.ParallelBatchProcess(batches, processor)
	if err != nil {
		return nil, nil, err
	}

	// For now, fall back to sequential training as parallel training
	// requires more careful synchronization of model parameters
	for i := range batches {
		loss, metric, err := pbt.TrainBatch(batches[i], targets[i])
		if err != nil {
			return nil, nil, err
		}
		losses[i] = loss
		metrics[i] = metric
	}

	return losses, metrics, nil
}

// PredictBatchesParallel performs parallel prediction on multiple batches.
func (pbt *ParallelBatchTrainer) PredictBatchesParallel(batches []core.Tensor) ([]core.Tensor, error) {
	if !pbt.parallel || len(batches) < 2 {
		// Sequential processing
		results := make([]core.Tensor, len(batches))
		for i, batch := range batches {
			result, _ := pbt.model.Predict(batch)
			results[i] = result
		}
		return results, nil
	}

	// Parallel processing
	processor := func(batch core.Tensor) (core.Tensor, error) {
		result, err := pbt.model.Predict(batch)
		return result, err
	}

	return core.ParallelBatchProcess(batches, processor)
}

// SetParallel enables or disables parallel processing.
func (pbt *ParallelBatchTrainer) SetParallel(parallel bool) {
	pbt.parallel = parallel
}

// OptimizedDataLoader provides memory-efficient data loading with prefetching.
type OptimizedDataLoader struct {
	*DataLoader
	prefetchSize int
	prefetchChan chan batchPair
	stopChan     chan struct{}
	started      bool
	mutex        sync.RWMutex
}

type batchPair struct {
	X   core.Tensor
	y   core.Tensor
	err error
}

// NewOptimizedDataLoader creates a new optimized data loader with prefetching.
func NewOptimizedDataLoader(X, y core.Tensor, batchSize int, shuffle bool, seed int64, prefetchSize int) (*OptimizedDataLoader, error) {
	dl, err := NewDataLoader(X, y, batchSize, shuffle, seed)
	if err != nil {
		return nil, err
	}

	if prefetchSize <= 0 {
		prefetchSize = 2 // Default prefetch size
	}

	return &OptimizedDataLoader{
		DataLoader:   dl,
		prefetchSize: prefetchSize,
		prefetchChan: make(chan batchPair, prefetchSize),
		stopChan:     make(chan struct{}),
	}, nil
}

// StartPrefetching starts the prefetching goroutine.
func (odl *OptimizedDataLoader) StartPrefetching() {
	odl.mutex.Lock()
	defer odl.mutex.Unlock()

	if odl.started {
		return
	}

	odl.started = true
	go odl.prefetchWorker()
}

// StopPrefetching stops the prefetching goroutine.
func (odl *OptimizedDataLoader) StopPrefetching() {
	odl.mutex.Lock()
	defer odl.mutex.Unlock()

	if !odl.started {
		return
	}

	close(odl.stopChan)
	odl.started = false
}

// prefetchWorker runs in a separate goroutine to prefetch batches.
func (odl *OptimizedDataLoader) prefetchWorker() {
	for {
		select {
		case <-odl.stopChan:
			return
		default:
			if odl.DataLoader.HasNext() {
				X, y, err := odl.DataLoader.Next()

				select {
				case odl.prefetchChan <- batchPair{X: X, y: y, err: err}:
				case <-odl.stopChan:
					return
				}
			} else {
				// No more batches, wait a bit before checking again
				select {
				case <-odl.stopChan:
					return
				default:
					// Continue loop
				}
			}
		}
	}
}

// NextOptimized returns the next prefetched batch.
func (odl *OptimizedDataLoader) NextOptimized() (core.Tensor, core.Tensor, error) {
	if !odl.started {
		// Fallback to regular Next if prefetching not started
		return odl.DataLoader.Next()
	}

	select {
	case batch := <-odl.prefetchChan:
		return batch.X, batch.y, batch.err
	default:
		// No prefetched batch available, use regular method
		return odl.DataLoader.Next()
	}
}

// MemoryOptimizedTensor provides memory-efficient tensor operations.
type MemoryOptimizedTensor struct {
	core.Tensor
	pool *MemoryPool
}

// NewMemoryOptimizedTensor creates a new memory-optimized tensor.
func NewMemoryOptimizedTensor(tensor core.Tensor, pool *MemoryPool) *MemoryOptimizedTensor {
	return &MemoryOptimizedTensor{
		Tensor: tensor,
		pool:   pool,
	}
}

// Release returns the tensor to the memory pool.
func (mot *MemoryOptimizedTensor) Release() {
	if mot.pool != nil {
		mot.pool.Put(mot.Tensor)
	}
}

// Copy creates a copy using the memory pool.
func (mot *MemoryOptimizedTensor) Copy() core.Tensor {
	rows, cols := mot.Dims()
	if mot.pool != nil {
		if pooledTensor := mot.pool.Get(rows, cols); pooledTensor != nil {
			// Copy data to pooled tensor
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					pooledTensor.Set(i, j, mot.At(i, j))
				}
			}
			return NewMemoryOptimizedTensor(pooledTensor, mot.pool)
		}
	}

	// Fallback to regular copy
	return mot.Tensor.Copy()
}
