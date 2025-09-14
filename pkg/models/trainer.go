package models

import (
	"fmt"
	"math"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// Trainer handles the training process with progress tracking and callbacks.
type Trainer struct {
	model     *Sequential
	optimizer core.Optimizer
	loss      core.Loss
	callbacks []core.Callback
	history   *core.History
}

// NewTrainer creates a new trainer.
func NewTrainer(model *Sequential, optimizer core.Optimizer, loss core.Loss) *Trainer {
	return &Trainer{
		model:     model,
		optimizer: optimizer,
		loss:      loss,
		callbacks: make([]core.Callback, 0),
		history:   &core.History{},
	}
}

// AddCallback adds a callback to the trainer.
func (t *Trainer) AddCallback(callback core.Callback) {
	t.callbacks = append(t.callbacks, callback)
}

// Train trains the model with the given configuration.
func (t *Trainer) Train(X, y core.Tensor, config core.TrainingConfig) (*core.History, error) {
	if t.model == nil {
		return nil, fmt.Errorf("model cannot be nil")
	}

	if !t.model.IsCompiled() {
		return nil, fmt.Errorf("model must be compiled before training")
	}

	// Initialize history
	t.initializeHistory(config)

	// Create data loaders
	trainLoader, valLoader, err := t.createDataLoaders(X, y, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create data loaders: %v", err)
	}

	// Training callbacks
	for _, callback := range t.callbacks {
		callback.OnTrainBegin()
	}

	startTime := time.Now()
	bestLoss := math.Inf(1)
	patience := 0

	// Main training loop
	for epoch := 0; epoch < config.Epochs; epoch++ {
		epochStart := time.Now()

		// Epoch begin callbacks
		for _, callback := range t.callbacks {
			callback.OnEpochBegin(epoch)
		}

		// Training phase
		trainLoss, trainMetrics, err := t.trainEpoch(trainLoader, config, epoch)
		if err != nil {
			return nil, fmt.Errorf("training epoch %d failed: %v", epoch, err)
		}

		// Validation phase
		var valLoss float64
		var valMetrics map[string]float64
		if valLoader != nil {
			valLoss, valMetrics, err = t.validateEpoch(valLoader, config)
			if err != nil {
				return nil, fmt.Errorf("validation epoch %d failed: %v", epoch, err)
			}
		}

		// Record metrics
		t.recordEpochMetrics(epoch+1, trainLoss, trainMetrics, valLoss, valMetrics)

		// Early stopping check
		if config.EarlyStopping.Enabled {
			currentLoss := trainLoss
			if valLoader != nil {
				currentLoss = valLoss
			}

			improved := false
			if config.EarlyStopping.Mode == "min" {
				improved = currentLoss < bestLoss-config.EarlyStopping.MinDelta
			} else {
				improved = currentLoss > bestLoss+config.EarlyStopping.MinDelta
			}

			if improved {
				bestLoss = currentLoss
				patience = 0
				t.history.BestEpoch = epoch + 1
				t.history.BestScore = bestLoss
			} else {
				patience++
				if patience >= config.EarlyStopping.Patience {
					if config.Verbose > 0 {
						fmt.Printf("Early stopping at epoch %d\n", epoch+1)
					}
					break
				}
			}
		}

		// Progress logging
		if config.Verbose > 0 && (epoch+1)%config.Verbose == 0 {
			elapsed := time.Since(epochStart)
			t.logProgress(epoch+1, config.Epochs, trainLoss, trainMetrics, valLoss, valMetrics, elapsed)
		}

		// Epoch end callbacks
		for _, callback := range t.callbacks {
			callback.OnEpochEnd(epoch, config.Epochs, trainLoss, trainMetrics)
		}
	}

	t.history.Duration = time.Since(startTime)

	// Training end callbacks
	for _, callback := range t.callbacks {
		callback.OnTrainEnd()
	}

	return t.history, nil
}

// initializeHistory initializes the training history.
func (t *Trainer) initializeHistory(config core.TrainingConfig) {
	t.history = &core.History{
		Epoch:      make([]int, 0),
		Loss:       make([]float64, 0),
		Metrics:    make(map[string][]float64),
		ValLoss:    make([]float64, 0),
		ValMetrics: make(map[string][]float64),
	}

	// Initialize metric slices
	for _, metric := range config.Metrics {
		t.history.Metrics[metric] = make([]float64, 0)
		t.history.ValMetrics[metric] = make([]float64, 0)
	}
}

// createDataLoaders creates training and validation data loaders.
func (t *Trainer) createDataLoaders(X, y core.Tensor, config core.TrainingConfig) (*DataLoader, *DataLoader, error) {
	var trainLoader, valLoader *DataLoader
	var err error

	if config.ValidationSplit > 0 {
		// Split data for validation
		split := t.splitData(X, y, config.ValidationSplit, config.Shuffle, config.Seed)

		trainLoader, err = NewDataLoader(split.XTrain, split.YTrain, config.BatchSize, config.Shuffle, config.Seed)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create training data loader: %v", err)
		}

		valLoader, err = NewDataLoader(split.XVal, split.YVal, config.BatchSize, false, config.Seed)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create validation data loader: %v", err)
		}
	} else {
		trainLoader, err = NewDataLoader(X, y, config.BatchSize, config.Shuffle, config.Seed)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create training data loader: %v", err)
		}
	}

	return trainLoader, valLoader, nil
}

// trainEpoch trains the model for one epoch.
func (t *Trainer) trainEpoch(trainLoader *DataLoader, config core.TrainingConfig, epoch int) (float64, map[string]float64, error) {
	trainLoader.Reset()

	totalLoss := 0.0
	totalMetrics := make(map[string]float64)
	batchCount := 0

	// Initialize metrics
	for _, metric := range config.Metrics {
		totalMetrics[metric] = 0.0
	}

	// Process all batches
	for trainLoader.HasNext() {
		// Batch begin callbacks
		for _, callback := range t.callbacks {
			callback.OnBatchBegin(batchCount)
		}

		// Get next batch
		XBatch, yBatch, err := trainLoader.Next()
		if err != nil {
			return 0, nil, fmt.Errorf("failed to get batch: %v", err)
		}

		// Train on batch
		batchLoss, batchMetrics, err := t.trainBatch(XBatch, yBatch, config.Metrics)
		if err != nil {
			return 0, nil, fmt.Errorf("failed to train batch: %v", err)
		}

		// Accumulate metrics
		totalLoss += batchLoss
		for metric, value := range batchMetrics {
			totalMetrics[metric] += value
		}
		batchCount++

		// Batch end callbacks
		for _, callback := range t.callbacks {
			callback.OnBatchEnd(batchCount, batchLoss)
		}
	}

	// Average metrics over batches
	avgLoss := totalLoss / float64(batchCount)
	avgMetrics := make(map[string]float64)
	for metric, total := range totalMetrics {
		avgMetrics[metric] = total / float64(batchCount)
	}

	return avgLoss, avgMetrics, nil
}

// validateEpoch validates the model for one epoch.
func (t *Trainer) validateEpoch(valLoader *DataLoader, config core.TrainingConfig) (float64, map[string]float64, error) {
	valLoader.Reset()

	totalLoss := 0.0
	totalMetrics := make(map[string]float64)
	batchCount := 0

	// Initialize metrics
	for _, metric := range config.Metrics {
		totalMetrics[metric] = 0.0
	}

	// Process all validation batches
	for valLoader.HasNext() {
		XBatch, yBatch, err := valLoader.Next()
		if err != nil {
			return 0, nil, fmt.Errorf("failed to get validation batch: %v", err)
		}

		// Validate on batch (no parameter updates)
		batchLoss, batchMetrics := t.validateBatch(XBatch, yBatch, config.Metrics)

		// Accumulate metrics
		totalLoss += batchLoss
		for metric, value := range batchMetrics {
			totalMetrics[metric] += value
		}
		batchCount++
	}

	// Average metrics over batches
	avgLoss := totalLoss / float64(batchCount)
	avgMetrics := make(map[string]float64)
	for metric, total := range totalMetrics {
		avgMetrics[metric] = total / float64(batchCount)
	}

	return avgLoss, avgMetrics, nil
}

// trainBatch trains the model on a single batch.
func (t *Trainer) trainBatch(XBatch, yBatch core.Tensor, metrics []string) (float64, map[string]float64, error) {
	// Forward pass
	yPred, err := t.model.Forward(XBatch)
	if err != nil {
		return 0, nil, err
	}

	// Compute loss
	loss := t.loss.Compute(yBatch, yPred)

	// Compute metrics
	batchMetrics := t.model.computeMetrics(yBatch, yPred, metrics)

	// Backward pass
	t.model.Backward(t.loss, yBatch, yPred)

	// Update parameters
	t.model.updateParameters()

	return loss, batchMetrics, nil
}

// validateBatch validates the model on a single batch.
func (t *Trainer) validateBatch(XBatch, yBatch core.Tensor, metrics []string) (float64, map[string]float64) {
	// Forward pass (no gradient computation)
	yPred, err := t.model.Forward(XBatch)
	if err != nil {
		return 0, nil
	}

	// Compute loss
	loss := t.loss.Compute(yBatch, yPred)

	// Compute metrics
	batchMetrics := t.model.computeMetrics(yBatch, yPred, metrics)

	return loss, batchMetrics
}

// recordEpochMetrics records metrics for the current epoch.
func (t *Trainer) recordEpochMetrics(epoch int, trainLoss float64, trainMetrics map[string]float64,
	valLoss float64, valMetrics map[string]float64) {

	t.history.Epoch = append(t.history.Epoch, epoch)
	t.history.Loss = append(t.history.Loss, trainLoss)

	// Record training metrics
	for metric, value := range trainMetrics {
		if _, exists := t.history.Metrics[metric]; !exists {
			t.history.Metrics[metric] = make([]float64, 0)
		}
		t.history.Metrics[metric] = append(t.history.Metrics[metric], value)
	}

	// Record validation metrics
	if valMetrics != nil {
		t.history.ValLoss = append(t.history.ValLoss, valLoss)
		for metric, value := range valMetrics {
			if _, exists := t.history.ValMetrics[metric]; !exists {
				t.history.ValMetrics[metric] = make([]float64, 0)
			}
			t.history.ValMetrics[metric] = append(t.history.ValMetrics[metric], value)
		}
	}
}

// logProgress logs training progress.
func (t *Trainer) logProgress(epoch, maxEpochs int, trainLoss float64, trainMetrics map[string]float64,
	valLoss float64, valMetrics map[string]float64, elapsed time.Duration) {

	logMsg := fmt.Sprintf("Epoch %d/%d [%.2fs] - loss: %.4f",
		epoch, maxEpochs, elapsed.Seconds(), trainLoss)

	// Add training metrics
	for metric, value := range trainMetrics {
		logMsg += fmt.Sprintf(" - %s: %.4f", metric, value)
	}

	// Add validation metrics if available
	if valMetrics != nil {
		logMsg += fmt.Sprintf(" - val_loss: %.4f", valLoss)
		for metric, value := range valMetrics {
			logMsg += fmt.Sprintf(" - val_%s: %.4f", metric, value)
		}
	}

	// Add learning rate if available
	if t.optimizer != nil {
		logMsg += fmt.Sprintf(" - lr: %.6f", t.optimizer.LearningRate())
	}

	fmt.Println(logMsg)
}

// splitData splits data into training and validation sets.
func (t *Trainer) splitData(X, y core.Tensor, validationSplit float64, shuffle bool, seed int64) *core.DataSplit {
	return t.model.splitData(X, y, validationSplit, shuffle, seed)
}

// GetHistory returns the training history.
func (t *Trainer) GetHistory() *core.History {
	return t.history
}

// ProgressTracker tracks training progress and provides statistics.
type ProgressTracker struct {
	startTime   time.Time
	epochTimes  []time.Duration
	losses      []float64
	valLosses   []float64
	bestLoss    float64
	bestEpoch   int
	patience    int
	maxPatience int
	earlyStop   bool
}

// NewProgressTracker creates a new progress tracker.
func NewProgressTracker(maxPatience int) *ProgressTracker {
	return &ProgressTracker{
		epochTimes:  make([]time.Duration, 0),
		losses:      make([]float64, 0),
		valLosses:   make([]float64, 0),
		bestLoss:    math.Inf(1),
		maxPatience: maxPatience,
	}
}

// StartTraining starts tracking training progress.
func (pt *ProgressTracker) StartTraining() {
	pt.startTime = time.Now()
}

// RecordEpoch records metrics for an epoch.
func (pt *ProgressTracker) RecordEpoch(epochStart time.Time, loss, valLoss float64) bool {
	elapsed := time.Since(epochStart)
	pt.epochTimes = append(pt.epochTimes, elapsed)
	pt.losses = append(pt.losses, loss)

	if !math.IsInf(valLoss, 1) {
		pt.valLosses = append(pt.valLosses, valLoss)
	}

	// Check for improvement
	currentLoss := loss
	if !math.IsInf(valLoss, 1) {
		currentLoss = valLoss
	}

	if currentLoss < pt.bestLoss {
		pt.bestLoss = currentLoss
		pt.bestEpoch = len(pt.losses)
		pt.patience = 0
		return false // Continue training
	}

	pt.patience++
	if pt.patience >= pt.maxPatience {
		pt.earlyStop = true
		return true // Stop training
	}

	return false // Continue training
}

// GetStats returns training statistics.
func (pt *ProgressTracker) GetStats() map[string]interface{} {
	stats := make(map[string]interface{})

	if len(pt.epochTimes) > 0 {
		totalTime := time.Duration(0)
		for _, t := range pt.epochTimes {
			totalTime += t
		}
		avgEpochTime := totalTime / time.Duration(len(pt.epochTimes))

		stats["total_time"] = time.Since(pt.startTime)
		stats["avg_epoch_time"] = avgEpochTime
		stats["epochs_completed"] = len(pt.epochTimes)
	}

	if len(pt.losses) > 0 {
		stats["final_loss"] = pt.losses[len(pt.losses)-1]
		stats["best_loss"] = pt.bestLoss
		stats["best_epoch"] = pt.bestEpoch
	}

	if len(pt.valLosses) > 0 {
		stats["final_val_loss"] = pt.valLosses[len(pt.valLosses)-1]
	}

	stats["early_stopped"] = pt.earlyStop
	stats["patience"] = pt.patience

	return stats
}

// EstimateTimeRemaining estimates remaining training time.
func (pt *ProgressTracker) EstimateTimeRemaining(currentEpoch, totalEpochs int) time.Duration {
	if len(pt.epochTimes) == 0 {
		return 0
	}

	// Calculate average epoch time
	totalTime := time.Duration(0)
	for _, t := range pt.epochTimes {
		totalTime += t
	}
	avgEpochTime := totalTime / time.Duration(len(pt.epochTimes))

	remainingEpochs := totalEpochs - currentEpoch
	return avgEpochTime * time.Duration(remainingEpochs)
}
