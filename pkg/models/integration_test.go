package models

import (
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// Integration test for the complete model training workflow
func TestModelTrainingWorkflow(t *testing.T) {
	// Create a simple model
	model := NewSequential()

	// Add layers
	layer1 := NewMockLayer("dense1", true, 50, []int{10, 5})
	layer2 := NewMockLayer("dense2", true, 10, []int{10, 2})

	err := model.AddLayer(layer1)
	if err != nil {
		t.Fatalf("Failed to add layer1: %v", err)
	}

	err = model.AddLayer(layer2)
	if err != nil {
		t.Fatalf("Failed to add layer2: %v", err)
	}

	// Compile model
	optimizer := NewMockOptimizer(0.001)
	loss := NewMockLoss()

	err = model.Compile(optimizer, loss)
	if err != nil {
		t.Fatalf("Failed to compile model: %v", err)
	}

	// Create training data
	X := NewMockTensor(100, 10) // 100 samples, 10 features
	y := NewMockTensor(100, 2)  // 100 samples, 2 classes

	// Fill with some dummy data
	for i := 0; i < 100; i++ {
		for j := 0; j < 10; j++ {
			X.Set(i, j, float64(i*j)/100.0)
		}
		// Binary classification labels
		if i < 50 {
			y.Set(i, 0, 1.0)
			y.Set(i, 1, 0.0)
		} else {
			y.Set(i, 0, 0.0)
			y.Set(i, 1, 1.0)
		}
	}

	// Configure training
	config := core.TrainingConfig{
		Epochs:          5,
		BatchSize:       10,
		ValidationSplit: 0.2,
		Metrics:         []string{"accuracy"},
		Verbose:         1,
		Shuffle:         true,
		Seed:            42,
		EarlyStopping: core.EarlyStoppingConfig{
			Enabled:  true,
			Monitor:  "val_loss",
			Patience: 3,
			MinDelta: 0.001,
			Mode:     "min",
		},
	}

	// Train the model
	history, err := model.Fit(X, y, config)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Verify training results
	if history == nil {
		t.Fatal("History should not be nil")
	}

	if len(history.Loss) == 0 {
		t.Error("History should contain loss values")
	}

	if len(history.Epoch) == 0 {
		t.Error("History should contain epoch numbers")
	}

	// Check that we have validation metrics
	if len(history.ValLoss) == 0 {
		t.Error("History should contain validation loss values")
	}

	// Check metrics
	if _, exists := history.Metrics["accuracy"]; !exists {
		t.Error("History should contain accuracy metrics")
	}

	if _, exists := history.ValMetrics["accuracy"]; !exists {
		t.Error("History should contain validation accuracy metrics")
	}

	// Test prediction
	testX := NewMockTensor(10, 10)
	predictions, err := model.Predict(testX)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	if predictions == nil {
		t.Fatal("Predictions should not be nil")
	}

	predRows, predCols := predictions.Dims()
	if predRows != 10 || predCols != 2 {
		t.Errorf("Expected predictions shape (10, 2), got (%d, %d)", predRows, predCols)
	}

	// Test evaluation
	metrics, err := model.Evaluate(X, y)
	if err != nil {
		t.Fatalf("Evaluation failed: %v", err)
	}

	if metrics == nil {
		t.Fatal("Metrics should not be nil")
	}

	// Test model summary
	summary := model.Summary()
	if summary == "" {
		t.Error("Summary should not be empty")
	}
}

func TestBatchProcessing(t *testing.T) {
	// Create test data
	X := NewMockTensor(50, 5)
	y := NewMockTensor(50, 1)

	// Fill with test data
	for i := 0; i < 50; i++ {
		for j := 0; j < 5; j++ {
			X.Set(i, j, float64(i+j))
		}
		y.Set(i, 0, float64(i%2)) // Binary labels
	}

	// Create data loader
	dataLoader, err := NewDataLoader(X, y, 10, true, 42)
	if err != nil {
		t.Fatalf("Failed to create data loader: %v", err)
	}

	// Test batch iteration
	batchCount := 0
	totalSamples := 0

	for dataLoader.HasNext() {
		XBatch, yBatch, err := dataLoader.Next()
		if err != nil {
			t.Fatalf("Failed to get batch: %v", err)
		}

		batchRows, _ := XBatch.Dims()
		totalSamples += batchRows
		batchCount++

		// Verify batch size (should be 10 except possibly the last batch)
		if batchCount < 5 && batchRows != 10 {
			t.Errorf("Expected batch size 10, got %d", batchRows)
		}

		// Verify y batch has same number of rows
		yRows, _ := yBatch.Dims()
		if yRows != batchRows {
			t.Errorf("X and y batch sizes don't match: %d vs %d", batchRows, yRows)
		}
	}

	// Verify we processed all samples
	if totalSamples != 50 {
		t.Errorf("Expected to process 50 samples, processed %d", totalSamples)
	}

	// Verify number of batches
	expectedBatches := 5 // 50 samples / 10 batch size
	if batchCount != expectedBatches {
		t.Errorf("Expected %d batches, got %d", expectedBatches, batchCount)
	}
}

func TestTrainerWorkflow(t *testing.T) {
	// Create model
	model := NewSequential()
	layer := NewMockLayer("test_layer", true, 20, []int{10, 2})
	model.AddLayer(layer)

	optimizer := NewMockOptimizer(0.01)
	loss := NewMockLoss()
	model.Compile(optimizer, loss)

	// Create trainer
	trainer := NewTrainer(model, optimizer, loss)

	// Add a mock callback
	callback := &MockCallback{}
	trainer.AddCallback(callback)

	// Create training data
	X := NewMockTensor(20, 5)
	y := NewMockTensor(20, 2)

	config := core.TrainingConfig{
		Epochs:    3,
		BatchSize: 5,
		Metrics:   []string{"accuracy"},
		Verbose:   0,
	}

	// Train using trainer
	history, err := trainer.Train(X, y, config)
	if err != nil {
		t.Fatalf("Trainer.Train() failed: %v", err)
	}

	if history == nil {
		t.Fatal("History should not be nil")
	}

	if len(history.Loss) != 3 {
		t.Errorf("Expected 3 loss values, got %d", len(history.Loss))
	}

	// Verify callback was called
	if !callback.trainBeginCalled {
		t.Error("OnTrainBegin should have been called")
	}

	if !callback.trainEndCalled {
		t.Error("OnTrainEnd should have been called")
	}
}

func TestMemoryPool(t *testing.T) {
	pool := NewMemoryPool(5)

	// Test initial state
	if pool.Size() != 0 {
		t.Errorf("Expected empty pool, got size %d", pool.Size())
	}

	// Create and put tensors
	tensor1 := NewMockTensor(3, 3)
	tensor2 := NewMockTensor(3, 3)
	tensor3 := NewMockTensor(2, 2)

	pool.Put(tensor1)
	pool.Put(tensor2)
	pool.Put(tensor3)

	if pool.Size() != 3 {
		t.Errorf("Expected pool size 3, got %d", pool.Size())
	}

	// Test getting tensors
	retrieved := pool.Get(3, 3)
	if retrieved == nil {
		t.Error("Should have retrieved a tensor from pool")
	}

	// Pool should have one less tensor of that size
	if pool.Size() != 2 {
		t.Errorf("Expected pool size 2 after retrieval, got %d", pool.Size())
	}

	// Test clearing pool
	pool.Clear()
	if pool.Size() != 0 {
		t.Errorf("Expected empty pool after clear, got size %d", pool.Size())
	}
}

// MockCallback for testing callback functionality
type MockCallback struct {
	core.BaseCallback
	trainBeginCalled bool
	trainEndCalled   bool
	epochBeginCount  int
	epochEndCount    int
	batchBeginCount  int
	batchEndCount    int
}

func (cb *MockCallback) OnTrainBegin() {
	cb.trainBeginCalled = true
}

func (cb *MockCallback) OnTrainEnd() {
	cb.trainEndCalled = true
}

func (cb *MockCallback) OnEpochBegin(epoch int) {
	cb.epochBeginCount++
}

func (cb *MockCallback) OnEpochEnd(epoch, maxEpochs int, loss float64, metrics map[string]float64) {
	cb.epochEndCount++
}

func (cb *MockCallback) OnBatchBegin(batch int) {
	cb.batchBeginCount++
}

func (cb *MockCallback) OnBatchEnd(batch int, loss float64) {
	cb.batchEndCount++
}
