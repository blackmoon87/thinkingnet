package models

import (
	"fmt"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// Example demonstrates how to use the Sequential model
func ExampleSequential() {
	// Create a new sequential model
	model := NewSequential(core.WithName("example_model"), core.WithSeed(42))

	// Add layers (using mock layers for this example)
	layer1 := NewMockLayer("dense1", true, 100, []int{32, 64})
	layer2 := NewMockLayer("dense2", true, 130, []int{32, 2})

	model.AddLayer(layer1)
	model.AddLayer(layer2)

	// Compile the model
	optimizer := NewMockOptimizer(0.001)
	loss := NewMockLoss()
	model.Compile(optimizer, loss)

	// Create training data
	X := NewMockTensor(100, 10) // 100 samples, 10 features
	y := NewMockTensor(100, 2)  // 100 samples, 2 classes (binary classification)

	// Configure training
	config := core.TrainingConfig{
		Epochs:          10,
		BatchSize:       16,
		ValidationSplit: 0.2,
		Metrics:         []string{"accuracy"},
		Verbose:         2, // Log every 2 epochs
		Shuffle:         true,
		Seed:            42,
		EarlyStopping: core.EarlyStoppingConfig{
			Enabled:  true,
			Monitor:  "val_loss",
			Patience: 5,
			MinDelta: 0.001,
			Mode:     "min",
		},
	}

	// Train the model
	history, err := model.Fit(X, y, config)
	if err != nil {
		fmt.Printf("Training failed: %v\n", err)
		return
	}

	// Print training results
	fmt.Printf("Training completed in %v\n", history.Duration)
	fmt.Printf("Final loss: %.4f\n", history.Loss[len(history.Loss)-1])

	if len(history.ValLoss) > 0 {
		fmt.Printf("Final validation loss: %.4f\n", history.ValLoss[len(history.ValLoss)-1])
	}

	// Make predictions
	testX := NewMockTensor(10, 10)
	predictions, err := model.Predict(testX)
	if err != nil {
		fmt.Printf("Prediction failed: %v\n", err)
		return
	}

	fmt.Printf("Predictions shape: %v\n", predictions.Shape())

	// Evaluate the model
	metrics, err := model.Evaluate(X, y)
	if err != nil {
		fmt.Printf("Evaluation failed: %v\n", err)
		return
	}

	fmt.Printf("Final accuracy: %.4f\n", metrics.Accuracy)

	// Print model summary
	fmt.Println("\nModel Summary:")
	fmt.Print(model.Summary())

	// Output:
	// Training completed
	// Model Summary contains layer information
}

// ExampleDataLoader demonstrates batch processing functionality
func ExampleDataLoader() {
	// Create sample data
	X := NewMockTensor(50, 5)
	y := NewMockTensor(50, 1)

	// Create a data loader with batch size 10
	dataLoader, err := NewDataLoader(X, y, 10, true, 42)
	if err != nil {
		fmt.Printf("Failed to create data loader: %v\n", err)
		return
	}

	fmt.Printf("Total samples: %d\n", dataLoader.NumSamples())
	fmt.Printf("Batch size: %d\n", dataLoader.BatchSize())
	fmt.Printf("Number of batches: %d\n", dataLoader.NumBatches())

	// Process batches
	batchCount := 0
	for dataLoader.HasNext() {
		XBatch, yBatch, err := dataLoader.Next()
		if err != nil {
			fmt.Printf("Error getting batch: %v\n", err)
			break
		}

		batchRows, batchCols := XBatch.Dims()
		fmt.Printf("Batch %d: shape (%d, %d)\n", batchCount+1, batchRows, batchCols)
		_ = yBatch // Use yBatch to avoid unused variable error
		batchCount++
	}

	// Output:
	// Total samples: 50
	// Batch size: 10
	// Number of batches: 5
	// Batch processing information
}

// ExampleTrainer demonstrates using the Trainer for more control over training
func ExampleTrainer() {
	// Create and compile model
	model := NewSequential()
	layer := NewMockLayer("dense", true, 50, []int{32, 1})
	model.AddLayer(layer)

	optimizer := NewMockOptimizer(0.01)
	loss := NewMockLoss()
	model.Compile(optimizer, loss)

	// Create trainer
	trainer := NewTrainer(model, optimizer, loss)

	// Add custom callback
	callback := &MockCallback{}
	trainer.AddCallback(callback)

	// Create training data
	X := NewMockTensor(100, 10)
	y := NewMockTensor(100, 1)

	// Configure training
	config := core.TrainingConfig{
		Epochs:    5,
		BatchSize: 20,
		Metrics:   []string{"accuracy"},
		Verbose:   1,
	}

	// Train using trainer
	history, err := trainer.Train(X, y, config)
	if err != nil {
		fmt.Printf("Training failed: %v\n", err)
		return
	}

	fmt.Printf("Training epochs: %d\n", len(history.Epoch))
	fmt.Printf("Callback train begin called: %v\n", callback.trainBeginCalled)
	fmt.Printf("Callback train end called: %v\n", callback.trainEndCalled)

	// Output:
	// Training epochs: 5
	// Callback train begin called: true
	// Callback train end called: true
}
