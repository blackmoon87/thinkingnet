package main

import (
	"fmt"

	"github.com/blackmoon87/thinkingnet/pkg/callbacks"
)

func main() {
	fmt.Println("ThinkingNet Callback System Demo")
	fmt.Println("=================================")

	// Create different types of callbacks
	verbose := callbacks.NewVerboseCallback(2) // One line per epoch
	earlyStopping := callbacks.NewEarlyStoppingCallback("loss", 3, 0.001)
	history := callbacks.NewHistoryCallback()

	// Demonstrate callback lifecycle
	fmt.Println("\n1. Training Begin Phase")
	verbose.OnTrainBegin()
	earlyStopping.OnTrainBegin()
	history.OnTrainBegin()

	// Simulate training epochs
	fmt.Println("\n2. Training Epochs")
	losses := []float64{1.0, 0.8, 0.6, 0.5, 0.51, 0.52, 0.53} // Should trigger early stopping

	for epoch, loss := range losses {
		// Epoch begin
		verbose.OnEpochBegin(epoch)
		earlyStopping.OnEpochBegin(epoch)
		history.OnEpochBegin(epoch)

		// Simulate metrics
		metrics := map[string]float64{
			"accuracy":  0.7 + float64(epoch)*0.03,
			"precision": 0.65 + float64(epoch)*0.035,
		}

		// Epoch end
		verbose.OnEpochEnd(epoch, len(losses), loss, metrics)
		earlyStopping.OnEpochEnd(epoch, len(losses), loss, metrics)
		history.OnEpochEnd(epoch, len(losses), loss, metrics)

		// Check if early stopping is triggered
		if earlyStopping.ShouldStop() {
			fmt.Printf("\nEarly stopping triggered at epoch %d!\n", epoch+1)
			break
		}
	}

	// Training end
	fmt.Println("\n3. Training End Phase")
	verbose.OnTrainEnd()
	earlyStopping.OnTrainEnd()
	history.OnTrainEnd()

	// Display results
	fmt.Println("\n4. Results Summary")
	fmt.Printf("Early stopping was triggered: %v\n", earlyStopping.WasStopped())
	fmt.Printf("Best epoch: %d\n", earlyStopping.GetBestEpoch())
	fmt.Printf("Best loss: %.6f\n", earlyStopping.GetBestValue())

	// Show history
	recordedHistory := history.GetHistory()
	fmt.Printf("Recorded %d epochs in history\n", len(recordedHistory.Epoch))
	fmt.Printf("Loss progression: %v\n", recordedHistory.Loss)

	if accuracyHistory, exists := recordedHistory.Metrics["accuracy"]; exists {
		fmt.Printf("Accuracy progression: %v\n", accuracyHistory)
	}

	fmt.Println("\n5. Callback Configuration Examples")

	// Show different verbose levels
	fmt.Println("\nVerbose Level 1 (Progress Bar):")
	progressCallback := callbacks.NewVerboseCallback(1)
	progressCallback.OnTrainBegin()
	progressCallback.OnEpochEnd(4, 5, 0.3, map[string]float64{"accuracy": 0.85})
	progressCallback.OnTrainEnd()

	// Show early stopping with different modes
	fmt.Println("\nEarly Stopping in Max Mode (for accuracy):")
	maxModeCallback := callbacks.NewEarlyStoppingCallback("accuracy", 2, 0.01).WithMode("max")
	maxModeCallback.OnTrainBegin()

	accuracies := []float64{0.7, 0.8, 0.85, 0.84, 0.83}
	for i, acc := range accuracies {
		metrics := map[string]float64{"accuracy": acc}
		maxModeCallback.OnEpochEnd(i, len(accuracies), 0.5, metrics)
		if maxModeCallback.ShouldStop() {
			fmt.Printf("Max mode early stopping triggered at epoch %d\n", i+1)
			break
		}
	}

	fmt.Printf("Best accuracy: %.3f at epoch %d\n",
		maxModeCallback.GetBestValue(), maxModeCallback.GetBestEpoch())

	fmt.Println("\nCallback system demonstration completed!")
}
