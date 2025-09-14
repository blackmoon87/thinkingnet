package callbacks

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestVerboseCallback(t *testing.T) {
	tests := []struct {
		name    string
		verbose int
	}{
		{"Silent", 0},
		{"ProgressBar", 1},
		{"OneLine", 2},
		{"Detailed", 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			callback := NewVerboseCallback(tt.verbose)

			// Test callback lifecycle
			callback.OnTrainBegin()

			for epoch := 0; epoch < 5; epoch++ {
				callback.OnEpochBegin(epoch)

				metrics := map[string]float64{
					"accuracy":  0.8 + float64(epoch)*0.02,
					"precision": 0.75 + float64(epoch)*0.03,
				}

				callback.OnEpochEnd(epoch, 5, 0.5-float64(epoch)*0.05, metrics)

				// Small delay to simulate training time
				time.Sleep(10 * time.Millisecond)
			}

			callback.OnTrainEnd()
		})
	}
}

func TestVerboseCallbackWithMetrics(t *testing.T) {
	callback := NewVerboseCallback(2).WithMetrics(false)

	callback.OnTrainBegin()

	metrics := map[string]float64{
		"accuracy": 0.85,
	}

	callback.OnEpochEnd(0, 1, 0.3, metrics)
	callback.OnTrainEnd()

	// Test that metrics are not shown when disabled
	// In a real test, we would capture stdout and verify the output
}

func TestEarlyStoppingCallback(t *testing.T) {
	t.Run("MinMode", func(t *testing.T) {
		callback := NewEarlyStoppingCallback("loss", 3, 0.001)

		callback.OnTrainBegin()

		// Simulate decreasing loss (improvement)
		losses := []float64{1.0, 0.8, 0.6, 0.5, 0.51, 0.52, 0.53} // Last 3 epochs no improvement

		for epoch, loss := range losses {
			callback.OnEpochEnd(epoch, len(losses), loss, nil)

			if epoch < 4 {
				if callback.ShouldStop() {
					t.Errorf("Should not stop at epoch %d", epoch)
				}
			}
		}

		// Should stop after patience is exceeded
		if !callback.ShouldStop() {
			t.Error("Should stop after patience exceeded")
		}

		if !callback.WasStopped() {
			t.Error("WasStopped should return true")
		}

		if callback.GetBestEpoch() != 4 {
			t.Errorf("Best epoch should be 4, got %d", callback.GetBestEpoch())
		}

		expectedBestValue := 0.5
		if math.Abs(callback.GetBestValue()-expectedBestValue) > 1e-6 {
			t.Errorf("Best value should be %f, got %f", expectedBestValue, callback.GetBestValue())
		}
	})

	t.Run("MaxMode", func(t *testing.T) {
		callback := NewEarlyStoppingCallback("accuracy", 2, 0.01).WithMode("max")

		callback.OnTrainBegin()

		// Simulate accuracy that stops improving
		accuracies := []float64{0.7, 0.8, 0.85, 0.84, 0.83} // Last 2 epochs no improvement

		for epoch, acc := range accuracies {
			metrics := map[string]float64{"accuracy": acc}
			callback.OnEpochEnd(epoch, len(accuracies), 0.5, metrics)
		}

		if !callback.ShouldStop() {
			t.Error("Should stop after patience exceeded in max mode")
		}

		if callback.GetBestEpoch() != 3 {
			t.Errorf("Best epoch should be 3, got %d", callback.GetBestEpoch())
		}
	})

	t.Run("WithMinDelta", func(t *testing.T) {
		callback := NewEarlyStoppingCallback("loss", 2, 0.1) // Large min_delta

		callback.OnTrainBegin()

		// Small improvements that don't meet min_delta
		losses := []float64{1.0, 0.95, 0.92, 0.91}

		for epoch, loss := range losses {
			callback.OnEpochEnd(epoch, len(losses), loss, nil)
		}

		if !callback.ShouldStop() {
			t.Error("Should stop when improvements are smaller than min_delta")
		}
	})
}

func TestEarlyStoppingCallbackInvalidValues(t *testing.T) {
	callback := NewEarlyStoppingCallback("loss", 2, 0.001).WithVerbose(false)

	callback.OnTrainBegin()

	// Test with NaN and Inf values
	invalidValues := []float64{math.NaN(), math.Inf(1), math.Inf(-1)}

	for epoch, loss := range invalidValues {
		callback.OnEpochEnd(epoch, len(invalidValues), loss, nil)

		// Should not stop due to invalid values
		if callback.ShouldStop() {
			t.Errorf("Should not stop due to invalid value at epoch %d", epoch)
		}
	}
}

func TestHistoryCallback(t *testing.T) {
	callback := NewHistoryCallback()

	callback.OnTrainBegin()

	// Simulate training epochs
	for epoch := 0; epoch < 3; epoch++ {
		loss := 1.0 - float64(epoch)*0.2
		metrics := map[string]float64{
			"accuracy":  0.7 + float64(epoch)*0.1,
			"precision": 0.6 + float64(epoch)*0.15,
		}

		callback.OnEpochEnd(epoch, 3, loss, metrics)

		// Add validation metrics
		valLoss := loss + 0.1
		valMetrics := map[string]float64{
			"accuracy":  metrics["accuracy"] - 0.05,
			"precision": metrics["precision"] - 0.05,
		}
		callback.SetValidationMetrics(valLoss, valMetrics)
	}

	callback.OnTrainEnd()

	history := callback.GetHistory()

	// Verify history structure
	if len(history.Epoch) != 3 {
		t.Errorf("Expected 3 epochs, got %d", len(history.Epoch))
	}

	if len(history.Loss) != 3 {
		t.Errorf("Expected 3 loss values, got %d", len(history.Loss))
	}

	if len(history.ValLoss) != 3 {
		t.Errorf("Expected 3 validation loss values, got %d", len(history.ValLoss))
	}

	// Verify metrics
	if _, exists := history.Metrics["accuracy"]; !exists {
		t.Error("Accuracy metric should exist in history")
	}

	if _, exists := history.ValMetrics["accuracy"]; !exists {
		t.Error("Validation accuracy metric should exist in history")
	}

	// Verify values
	expectedLoss := 1.0
	if math.Abs(history.Loss[0]-expectedLoss) > 1e-6 {
		t.Errorf("First loss should be %f, got %f", expectedLoss, history.Loss[0])
	}

	expectedAccuracy := 0.7
	if math.Abs(history.Metrics["accuracy"][0]-expectedAccuracy) > 1e-6 {
		t.Errorf("First accuracy should be %f, got %f", expectedAccuracy, history.Metrics["accuracy"][0])
	}
}

func TestModelCheckpointCallback(t *testing.T) {
	// Create temporary directory for checkpoints
	tempDir := t.TempDir()
	checkpointPath := filepath.Join(tempDir, "model_epoch_%d_%.4f.ckpt")

	t.Run("SaveBest", func(t *testing.T) {
		callback := NewModelCheckpointCallback(checkpointPath).
			WithMonitor("val_loss").
			WithSaveBest(true)

		callback.OnTrainBegin()

		// Simulate training with improving then worsening validation loss
		valLosses := []float64{1.0, 0.8, 0.6, 0.7, 0.5, 0.9} // Best at epoch 5 (0.5)

		for epoch, valLoss := range valLosses {
			metrics := map[string]float64{"val_loss": valLoss}
			callback.OnEpochEnd(epoch, len(valLosses), valLoss, metrics)
		}

		// Check that best values are recorded correctly
		if callback.GetBestEpoch() != 5 {
			t.Errorf("Best epoch should be 5, got %d", callback.GetBestEpoch())
		}

		expectedBestValue := 0.5
		if math.Abs(callback.GetBestValue()-expectedBestValue) > 1e-6 {
			t.Errorf("Best value should be %f, got %f", expectedBestValue, callback.GetBestValue())
		}

		// Check that checkpoint files were created (at least for the best model)
		files, err := os.ReadDir(tempDir)
		if err != nil {
			t.Fatalf("Failed to read temp directory: %v", err)
		}

		if len(files) == 0 {
			t.Error("No checkpoint files were created")
		}

		// Verify that at least one file contains the best epoch info
		foundBestCheckpoint := false
		for _, file := range files {
			if strings.Contains(file.Name(), "0.5000") {
				foundBestCheckpoint = true
				break
			}
		}

		if !foundBestCheckpoint {
			t.Error("Best checkpoint file not found")
		}
	})

	t.Run("SaveAll", func(t *testing.T) {
		tempDir2 := t.TempDir()
		checkpointPath2 := filepath.Join(tempDir2, "model_epoch_%d.ckpt")

		callback := NewModelCheckpointCallback(checkpointPath2).
			WithSaveBest(false)

		callback.OnTrainBegin()

		// Simulate 3 epochs
		for epoch := 0; epoch < 3; epoch++ {
			loss := 1.0 - float64(epoch)*0.2
			callback.OnEpochEnd(epoch, 3, loss, nil)
		}

		// Check that all checkpoint files were created
		files, err := os.ReadDir(tempDir2)
		if err != nil {
			t.Fatalf("Failed to read temp directory: %v", err)
		}

		if len(files) != 3 {
			t.Errorf("Expected 3 checkpoint files, got %d", len(files))
		}
	})

	t.Run("MaxMode", func(t *testing.T) {
		tempDir3 := t.TempDir()
		checkpointPath3 := filepath.Join(tempDir3, "model_acc_%.4f.ckpt")

		callback := NewModelCheckpointCallback(checkpointPath3).
			WithMonitor("accuracy").
			WithSaveBest(true)

		callback.OnTrainBegin()

		// Simulate improving accuracy
		accuracies := []float64{0.7, 0.8, 0.85, 0.82} // Best at epoch 3 (0.85)

		for epoch, acc := range accuracies {
			metrics := map[string]float64{"accuracy": acc}
			callback.OnEpochEnd(epoch, len(accuracies), 0.5, metrics)
		}

		if callback.GetBestEpoch() != 3 {
			t.Errorf("Best epoch should be 3, got %d", callback.GetBestEpoch())
		}

		expectedBestValue := 0.85
		if math.Abs(callback.GetBestValue()-expectedBestValue) > 1e-6 {
			t.Errorf("Best value should be %f, got %f", expectedBestValue, callback.GetBestValue())
		}
	})
}

func TestCallbackIntegration(t *testing.T) {
	// Test multiple callbacks working together
	verbose := NewVerboseCallback(1)
	earlyStopping := NewEarlyStoppingCallback("loss", 2, 0.001)
	history := NewHistoryCallback()

	callbacks := []core.Callback{verbose, earlyStopping, history}

	// Simulate training begin
	for _, cb := range callbacks {
		cb.OnTrainBegin()
	}

	// Simulate training epochs
	losses := []float64{1.0, 0.8, 0.6, 0.61, 0.62} // Should trigger early stopping

	for epoch, loss := range losses {
		for _, cb := range callbacks {
			cb.OnEpochBegin(epoch)
		}

		metrics := map[string]float64{"accuracy": 0.7 + float64(epoch)*0.05}

		for _, cb := range callbacks {
			cb.OnEpochEnd(epoch, len(losses), loss, metrics)
		}

		// Check if early stopping is triggered
		if earlyStopping.ShouldStop() {
			break
		}
	}

	// Simulate training end
	for _, cb := range callbacks {
		cb.OnTrainEnd()
	}

	// Verify early stopping was triggered
	if !earlyStopping.WasStopped() {
		t.Error("Early stopping should have been triggered")
	}

	// Verify history was recorded
	recordedHistory := history.GetHistory()
	if len(recordedHistory.Loss) == 0 {
		t.Error("History should have recorded loss values")
	}
}

func TestCallbackErrorHandling(t *testing.T) {
	// Test callbacks with nil/empty inputs
	verbose := NewVerboseCallback(2)

	verbose.OnTrainBegin()

	// Test with nil metrics
	verbose.OnEpochEnd(0, 1, 0.5, nil)

	// Test with empty metrics
	verbose.OnEpochEnd(1, 2, 0.4, map[string]float64{})

	verbose.OnTrainEnd()

	// Test early stopping with missing monitored metric
	earlyStopping := NewEarlyStoppingCallback("nonexistent_metric", 2, 0.001)
	earlyStopping.OnTrainBegin()

	metrics := map[string]float64{"accuracy": 0.8}
	earlyStopping.OnEpochEnd(0, 1, 0.5, metrics)

	// Should fallback to loss when monitored metric doesn't exist
	if earlyStopping.ShouldStop() {
		t.Error("Should not stop on first epoch with fallback metric")
	}
}

// Benchmark tests
func BenchmarkVerboseCallback(b *testing.B) {
	callback := NewVerboseCallback(0) // Silent mode for benchmarking

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		callback.OnTrainBegin()

		for epoch := 0; epoch < 100; epoch++ {
			callback.OnEpochBegin(epoch)

			metrics := map[string]float64{
				"accuracy":  0.8,
				"precision": 0.75,
				"recall":    0.7,
			}

			callback.OnEpochEnd(epoch, 100, 0.5, metrics)
		}

		callback.OnTrainEnd()
	}
}

func BenchmarkEarlyStoppingCallback(b *testing.B) {
	callback := NewEarlyStoppingCallback("loss", 10, 0.001).WithVerbose(false)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		callback.OnTrainBegin()

		for epoch := 0; epoch < 100; epoch++ {
			loss := 1.0 - float64(epoch)*0.01
			callback.OnEpochEnd(epoch, 100, loss, nil)

			if callback.ShouldStop() {
				break
			}
		}
	}
}

func BenchmarkHistoryCallback(b *testing.B) {
	callback := NewHistoryCallback()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		callback.OnTrainBegin()

		for epoch := 0; epoch < 100; epoch++ {
			metrics := map[string]float64{
				"accuracy":  0.8 + float64(epoch)*0.001,
				"precision": 0.75 + float64(epoch)*0.001,
				"recall":    0.7 + float64(epoch)*0.001,
				"f1_score":  0.72 + float64(epoch)*0.001,
			}

			callback.OnEpochEnd(epoch, 100, 1.0-float64(epoch)*0.01, metrics)
		}

		callback.OnTrainEnd()
	}
}
