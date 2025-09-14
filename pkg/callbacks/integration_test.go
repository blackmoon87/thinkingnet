package callbacks

import (
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// MockModel simulates a model for integration testing
type MockModel struct {
	callbacks     []core.Callback
	compiled      bool
	simulateEpoch func(epoch int) (float64, map[string]float64)
}

func NewMockModel() *MockModel {
	m := &MockModel{
		callbacks: make([]core.Callback, 0),
		compiled:  false,
	}
	m.simulateEpoch = m.defaultSimulateEpoch
	return m
}

func (m *MockModel) AddCallback(callback core.Callback) {
	m.callbacks = append(m.callbacks, callback)
}

func (m *MockModel) Compile() {
	m.compiled = true
}

func (m *MockModel) Train(epochs int) *core.History {
	if !m.compiled {
		panic("model must be compiled before training")
	}

	// Initialize history
	history := &core.History{
		Epoch:      make([]int, 0),
		Loss:       make([]float64, 0),
		Metrics:    make(map[string][]float64),
		ValLoss:    make([]float64, 0),
		ValMetrics: make(map[string][]float64),
	}

	// Training begin callbacks
	for _, callback := range m.callbacks {
		callback.OnTrainBegin()
	}

	startTime := time.Now()

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		// Epoch begin callbacks
		for _, callback := range m.callbacks {
			callback.OnEpochBegin(epoch)
		}

		// Check for early stopping
		shouldStop := false
		for _, callback := range m.callbacks {
			if earlyStop, ok := callback.(interface{ ShouldStop() bool }); ok {
				if earlyStop.ShouldStop() {
					shouldStop = true
					break
				}
			}
		}
		if shouldStop {
			break
		}

		// Simulate training epoch
		loss, metrics := m.simulateEpoch(epoch)

		// Record in history
		history.Epoch = append(history.Epoch, epoch+1)
		history.Loss = append(history.Loss, loss)

		for metric, value := range metrics {
			if _, exists := history.Metrics[metric]; !exists {
				history.Metrics[metric] = make([]float64, 0)
			}
			history.Metrics[metric] = append(history.Metrics[metric], value)
		}

		// Epoch end callbacks
		for _, callback := range m.callbacks {
			callback.OnEpochEnd(epoch, epochs, loss, metrics)
		}

		// Small delay to simulate training time
		time.Sleep(10 * time.Millisecond)
	}

	history.Duration = time.Since(startTime)

	// Training end callbacks
	for _, callback := range m.callbacks {
		callback.OnTrainEnd()
	}

	return history
}

func (m *MockModel) defaultSimulateEpoch(epoch int) (float64, map[string]float64) {
	// Simulate decreasing loss with some noise
	loss := 1.0 - float64(epoch)*0.1 + (float64(epoch%3)-1)*0.02

	// Simulate improving metrics
	metrics := map[string]float64{
		"accuracy":  0.7 + float64(epoch)*0.03,
		"precision": 0.65 + float64(epoch)*0.035,
		"recall":    0.6 + float64(epoch)*0.04,
	}

	return loss, metrics
}

func TestCallbackIntegrationWithModel(t *testing.T) {
	model := NewMockModel()
	model.Compile()

	// Add multiple callbacks
	verbose := NewVerboseCallback(0) // Silent for testing
	earlyStopping := NewEarlyStoppingCallback("loss", 3, 0.001)
	history := NewHistoryCallback()

	model.AddCallback(verbose)
	model.AddCallback(earlyStopping)
	model.AddCallback(history)

	// Train the model
	result := model.Train(10)

	// Verify training completed
	if len(result.Epoch) == 0 {
		t.Error("Training should have completed at least one epoch")
	}

	// Verify history was recorded
	recordedHistory := history.GetHistory()
	if len(recordedHistory.Loss) != len(result.Epoch) {
		t.Errorf("History loss length (%d) should match result epochs (%d)",
			len(recordedHistory.Loss), len(result.Epoch))
	}

	// Verify metrics were recorded
	if _, exists := recordedHistory.Metrics["accuracy"]; !exists {
		t.Error("Accuracy metric should be recorded in history")
	}
}

func TestEarlyStoppingIntegration(t *testing.T) {
	model := NewMockModel()
	model.Compile()

	// Create early stopping with very low patience
	earlyStopping := NewEarlyStoppingCallback("loss", 2, 0.001).WithVerbose(false)
	model.AddCallback(earlyStopping)

	// Override the model's epoch simulation to create plateauing loss
	originalSimulate := model.simulateEpoch
	model.simulateEpoch = func(epoch int) (float64, map[string]float64) {
		// Loss improves for first 3 epochs, then plateaus
		var loss float64
		if epoch < 3 {
			loss = 1.0 - float64(epoch)*0.2
		} else {
			loss = 0.4 + float64(epoch%2)*0.001 // Very small changes
		}

		metrics := map[string]float64{
			"accuracy": 0.7 + float64(epoch)*0.01,
		}

		return loss, metrics
	}

	// Train the model
	result := model.Train(10)

	// Restore original simulation
	model.simulateEpoch = originalSimulate

	// Verify early stopping was triggered
	if !earlyStopping.WasStopped() {
		t.Error("Early stopping should have been triggered")
	}

	// Verify training stopped before all epochs
	if len(result.Epoch) >= 10 {
		t.Errorf("Training should have stopped early, but completed %d epochs", len(result.Epoch))
	}

	// Verify best epoch is recorded
	if earlyStopping.GetBestEpoch() == 0 {
		t.Error("Best epoch should be recorded")
	}
}

func TestHistoryIntegrationWithValidation(t *testing.T) {
	model := NewMockModel()
	model.Compile()

	history := NewHistoryCallback()
	model.AddCallback(history)

	// Override epoch simulation to include validation metrics
	originalSimulate := model.simulateEpoch
	model.simulateEpoch = func(epoch int) (float64, map[string]float64) {
		loss, metrics := originalSimulate(epoch)

		// Simulate validation metrics in the callback
		valLoss := loss + 0.05
		valMetrics := map[string]float64{
			"accuracy":  metrics["accuracy"] - 0.02,
			"precision": metrics["precision"] - 0.03,
		}

		// Set validation metrics in history callback
		history.SetValidationMetrics(valLoss, valMetrics)

		return loss, metrics
	}

	// Train the model
	result := model.Train(5)

	// Restore original simulation
	model.simulateEpoch = originalSimulate

	// Verify validation metrics were recorded
	recordedHistory := history.GetHistory()

	if len(recordedHistory.ValLoss) != len(result.Epoch) {
		t.Errorf("Validation loss length (%d) should match epochs (%d)",
			len(recordedHistory.ValLoss), len(result.Epoch))
	}

	if _, exists := recordedHistory.ValMetrics["accuracy"]; !exists {
		t.Error("Validation accuracy should be recorded")
	}

	// Verify validation values are different from training values
	if len(recordedHistory.ValLoss) > 0 && len(recordedHistory.Loss) > 0 {
		if recordedHistory.ValLoss[0] == recordedHistory.Loss[0] {
			t.Error("Validation loss should be different from training loss")
		}
	}
}

func TestCheckpointIntegrationWithFileSystem(t *testing.T) {
	// Create temporary directory
	tempDir := t.TempDir()
	checkpointPath := filepath.Join(tempDir, "model_epoch_%d.ckpt")

	model := NewMockModel()
	model.Compile()

	checkpoint := NewModelCheckpointCallback(checkpointPath).
		WithSaveBest(false). // Save all epochs
		WithVerbose(false)   // Silent for testing

	model.AddCallback(checkpoint)

	// Train the model
	result := model.Train(3)

	// Verify checkpoint files were created
	files, err := os.ReadDir(tempDir)
	if err != nil {
		t.Fatalf("Failed to read temp directory: %v", err)
	}

	expectedFiles := len(result.Epoch)
	if len(files) != expectedFiles {
		t.Errorf("Expected %d checkpoint files, got %d", expectedFiles, len(files))
	}

	// Verify file names contain epoch information
	for i, file := range files {
		expectedPattern := "model_epoch_"
		if !contains(file.Name(), expectedPattern) {
			t.Errorf("File %d name should contain '%s', got '%s'", i, expectedPattern, file.Name())
		}
	}
}

func TestMultipleCallbacksIntegration(t *testing.T) {
	model := NewMockModel()
	model.Compile()

	// Create multiple callbacks with different behaviors
	verbose := NewVerboseCallback(0) // Silent
	earlyStopping := NewEarlyStoppingCallback("loss", 5, 0.001).WithVerbose(false)
	history := NewHistoryCallback()

	tempDir := t.TempDir()
	checkpoint := NewModelCheckpointCallback(filepath.Join(tempDir, "model.ckpt")).
		WithSaveBest(true).
		WithVerbose(false)

	model.AddCallback(verbose)
	model.AddCallback(earlyStopping)
	model.AddCallback(history)
	model.AddCallback(checkpoint)

	// Train the model
	result := model.Train(8)

	// Verify all callbacks worked together
	recordedHistory := history.GetHistory()

	// History should match training result
	if len(recordedHistory.Epoch) != len(result.Epoch) {
		t.Errorf("History epochs (%d) should match result epochs (%d)",
			len(recordedHistory.Epoch), len(result.Epoch))
	}

	// Checkpoint should have saved at least one file
	files, err := os.ReadDir(tempDir)
	if err != nil {
		t.Fatalf("Failed to read temp directory: %v", err)
	}

	if len(files) == 0 {
		t.Error("At least one checkpoint file should have been created")
	}

	// Early stopping should have recorded best values
	if earlyStopping.GetBestEpoch() == 0 {
		t.Error("Early stopping should have recorded a best epoch")
	}

	if math.IsInf(earlyStopping.GetBestValue(), 0) {
		t.Error("Early stopping should have recorded a finite best value")
	}
}

func TestCallbackErrorHandlingIntegration(t *testing.T) {
	model := NewMockModel()
	model.Compile()

	// Test with callbacks that might encounter errors
	earlyStopping := NewEarlyStoppingCallback("nonexistent_metric", 2, 0.001).WithVerbose(false)
	history := NewHistoryCallback()

	model.AddCallback(earlyStopping)
	model.AddCallback(history)

	// Override simulation to return nil metrics occasionally
	originalSimulate := model.simulateEpoch
	model.simulateEpoch = func(epoch int) (float64, map[string]float64) {
		loss, metrics := originalSimulate(epoch)

		// Return nil metrics on odd epochs
		if epoch%2 == 1 {
			return loss, nil
		}

		return loss, metrics
	}

	// Train should complete without panicking
	result := model.Train(4)

	// Restore original simulation
	model.simulateEpoch = originalSimulate

	// Verify training completed
	if len(result.Epoch) == 0 {
		t.Error("Training should have completed despite nil metrics")
	}

	// History should still record what it can
	recordedHistory := history.GetHistory()
	if len(recordedHistory.Loss) == 0 {
		t.Error("History should record loss values even with nil metrics")
	}
}

func TestCallbackPerformance(t *testing.T) {
	model := NewMockModel()
	model.Compile()

	// Add multiple callbacks to test performance impact
	callbacks := []core.Callback{
		NewVerboseCallback(0), // Silent
		NewEarlyStoppingCallback("loss", 100, 0.001).WithVerbose(false),
		NewHistoryCallback(),
		NewModelCheckpointCallback(filepath.Join(t.TempDir(), "model.ckpt")).WithVerbose(false),
	}

	for _, cb := range callbacks {
		model.AddCallback(cb)
	}

	// Measure training time
	start := time.Now()
	result := model.Train(50)
	elapsed := time.Since(start)

	// Verify training completed
	if len(result.Epoch) == 0 {
		t.Error("Training should have completed")
	}

	// Performance check - callbacks shouldn't add significant overhead
	// This is a rough check; in practice, you'd have more sophisticated benchmarks
	if elapsed > 2*time.Second {
		t.Errorf("Training with callbacks took too long: %v", elapsed)
	}

	t.Logf("Training with %d callbacks completed in %v", len(callbacks), elapsed)
}

// Helper function to check if a string contains a substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && s[:len(substr)] == substr ||
		(len(s) > len(substr) && contains(s[1:], substr))
}

// Benchmark callback overhead
func BenchmarkCallbackOverhead(b *testing.B) {
	model := NewMockModel()
	model.Compile()

	// Add typical callbacks
	model.AddCallback(NewVerboseCallback(0))
	model.AddCallback(NewEarlyStoppingCallback("loss", 100, 0.001).WithVerbose(false))
	model.AddCallback(NewHistoryCallback())

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		model.Train(10)
	}
}

// Benchmark individual callback performance
func BenchmarkIndividualCallbacks(b *testing.B) {
	callbacks := map[string]core.Callback{
		"Verbose":       NewVerboseCallback(0),
		"EarlyStopping": NewEarlyStoppingCallback("loss", 10, 0.001).WithVerbose(false),
		"History":       NewHistoryCallback(),
	}

	for name, callback := range callbacks {
		b.Run(name, func(b *testing.B) {
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				callback.OnTrainBegin()

				for epoch := 0; epoch < 100; epoch++ {
					callback.OnEpochBegin(epoch)

					metrics := map[string]float64{
						"accuracy":  0.8,
						"precision": 0.75,
					}

					callback.OnEpochEnd(epoch, 100, 0.5, metrics)
				}

				callback.OnTrainEnd()
			}
		})
	}
}
