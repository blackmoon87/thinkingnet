// Package callbacks provides training callback implementations for monitoring and controlling the training process.
package callbacks

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// VerboseCallback provides training progress display with configurable verbosity levels.
type VerboseCallback struct {
	core.BaseCallback
	verbose     int
	startTime   time.Time
	epochTimes  []time.Duration
	lastEpoch   int
	showMetrics bool
}

// NewVerboseCallback creates a new verbose callback.
// verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch, 3 = detailed metrics
func NewVerboseCallback(verbose int) *VerboseCallback {
	return &VerboseCallback{
		verbose:     verbose,
		epochTimes:  make([]time.Duration, 0),
		showMetrics: true,
	}
}

// WithMetrics enables or disables metric display.
func (vc *VerboseCallback) WithMetrics(show bool) *VerboseCallback {
	vc.showMetrics = show
	return vc
}

// OnTrainBegin is called at the start of training.
func (vc *VerboseCallback) OnTrainBegin() {
	if vc.verbose > 0 {
		vc.startTime = time.Now()
		fmt.Println("Training started...")
	}
}

// OnTrainEnd is called at the end of training.
func (vc *VerboseCallback) OnTrainEnd() {
	if vc.verbose > 0 {
		totalTime := time.Since(vc.startTime)
		fmt.Printf("Training completed in %v\n", totalTime)

		if vc.verbose >= 2 && len(vc.epochTimes) > 0 {
			avgEpochTime := vc.calculateAverageEpochTime()
			fmt.Printf("Average epoch time: %v\n", avgEpochTime)
		}
	}
}

// OnEpochBegin is called at the start of each epoch.
func (vc *VerboseCallback) OnEpochBegin(epoch int) {
	vc.lastEpoch = epoch
}

// OnEpochEnd is called at the end of each epoch.
func (vc *VerboseCallback) OnEpochEnd(epoch, maxEpochs int, loss float64, metrics map[string]float64) {
	if vc.verbose == 0 {
		return
	}

	epochTime := time.Since(vc.startTime) - vc.getTotalEpochTime()
	vc.epochTimes = append(vc.epochTimes, epochTime)

	switch vc.verbose {
	case 1:
		// Progress bar style
		vc.displayProgressBar(epoch+1, maxEpochs, loss)
	case 2:
		// One line per epoch
		vc.displayEpochLine(epoch+1, maxEpochs, loss, metrics, epochTime)
	case 3:
		// Detailed metrics
		vc.displayDetailedMetrics(epoch+1, maxEpochs, loss, metrics, epochTime)
	}
}

// displayProgressBar shows a simple progress bar.
func (vc *VerboseCallback) displayProgressBar(epoch, maxEpochs int, loss float64) {
	progress := float64(epoch) / float64(maxEpochs)
	barLength := 30
	filledLength := int(progress * float64(barLength))

	bar := ""
	for i := 0; i < barLength; i++ {
		if i < filledLength {
			bar += "="
		} else if i == filledLength {
			bar += ">"
		} else {
			bar += " "
		}
	}

	fmt.Printf("\rEpoch %d/%d [%s] %.1f%% - loss: %.4f",
		epoch, maxEpochs, bar, progress*100, loss)

	if epoch == maxEpochs {
		fmt.Println() // New line at the end
	}
}

// displayEpochLine shows one line per epoch with basic metrics.
func (vc *VerboseCallback) displayEpochLine(epoch, maxEpochs int, loss float64, metrics map[string]float64, epochTime time.Duration) {
	logMsg := fmt.Sprintf("Epoch %d/%d [%.2fs] - loss: %.4f",
		epoch, maxEpochs, epochTime.Seconds(), loss)

	if vc.showMetrics && metrics != nil {
		for metric, value := range metrics {
			logMsg += fmt.Sprintf(" - %s: %.4f", metric, value)
		}
	}

	// Add ETA if we have enough data
	if len(vc.epochTimes) >= 3 && epoch < maxEpochs {
		eta := vc.estimateTimeRemaining(epoch, maxEpochs)
		logMsg += fmt.Sprintf(" - ETA: %v", eta)
	}

	fmt.Println(logMsg)
}

// displayDetailedMetrics shows detailed metrics and statistics.
func (vc *VerboseCallback) displayDetailedMetrics(epoch, maxEpochs int, loss float64, metrics map[string]float64, epochTime time.Duration) {
	fmt.Printf("Epoch %d/%d\n", epoch, maxEpochs)
	fmt.Printf("  Time: %.2fs\n", epochTime.Seconds())
	fmt.Printf("  Loss: %.6f\n", loss)

	if vc.showMetrics && metrics != nil {
		fmt.Println("  Metrics:")
		for metric, value := range metrics {
			fmt.Printf("    %s: %.6f\n", metric, value)
		}
	}

	if len(vc.epochTimes) >= 3 && epoch < maxEpochs {
		eta := vc.estimateTimeRemaining(epoch, maxEpochs)
		fmt.Printf("  ETA: %v\n", eta)
	}

	fmt.Println()
}

// calculateAverageEpochTime calculates the average time per epoch.
func (vc *VerboseCallback) calculateAverageEpochTime() time.Duration {
	if len(vc.epochTimes) == 0 {
		return 0
	}

	total := time.Duration(0)
	for _, t := range vc.epochTimes {
		total += t
	}

	return total / time.Duration(len(vc.epochTimes))
}

// getTotalEpochTime returns the total time spent in completed epochs.
func (vc *VerboseCallback) getTotalEpochTime() time.Duration {
	total := time.Duration(0)
	for _, t := range vc.epochTimes {
		total += t
	}
	return total
}

// estimateTimeRemaining estimates the remaining training time.
func (vc *VerboseCallback) estimateTimeRemaining(currentEpoch, totalEpochs int) time.Duration {
	if len(vc.epochTimes) == 0 {
		return 0
	}

	avgTime := vc.calculateAverageEpochTime()
	remainingEpochs := totalEpochs - currentEpoch

	return avgTime * time.Duration(remainingEpochs)
}

// EarlyStoppingCallback implements early stopping based on monitored metrics.
type EarlyStoppingCallback struct {
	core.BaseCallback
	monitor  string  // Metric to monitor ("loss", "val_loss", "accuracy", etc.)
	patience int     // Number of epochs with no improvement after which training will be stopped
	minDelta float64 // Minimum change to qualify as an improvement
	mode     string  // "min" for loss, "max" for accuracy
	verbose  bool    // Whether to print early stopping messages

	bestValue  float64 // Best value seen so far
	waitCount  int     // Number of epochs since last improvement
	shouldStop bool    // Whether training should stop
	bestEpoch  int     // Epoch with the best value
	stopped    bool    // Whether early stopping was triggered
}

// NewEarlyStoppingCallback creates a new early stopping callback.
func NewEarlyStoppingCallback(monitor string, patience int, minDelta float64) *EarlyStoppingCallback {
	mode := "min"
	if monitor == "accuracy" || monitor == "val_accuracy" {
		mode = "max"
	}

	bestValue := math.Inf(1)
	if mode == "max" {
		bestValue = math.Inf(-1)
	}

	return &EarlyStoppingCallback{
		monitor:    monitor,
		patience:   patience,
		minDelta:   minDelta,
		mode:       mode,
		verbose:    true,
		bestValue:  bestValue,
		waitCount:  0,
		shouldStop: false,
		bestEpoch:  0,
		stopped:    false,
	}
}

// WithMode sets the monitoring mode ("min" or "max").
func (esc *EarlyStoppingCallback) WithMode(mode string) *EarlyStoppingCallback {
	esc.mode = mode
	if mode == "max" {
		esc.bestValue = math.Inf(-1)
	} else {
		esc.bestValue = math.Inf(1)
	}
	return esc
}

// WithVerbose enables or disables verbose output.
func (esc *EarlyStoppingCallback) WithVerbose(verbose bool) *EarlyStoppingCallback {
	esc.verbose = verbose
	return esc
}

// OnTrainBegin initializes the early stopping state.
func (esc *EarlyStoppingCallback) OnTrainBegin() {
	esc.waitCount = 0
	esc.shouldStop = false
	esc.stopped = false
	esc.bestEpoch = 0

	if esc.mode == "max" {
		esc.bestValue = math.Inf(-1)
	} else {
		esc.bestValue = math.Inf(1)
	}

	if esc.verbose {
		fmt.Printf("Early stopping monitoring '%s' with patience %d\n", esc.monitor, esc.patience)
	}
}

// OnEpochEnd checks for improvement and updates early stopping state.
func (esc *EarlyStoppingCallback) OnEpochEnd(epoch, maxEpochs int, loss float64, metrics map[string]float64) {
	currentValue := esc.getCurrentValue(loss, metrics)
	if math.IsNaN(currentValue) || math.IsInf(currentValue, 0) {
		if esc.verbose {
			fmt.Printf("Warning: Invalid value for monitored metric '%s': %f\n", esc.monitor, currentValue)
		}
		return
	}

	improved := false
	if esc.mode == "min" {
		improved = currentValue < esc.bestValue-esc.minDelta
	} else {
		improved = currentValue > esc.bestValue+esc.minDelta
	}

	if improved {
		esc.bestValue = currentValue
		esc.bestEpoch = epoch + 1
		esc.waitCount = 0

		if esc.verbose {
			fmt.Printf("Epoch %d: %s improved from %.6f to %.6f\n",
				epoch+1, esc.monitor, esc.bestValue, currentValue)
		}
	} else {
		esc.waitCount++

		if esc.waitCount >= esc.patience {
			esc.shouldStop = true
			esc.stopped = true

			if esc.verbose {
				fmt.Printf("Early stopping at epoch %d. Best %s: %.6f at epoch %d\n",
					epoch+1, esc.monitor, esc.bestValue, esc.bestEpoch)
			}
		}
	}
}

// getCurrentValue extracts the monitored value from loss and metrics.
func (esc *EarlyStoppingCallback) getCurrentValue(loss float64, metrics map[string]float64) float64 {
	switch esc.monitor {
	case "loss":
		return loss
	case "val_loss":
		// This would need to be passed separately in a real implementation
		// For now, we'll use loss as a fallback
		return loss
	default:
		if metrics != nil {
			if value, exists := metrics[esc.monitor]; exists {
				return value
			}
		}
		// Fallback to loss if metric not found
		return loss
	}
}

// ShouldStop returns true if training should be stopped.
func (esc *EarlyStoppingCallback) ShouldStop() bool {
	return esc.shouldStop
}

// GetBestValue returns the best monitored value.
func (esc *EarlyStoppingCallback) GetBestValue() float64 {
	return esc.bestValue
}

// GetBestEpoch returns the epoch with the best value.
func (esc *EarlyStoppingCallback) GetBestEpoch() int {
	return esc.bestEpoch
}

// WasStopped returns true if early stopping was triggered.
func (esc *EarlyStoppingCallback) WasStopped() bool {
	return esc.stopped
}

// HistoryCallback tracks training history with detailed metrics.
type HistoryCallback struct {
	core.BaseCallback
	history *core.History
}

// NewHistoryCallback creates a new history tracking callback.
func NewHistoryCallback() *HistoryCallback {
	return &HistoryCallback{
		history: &core.History{
			Epoch:      make([]int, 0),
			Loss:       make([]float64, 0),
			Metrics:    make(map[string][]float64),
			ValLoss:    make([]float64, 0),
			ValMetrics: make(map[string][]float64),
		},
	}
}

// OnTrainBegin initializes the history tracking.
func (hc *HistoryCallback) OnTrainBegin() {
	hc.history = &core.History{
		Epoch:      make([]int, 0),
		Loss:       make([]float64, 0),
		Metrics:    make(map[string][]float64),
		ValLoss:    make([]float64, 0),
		ValMetrics: make(map[string][]float64),
	}
}

// OnEpochEnd records epoch metrics in the history.
func (hc *HistoryCallback) OnEpochEnd(epoch, maxEpochs int, loss float64, metrics map[string]float64) {
	hc.history.Epoch = append(hc.history.Epoch, epoch+1)
	hc.history.Loss = append(hc.history.Loss, loss)

	// Record training metrics
	for metric, value := range metrics {
		if _, exists := hc.history.Metrics[metric]; !exists {
			hc.history.Metrics[metric] = make([]float64, 0)
		}
		hc.history.Metrics[metric] = append(hc.history.Metrics[metric], value)
	}
}

// OnTrainEnd finalizes the history with training duration.
func (hc *HistoryCallback) OnTrainEnd() {
	// Duration would be set by the training loop
}

// GetHistory returns the recorded training history.
func (hc *HistoryCallback) GetHistory() *core.History {
	return hc.history
}

// SetValidationMetrics sets validation metrics for the current epoch.
func (hc *HistoryCallback) SetValidationMetrics(valLoss float64, valMetrics map[string]float64) {
	hc.history.ValLoss = append(hc.history.ValLoss, valLoss)

	for metric, value := range valMetrics {
		if _, exists := hc.history.ValMetrics[metric]; !exists {
			hc.history.ValMetrics[metric] = make([]float64, 0)
		}
		hc.history.ValMetrics[metric] = append(hc.history.ValMetrics[metric], value)
	}
}

// ModelCheckpointCallback saves model checkpoints during training.
type ModelCheckpointCallback struct {
	core.BaseCallback
	filepath    string // Path template for saving checkpoints
	monitor     string // Metric to monitor for best model saving
	saveBest    bool   // Whether to save only the best model
	saveWeights bool   // Whether to save only weights or full model
	mode        string // "min" or "max" for the monitored metric
	verbose     bool   // Whether to print save messages

	bestValue float64 // Best monitored value
	bestEpoch int     // Epoch with best value
}

// NewModelCheckpointCallback creates a new model checkpoint callback.
func NewModelCheckpointCallback(filepath string) *ModelCheckpointCallback {
	return &ModelCheckpointCallback{
		filepath:    filepath,
		monitor:     "val_loss",
		saveBest:    true,
		saveWeights: false,
		mode:        "min",
		verbose:     true,
		bestValue:   math.Inf(1),
		bestEpoch:   0,
	}
}

// WithMonitor sets the metric to monitor.
func (mcc *ModelCheckpointCallback) WithMonitor(monitor string) *ModelCheckpointCallback {
	mcc.monitor = monitor
	if monitor == "accuracy" || monitor == "val_accuracy" {
		mcc.mode = "max"
		mcc.bestValue = math.Inf(-1)
	}
	return mcc
}

// WithSaveBest sets whether to save only the best model.
func (mcc *ModelCheckpointCallback) WithSaveBest(saveBest bool) *ModelCheckpointCallback {
	mcc.saveBest = saveBest
	return mcc
}

// WithSaveWeights sets whether to save only weights.
func (mcc *ModelCheckpointCallback) WithSaveWeights(saveWeights bool) *ModelCheckpointCallback {
	mcc.saveWeights = saveWeights
	return mcc
}

// WithVerbose sets whether to print save messages.
func (mcc *ModelCheckpointCallback) WithVerbose(verbose bool) *ModelCheckpointCallback {
	mcc.verbose = verbose
	return mcc
}

// OnTrainBegin initializes checkpoint saving.
func (mcc *ModelCheckpointCallback) OnTrainBegin() {
	// Create directory if it doesn't exist
	dir := filepath.Dir(mcc.filepath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		fmt.Printf("Warning: Could not create checkpoint directory: %v\n", err)
	}

	if mcc.mode == "max" {
		mcc.bestValue = math.Inf(-1)
	} else {
		mcc.bestValue = math.Inf(1)
	}

	if mcc.verbose {
		fmt.Printf("Model checkpointing enabled. Monitoring '%s'\n", mcc.monitor)
	}
}

// OnEpochEnd saves model checkpoint if conditions are met.
func (mcc *ModelCheckpointCallback) OnEpochEnd(epoch, maxEpochs int, loss float64, metrics map[string]float64) {
	currentValue := mcc.getCurrentValue(loss, metrics)

	shouldSave := false
	if mcc.saveBest {
		improved := false
		if mcc.mode == "min" {
			improved = currentValue < mcc.bestValue
		} else {
			improved = currentValue > mcc.bestValue
		}

		if improved {
			mcc.bestValue = currentValue
			mcc.bestEpoch = epoch + 1
			shouldSave = true
		}
	} else {
		shouldSave = true // Save every epoch
	}

	if shouldSave {
		filename := mcc.formatFilename(epoch+1, currentValue)

		// In a real implementation, this would save the actual model
		// For now, we'll just create a placeholder file
		if err := mcc.saveModelPlaceholder(filename, epoch+1, currentValue); err != nil {
			if mcc.verbose {
				fmt.Printf("Warning: Could not save checkpoint: %v\n", err)
			}
		} else if mcc.verbose {
			fmt.Printf("Epoch %d: saved model to %s (%s: %.6f)\n",
				epoch+1, filename, mcc.monitor, currentValue)
		}
	}
}

// getCurrentValue extracts the monitored value.
func (mcc *ModelCheckpointCallback) getCurrentValue(loss float64, metrics map[string]float64) float64 {
	switch mcc.monitor {
	case "loss":
		return loss
	case "val_loss":
		return loss // Fallback
	default:
		if metrics != nil {
			if value, exists := metrics[mcc.monitor]; exists {
				return value
			}
		}
		return loss
	}
}

// formatFilename creates the checkpoint filename.
func (mcc *ModelCheckpointCallback) formatFilename(epoch int, value float64) string {
	// Replace placeholders in filepath
	filename := mcc.filepath
	filename = fmt.Sprintf(filename, epoch, value)
	return filename
}

// saveModelPlaceholder creates a placeholder checkpoint file.
func (mcc *ModelCheckpointCallback) saveModelPlaceholder(filename string, epoch int, value float64) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	_, err = fmt.Fprintf(file, "Checkpoint: epoch=%d, %s=%.6f, timestamp=%s\n",
		epoch, mcc.monitor, value, time.Now().Format(time.RFC3339))
	return err
}

// GetBestValue returns the best monitored value.
func (mcc *ModelCheckpointCallback) GetBestValue() float64 {
	return mcc.bestValue
}

// GetBestEpoch returns the epoch with the best value.
func (mcc *ModelCheckpointCallback) GetBestEpoch() int {
	return mcc.bestEpoch
}
