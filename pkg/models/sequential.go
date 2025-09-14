// Package models provides neural network model implementations.
package models

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// Sequential represents a sequential neural network model.
type Sequential struct {
	layers    []core.Layer
	optimizer core.Optimizer
	loss      core.Loss
	compiled  bool
	history   *core.History
	config    *core.ModelConfig
	callbacks []core.Callback
}

// NewSequential creates a new sequential model.
func NewSequential(options ...core.ModelOption) *Sequential {
	config := core.NewModelConfig(options...)

	return &Sequential{
		layers:    make([]core.Layer, 0),
		history:   &core.History{},
		config:    config,
		callbacks: make([]core.Callback, 0),
	}
}

// AddLayer adds a layer to the model.
func (m *Sequential) AddLayer(layer core.Layer) error {
	if layer == nil {
		return core.NewError(core.ErrInvalidInput, "layer cannot be nil")
	}

	// Set layer name if not already set
	if layer.Name() == "" {
		layerType := strings.Split(fmt.Sprintf("%T", layer), ".")[1]
		layerName := fmt.Sprintf("%s_%d", strings.ToLower(layerType), len(m.layers))
		layer.SetName(layerName)
	}

	m.layers = append(m.layers, layer)
	return nil
}

// Layers returns all layers in the model.
func (m *Sequential) Layers() []core.Layer {
	return m.layers
}

// Compile compiles the model with optimizer and loss function.
func (m *Sequential) Compile(optimizer core.Optimizer, loss core.Loss) error {
	if optimizer == nil {
		return core.NewError(core.ErrInvalidInput, "optimizer cannot be nil")
	}
	if loss == nil {
		return core.NewError(core.ErrInvalidInput, "loss function cannot be nil")
	}

	if len(m.layers) == 0 {
		return core.NewError(core.ErrConfigurationError, "model must have at least one layer before compilation")
	}

	m.optimizer = optimizer
	m.loss = loss
	m.compiled = true

	return nil
}

// Forward performs forward pass through all layers.
func (m *Sequential) Forward(input core.Tensor) (core.Tensor, error) {
	if len(m.layers) == 0 {
		return input, nil
	}

	current := input
	for _, layer := range m.layers {
		var err error
		current, err = layer.Forward(current)
		if err != nil {
			return nil, err
		}
	}

	return current, nil
}

// Backward performs backward pass through all layers.
func (m *Sequential) Backward(loss core.Loss, yTrue, yPred core.Tensor) error {
	if err := core.ValidateCompiled(m.compiled); err != nil {
		return err
	}

	if err := core.ValidateTrainingData(yTrue, yPred); err != nil {
		return err
	}

	// Compute loss gradient
	gradient := loss.Gradient(yTrue, yPred)

	// Propagate gradients backward through layers
	for i := len(m.layers) - 1; i >= 0; i-- {
		var err error
		gradient, err = m.layers[i].Backward(gradient)
		if err != nil {
			return err
		}
	}

	return nil
}

// Predict makes predictions on input data.
func (m *Sequential) Predict(X core.Tensor) (core.Tensor, error) {
	if err := core.ValidateCompiled(m.compiled); err != nil {
		return nil, err
	}

	if err := core.ValidateNonEmpty(X, "input"); err != nil {
		return nil, err
	}

	if err := core.ValidateTensorFinite(X, "input"); err != nil {
		return nil, err
	}

	result, err := m.Forward(X)
	return result, err
}

// PredictBatch makes predictions on multiple input tensors in parallel.
func (m *Sequential) PredictBatch(inputs []core.Tensor) ([]core.Tensor, error) {
	if err := core.ValidateCompiled(m.compiled); err != nil {
		return nil, err
	}

	if len(inputs) == 0 {
		return nil, core.NewError(core.ErrInvalidInput, "input batch cannot be empty")
	}

	// Validate all inputs
	for i, input := range inputs {
		if err := core.ValidateNonEmpty(input, fmt.Sprintf("input[%d]", i)); err != nil {
			return nil, err
		}
		if err := core.ValidateTensorFinite(input, fmt.Sprintf("input[%d]", i)); err != nil {
			return nil, err
		}
	}

	// Use batch processor for parallel prediction
	batchProcessor := core.GetBatchProcessor()
	results := batchProcessor.ProcessBatches(inputs, func(input core.Tensor) core.Tensor {
		result, err := m.Forward(input)
		if err != nil {
			// Return zero tensor on error - could be improved with better error handling
			rows, cols := input.Dims()
			return core.NewZerosTensor(rows, cols)
		}
		return result
	})

	return results, nil
}

// Fit trains the model on the provided data.
func (m *Sequential) Fit(X, y core.Tensor, config core.TrainingConfig) (*core.History, error) {
	if err := core.ValidateCompiled(m.compiled); err != nil {
		return nil, err
	}

	if err := core.ValidateTrainingData(X, y); err != nil {
		return nil, err
	}

	if err := core.ValidateTensorFinite(X, "training input"); err != nil {
		return nil, err
	}

	if err := core.ValidateTensorFinite(y, "training target"); err != nil {
		return nil, err
	}

	if err := validateTrainingConfig(config); err != nil {
		return nil, err
	}

	// Set random seed for reproducibility
	if config.Seed != 0 {
		rand.Seed(config.Seed)
	}

	// Initialize history
	m.history = &core.History{
		Epoch:      make([]int, 0),
		Loss:       make([]float64, 0),
		Metrics:    make(map[string][]float64),
		ValLoss:    make([]float64, 0),
		ValMetrics: make(map[string][]float64),
	}

	// Initialize metrics maps
	for _, metric := range config.Metrics {
		m.history.Metrics[metric] = make([]float64, 0)
		m.history.ValMetrics[metric] = make([]float64, 0)
	}

	// Split data for validation if needed
	var XTrain, yTrain, XVal, yVal core.Tensor
	if config.ValidationSplit > 0 {
		split := m.splitData(X, y, config.ValidationSplit, config.Shuffle, config.Seed)
		XTrain, yTrain = split.XTrain, split.YTrain
		XVal, yVal = split.XVal, split.YVal
	} else {
		XTrain, yTrain = X, y
	}

	// Training callbacks
	for _, callback := range m.callbacks {
		callback.OnTrainBegin()
	}

	startTime := time.Now()
	bestLoss := math.Inf(1)
	patience := 0

	// Training loop
	for epoch := 0; epoch < config.Epochs; epoch++ {
		epochStart := time.Now()

		// Epoch callbacks
		for _, callback := range m.callbacks {
			callback.OnEpochBegin(epoch)
		}

		// Check for early stopping before training epoch
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
			if config.Verbose > 0 {
				fmt.Printf("Early stopping triggered at epoch %d\n", epoch+1)
			}
			break
		}

		// Train for one epoch
		epochLoss, epochMetrics := m.trainEpoch(XTrain, yTrain, config)

		// Record training metrics
		m.history.Epoch = append(m.history.Epoch, epoch+1)
		m.history.Loss = append(m.history.Loss, epochLoss)

		for metric, value := range epochMetrics {
			if _, exists := m.history.Metrics[metric]; !exists {
				m.history.Metrics[metric] = make([]float64, 0)
			}
			m.history.Metrics[metric] = append(m.history.Metrics[metric], value)
		}

		// Validation
		var valLoss float64
		var valMetrics map[string]float64
		if XVal != nil && yVal != nil {
			valLoss, valMetrics = m.validateEpoch(XVal, yVal)
			m.history.ValLoss = append(m.history.ValLoss, valLoss)

			for metric, value := range valMetrics {
				if _, exists := m.history.ValMetrics[metric]; !exists {
					m.history.ValMetrics[metric] = make([]float64, 0)
				}
				m.history.ValMetrics[metric] = append(m.history.ValMetrics[metric], value)
			}
		}

		// Early stopping check
		if config.EarlyStopping.Enabled {
			currentLoss := epochLoss
			if XVal != nil {
				currentLoss = valLoss
			}

			if currentLoss < bestLoss-config.EarlyStopping.MinDelta {
				bestLoss = currentLoss
				patience = 0
				m.history.BestEpoch = epoch + 1
				m.history.BestScore = bestLoss
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

		// Verbose logging
		if config.Verbose > 0 && (epoch+1)%config.Verbose == 0 {
			elapsed := time.Since(epochStart)
			m.logEpoch(epoch+1, config.Epochs, epochLoss, epochMetrics, valLoss, valMetrics, elapsed)
		}

		// Epoch end callbacks
		for _, callback := range m.callbacks {
			callback.OnEpochEnd(epoch, config.Epochs, epochLoss, epochMetrics)
		}
	}

	m.history.Duration = time.Since(startTime)

	// Training end callbacks
	for _, callback := range m.callbacks {
		callback.OnTrainEnd()
	}

	return m.history, nil
}

// trainEpoch trains the model for one epoch.
func (m *Sequential) trainEpoch(X, y core.Tensor, config core.TrainingConfig) (float64, map[string]float64) {
	numSamples, _ := X.Dims()
	numBatches := (numSamples + config.BatchSize - 1) / config.BatchSize

	totalLoss := 0.0
	totalMetrics := make(map[string]float64)

	// Initialize metrics
	for _, metric := range config.Metrics {
		totalMetrics[metric] = 0.0
	}

	// Process batches
	for batch := 0; batch < numBatches; batch++ {
		// Batch callbacks
		for _, callback := range m.callbacks {
			callback.OnBatchBegin(batch)
		}

		// Get batch data
		startIdx := batch * config.BatchSize
		endIdx := startIdx + config.BatchSize
		if endIdx > numSamples {
			endIdx = numSamples
		}

		_, xCols := X.Dims()
		_, yCols := y.Dims()
		XBatch := X.Slice(startIdx, endIdx, 0, xCols)
		yBatch := y.Slice(startIdx, endIdx, 0, yCols)

		// Forward pass
		yPred, err := m.Forward(XBatch)
		if err != nil {
			return 0, make(map[string]float64)
		}

		// Compute loss
		batchLoss := m.loss.Compute(yBatch, yPred)
		totalLoss += batchLoss

		// Compute metrics
		batchMetrics := m.computeMetrics(yBatch, yPred, config.Metrics)
		for metric, value := range batchMetrics {
			totalMetrics[metric] += value
		}

		// Backward pass
		if err := m.Backward(m.loss, yBatch, yPred); err != nil {
			return 0, nil // Return zero loss on error - could be improved with better error handling
		}

		// Update parameters
		m.updateParameters()

		// Batch end callbacks
		for _, callback := range m.callbacks {
			callback.OnBatchEnd(batch, batchLoss)
		}
	}

	// Average metrics over batches
	avgLoss := totalLoss / float64(numBatches)
	avgMetrics := make(map[string]float64)
	for metric, total := range totalMetrics {
		avgMetrics[metric] = total / float64(numBatches)
	}

	return avgLoss, avgMetrics
}

// validateEpoch validates the model for one epoch.
func (m *Sequential) validateEpoch(XVal, yVal core.Tensor) (float64, map[string]float64) {
	// Forward pass (no training mode)
	yPred, err := m.Forward(XVal)
	if err != nil {
		return 0, make(map[string]float64)
	}

	// Compute validation loss
	valLoss := m.loss.Compute(yVal, yPred)

	// Compute validation metrics
	valMetrics := m.computeMetrics(yVal, yPred, []string{"accuracy"})

	return valLoss, valMetrics
}

// updateParameters updates model parameters using the optimizer.
func (m *Sequential) updateParameters() {
	// Collect all parameters and gradients from all layers
	var allParams []core.Tensor
	var allGrads []core.Tensor

	for _, layer := range m.layers {
		if layer.IsTrainable() {
			params := layer.Parameters()
			grads := layer.Gradients()

			allParams = append(allParams, params...)
			allGrads = append(allGrads, grads...)
		}
	}

	// Update all parameters at once if we have any
	if len(allParams) > 0 && len(allGrads) > 0 {
		m.optimizer.Update(allParams, allGrads)
	}

	m.optimizer.Step()
}

// computeMetrics computes evaluation metrics.
func (m *Sequential) computeMetrics(yTrue, yPred core.Tensor, metrics []string) map[string]float64 {
	result := make(map[string]float64)

	for _, metric := range metrics {
		switch metric {
		case "accuracy":
			result[metric] = m.computeAccuracy(yTrue, yPred)
		case "mse":
			result[metric] = m.computeMSE(yTrue, yPred)
		case "mae":
			result[metric] = m.computeMAE(yTrue, yPred)
		}
	}

	return result
}

// computeAccuracy computes classification accuracy.
func (m *Sequential) computeAccuracy(yTrue, yPred core.Tensor) float64 {
	rows, cols := yTrue.Dims()
	correct := 0

	if cols == 1 {
		// Binary classification
		for i := 0; i < rows; i++ {
			pred := 0.0
			if yPred.At(i, 0) >= 0.5 {
				pred = 1.0
			}
			if pred == yTrue.At(i, 0) {
				correct++
			}
		}
	} else {
		// Multi-class classification
		for i := 0; i < rows; i++ {
			// Find predicted class (argmax)
			predClass := 0
			maxPred := yPred.At(i, 0)
			for j := 1; j < cols; j++ {
				if yPred.At(i, j) > maxPred {
					maxPred = yPred.At(i, j)
					predClass = j
				}
			}

			// Find true class (argmax)
			trueClass := 0
			maxTrue := yTrue.At(i, 0)
			for j := 1; j < cols; j++ {
				if yTrue.At(i, j) > maxTrue {
					maxTrue = yTrue.At(i, j)
					trueClass = j
				}
			}

			if predClass == trueClass {
				correct++
			}
		}
	}

	return float64(correct) / float64(rows)
}

// computeMSE computes mean squared error.
func (m *Sequential) computeMSE(yTrue, yPred core.Tensor) float64 {
	rows, cols := yTrue.Dims()
	totalError := 0.0

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			diff := yTrue.At(i, j) - yPred.At(i, j)
			totalError += diff * diff
		}
	}

	return totalError / float64(rows*cols)
}

// computeMAE computes mean absolute error.
func (m *Sequential) computeMAE(yTrue, yPred core.Tensor) float64 {
	rows, cols := yTrue.Dims()
	totalError := 0.0

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			diff := yTrue.At(i, j) - yPred.At(i, j)
			totalError += math.Abs(diff)
		}
	}

	return totalError / float64(rows*cols)
}

// splitData splits data into training and validation sets.
func (m *Sequential) splitData(X, y core.Tensor, validationSplit float64, shuffle bool, seed int64) *core.DataSplit {
	numSamples, _ := X.Dims()
	numVal := int(float64(numSamples) * validationSplit)
	numTrain := numSamples - numVal

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

	// Create training data
	XTrain := m.createSubset(X, indices[:numTrain])
	yTrain := m.createSubset(y, indices[:numTrain])

	// Create validation data
	XVal := m.createSubset(X, indices[numTrain:])
	yVal := m.createSubset(y, indices[numTrain:])

	return &core.DataSplit{
		XTrain: XTrain,
		YTrain: yTrain,
		XVal:   XVal,
		YVal:   yVal,
	}
}

// createSubset creates a subset of tensor based on indices.
func (m *Sequential) createSubset(tensor core.Tensor, indices []int) core.Tensor {
	_, cols := tensor.Dims()
	subset := tensor.Copy()
	subset = subset.Reshape(len(indices), cols)

	for i, idx := range indices {
		for j := 0; j < cols; j++ {
			subset.Set(i, j, tensor.At(idx, j))
		}
	}

	return subset
}

// logEpoch logs training progress for an epoch.
func (m *Sequential) logEpoch(epoch, maxEpochs int, loss float64, metrics map[string]float64,
	valLoss float64, valMetrics map[string]float64, elapsed time.Duration) {

	logMsg := fmt.Sprintf("Epoch %d/%d [%.2fs] - loss: %.4f",
		epoch, maxEpochs, elapsed.Seconds(), loss)

	// Add training metrics
	for metric, value := range metrics {
		logMsg += fmt.Sprintf(" - %s: %.4f", metric, value)
	}

	// Add validation metrics if available
	if valMetrics != nil {
		logMsg += fmt.Sprintf(" - val_loss: %.4f", valLoss)
		for metric, value := range valMetrics {
			logMsg += fmt.Sprintf(" - val_%s: %.4f", metric, value)
		}
	}

	fmt.Println(logMsg)
}

// Evaluate evaluates the model on test data.
func (m *Sequential) Evaluate(X, y core.Tensor) (*core.Metrics, error) {
	if err := core.ValidateCompiled(m.compiled); err != nil {
		return nil, err
	}

	if err := core.ValidateTrainingData(X, y); err != nil {
		return nil, err
	}

	if err := core.ValidateTensorFinite(X, "evaluation input"); err != nil {
		return nil, err
	}

	if err := core.ValidateTensorFinite(y, "evaluation target"); err != nil {
		return nil, err
	}

	// Make predictions
	yPred, err := m.Predict(X)
	if err != nil {
		return nil, core.NewErrorWithCause(core.ErrModelNotCompiled, "prediction failed during evaluation", err)
	}

	// Compute metrics
	metrics := &core.Metrics{}

	// Compute accuracy
	metrics.Accuracy = m.computeAccuracy(y, yPred)

	// Compute MSE and RMSE
	metrics.MSE = m.computeMSE(y, yPred)
	metrics.RMSE = math.Sqrt(metrics.MSE)

	// Compute MAE
	metrics.MAE = m.computeMAE(y, yPred)

	return metrics, nil
}

// Summary returns a string representation of the model architecture.
func (m *Sequential) Summary() string {
	var summary strings.Builder

	summary.WriteString("Model: Sequential\n")
	summary.WriteString("_________________________________________________________________\n")
	summary.WriteString(fmt.Sprintf("%-25s %-20s %-15s\n", "Layer (type)", "Output Shape", "Param #"))
	summary.WriteString("=================================================================\n")

	totalParams := 0
	trainableParams := 0

	// Assume input shape for first layer
	currentShape := []int{1, 784} // Default input shape

	for i, layer := range m.layers {
		layerName := layer.Name()
		if layerName == "" {
			layerType := strings.Split(fmt.Sprintf("%T", layer), ".")[1]
			layerName = fmt.Sprintf("%s_%d", strings.ToLower(layerType), i)
		}

		layerType := fmt.Sprintf("(%s)", strings.Split(fmt.Sprintf("%T", layer), ".")[1])
		fullLayerName := fmt.Sprintf("%s %s", layerName, layerType)

		params := layer.ParameterCount()
		totalParams += params
		if layer.IsTrainable() {
			trainableParams += params
		}

		outputShape, err := layer.OutputShape(currentShape)
		if err != nil {
			return fmt.Sprintf("Error computing output shape: %v", err)
		}
		shapeStr := fmt.Sprintf("(%s)", formatShape(outputShape))

		summary.WriteString(fmt.Sprintf("%-25s %-20s %-15d\n", fullLayerName, shapeStr, params))
		currentShape = outputShape
	}

	summary.WriteString("=================================================================\n")
	summary.WriteString(fmt.Sprintf("Total params: %d\n", totalParams))
	summary.WriteString(fmt.Sprintf("Trainable params: %d\n", trainableParams))
	summary.WriteString(fmt.Sprintf("Non-trainable params: %d\n", totalParams-trainableParams))
	summary.WriteString("_________________________________________________________________\n")

	return summary.String()
}

// formatShape formats a shape slice as a string.
func formatShape(shape []int) string {
	if len(shape) == 0 {
		return "None"
	}

	parts := make([]string, len(shape))
	for i, dim := range shape {
		if dim < 0 {
			parts[i] = "None"
		} else {
			parts[i] = fmt.Sprintf("%d", dim)
		}
	}

	return strings.Join(parts, ", ")
}

// Save saves the model to a file.
func (m *Sequential) Save(path string) error {
	// TODO: Implement model serialization
	return fmt.Errorf("model saving not yet implemented")
}

// Load loads the model from a file.
func (m *Sequential) Load(path string) error {
	// TODO: Implement model deserialization
	return fmt.Errorf("model loading not yet implemented")
}

// AddCallback adds a callback to the model.
func (m *Sequential) AddCallback(callback core.Callback) error {
	if callback == nil {
		return core.NewError(core.ErrInvalidInput, "callback cannot be nil")
	}
	m.callbacks = append(m.callbacks, callback)
	return nil
}

// GetHistory returns the training history.
func (m *Sequential) GetHistory() *core.History {
	return m.history
}

// IsCompiled returns true if the model has been compiled.
func (m *Sequential) IsCompiled() bool {
	return m.compiled
}

// EasyTrain trains the model with sensible defaults and better error messages.
// This is a simplified version of Fit() that provides good default parameters
// for beginners and includes Arabic error messages.
func (m *Sequential) EasyTrain(X, y core.Tensor) (*core.History, error) {
	// Validate model is compiled with Arabic error message
	if !m.compiled {
		return nil, core.NewError(core.ErrModelNotCompiled,
			"يجب تجميع النموذج أولاً باستخدام Compile() - Model must be compiled first using Compile()")
	}

	// Validate input data with Arabic error messages
	if X == nil {
		return nil, core.NewError(core.ErrInvalidInput,
			"بيانات الإدخال لا يمكن أن تكون فارغة - Input data cannot be nil")
	}

	if y == nil {
		return nil, core.NewError(core.ErrInvalidInput,
			"بيانات الهدف لا يمكن أن تكون فارغة - Target data cannot be nil")
	}

	// Validate training data compatibility
	if err := core.ValidateTrainingData(X, y); err != nil {
		return nil, core.NewErrorWithCause(core.ErrInvalidInput,
			"بيانات التدريب غير متوافقة - Training data is incompatible", err)
	}

	// Validate tensor values are finite
	if err := core.ValidateTensorFinite(X, "input"); err != nil {
		return nil, core.NewErrorWithCause(core.ErrNumericalInstability,
			"بيانات الإدخال تحتوي على قيم غير صحيحة - Input data contains invalid values", err)
	}

	if err := core.ValidateTensorFinite(y, "target"); err != nil {
		return nil, core.NewErrorWithCause(core.ErrNumericalInstability,
			"بيانات الهدف تحتوي على قيم غير صحيحة - Target data contains invalid values", err)
	}

	// Create training configuration with sensible defaults
	// Note: Using ValidationSplit = 0 to avoid data splitting issues for simplicity
	config := core.TrainingConfig{
		Epochs:          10,                   // Smaller number for quick testing
		BatchSize:       4,                    // Small batch size for small datasets
		ValidationSplit: 0.0,                  // No validation split to keep it simple
		Shuffle:         true,                 // Always shuffle for better training
		Verbose:         1,                    // Show progress
		Seed:            42,                   // Reproducible results
		Metrics:         []string{"accuracy"}, // Basic accuracy metric
		EarlyStopping: core.EarlyStoppingConfig{
			Enabled:  false, // Keep it simple for beginners
			Monitor:  "val_loss",
			Patience: 10,
			MinDelta: 0.0001,
			Mode:     "min",
		},
	}

	// Call the main Fit method with our configuration
	history, err := m.Fit(X, y, config)
	if err != nil {
		return nil, core.NewErrorWithCause(core.ErrConfigurationError,
			"فشل في تدريب النموذج - Model training failed", err)
	}

	return history, nil
}

// EasyPredict makes predictions with better error messages and validation.
// This is a simplified version of Predict() with enhanced error handling
// and Arabic error messages for better user experience.
func (m *Sequential) EasyPredict(X core.Tensor) (core.Tensor, error) {
	// Validate model is compiled with Arabic error message
	if !m.compiled {
		return nil, core.NewError(core.ErrModelNotCompiled,
			"يجب تجميع النموذج أولاً باستخدام Compile() - Model must be compiled first using Compile()")
	}

	// Validate input data with Arabic error messages
	if X == nil {
		return nil, core.NewError(core.ErrInvalidInput,
			"بيانات الإدخال لا يمكن أن تكون فارغة - Input data cannot be nil")
	}

	// Check if model has layers
	if len(m.layers) == 0 {
		return nil, core.NewError(core.ErrConfigurationError,
			"النموذج يجب أن يحتوي على طبقات - Model must have layers")
	}

	// Validate input is not empty
	if err := core.ValidateNonEmpty(X, "input"); err != nil {
		return nil, core.NewErrorWithCause(core.ErrInvalidInput,
			"بيانات الإدخال فارغة - Input data is empty", err)
	}

	// Validate input contains finite values
	if err := core.ValidateTensorFinite(X, "input"); err != nil {
		return nil, core.NewErrorWithCause(core.ErrNumericalInstability,
			"بيانات الإدخال تحتوي على قيم غير صحيحة (NaN أو لا نهائية) - Input data contains invalid values (NaN or infinite)", err)
	}

	// Validate input dimensions are reasonable
	rows, cols := X.Dims()
	if rows <= 0 || cols <= 0 {
		return nil, core.NewError(core.ErrInvalidInput,
			"أبعاد بيانات الإدخال غير صحيحة - Input data dimensions are invalid").
			WithContext("rows", rows).WithContext("cols", cols)
	}

	// Make the prediction using the existing Predict method
	result, err := m.Predict(X)
	if err != nil {
		return nil, core.NewErrorWithCause(core.ErrConfigurationError,
			"فشل في التنبؤ - Prediction failed", err)
	}

	// Validate output is finite
	if err := core.ValidateTensorFinite(result, "prediction output"); err != nil {
		return nil, core.NewErrorWithCause(core.ErrNumericalInstability,
			"نتائج التنبؤ تحتوي على قيم غير صحيحة - Prediction results contain invalid values", err)
	}

	return result, nil
}

// validateTrainingConfig validates the training configuration parameters.
func validateTrainingConfig(config core.TrainingConfig) error {
	if config.Epochs <= 0 {
		return core.NewError(core.ErrInvalidInput, "epochs must be positive").
			WithContext("epochs", config.Epochs)
	}

	if config.BatchSize <= 0 {
		return core.NewError(core.ErrInvalidInput, "batch size must be positive").
			WithContext("batch_size", config.BatchSize)
	}

	if err := core.ValidateRange(config.ValidationSplit, 0.0, 1.0, "validation_split"); err != nil {
		return err
	}

	if config.EarlyStopping.Enabled {
		if config.EarlyStopping.Patience <= 0 {
			return core.NewError(core.ErrInvalidInput, "early stopping patience must be positive").
				WithContext("patience", config.EarlyStopping.Patience)
		}

		if err := core.ValidateNonNegative(config.EarlyStopping.MinDelta, "min_delta"); err != nil {
			return err
		}
	}

	if config.Verbose < 0 {
		return core.NewError(core.ErrInvalidInput, "verbose level cannot be negative").
			WithContext("verbose", config.Verbose)
	}

	return nil
}
