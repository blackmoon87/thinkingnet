package layers

import (
	"fmt"
	"math"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// Dense represents a fully connected (dense) neural network layer.
type Dense struct {
	*BaseLayer

	// Layer configuration
	units       int
	activation  core.Activation
	useBias     bool
	initializer WeightInitializer

	// Layer parameters
	weights core.Tensor
	biases  core.Tensor

	// Gradients
	weightGrads core.Tensor
	biasGrads   core.Tensor

	// Forward pass cache
	lastInput     core.Tensor
	preActivation core.Tensor

	// Built flag
	built    bool
	inputDim int
}

// DenseConfig holds configuration options for Dense layer.
type DenseConfig struct {
	Activation  core.Activation
	UseBias     bool
	Initializer WeightInitializer
}

// NewDense creates a new Dense layer.
func NewDense(units int, config *DenseConfig) *Dense {
	if config == nil {
		config = &DenseConfig{
			Activation:  nil, // Linear by default
			UseBias:     true,
			Initializer: XavierUniform,
		}
	}

	return &Dense{
		BaseLayer:   NewBaseLayer(fmt.Sprintf("dense_%d", units)),
		units:       units,
		activation:  config.Activation,
		useBias:     config.UseBias,
		initializer: config.Initializer,
		built:       false,
	}
}

// Build initializes the layer parameters based on input dimensions.
func (d *Dense) Build(inputDim int) error {
	if d.built {
		return nil
	}

	if inputDim <= 0 {
		return core.NewError(core.ErrInvalidInput, "input dimension must be positive").
			WithContext("input_dim", inputDim)
	}

	if d.units <= 0 {
		return core.NewError(core.ErrInvalidInput, "number of units must be positive").
			WithContext("units", d.units)
	}

	d.inputDim = inputDim

	// Initialize weights
	d.weights = InitializeWeights(inputDim, d.units, d.initializer)
	if d.weights == nil {
		return core.NewError(core.ErrMemory, "failed to initialize weights")
	}
	d.weights.SetName(fmt.Sprintf("%s_weights", d.name))

	// Initialize biases if enabled
	if d.useBias {
		d.biases = core.NewZerosTensor(1, d.units)
		if d.biases == nil {
			return core.NewError(core.ErrMemory, "failed to initialize biases")
		}
		d.biases.SetName(fmt.Sprintf("%s_biases", d.name))
	}

	// Initialize gradient tensors
	d.weightGrads = core.NewZerosTensor(inputDim, d.units)
	if d.weightGrads == nil {
		return core.NewError(core.ErrMemory, "failed to initialize weight gradients")
	}

	if d.useBias {
		d.biasGrads = core.NewZerosTensor(1, d.units)
		if d.biasGrads == nil {
			return core.NewError(core.ErrMemory, "failed to initialize bias gradients")
		}
	}

	d.built = true
	return nil
}

// Forward performs the forward pass through the dense layer.
func (d *Dense) Forward(input core.Tensor) (core.Tensor, error) {
	if err := core.ValidateNonEmpty(input, "input"); err != nil {
		return nil, err
	}

	if err := core.ValidateTensorFinite(input, "input"); err != nil {
		return nil, err
	}

	// Build layer if not already built
	if !d.built {
		_, inputDim := input.Dims()
		if err := d.Build(inputDim); err != nil {
			return nil, core.NewErrorWithCause(core.ErrConfigurationError, "failed to build dense layer", err)
		}
	}

	// Validate input dimensions
	_, cols := input.Dims()
	if cols != d.inputDim {
		return nil, core.NewError(core.ErrDimensionMismatch,
			fmt.Sprintf("input dimension %d doesn't match expected %d", cols, d.inputDim)).
			WithContext("input_dim", cols).
			WithContext("expected_dim", d.inputDim)
	}

	// Cache input for backward pass
	d.lastInput = input.Copy()

	// Compute pre-activation: input @ weights
	d.preActivation = input.Mul(d.weights)

	// Add bias if enabled
	if d.useBias {
		// Broadcast bias across all batch samples
		batchSize, _ := d.preActivation.Dims()
		for i := 0; i < batchSize; i++ {
			for j := 0; j < d.units; j++ {
				current := d.preActivation.At(i, j)
				bias := d.biases.At(0, j)
				d.preActivation.Set(i, j, current+bias)
			}
		}
	}

	// Apply activation function if specified
	if d.activation != nil {
		output := d.preActivation.Copy()
		batchSize, units := output.Dims()

		// Handle softmax specially (operates on vectors)
		if d.activation.Name() == "softmax" {
			for i := 0; i < batchSize; i++ {
				// Extract row for softmax
				row := make([]float64, units)
				for j := 0; j < units; j++ {
					row[j] = d.preActivation.At(i, j)
				}

				// Apply softmax
				softmaxRow := applySoftmax(row)

				// Set result
				for j := 0; j < units; j++ {
					output.Set(i, j, softmaxRow[j])
				}
			}
		} else {
			// Use optimized parallel activation processing for large tensors
			totalElements := batchSize * units
			if totalElements >= 10000 {
				// Extract data for parallel processing
				inputData := make([]float64, totalElements)
				outputData := make([]float64, totalElements)

				idx := 0
				for i := 0; i < batchSize; i++ {
					for j := 0; j < units; j++ {
						inputData[idx] = d.preActivation.At(i, j)
						idx++
					}
				}

				// Apply parallel activation
				activationProcessor := core.GetParallelActivationProcessor()
				switch d.activation.Name() {
				case "relu":
					activationProcessor.ProcessReLU(inputData, outputData)
				case "sigmoid":
					activationProcessor.ProcessSigmoid(inputData, outputData)
				case "tanh":
					activationProcessor.ProcessTanh(inputData, outputData)
				default:
					// Fallback to sequential for other activations
					for i := 0; i < len(inputData); i++ {
						outputData[i] = d.activation.Forward(inputData[i])
					}
				}

				// Copy results back
				idx = 0
				for i := 0; i < batchSize; i++ {
					for j := 0; j < units; j++ {
						output.Set(i, j, outputData[idx])
						idx++
					}
				}
			} else {
				// Apply activation element-wise for smaller tensors
				for i := 0; i < batchSize; i++ {
					for j := 0; j < units; j++ {
						val := d.preActivation.At(i, j)
						activated := d.activation.Forward(val)
						output.Set(i, j, activated)
					}
				}
			}
		}

		// Validate output
		if err := core.ValidateTensorFinite(output, "layer_output"); err != nil {
			return nil, core.NewErrorWithCause(core.ErrNumericalInstability, "layer output contains non-finite values", err)
		}

		return output, nil
	}

	// Return pre-activation if no activation function
	result := d.preActivation.Copy()

	// Validate output
	if err := core.ValidateTensorFinite(result, "layer_output"); err != nil {
		return nil, core.NewErrorWithCause(core.ErrNumericalInstability, "layer output contains non-finite values", err)
	}

	return result, nil
}

// Backward performs the backward pass through the dense layer.
func (d *Dense) Backward(gradient core.Tensor) (core.Tensor, error) {
	if err := core.ValidateNotFitted(d.built, "dense layer"); err != nil {
		return nil, err
	}

	if err := core.ValidateNonEmpty(gradient, "gradient"); err != nil {
		return nil, err
	}

	if err := core.ValidateTensorFinite(gradient, "gradient"); err != nil {
		return nil, err
	}

	batchSize, units := gradient.Dims()

	// Validate gradient dimensions
	if units != d.units {
		return nil, core.NewError(core.ErrDimensionMismatch,
			fmt.Sprintf("gradient units %d doesn't match layer units %d", units, d.units)).
			WithContext("gradient_units", units).
			WithContext("layer_units", d.units)
	}

	// Compute activation gradient
	activationGrad := gradient.Copy()

	if d.activation != nil {
		if d.activation.Name() == "softmax" {
			// For softmax, gradient is already computed correctly in loss function
			// (assuming categorical crossentropy), so we pass it through
		} else {
			// Apply activation derivative element-wise
			for i := 0; i < batchSize; i++ {
				for j := 0; j < units; j++ {
					preActVal := d.preActivation.At(i, j)
					gradVal := gradient.At(i, j)
					actDeriv := d.activation.Backward(preActVal)
					activationGrad.Set(i, j, gradVal*actDeriv)
				}
			}
		}
	}

	// Compute weight gradients: input^T @ activation_grad
	d.weightGrads = d.lastInput.T().Mul(activationGrad)

	// Compute bias gradients if enabled
	if d.useBias {
		// Sum gradients across batch dimension
		for j := 0; j < units; j++ {
			var sum float64
			for i := 0; i < batchSize; i++ {
				sum += activationGrad.At(i, j)
			}
			d.biasGrads.Set(0, j, sum)
		}
	}

	// Compute input gradients: activation_grad @ weights^T
	inputGrad := activationGrad.Mul(d.weights.T())

	// Validate output gradient
	if err := core.ValidateTensorFinite(inputGrad, "input_gradient"); err != nil {
		return nil, core.NewErrorWithCause(core.ErrNumericalInstability, "input gradient contains non-finite values", err)
	}

	return inputGrad, nil
}

// Parameters returns all trainable parameters.
func (d *Dense) Parameters() []core.Tensor {
	if !d.built {
		return nil
	}

	params := []core.Tensor{d.weights}
	if d.useBias {
		params = append(params, d.biases)
	}
	return params
}

// Gradients returns gradients for all parameters.
func (d *Dense) Gradients() []core.Tensor {
	if !d.built {
		return nil
	}

	grads := []core.Tensor{d.weightGrads}
	if d.useBias {
		grads = append(grads, d.biasGrads)
	}
	return grads
}

// IsTrainable returns true since Dense layer has trainable parameters.
func (d *Dense) IsTrainable() bool {
	return true
}

// OutputShape returns the output shape given input shape.
func (d *Dense) OutputShape(inputShape []int) ([]int, error) {
	if len(inputShape) != 2 {
		return nil, core.NewError(core.ErrInvalidInput, "input shape must be 2D [batch_size, features]").
			WithContext("input_shape_length", len(inputShape)).
			WithContext("input_shape", inputShape)
	}

	if inputShape[1] <= 0 {
		return nil, core.NewError(core.ErrInvalidInput, "feature dimension must be positive").
			WithContext("feature_dim", inputShape[1])
	}

	return []int{inputShape[0], d.units}, nil
}

// ParameterCount returns the number of trainable parameters.
func (d *Dense) ParameterCount() int {
	if !d.built {
		return 0
	}

	count := d.inputDim * d.units
	if d.useBias {
		count += d.units
	}
	return count
}

// GetUnits returns the number of units in the layer.
func (d *Dense) GetUnits() int {
	return d.units
}

// GetActivation returns the activation function.
func (d *Dense) GetActivation() core.Activation {
	return d.activation
}

// SetActivation sets the activation function.
func (d *Dense) SetActivation(activation core.Activation) {
	d.activation = activation
}

// applySoftmax applies softmax to a vector.
func applySoftmax(x []float64) []float64 {
	if len(x) == 0 {
		return x
	}

	// Find max for numerical stability
	max := x[0]
	for _, val := range x[1:] {
		if val > max {
			max = val
		}
	}

	// Compute exp(x - max)
	result := make([]float64, len(x))
	var sum float64
	for i, val := range x {
		exp := math.Exp(val - max)
		result[i] = exp
		sum += exp
	}

	// Normalize
	for i := range result {
		result[i] /= sum
	}

	return result
}
