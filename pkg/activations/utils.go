package activations

import (
	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// ApplyActivation applies an activation function element-wise to a tensor.
func ApplyActivation(activation core.Activation, input core.Tensor) core.Tensor {
	rows, cols := input.Dims()
	result := core.NewZerosTensor(rows, cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := input.At(i, j)
			activated := activation.Forward(val)
			result.Set(i, j, activated)
		}
	}

	return result
}

// ApplyActivationBackward applies the backward pass of an activation function element-wise.
func ApplyActivationBackward(activation core.Activation, input, gradOutput core.Tensor) core.Tensor {
	if err := core.ValidateDimensions(input, gradOutput, "activation_backward"); err != nil {
		panic(err)
	}

	rows, cols := input.Dims()
	result := core.NewZerosTensor(rows, cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := input.At(i, j)
			grad := gradOutput.At(i, j)
			derivative := activation.Backward(val)
			result.Set(i, j, grad*derivative)
		}
	}

	return result
}

// ApplyActivationInPlace applies an activation function in-place to a tensor.
// This is more memory efficient but modifies the input tensor.
func ApplyActivationInPlace(activation core.Activation, tensor core.Tensor) {
	rows, cols := tensor.Dims()

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := tensor.At(i, j)
			activated := activation.Forward(val)
			tensor.Set(i, j, activated)
		}
	}
}

// BatchApplyActivation applies an activation function to multiple tensors.
func BatchApplyActivation(activation core.Activation, inputs []core.Tensor) []core.Tensor {
	results := make([]core.Tensor, len(inputs))
	for i, input := range inputs {
		results[i] = ApplyActivation(activation, input)
	}
	return results
}

// CompareActivations compares multiple activation functions on the same input.
func CompareActivations(activations []core.Activation, input core.Tensor) map[string]core.Tensor {
	results := make(map[string]core.Tensor)

	for _, activation := range activations {
		results[activation.Name()] = ApplyActivation(activation, input)
	}

	return results
}

// ActivationLayer represents a standalone activation layer.
type ActivationLayer struct {
	activation core.Activation
	lastInput  core.Tensor
	name       string
}

// NewActivationLayer creates a new activation layer.
func NewActivationLayer(activation core.Activation) *ActivationLayer {
	return &ActivationLayer{
		activation: activation,
		name:       "activation_" + activation.Name(),
	}
}

// Forward performs the forward pass through the activation layer.
func (a *ActivationLayer) Forward(input core.Tensor) core.Tensor {
	a.lastInput = input.Copy()

	// Special handling for Softmax
	if softmax, ok := a.activation.(*Softmax); ok {
		return softmax.ApplyTensorwise(input)
	}

	return ApplyActivation(a.activation, input)
}

// Backward performs the backward pass through the activation layer.
func (a *ActivationLayer) Backward(gradient core.Tensor) core.Tensor {
	if a.lastInput == nil {
		panic(core.NewError(core.ErrInvalidInput, "backward called before forward"))
	}

	// Special handling for Softmax
	if _, ok := a.activation.(*Softmax); ok {
		// For softmax, the gradient computation is more complex
		// This is a simplified version - in practice, you'd need the full Jacobian
		return gradient.Copy()
	}

	return ApplyActivationBackward(a.activation, a.lastInput, gradient)
}

// Parameters returns the parameters (none for activation layers).
func (a *ActivationLayer) Parameters() []core.Tensor {
	return nil
}

// Gradients returns the gradients (none for activation layers).
func (a *ActivationLayer) Gradients() []core.Tensor {
	return nil
}

// IsTrainable returns false as activation layers have no trainable parameters.
func (a *ActivationLayer) IsTrainable() bool {
	return false
}

// Name returns the layer name.
func (a *ActivationLayer) Name() string {
	return a.name
}

// SetName sets the layer name.
func (a *ActivationLayer) SetName(name string) {
	a.name = name
}

// OutputShape returns the output shape (same as input for activation layers).
func (a *ActivationLayer) OutputShape(inputShape []int) []int {
	return inputShape
}

// ParameterCount returns 0 as activation layers have no parameters.
func (a *ActivationLayer) ParameterCount() int {
	return 0
}
