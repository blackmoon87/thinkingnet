package layers

import (
	"fmt"
	"math/rand"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// Dropout represents a dropout regularization layer.
type Dropout struct {
	*BaseLayer

	// Layer configuration
	rate float64 // Dropout rate (probability of setting to zero)

	// Forward pass cache
	mask      core.Tensor // Dropout mask used in forward pass
	lastInput core.Tensor // Input from forward pass

	// Built flag
	built bool
}

// DropoutConfig holds configuration options for Dropout layer.
type DropoutConfig struct {
	Rate float64
}

// NewDropout creates a new Dropout layer.
func NewDropout(rate float64, config *DropoutConfig) *Dropout {
	if rate < 0.0 || rate >= 1.0 {
		panic(core.NewError(core.ErrInvalidInput,
			fmt.Sprintf("dropout rate must be in [0, 1), got %f", rate)))
	}

	return &Dropout{
		BaseLayer: NewBaseLayer(fmt.Sprintf("dropout_%.2f", rate)),
		rate:      rate,
		built:     true, // Dropout doesn't need building
	}
}

// Forward performs the forward pass through the dropout layer.
func (d *Dropout) Forward(input core.Tensor) (core.Tensor, error) {
	// Cache input for backward pass
	d.lastInput = input.Copy()

	// If not training or rate is 0, return input unchanged
	if !d.training || d.rate == 0.0 {
		return input.Copy(), nil
	}

	rows, cols := input.Dims()

	// Create dropout mask
	d.mask = core.NewZerosTensor(rows, cols)
	scale := 1.0 / (1.0 - d.rate) // Inverted dropout scaling

	// Generate random mask
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if rand.Float64() > d.rate {
				d.mask.Set(i, j, scale) // Keep and scale
			} else {
				d.mask.Set(i, j, 0.0) // Drop
			}
		}
	}

	// Apply mask to input
	output := input.MulElem(d.mask)
	return output, nil
}

// Backward performs the backward pass through the dropout layer.
func (d *Dropout) Backward(gradient core.Tensor) (core.Tensor, error) {
	// If not training or rate is 0, pass gradient unchanged
	if !d.training || d.rate == 0.0 {
		return gradient.Copy(), nil
	}

	// Apply the same mask to gradients
	if d.mask == nil {
		panic(core.NewError(core.ErrNotFitted, "dropout mask not available for backward pass"))
	}

	// Gradient flows through the same mask
	inputGrad := gradient.MulElem(d.mask)
	return inputGrad, nil
}

// Parameters returns empty slice since Dropout has no trainable parameters.
func (d *Dropout) Parameters() []core.Tensor {
	return []core.Tensor{}
}

// Gradients returns empty slice since Dropout has no trainable parameters.
func (d *Dropout) Gradients() []core.Tensor {
	return []core.Tensor{}
}

// IsTrainable returns false since Dropout has no trainable parameters.
func (d *Dropout) IsTrainable() bool {
	return false
}

// OutputShape returns the same shape as input (dropout doesn't change shape).
func (d *Dropout) OutputShape(inputShape []int) ([]int, error) {
	return inputShape, nil
}

// ParameterCount returns 0 since Dropout has no trainable parameters.
func (d *Dropout) ParameterCount() int {
	return 0
}

// GetRate returns the dropout rate.
func (d *Dropout) GetRate() float64 {
	return d.rate
}

// SetRate sets the dropout rate.
func (d *Dropout) SetRate(rate float64) error {
	if rate < 0.0 || rate >= 1.0 {
		return core.NewError(core.ErrInvalidInput,
			fmt.Sprintf("dropout rate must be in [0, 1), got %f", rate))
	}
	d.rate = rate
	d.name = fmt.Sprintf("dropout_%.2f", rate)
	return nil
}
