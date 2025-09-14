// Package optimizers provides optimization algorithms for neural network training.
package optimizers

import (
	"fmt"
	"math"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// BaseOptimizer provides common functionality for all optimizers.
type BaseOptimizer struct {
	name         string
	learningRate float64
	stepCount    int
}

// Name returns the optimizer name.
func (o *BaseOptimizer) Name() string {
	return o.name
}

// LearningRate returns the current learning rate.
func (o *BaseOptimizer) LearningRate() float64 {
	return o.learningRate
}

// SetLearningRate sets the learning rate.
func (o *BaseOptimizer) SetLearningRate(lr float64) error {
	if err := core.ValidatePositive(lr, "learning_rate"); err != nil {
		return err
	}
	o.learningRate = lr
	return nil
}

// Step increments the step counter.
func (o *BaseOptimizer) Step() {
	o.stepCount++
}

// Reset resets the optimizer state.
func (o *BaseOptimizer) Reset() {
	o.stepCount = 0
}

// validateParameters validates that parameters and gradients have matching dimensions.
func validateParameters(params, grads []core.Tensor) error {
	if len(params) != len(grads) {
		return core.NewError(core.ErrDimensionMismatch,
			fmt.Sprintf("parameter count (%d) does not match gradient count (%d)", len(params), len(grads))).
			WithContext("param_count", len(params)).
			WithContext("grad_count", len(grads))
	}

	for i, param := range params {
		if param == nil {
			return core.NewError(core.ErrInvalidInput, fmt.Sprintf("parameter %d is nil", i)).
				WithContext("parameter_index", i)
		}

		grad := grads[i]
		if grad == nil {
			return core.NewError(core.ErrInvalidInput, fmt.Sprintf("gradient %d is nil", i)).
				WithContext("gradient_index", i)
		}

		pRows, pCols := param.Dims()
		gRows, gCols := grad.Dims()

		if pRows != gRows || pCols != gCols {
			return core.NewError(core.ErrDimensionMismatch,
				fmt.Sprintf("parameter %d dimensions (%d, %d) do not match gradient dimensions (%d, %d)",
					i, pRows, pCols, gRows, gCols)).
				WithContext("parameter_index", i).
				WithContext("param_shape", []int{pRows, pCols}).
				WithContext("grad_shape", []int{gRows, gCols})
		}

		// Validate that tensors contain finite values
		if err := core.ValidateTensorFinite(param, fmt.Sprintf("parameter_%d", i)); err != nil {
			return err
		}

		if err := core.ValidateTensorFinite(grad, fmt.Sprintf("gradient_%d", i)); err != nil {
			return err
		}
	}

	return nil
}

// clampGradients applies gradient clipping to prevent exploding gradients.
func clampGradients(grads []core.Tensor, maxNorm float64) error {
	if err := core.ValidatePositive(maxNorm, "max_norm"); err != nil {
		return err
	}

	if len(grads) == 0 {
		return nil
	}

	// Validate all gradients first
	for i, grad := range grads {
		if grad == nil {
			return core.NewError(core.ErrInvalidInput, fmt.Sprintf("gradient %d is nil", i)).
				WithContext("gradient_index", i)
		}

		if err := core.ValidateTensorFinite(grad, fmt.Sprintf("gradient_%d", i)); err != nil {
			return err
		}
	}

	// Calculate total gradient norm
	totalNorm := 0.0
	for _, grad := range grads {
		norm := grad.Norm()
		if math.IsNaN(norm) || math.IsInf(norm, 0) {
			return core.NewError(core.ErrNumericalInstability, "gradient norm is not finite")
		}
		totalNorm += math.Pow(norm, 2)
	}
	totalNorm = math.Sqrt(totalNorm)

	// Apply clipping if necessary
	if totalNorm > maxNorm {
		clipCoeff := maxNorm / totalNorm
		for _, grad := range grads {
			grad = grad.Scale(clipCoeff)
		}
	}

	return nil
}
