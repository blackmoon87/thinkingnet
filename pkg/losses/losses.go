// Package losses provides loss function implementations for the ThinkingNet AI library.
package losses

import (
	"math"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// clamp constrains a value between min and max.
func clamp(x, min, max float64) float64 {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}

// BinaryCrossEntropy implements binary cross-entropy loss with numerical stability.
type BinaryCrossEntropy struct {
	name string
}

// NewBinaryCrossEntropy creates a new binary cross-entropy loss function.
func NewBinaryCrossEntropy() *BinaryCrossEntropy {
	return &BinaryCrossEntropy{
		name: "binary_crossentropy",
	}
}

// Name returns the loss function name.
func (bce *BinaryCrossEntropy) Name() string {
	return bce.name
}

// Compute calculates the binary cross-entropy loss.
func (bce *BinaryCrossEntropy) Compute(yTrue, yPred core.Tensor) float64 {
	r1, c1 := yTrue.Dims()
	r2, c2 := yPred.Dims()
	if r1 != r2 || c1 != c2 {
		panic("yTrue and yPred must have the same dimensions")
	}

	rows, cols := yPred.Dims()
	var total float64

	for i := range rows {
		for j := range cols {
			// Clamp predictions to prevent log(0)
			p := clamp(yPred.At(i, j), core.GetEpsilon(), 1-core.GetEpsilon())
			t := yTrue.At(i, j)

			// Binary cross-entropy formula: -[t*log(p) + (1-t)*log(1-p)]
			total += t*math.Log(p) + (1-t)*math.Log(1-p)
		}
	}

	return -total / float64(rows)
}

// Gradient computes the gradient of binary cross-entropy loss.
func (bce *BinaryCrossEntropy) Gradient(yTrue, yPred core.Tensor) core.Tensor {
	r1, c1 := yTrue.Dims()
	r2, c2 := yPred.Dims()
	if r1 != r2 || c1 != c2 {
		panic("yTrue and yPred must have the same dimensions")
	}

	rows, cols := yPred.Dims()
	grad := yPred.Copy()
	grad.Zero()

	for i := range rows {
		for j := range cols {
			// Clamp predictions to prevent division by zero
			p := clamp(yPred.At(i, j), core.GetEpsilon(), 1-core.GetEpsilon())
			t := yTrue.At(i, j)

			// Gradient: (p - t) / (p * (1 - p)) / batch_size
			gradValue := (p - t) / (p * (1 - p) * float64(rows))
			grad.Set(i, j, gradValue)
		}
	}

	return grad
}

// CategoricalCrossEntropy implements categorical cross-entropy loss with softmax integration.
type CategoricalCrossEntropy struct {
	name string
}

// NewCategoricalCrossEntropy creates a new categorical cross-entropy loss function.
func NewCategoricalCrossEntropy() *CategoricalCrossEntropy {
	return &CategoricalCrossEntropy{
		name: "categorical_crossentropy",
	}
}

// Name returns the loss function name.
func (cce *CategoricalCrossEntropy) Name() string {
	return cce.name
}

// Compute calculates the categorical cross-entropy loss.
func (cce *CategoricalCrossEntropy) Compute(yTrue, yPred core.Tensor) float64 {
	r1, c1 := yTrue.Dims()
	r2, c2 := yPred.Dims()
	if r1 != r2 || c1 != c2 {
		panic("yTrue and yPred must have the same dimensions")
	}

	rows, cols := yPred.Dims()
	var total float64

	for i := range rows {
		for j := range cols {
			// Clamp predictions to prevent log(0)
			p := clamp(yPred.At(i, j), core.GetEpsilon(), 1-core.GetEpsilon())
			t := yTrue.At(i, j)

			// Only compute loss for true class (t > 0)
			if t > 0 {
				total += t * math.Log(p)
			}
		}
	}

	return -total / float64(rows)
}

// Gradient computes the gradient of categorical cross-entropy loss.
// This assumes softmax activation in the final layer.
func (cce *CategoricalCrossEntropy) Gradient(yTrue, yPred core.Tensor) core.Tensor {
	r1, c1 := yTrue.Dims()
	r2, c2 := yPred.Dims()
	if r1 != r2 || c1 != c2 {
		panic("yTrue and yPred must have the same dimensions")
	}

	rows, _ := yPred.Dims()
	grad := yPred.Copy()

	// For softmax + categorical cross-entropy, gradient is simply (yPred - yTrue) / batch_size
	grad = grad.Sub(yTrue)
	grad = grad.Scale(1.0 / float64(rows))

	return grad
}

// MeanSquaredError implements mean squared error loss for regression tasks.
type MeanSquaredError struct {
	name string
}

// NewMeanSquaredError creates a new mean squared error loss function.
func NewMeanSquaredError() *MeanSquaredError {
	return &MeanSquaredError{
		name: "mean_squared_error",
	}
}

// Name returns the loss function name.
func (mse *MeanSquaredError) Name() string {
	return mse.name
}

// Compute calculates the mean squared error loss.
func (mse *MeanSquaredError) Compute(yTrue, yPred core.Tensor) float64 {
	r1, c1 := yTrue.Dims()
	r2, c2 := yPred.Dims()
	if r1 != r2 || c1 != c2 {
		panic("yTrue and yPred must have the same dimensions")
	}

	rows, cols := yPred.Dims()
	var total float64

	for i := range rows {
		for j := range cols {
			diff := yTrue.At(i, j) - yPred.At(i, j)
			total += diff * diff
		}
	}

	return total / (2.0 * float64(rows))
}

// Gradient computes the gradient of mean squared error loss.
func (mse *MeanSquaredError) Gradient(yTrue, yPred core.Tensor) core.Tensor {
	r1, c1 := yTrue.Dims()
	r2, c2 := yPred.Dims()
	if r1 != r2 || c1 != c2 {
		panic("yTrue and yPred must have the same dimensions")
	}

	rows, _ := yPred.Dims()
	grad := yPred.Copy()

	// Gradient: (yPred - yTrue) / batch_size
	grad = grad.Sub(yTrue)
	grad = grad.Scale(1.0 / float64(rows))

	return grad
}
