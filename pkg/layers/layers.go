// Package layers provides neural network layer implementations.
package layers

import (
	"math"
	"math/rand"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// BaseLayer provides common functionality for all layers.
type BaseLayer struct {
	name     string
	training bool
}

// NewBaseLayer creates a new base layer.
func NewBaseLayer(name string) *BaseLayer {
	return &BaseLayer{
		name:     name,
		training: true,
	}
}

// Name returns the layer name.
func (l *BaseLayer) Name() string {
	return l.name
}

// SetName sets the layer name.
func (l *BaseLayer) SetName(name string) {
	l.name = name
}

// SetTraining sets the training mode.
func (l *BaseLayer) SetTraining(training bool) {
	l.training = training
}

// IsTraining returns true if the layer is in training mode.
func (l *BaseLayer) IsTraining() bool {
	return l.training
}

// WeightInitializer represents different weight initialization strategies.
type WeightInitializer string

const (
	// Xavier/Glorot uniform initialization
	XavierUniform WeightInitializer = "xavier_uniform"
	// Xavier/Glorot normal initialization
	XavierNormal WeightInitializer = "xavier_normal"
	// He uniform initialization (good for ReLU)
	HeUniform WeightInitializer = "he_uniform"
	// He normal initialization (good for ReLU)
	HeNormal WeightInitializer = "he_normal"
	// Random normal initialization
	RandomNormal WeightInitializer = "random_normal"
	// Zero initialization
	Zeros WeightInitializer = "zeros"
)

// InitializeWeights initializes a weight matrix using the specified strategy.
func InitializeWeights(rows, cols int, initializer WeightInitializer) core.Tensor {
	var std float64
	var data []float64

	switch initializer {
	case XavierUniform:
		// Xavier uniform: [-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out))]
		limit := math.Sqrt(6.0 / float64(rows+cols))
		data = make([]float64, rows*cols)
		for i := range data {
			data[i] = (rand.Float64()*2 - 1) * limit
		}
	case XavierNormal:
		// Xavier normal: std = sqrt(2/(fan_in + fan_out))
		std = math.Sqrt(2.0 / float64(rows+cols))
		data = make([]float64, rows*cols)
		for i := range data {
			data[i] = rand.NormFloat64() * std
		}
	case HeUniform:
		// He uniform: [-sqrt(6/fan_in), sqrt(6/fan_in)]
		limit := math.Sqrt(6.0 / float64(rows))
		data = make([]float64, rows*cols)
		for i := range data {
			data[i] = (rand.Float64()*2 - 1) * limit
		}
	case HeNormal:
		// He normal: std = sqrt(2/fan_in)
		std = math.Sqrt(2.0 / float64(rows))
		data = make([]float64, rows*cols)
		for i := range data {
			data[i] = rand.NormFloat64() * std
		}
	case RandomNormal:
		// Random normal with std = 0.01
		std = 0.01
		data = make([]float64, rows*cols)
		for i := range data {
			data[i] = rand.NormFloat64() * std
		}
	case Zeros:
		// Zero initialization
		data = make([]float64, rows*cols)
		// Already zeros by default
	default:
		// Default to Xavier uniform
		limit := math.Sqrt(6.0 / float64(rows+cols))
		data = make([]float64, rows*cols)
		for i := range data {
			data[i] = (rand.Float64()*2 - 1) * limit
		}
	}

	return core.NewTensorFromData(rows, cols, data)
}
