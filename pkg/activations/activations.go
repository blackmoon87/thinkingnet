// Package activations provides activation functions for neural networks.
package activations

import (
	"math"

	"github.com/blackmoon87/thinkingnet/pkg/core"
	"github.com/blackmoon87/thinkingnet/pkg/utils"
)

// ReLU (Rectified Linear Unit) activation function.
type ReLU struct{}

// NewReLU creates a new ReLU activation function.
func NewReLU() *ReLU {
	return &ReLU{}
}

// Name returns the activation function name.
func (r *ReLU) Name() string {
	return "relu"
}

// Forward applies the ReLU function: f(x) = max(0, x).
func (r *ReLU) Forward(x float64) float64 {
	return math.Max(0, x)
}

// Backward computes the derivative: f'(x) = 1 if x > 0, else 0.
func (r *ReLU) Backward(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// LeakyReLU activation function with configurable negative slope.
type LeakyReLU struct {
	Alpha float64
}

// NewLeakyReLU creates a new LeakyReLU activation function.
func NewLeakyReLU(alpha float64) *LeakyReLU {
	if alpha <= 0 {
		alpha = 0.01 // Default value
	}
	return &LeakyReLU{Alpha: alpha}
}

// Name returns the activation function name.
func (l *LeakyReLU) Name() string {
	return "leaky_relu"
}

// Forward applies the LeakyReLU function: f(x) = x if x > 0, else alpha * x.
func (l *LeakyReLU) Forward(x float64) float64 {
	if x > 0 {
		return x
	}
	return l.Alpha * x
}

// Backward computes the derivative: f'(x) = 1 if x > 0, else alpha.
func (l *LeakyReLU) Backward(x float64) float64 {
	if x > 0 {
		return 1
	}
	return l.Alpha
}

// ELU (Exponential Linear Unit) activation function.
type ELU struct {
	Alpha float64
}

// NewELU creates a new ELU activation function.
func NewELU(alpha float64) *ELU {
	if alpha <= 0 {
		alpha = 1.0 // Default value
	}
	return &ELU{Alpha: alpha}
}

// Name returns the activation function name.
func (e *ELU) Name() string {
	return "elu"
}

// Forward applies the ELU function: f(x) = x if x > 0, else alpha * (exp(x) - 1).
func (e *ELU) Forward(x float64) float64 {
	if x > 0 {
		return x
	}
	return e.Alpha * (utils.SafeExp(x) - 1)
}

// Backward computes the derivative: f'(x) = 1 if x > 0, else alpha * exp(x).
func (e *ELU) Backward(x float64) float64 {
	if x > 0 {
		return 1
	}
	return e.Alpha * utils.SafeExp(x)
}

// Swish activation function (also known as SiLU).
type Swish struct{}

// NewSwish creates a new Swish activation function.
func NewSwish() *Swish {
	return &Swish{}
}

// Name returns the activation function name.
func (s *Swish) Name() string {
	return "swish"
}

// Forward applies the Swish function: f(x) = x * sigmoid(x).
func (s *Swish) Forward(x float64) float64 {
	sigmoid := utils.Sigmoid(x)
	return x * sigmoid
}

// Backward computes the derivative: f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
func (s *Swish) Backward(x float64) float64 {
	sigmoid := utils.Sigmoid(x)
	return sigmoid + x*sigmoid*(1-sigmoid)
}

// GELU (Gaussian Error Linear Unit) activation function.
type GELU struct{}

// NewGELU creates a new GELU activation function.
func NewGELU() *GELU {
	return &GELU{}
}

// Name returns the activation function name.
func (g *GELU) Name() string {
	return "gelu"
}

// Forward applies the GELU function using the approximation.
func (g *GELU) Forward(x float64) float64 {
	return 0.5 * x * (1.0 + math.Tanh(math.Sqrt(2.0/math.Pi)*(x+0.044715*math.Pow(x, 3))))
}

// Backward computes the derivative of GELU (approximation).
func (g *GELU) Backward(x float64) float64 {
	tanhInput := math.Sqrt(2.0/math.Pi) * (x + 0.044715*math.Pow(x, 3))
	tanhVal := math.Tanh(tanhInput)
	sech2 := 1 - tanhVal*tanhVal
	return 0.5*(1+tanhVal) + 0.5*x*sech2*math.Sqrt(2.0/math.Pi)*(1+3*0.044715*x*x)
}

// Sigmoid activation function.
type Sigmoid struct{}

// NewSigmoid creates a new Sigmoid activation function.
func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

// Name returns the activation function name.
func (s *Sigmoid) Name() string {
	return "sigmoid"
}

// Forward applies the Sigmoid function: f(x) = 1 / (1 + exp(-x)).
func (s *Sigmoid) Forward(x float64) float64 {
	return utils.Sigmoid(x)
}

// Backward computes the derivative: f'(x) = sigmoid(x) * (1 - sigmoid(x)).
func (s *Sigmoid) Backward(x float64) float64 {
	sig := s.Forward(x)
	return sig * (1 - sig)
}

// Tanh (Hyperbolic Tangent) activation function.
type Tanh struct{}

// NewTanh creates a new Tanh activation function.
func NewTanh() *Tanh {
	return &Tanh{}
}

// Name returns the activation function name.
func (t *Tanh) Name() string {
	return "tanh"
}

// Forward applies the Tanh function: f(x) = tanh(x).
func (t *Tanh) Forward(x float64) float64 {
	return math.Tanh(x)
}

// Backward computes the derivative: f'(x) = 1 - tanh²(x).
func (t *Tanh) Backward(x float64) float64 {
	tanh := math.Tanh(x)
	return 1 - tanh*tanh
}

// Linear (Identity) activation function.
type Linear struct{}

// NewLinear creates a new Linear activation function.
func NewLinear() *Linear {
	return &Linear{}
}

// Name returns the activation function name.
func (l *Linear) Name() string {
	return "linear"
}

// Forward applies the Linear function: f(x) = x.
func (l *Linear) Forward(x float64) float64 {
	return x
}

// Backward computes the derivative: f'(x) = 1.
func (l *Linear) Backward(x float64) float64 {
	return 1
}

// Softmax activation function (for multi-class classification).
// Note: Softmax is typically applied to vectors, not individual scalars.
type Softmax struct{}

// NewSoftmax creates a new Softmax activation function.
func NewSoftmax() *Softmax {
	return &Softmax{}
}

// Name returns the activation function name.
func (s *Softmax) Name() string {
	return "softmax"
}

// Forward for individual values (identity for softmax preparation).
func (s *Softmax) Forward(x float64) float64 {
	return x
}

// Backward for individual values (identity for softmax preparation).
func (s *Softmax) Backward(x float64) float64 {
	return 1
}

// ApplyTensorwise applies softmax to a tensor along the last dimension.
func (s *Softmax) ApplyTensorwise(input core.Tensor) core.Tensor {
	rows, cols := input.Dims()
	result := core.NewZerosTensor(rows, cols)

	for i := 0; i < rows; i++ {
		// Extract row
		row := make([]float64, cols)
		for j := 0; j < cols; j++ {
			row[j] = input.At(i, j)
		}

		// Apply softmax
		softmaxRow := utils.Softmax(row)

		// Set result
		for j := 0; j < cols; j++ {
			result.Set(i, j, softmaxRow[j])
		}
	}

	return result
}

// ActivationRegistry provides a registry for activation functions.
type ActivationRegistry struct {
	activations map[string]func() core.Activation
}

// NewActivationRegistry creates a new activation registry.
func NewActivationRegistry() *ActivationRegistry {
	registry := &ActivationRegistry{
		activations: make(map[string]func() core.Activation),
	}

	// Register default activations
	registry.Register("relu", func() core.Activation { return NewReLU() })
	registry.Register("leaky_relu", func() core.Activation { return NewLeakyReLU(0.01) })
	registry.Register("elu", func() core.Activation { return NewELU(1.0) })
	registry.Register("swish", func() core.Activation { return NewSwish() })
	registry.Register("gelu", func() core.Activation { return NewGELU() })
	registry.Register("sigmoid", func() core.Activation { return NewSigmoid() })
	registry.Register("tanh", func() core.Activation { return NewTanh() })
	registry.Register("linear", func() core.Activation { return NewLinear() })
	registry.Register("softmax", func() core.Activation { return NewSoftmax() })

	return registry
}

// Register adds a new activation function to the registry.
func (r *ActivationRegistry) Register(name string, factory func() core.Activation) {
	r.activations[name] = factory
}

// Get retrieves an activation function by name.
func (r *ActivationRegistry) Get(name string) (core.Activation, error) {
	factory, exists := r.activations[name]
	if !exists {
		return nil, core.NewError(core.ErrInvalidInput, "unknown activation function: "+name)
	}
	return factory(), nil
}

// List returns all registered activation function names.
func (r *ActivationRegistry) List() []string {
	names := make([]string, 0, len(r.activations))
	for name := range r.activations {
		names = append(names, name)
	}
	return names
}

// Global registry instance
var defaultRegistry = NewActivationRegistry()

// GetActivation retrieves an activation function from the default registry.
func GetActivation(name string) (core.Activation, error) {
	return defaultRegistry.Get(name)
}

// RegisterActivation adds an activation function to the default registry.
func RegisterActivation(name string, factory func() core.Activation) {
	defaultRegistry.Register(name, factory)
}

// ListActivations returns all available activation function names.
func ListActivations() []string {
	return defaultRegistry.List()
}
