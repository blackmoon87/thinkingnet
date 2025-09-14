package optimizers

import (
	"fmt"
	"math"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// RMSpropConfig holds configuration for the RMSprop optimizer.
type RMSpropConfig struct {
	LearningRate float64 `json:"learning_rate"`
	Alpha        float64 `json:"alpha"`        // Smoothing constant (decay rate)
	Epsilon      float64 `json:"epsilon"`      // Small constant for numerical stability
	WeightDecay  float64 `json:"weight_decay"` // L2 penalty
	Momentum     float64 `json:"momentum"`     // Momentum factor
	Centered     bool    `json:"centered"`     // Whether to use centered RMSprop
	MaxGradNorm  float64 `json:"max_grad_norm"`
}

// DefaultRMSpropConfig returns the default RMSprop configuration.
func DefaultRMSpropConfig() RMSpropConfig {
	return RMSpropConfig{
		LearningRate: 0.01,
		Alpha:        0.99,
		Epsilon:      1e-8,
		WeightDecay:  0.0,
		Momentum:     0.0,
		Centered:     false,
		MaxGradNorm:  0.0, // 0 means no clipping
	}
}

// RMSprop implements the RMSprop optimization algorithm.
// Reference: "Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude" (Hinton, 2012)
type RMSprop struct {
	BaseOptimizer
	config RMSpropConfig

	// State variables for each parameter
	squareAvg []core.Tensor // Running average of squared gradients
	gradAvg   []core.Tensor // Running average of gradients (for centered variant)
	momentum  []core.Tensor // Momentum buffers

	initialized bool
}

// NewRMSprop creates a new RMSprop optimizer with the given configuration.
func NewRMSprop(config RMSpropConfig) *RMSprop {
	if err := validateRMSpropConfig(config); err != nil {
		panic(fmt.Sprintf("invalid RMSprop config: %v", err))
	}

	return &RMSprop{
		BaseOptimizer: BaseOptimizer{
			name:         "rmsprop",
			learningRate: config.LearningRate,
			stepCount:    0,
		},
		config:      config,
		initialized: false,
	}
}

// NewRMSpropWithDefaults creates a new RMSprop optimizer with default configuration.
func NewRMSpropWithDefaults(learningRate float64) *RMSprop {
	config := DefaultRMSpropConfig()
	config.LearningRate = learningRate
	return NewRMSprop(config)
}

// NewRMSpropCentered creates a new centered RMSprop optimizer.
func NewRMSpropCentered(learningRate, alpha float64) *RMSprop {
	config := DefaultRMSpropConfig()
	config.LearningRate = learningRate
	config.Alpha = alpha
	config.Centered = true
	return NewRMSprop(config)
}

// validateRMSpropConfig validates the RMSprop configuration parameters.
func validateRMSpropConfig(config RMSpropConfig) error {
	if config.LearningRate <= 0 {
		return fmt.Errorf("learning rate must be positive, got %f", config.LearningRate)
	}
	if config.Alpha < 0 || config.Alpha >= 1 {
		return fmt.Errorf("alpha must be in [0, 1), got %f", config.Alpha)
	}
	if config.Epsilon <= 0 {
		return fmt.Errorf("epsilon must be positive, got %f", config.Epsilon)
	}
	if config.WeightDecay < 0 {
		return fmt.Errorf("weight decay must be non-negative, got %f", config.WeightDecay)
	}
	if config.Momentum < 0 || config.Momentum >= 1 {
		return fmt.Errorf("momentum must be in [0, 1), got %f", config.Momentum)
	}
	if config.MaxGradNorm < 0 {
		return fmt.Errorf("max gradient norm must be non-negative, got %f", config.MaxGradNorm)
	}
	return nil
}

// initializeState initializes the optimizer state for the given parameters.
func (r *RMSprop) initializeState(params []core.Tensor) {
	r.squareAvg = make([]core.Tensor, len(params))

	if r.config.Centered {
		r.gradAvg = make([]core.Tensor, len(params))
	}

	if r.config.Momentum > 0 {
		r.momentum = make([]core.Tensor, len(params))
	}

	for i, param := range params {
		// Initialize running average of squared gradients to zero
		r.squareAvg[i] = param.Copy()
		r.squareAvg[i].Zero()

		// Initialize running average of gradients for centered variant
		if r.config.Centered {
			r.gradAvg[i] = param.Copy()
			r.gradAvg[i].Zero()
		}

		// Initialize momentum buffer
		if r.config.Momentum > 0 {
			r.momentum[i] = param.Copy()
			r.momentum[i].Zero()
		}
	}

	r.initialized = true
}

// Update updates the parameters using the RMSprop algorithm.
func (r *RMSprop) Update(params []core.Tensor, grads []core.Tensor) {
	if err := validateParameters(params, grads); err != nil {
		panic(fmt.Sprintf("RMSprop update failed: %v", err))
	}

	// Initialize state on first call
	if !r.initialized {
		r.initializeState(params)
	}

	// Apply gradient clipping if specified
	if r.config.MaxGradNorm > 0 {
		clampGradients(grads, r.config.MaxGradNorm)
	}

	// Increment step counter
	r.Step()

	// Update each parameter
	for i, param := range params {
		grad := grads[i]

		// Apply weight decay if specified (L2 regularization)
		if r.config.WeightDecay > 0 {
			// Add weight decay to gradient: grad = grad + weight_decay * param
			weightDecayTerm := param.Copy()
			weightDecayTerm.Scale(r.config.WeightDecay)
			grad = grad.Add(weightDecayTerm)
		}

		// Update running average of squared gradients
		// squareAvg = alpha * squareAvg + (1 - alpha) * grad^2
		r.squareAvg[i].Scale(r.config.Alpha)
		gradSquared := grad.Copy()
		gradSquared = gradSquared.MulElem(grad)
		gradSquared.Scale(1.0 - r.config.Alpha)
		r.squareAvg[i] = r.squareAvg[i].Add(gradSquared)

		var denominator core.Tensor

		if r.config.Centered {
			// Centered RMSprop: use variance instead of second moment
			// Update running average of gradients
			// gradAvg = alpha * gradAvg + (1 - alpha) * grad
			r.gradAvg[i].Scale(r.config.Alpha)
			gradTerm := grad.Copy()
			gradTerm.Scale(1.0 - r.config.Alpha)
			r.gradAvg[i] = r.gradAvg[i].Add(gradTerm)

			// Compute variance: var = squareAvg - gradAvg^2
			variance := r.squareAvg[i].Copy()
			gradAvgSquared := r.gradAvg[i].Copy()
			gradAvgSquared = gradAvgSquared.MulElem(r.gradAvg[i])
			variance = variance.Sub(gradAvgSquared)

			// Denominator: sqrt(var + epsilon)
			denominator = variance.Apply(func(i, j int, v float64) float64 {
				return math.Sqrt(math.Max(v, 0) + r.config.Epsilon)
			})
		} else {
			// Standard RMSprop: denominator = sqrt(squareAvg + epsilon)
			denominator = r.squareAvg[i].Apply(func(i, j int, v float64) float64 {
				return math.Sqrt(v + r.config.Epsilon)
			})
		}

		// Compute scaled gradient: scaledGrad = grad / denominator
		scaledGrad := grad.Apply(func(i, j int, g float64) float64 {
			return g / denominator.At(i, j)
		})

		var update core.Tensor

		if r.config.Momentum > 0 {
			// Apply momentum: momentum = momentum_factor * momentum + scaledGrad
			r.momentum[i].Scale(r.config.Momentum)
			r.momentum[i] = r.momentum[i].Add(scaledGrad)
			update = r.momentum[i].Copy()
		} else {
			// No momentum: update = scaledGrad
			update = scaledGrad
		}

		// Apply learning rate and update parameters: param = param - lr * update
		update.Scale(r.config.LearningRate)

		// Update parameters in place
		rows, cols := param.Dims()
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				newVal := param.At(r, c) - update.At(r, c)
				param.Set(r, c, newVal)
			}
		}
	}
}

// Config returns the optimizer configuration.
func (r *RMSprop) Config() core.OptimizerConfig {
	return core.OptimizerConfig{
		Name:         r.name,
		LearningRate: r.learningRate,
		Parameters: map[string]interface{}{
			"alpha":         r.config.Alpha,
			"epsilon":       r.config.Epsilon,
			"weight_decay":  r.config.WeightDecay,
			"momentum":      r.config.Momentum,
			"centered":      r.config.Centered,
			"max_grad_norm": r.config.MaxGradNorm,
		},
	}
}

// Reset resets the optimizer state.
func (r *RMSprop) Reset() {
	r.BaseOptimizer.Reset()
	r.initialized = false
	r.squareAvg = nil
	r.gradAvg = nil
	r.momentum = nil
}

// SetLearningRate updates the learning rate and the internal config.
func (r *RMSprop) SetLearningRate(lr float64) {
	r.BaseOptimizer.SetLearningRate(lr)
	r.config.LearningRate = lr
}
