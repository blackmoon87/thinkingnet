package optimizers

import (
	"fmt"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// SGDConfig holds configuration for the SGD optimizer.
type SGDConfig struct {
	LearningRate float64 `json:"learning_rate"`
	Momentum     float64 `json:"momentum"`
	Dampening    float64 `json:"dampening"`
	WeightDecay  float64 `json:"weight_decay"`
	Nesterov     bool    `json:"nesterov"`
	MaxGradNorm  float64 `json:"max_grad_norm"`
}

// DefaultSGDConfig returns the default SGD configuration.
func DefaultSGDConfig() SGDConfig {
	return SGDConfig{
		LearningRate: 0.01,
		Momentum:     0.0,
		Dampening:    0.0,
		WeightDecay:  0.0,
		Nesterov:     false,
		MaxGradNorm:  0.0, // 0 means no clipping
	}
}

// SGD implements the Stochastic Gradient Descent optimization algorithm.
// Supports momentum, dampening, weight decay, and Nesterov acceleration.
type SGD struct {
	BaseOptimizer
	config SGDConfig

	// State variables for each parameter
	velocity []core.Tensor // Momentum buffers

	initialized bool
}

// NewSGD creates a new SGD optimizer with the given configuration.
func NewSGD(config SGDConfig) *SGD {
	if err := validateSGDConfig(config); err != nil {
		panic(fmt.Sprintf("invalid SGD config: %v", err))
	}

	return &SGD{
		BaseOptimizer: BaseOptimizer{
			name:         "sgd",
			learningRate: config.LearningRate,
			stepCount:    0,
		},
		config:      config,
		initialized: false,
	}
}

// NewSGDWithDefaults creates a new SGD optimizer with default configuration.
func NewSGDWithDefaults(learningRate float64) *SGD {
	config := DefaultSGDConfig()
	config.LearningRate = learningRate
	return NewSGD(config)
}

// NewSGDWithMomentum creates a new SGD optimizer with momentum.
func NewSGDWithMomentum(learningRate, momentum float64) *SGD {
	config := DefaultSGDConfig()
	config.LearningRate = learningRate
	config.Momentum = momentum
	return NewSGD(config)
}

// validateSGDConfig validates the SGD configuration parameters.
func validateSGDConfig(config SGDConfig) error {
	if config.LearningRate <= 0 {
		return fmt.Errorf("learning rate must be positive, got %f", config.LearningRate)
	}
	if config.Momentum < 0 || config.Momentum >= 1 {
		return fmt.Errorf("momentum must be in [0, 1), got %f", config.Momentum)
	}
	if config.Dampening < 0 || config.Dampening > 1 {
		return fmt.Errorf("dampening must be in [0, 1], got %f", config.Dampening)
	}
	if config.WeightDecay < 0 {
		return fmt.Errorf("weight decay must be non-negative, got %f", config.WeightDecay)
	}
	if config.MaxGradNorm < 0 {
		return fmt.Errorf("max gradient norm must be non-negative, got %f", config.MaxGradNorm)
	}
	if config.Nesterov && config.Momentum <= 0 {
		return fmt.Errorf("Nesterov momentum requires momentum > 0, got %f", config.Momentum)
	}
	return nil
}

// initializeState initializes the optimizer state for the given parameters.
func (s *SGD) initializeState(params []core.Tensor) {
	if s.config.Momentum > 0 {
		s.velocity = make([]core.Tensor, len(params))

		for i, param := range params {
			// Initialize velocity buffer to zero
			s.velocity[i] = param.Copy()
			s.velocity[i].Zero()
		}
	}

	s.initialized = true
}

// Update updates the parameters using the SGD algorithm.
func (s *SGD) Update(params []core.Tensor, grads []core.Tensor) {
	if err := validateParameters(params, grads); err != nil {
		panic(fmt.Sprintf("SGD update failed: %v", err))
	}

	// Initialize state on first call
	if !s.initialized {
		s.initializeState(params)
	}

	// Apply gradient clipping if specified
	if s.config.MaxGradNorm > 0 {
		clampGradients(grads, s.config.MaxGradNorm)
	}

	// Increment step counter
	s.Step()

	// Update each parameter
	for i, param := range params {
		grad := grads[i]

		// Apply weight decay if specified (L2 regularization)
		if s.config.WeightDecay > 0 {
			// Add weight decay to gradient: grad = grad + weight_decay * param
			weightDecayTerm := param.Copy()
			weightDecayTerm.Scale(s.config.WeightDecay)
			grad = grad.Add(weightDecayTerm)
		}

		var update core.Tensor

		if s.config.Momentum > 0 {
			// Momentum-based update
			velocity := s.velocity[i]

			// Update velocity: v = momentum * v + (1 - dampening) * grad
			velocity.Scale(s.config.Momentum)
			gradTerm := grad.Copy()
			gradTerm.Scale(1.0 - s.config.Dampening)
			velocity = velocity.Add(gradTerm)
			s.velocity[i] = velocity

			if s.config.Nesterov {
				// Nesterov momentum: update = momentum * v + grad
				update = velocity.Copy()
				update.Scale(s.config.Momentum)
				update = update.Add(grad)
			} else {
				// Standard momentum: update = v
				update = velocity.Copy()
			}
		} else {
			// Standard SGD without momentum: update = grad
			update = grad.Copy()
		}

		// Apply learning rate and update parameters: param = param - lr * update
		update.Scale(s.config.LearningRate)

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
func (s *SGD) Config() core.OptimizerConfig {
	return core.OptimizerConfig{
		Name:         s.name,
		LearningRate: s.learningRate,
		Parameters: map[string]interface{}{
			"momentum":      s.config.Momentum,
			"dampening":     s.config.Dampening,
			"weight_decay":  s.config.WeightDecay,
			"nesterov":      s.config.Nesterov,
			"max_grad_norm": s.config.MaxGradNorm,
		},
	}
}

// Reset resets the optimizer state.
func (s *SGD) Reset() {
	s.BaseOptimizer.Reset()
	s.initialized = false
	s.velocity = nil
}

// SetLearningRate updates the learning rate and the internal config.
func (s *SGD) SetLearningRate(lr float64) {
	s.BaseOptimizer.SetLearningRate(lr)
	s.config.LearningRate = lr
}
