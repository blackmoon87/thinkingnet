package optimizers

import (
	"fmt"
	"math"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// AdamConfig holds configuration for the Adam optimizer.
type AdamConfig struct {
	LearningRate float64 `json:"learning_rate"`
	Beta1        float64 `json:"beta1"`
	Beta2        float64 `json:"beta2"`
	Epsilon      float64 `json:"epsilon"`
	WeightDecay  float64 `json:"weight_decay"`
	AMSGrad      bool    `json:"amsgrad"`
	MaxGradNorm  float64 `json:"max_grad_norm"`
}

// DefaultAdamConfig returns the default Adam configuration.
func DefaultAdamConfig() AdamConfig {
	return AdamConfig{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  0.0,
		AMSGrad:      false,
		MaxGradNorm:  0.0, // 0 means no clipping
	}
}

// Adam implements the Adam optimization algorithm.
// Reference: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
type Adam struct {
	BaseOptimizer
	config AdamConfig

	// State variables for each parameter
	m    []core.Tensor // First moment estimates
	v    []core.Tensor // Second moment estimates
	vMax []core.Tensor // Maximum of second moment estimates (for AMSGrad)

	initialized bool
}

// NewAdam creates a new Adam optimizer with the given configuration.
func NewAdam(config AdamConfig) (*Adam, error) {
	if err := validateAdamConfig(config); err != nil {
		return nil, core.NewErrorWithCause(core.ErrConfigurationError, "invalid Adam configuration", err)
	}

	return &Adam{
		BaseOptimizer: BaseOptimizer{
			name:         "adam",
			learningRate: config.LearningRate,
			stepCount:    0,
		},
		config:      config,
		initialized: false,
	}, nil
}

// NewAdamWithDefaults creates a new Adam optimizer with default configuration.
func NewAdamWithDefaults(learningRate float64) (*Adam, error) {
	config := DefaultAdamConfig()
	config.LearningRate = learningRate
	return NewAdam(config)
}

// validateAdamConfig validates the Adam configuration parameters.
func validateAdamConfig(config AdamConfig) error {
	if err := core.ValidatePositive(config.LearningRate, "learning_rate"); err != nil {
		return err
	}

	if err := core.ValidateRange(config.Beta1, 0.0, 0.999, "beta1"); err != nil {
		return err
	}

	if err := core.ValidateRange(config.Beta2, 0.0, 0.999, "beta2"); err != nil {
		return err
	}

	if err := core.ValidatePositive(config.Epsilon, "epsilon"); err != nil {
		return err
	}

	if err := core.ValidateNonNegative(config.WeightDecay, "weight_decay"); err != nil {
		return err
	}

	if err := core.ValidateNonNegative(config.MaxGradNorm, "max_grad_norm"); err != nil {
		return err
	}

	return nil
}

// initializeState initializes the optimizer state for the given parameters.
func (a *Adam) initializeState(params []core.Tensor) {
	a.m = make([]core.Tensor, len(params))
	a.v = make([]core.Tensor, len(params))

	if a.config.AMSGrad {
		a.vMax = make([]core.Tensor, len(params))
	}

	for i, param := range params {
		// Initialize first moment estimate to zero
		a.m[i] = param.Copy()
		a.m[i].Zero()

		// Initialize second moment estimate to zero
		a.v[i] = param.Copy()
		a.v[i].Zero()

		// Initialize maximum second moment estimate for AMSGrad
		if a.config.AMSGrad {
			a.vMax[i] = param.Copy()
			a.vMax[i].Zero()
		}
	}

	a.initialized = true
}

// Update updates the parameters using the Adam algorithm.
func (a *Adam) Update(params []core.Tensor, grads []core.Tensor) error {
	if err := validateParameters(params, grads); err != nil {
		return core.NewErrorWithCause(core.ErrInvalidInput, "Adam update failed", err)
	}

	// Initialize state on first call
	if !a.initialized {
		a.initializeState(params)
	}

	// Apply gradient clipping if specified
	if a.config.MaxGradNorm > 0 {
		if err := clampGradients(grads, a.config.MaxGradNorm); err != nil {
			return core.NewErrorWithCause(core.ErrNumericalInstability, "gradient clipping failed", err)
		}
	}

	// Increment step counter
	a.Step()

	// Bias correction terms
	beta1Power := math.Pow(a.config.Beta1, float64(a.stepCount))
	beta2Power := math.Pow(a.config.Beta2, float64(a.stepCount))
	biasCorrection1 := 1.0 - beta1Power
	biasCorrection2 := 1.0 - beta2Power

	// Update each parameter
	for i, param := range params {
		grad := grads[i]

		// Apply weight decay if specified (L2 regularization)
		if a.config.WeightDecay > 0 {
			// Add weight decay to gradient: grad = grad + weight_decay * param
			weightDecayTerm := param.Scale(a.config.WeightDecay)
			grad = grad.Add(weightDecayTerm)
		}

		// Update first moment estimate: m = beta1 * m + (1 - beta1) * grad
		a.m[i] = a.m[i].Scale(a.config.Beta1)
		gradTerm := grad.Scale(1.0 - a.config.Beta1)
		a.m[i] = a.m[i].Add(gradTerm)

		// Update second moment estimate: v = beta2 * v + (1 - beta2) * grad^2
		a.v[i] = a.v[i].Scale(a.config.Beta2)
		gradSquared := grad.MulElem(grad)
		gradSquared = gradSquared.Scale(1.0 - a.config.Beta2)
		a.v[i] = a.v[i].Add(gradSquared)

		// Compute bias-corrected estimates
		mHat := a.m[i].Scale(1.0 / biasCorrection1)

		var vHat core.Tensor
		if a.config.AMSGrad {
			// AMSGrad: use maximum of current and previous second moment estimates
			a.vMax[i] = a.vMax[i].Apply(func(r, c int, oldVal float64) float64 {
				newVal := a.v[i].At(r, c)
				return math.Max(oldVal, newVal)
			})
			vHat = a.vMax[i].Scale(1.0 / biasCorrection2)
		} else {
			vHat = a.v[i].Scale(1.0 / biasCorrection2)
		}

		// Update parameters in place: param = param - lr * mHat / (sqrt(vHat) + epsilon)
		rows, cols := param.Dims()
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				m := mHat.At(r, c)
				v := vHat.At(r, c)
				update := a.config.LearningRate * m / (math.Sqrt(v) + a.config.Epsilon)
				newVal := param.At(r, c) - update

				// Validate the updated value
				if math.IsNaN(newVal) || math.IsInf(newVal, 0) {
					return core.NewError(core.ErrNumericalInstability,
						fmt.Sprintf("parameter update resulted in non-finite value at (%d,%d)", r, c)).
						WithContext("parameter_index", i).
						WithContext("row", r).
						WithContext("col", c).
						WithContext("old_value", param.At(r, c)).
						WithContext("update", update)
				}

				param.Set(r, c, newVal)
			}
		}
	}

	return nil
}

// Config returns the optimizer configuration.
func (a *Adam) Config() core.OptimizerConfig {
	return core.OptimizerConfig{
		Name:         a.name,
		LearningRate: a.learningRate,
		Parameters: map[string]interface{}{
			"beta1":         a.config.Beta1,
			"beta2":         a.config.Beta2,
			"epsilon":       a.config.Epsilon,
			"weight_decay":  a.config.WeightDecay,
			"amsgrad":       a.config.AMSGrad,
			"max_grad_norm": a.config.MaxGradNorm,
		},
	}
}

// Reset resets the optimizer state.
func (a *Adam) Reset() {
	a.BaseOptimizer.Reset()
	a.initialized = false
	a.m = nil
	a.v = nil
	a.vMax = nil
}

// SetLearningRate updates the learning rate and the internal config.
func (a *Adam) SetLearningRate(lr float64) {
	a.BaseOptimizer.SetLearningRate(lr)
	a.config.LearningRate = lr
}
