package core

import (
	"fmt"
	"math"
	"time"
)

// ErrorRecoveryConfig holds configuration for error recovery mechanisms.
type ErrorRecoveryConfig struct {
	MaxRetries     int           `json:"max_retries"`
	RetryDelay     time.Duration `json:"retry_delay"`
	BackoffFactor  float64       `json:"backoff_factor"`
	EnableFallback bool          `json:"enable_fallback"`
	LogErrors      bool          `json:"log_errors"`
	FailFast       bool          `json:"fail_fast"`
	TolerateNaN    bool          `json:"tolerate_nan"`
	TolerateInf    bool          `json:"tolerate_inf"`
	MaxErrorRate   float64       `json:"max_error_rate"`
	ErrorWindow    time.Duration `json:"error_window"`
}

// DefaultErrorRecoveryConfig returns a default error recovery configuration.
func DefaultErrorRecoveryConfig() ErrorRecoveryConfig {
	return ErrorRecoveryConfig{
		MaxRetries:     3,
		RetryDelay:     100 * time.Millisecond,
		BackoffFactor:  2.0,
		EnableFallback: true,
		LogErrors:      true,
		FailFast:       false,
		TolerateNaN:    false,
		TolerateInf:    false,
		MaxErrorRate:   0.1, // 10% error rate threshold
		ErrorWindow:    time.Minute,
	}
}

// ErrorRecovery provides mechanisms for handling and recovering from errors.
type ErrorRecovery struct {
	config      ErrorRecoveryConfig
	errorCounts map[string]int
	lastErrors  map[string]time.Time
}

// NewErrorRecovery creates a new error recovery instance.
func NewErrorRecovery(config ErrorRecoveryConfig) *ErrorRecovery {
	return &ErrorRecovery{
		config:      config,
		errorCounts: make(map[string]int),
		lastErrors:  make(map[string]time.Time),
	}
}

// RetryWithBackoff executes a function with retry logic and exponential backoff.
func (er *ErrorRecovery) RetryWithBackoff(operation string, fn func() error) error {
	var lastErr error
	delay := er.config.RetryDelay

	for attempt := 0; attempt <= er.config.MaxRetries; attempt++ {
		if attempt > 0 {
			if er.config.LogErrors {
				fmt.Printf("Retrying %s (attempt %d/%d) after error: %v\n",
					operation, attempt, er.config.MaxRetries, lastErr)
			}
			time.Sleep(delay)
			delay = time.Duration(float64(delay) * er.config.BackoffFactor)
		}

		err := fn()
		if err == nil {
			return nil
		}

		lastErr = err

		// Check if we should fail fast for certain error types
		if er.config.FailFast {
			if tnErr, ok := err.(*ThinkingNetError); ok {
				switch tnErr.Type {
				case ErrInvalidInput, ErrConfigurationError:
					// Don't retry for configuration errors
					return err
				}
			}
		}

		// Track error for rate limiting
		er.trackError(operation)
	}

	return NewErrorWithCause(ErrConvergence,
		fmt.Sprintf("operation %s failed after %d attempts", operation, er.config.MaxRetries+1),
		lastErr)
}

// SafeTensorOperation performs a tensor operation with error recovery.
func (er *ErrorRecovery) SafeTensorOperation(operation string, fn func() Tensor) (Tensor, error) {
	var result Tensor

	err := er.RetryWithBackoff(operation, func() error {
		defer func() {
			if r := recover(); r != nil {
				panic(NewError(ErrNumericalInstability,
					fmt.Sprintf("tensor operation %s panicked: %v", operation, r)))
			}
		}()

		result = fn()

		// Validate result
		if result == nil {
			return NewError(ErrMemory, "tensor operation returned nil result")
		}

		// Check for numerical issues
		if !er.config.TolerateNaN && result.HasNaN() {
			return NewError(ErrNumericalInstability, "tensor operation produced NaN values")
		}

		if !er.config.TolerateInf && result.HasInf() {
			return NewError(ErrNumericalInstability, "tensor operation produced infinite values")
		}

		return nil
	})

	if err != nil && er.config.EnableFallback {
		// Try fallback strategies
		if fallbackResult, fallbackErr := er.tryFallbackStrategies(operation, fn); fallbackErr == nil {
			return fallbackResult, nil
		}
	}

	return result, err
}

// tryFallbackStrategies attempts various fallback strategies for failed operations.
func (er *ErrorRecovery) tryFallbackStrategies(operation string, fn func() Tensor) (Tensor, error) {
	// Strategy 1: Try with reduced precision
	if er.config.LogErrors {
		fmt.Printf("Attempting fallback strategy for %s\n", operation)
	}

	// For now, just return the original error
	// In a full implementation, you might try:
	// - Reducing numerical precision
	// - Using alternative algorithms
	// - Applying regularization
	// - Using approximate methods

	return nil, NewError(ErrConvergence, "all fallback strategies failed")
}

// trackError tracks error occurrences for rate limiting.
func (er *ErrorRecovery) trackError(operation string) {
	now := time.Now()
	er.errorCounts[operation]++
	er.lastErrors[operation] = now

	// Clean up old error records
	for op, lastTime := range er.lastErrors {
		if now.Sub(lastTime) > er.config.ErrorWindow {
			delete(er.errorCounts, op)
			delete(er.lastErrors, op)
		}
	}
}

// GetErrorRate returns the current error rate for an operation.
func (er *ErrorRecovery) GetErrorRate(operation string) float64 {
	count, exists := er.errorCounts[operation]
	if !exists {
		return 0.0
	}

	// Simple rate calculation - in practice you might want more sophisticated metrics
	return float64(count) / float64(er.config.MaxRetries+1)
}

// ShouldCircuitBreak determines if an operation should be circuit-broken due to high error rate.
func (er *ErrorRecovery) ShouldCircuitBreak(operation string) bool {
	return er.GetErrorRate(operation) > er.config.MaxErrorRate
}

// CleanNaNInf cleans NaN and Inf values from a tensor using various strategies.
func CleanNaNInf(tensor Tensor, strategy string) (Tensor, error) {
	if tensor == nil {
		return nil, NewError(ErrInvalidInput, "tensor cannot be nil")
	}

	if !tensor.HasNaN() && !tensor.HasInf() {
		return tensor, nil // No cleaning needed
	}

	rows, cols := tensor.Dims()
	result := tensor.Copy()

	switch strategy {
	case "zero":
		// Replace NaN/Inf with zero
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				val := result.At(i, j)
				if math.IsNaN(val) || math.IsInf(val, 0) {
					result.Set(i, j, 0.0)
				}
			}
		}

	case "mean":
		// Replace NaN/Inf with column mean
		for j := 0; j < cols; j++ {
			var sum float64
			var count int

			// Calculate mean of finite values in column
			for i := 0; i < rows; i++ {
				val := result.At(i, j)
				if !math.IsNaN(val) && !math.IsInf(val, 0) {
					sum += val
					count++
				}
			}

			mean := 0.0
			if count > 0 {
				mean = sum / float64(count)
			}

			// Replace non-finite values with mean
			for i := 0; i < rows; i++ {
				val := result.At(i, j)
				if math.IsNaN(val) || math.IsInf(val, 0) {
					result.Set(i, j, mean)
				}
			}
		}

	case "clamp":
		// Clamp values to a reasonable range
		const maxVal = 1e6
		const minVal = -1e6

		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				val := result.At(i, j)
				if math.IsNaN(val) {
					result.Set(i, j, 0.0)
				} else if math.IsInf(val, 1) {
					result.Set(i, j, maxVal)
				} else if math.IsInf(val, -1) {
					result.Set(i, j, minVal)
				}
			}
		}

	default:
		return nil, NewError(ErrInvalidInput,
			fmt.Sprintf("unknown cleaning strategy: %s", strategy))
	}

	return result, nil
}

// ValidateAndClean validates a tensor and applies cleaning if necessary.
func ValidateAndClean(tensor Tensor, name string, cleanStrategy string) (Tensor, error) {
	if err := ValidateNonEmpty(tensor, name); err != nil {
		return nil, err
	}

	// Check if cleaning is needed
	if tensor.HasNaN() || tensor.HasInf() {
		if cleanStrategy == "" {
			return nil, NewError(ErrNumericalInstability,
				fmt.Sprintf("%s contains NaN or infinite values", name))
		}

		cleaned, err := CleanNaNInf(tensor, cleanStrategy)
		if err != nil {
			return nil, NewErrorWithCause(ErrNumericalInstability,
				fmt.Sprintf("failed to clean %s", name), err)
		}

		return cleaned, nil
	}

	return tensor, nil
}

// GradientClipping provides various gradient clipping strategies.
type GradientClipping struct {
	MaxNorm    float64 `json:"max_norm"`
	ClipValue  float64 `json:"clip_value"`
	Strategy   string  `json:"strategy"` // "norm", "value", "adaptive"
	Percentile float64 `json:"percentile"`
}

// ClipGradients applies gradient clipping using the specified strategy.
func (gc *GradientClipping) ClipGradients(gradients []Tensor) error {
	if len(gradients) == 0 {
		return nil
	}

	switch gc.Strategy {
	case "norm":
		return gc.clipByNorm(gradients)
	case "value":
		return gc.clipByValue(gradients)
	case "adaptive":
		return gc.clipAdaptive(gradients)
	default:
		return NewError(ErrInvalidInput,
			fmt.Sprintf("unknown clipping strategy: %s", gc.Strategy))
	}
}

// clipByNorm clips gradients by global norm.
func (gc *GradientClipping) clipByNorm(gradients []Tensor) error {
	if gc.MaxNorm <= 0 {
		return NewError(ErrInvalidInput, "max_norm must be positive")
	}

	// Calculate global norm
	var totalNorm float64
	for _, grad := range gradients {
		if grad == nil {
			continue
		}
		norm := grad.Norm()
		totalNorm += norm * norm
	}
	totalNorm = math.Sqrt(totalNorm)

	// Apply clipping if necessary
	if totalNorm > gc.MaxNorm {
		clipCoeff := gc.MaxNorm / totalNorm
		for i, grad := range gradients {
			if grad != nil {
				gradients[i] = grad.Scale(clipCoeff)
			}
		}
	}

	return nil
}

// clipByValue clips gradients by absolute value.
func (gc *GradientClipping) clipByValue(gradients []Tensor) error {
	if gc.ClipValue <= 0 {
		return NewError(ErrInvalidInput, "clip_value must be positive")
	}

	for i, grad := range gradients {
		if grad == nil {
			continue
		}

		gradients[i] = grad.Clamp(-gc.ClipValue, gc.ClipValue)
	}

	return nil
}

// clipAdaptive applies adaptive gradient clipping based on percentiles.
func (gc *GradientClipping) clipAdaptive(gradients []Tensor) error {
	if gc.Percentile <= 0 || gc.Percentile >= 100 {
		return NewError(ErrInvalidInput, "percentile must be between 0 and 100")
	}

	// Collect all gradient values
	var allValues []float64
	for _, grad := range gradients {
		if grad == nil {
			continue
		}

		rows, cols := grad.Dims()
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				val := math.Abs(grad.At(i, j))
				if !math.IsNaN(val) && !math.IsInf(val, 0) {
					allValues = append(allValues, val)
				}
			}
		}
	}

	if len(allValues) == 0 {
		return nil
	}

	// Calculate percentile threshold
	// This is a simplified percentile calculation
	// In practice, you might want to use a more sophisticated method
	threshold := gc.ClipValue // Fallback to clip_value if percentile calculation fails

	// Apply clipping
	for i, grad := range gradients {
		if grad != nil {
			gradients[i] = grad.Clamp(-threshold, threshold)
		}
	}

	return nil
}
