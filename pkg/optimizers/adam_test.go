package optimizers

import (
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestAdamConfig(t *testing.T) {
	// Test default config
	config := DefaultAdamConfig()
	if config.LearningRate != 0.001 {
		t.Errorf("Expected default learning rate 0.001, got %f", config.LearningRate)
	}
	if config.Beta1 != 0.9 {
		t.Errorf("Expected default beta1 0.9, got %f", config.Beta1)
	}
	if config.Beta2 != 0.999 {
		t.Errorf("Expected default beta2 0.999, got %f", config.Beta2)
	}
	if config.Epsilon != 1e-8 {
		t.Errorf("Expected default epsilon 1e-8, got %f", config.Epsilon)
	}

	// Test config validation
	validConfig := AdamConfig{
		LearningRate: 0.01,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  0.0,
		AMSGrad:      false,
		MaxGradNorm:  0.0,
	}
	if err := validateAdamConfig(validConfig); err != nil {
		t.Errorf("Valid config should not produce error: %v", err)
	}

	// Test invalid learning rate
	invalidConfig := validConfig
	invalidConfig.LearningRate = -0.01
	if err := validateAdamConfig(invalidConfig); err == nil {
		t.Error("Expected error for negative learning rate")
	}

	// Test invalid beta1
	invalidConfig = validConfig
	invalidConfig.Beta1 = 1.0
	if err := validateAdamConfig(invalidConfig); err == nil {
		t.Error("Expected error for beta1 >= 1")
	}

	// Test invalid beta2
	invalidConfig = validConfig
	invalidConfig.Beta2 = -0.1
	if err := validateAdamConfig(invalidConfig); err == nil {
		t.Error("Expected error for negative beta2")
	}

	// Test invalid epsilon
	invalidConfig = validConfig
	invalidConfig.Epsilon = 0.0
	if err := validateAdamConfig(invalidConfig); err == nil {
		t.Error("Expected error for zero epsilon")
	}

	// Test invalid weight decay
	invalidConfig = validConfig
	invalidConfig.WeightDecay = -0.1
	if err := validateAdamConfig(invalidConfig); err == nil {
		t.Error("Expected error for negative weight decay")
	}
}

func TestAdamCreation(t *testing.T) {
	// Test creation with valid config
	config := DefaultAdamConfig()
	config.LearningRate = 0.01
	adam, err := NewAdam(config)
	if err != nil {
		t.Fatalf("Failed to create Adam optimizer: %v", err)
	}

	if adam.Name() != "adam" {
		t.Errorf("Expected name 'adam', got '%s'", adam.Name())
	}
	if adam.LearningRate() != 0.01 {
		t.Errorf("Expected learning rate 0.01, got %f", adam.LearningRate())
	}

	// Test creation with defaults
	adamDefault, err := NewAdamWithDefaults(0.02)
	if err != nil {
		t.Fatalf("Failed to create Adam optimizer with defaults: %v", err)
	}
	if adamDefault.LearningRate() != 0.02 {
		t.Errorf("Expected learning rate 0.02, got %f", adamDefault.LearningRate())
	}

	// Test creation with invalid config
	invalidConfig := config
	invalidConfig.LearningRate = -0.01
	_, err = NewAdam(invalidConfig)
	if err == nil {
		t.Error("Expected error for invalid config")
	}
}

func TestAdamUpdate(t *testing.T) {
	config := AdamConfig{
		LearningRate: 0.1,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  0.0,
		AMSGrad:      false,
		MaxGradNorm:  0.0,
	}
	adam, err := NewAdam(config)
	if err != nil {
		t.Fatalf("Failed to create Adam optimizer: %v", err)
	}

	// Create test parameters and gradients
	params := []core.Tensor{
		newMockTensor(2, 2, []float64{1.0, 2.0, 3.0, 4.0}),
	}
	grads := []core.Tensor{
		newMockTensor(2, 2, []float64{0.1, 0.2, 0.3, 0.4}),
	}

	// Store original parameter values
	originalParam := params[0].Copy()

	// Perform update
	err = adam.Update(params, grads)
	if err != nil {
		t.Fatalf("Failed to update parameters: %v", err)
	}

	// Check that parameters were updated
	tolerance := 1e-6
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			original := originalParam.At(i, j)
			updated := params[0].At(i, j)
			if almostEqual(original, updated, tolerance) {
				t.Errorf("Parameter at (%d, %d) was not updated: %f == %f", i, j, original, updated)
			}
			// Parameters should decrease (gradient descent)
			if updated >= original {
				t.Errorf("Parameter at (%d, %d) should decrease: %f -> %f", i, j, original, updated)
			}
		}
	}

	// Check that step count increased
	if adam.stepCount != 1 {
		t.Errorf("Expected step count 1, got %d", adam.stepCount)
	}
}

func TestAdamBiasCorrection(t *testing.T) {
	config := AdamConfig{
		LearningRate: 0.1, // More reasonable learning rate
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  0.0,
		AMSGrad:      false,
		MaxGradNorm:  0.0,
	}
	adam, err := NewAdam(config)
	if err != nil {
		t.Fatalf("Failed to create Adam optimizer: %v", err)
	}

	params := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}
	grads := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}

	// First update - bias correction should have significant effect
	originalParam := params[0].At(0, 0)
	err = adam.Update(params, grads)
	if err != nil {
		t.Fatalf("Failed to update parameters: %v", err)
	}
	firstUpdate := originalParam - params[0].At(0, 0)

	// Bias correction should produce a reasonable update
	// With bias correction in first step, update should be substantial but not extreme
	if firstUpdate <= 0 {
		t.Error("Adam should produce positive update for positive gradient")
	}

	if firstUpdate > 1.0 {
		t.Errorf("Update seems too large: %f", firstUpdate)
	}

	// Test multiple updates to ensure convergence behavior
	for i := 0; i < 5; i++ {
		err = adam.Update(params, grads)
		if err != nil {
			t.Fatalf("Failed to update parameters on iteration %d: %v", i, err)
		}
	}

	// Parameter should have moved significantly from original
	finalParam := params[0].At(0, 0)
	totalChange := originalParam - finalParam
	if totalChange <= 0 {
		t.Error("Adam should decrease parameter with positive gradient")
	}
}

func TestAdamAMSGrad(t *testing.T) {
	// Test AMSGrad variant
	config := AdamConfig{
		LearningRate: 0.1,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  0.0,
		AMSGrad:      true,
		MaxGradNorm:  0.0,
	}
	adam, err := NewAdam(config)
	if err != nil {
		t.Fatalf("Failed to create Adam optimizer: %v", err)
	}

	params := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}
	grads := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}

	// Perform multiple updates to test AMSGrad behavior
	for i := 0; i < 5; i++ {
		err := adam.Update(params, grads)
		if err != nil {
			t.Fatalf("Failed to update parameters: %v", err)
		}
	}

	// Check that vMax was initialized (this is indirect since we can't access it directly)
	if len(adam.vMax) != 1 {
		t.Errorf("Expected vMax to be initialized with 1 tensor, got %d", len(adam.vMax))
	}
}

func TestAdamWeightDecay(t *testing.T) {
	config := AdamConfig{
		LearningRate: 0.1,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  0.1, // Add weight decay
		AMSGrad:      false,
		MaxGradNorm:  0.0,
	}
	adam, err := NewAdam(config)
	if err != nil {
		t.Fatalf("Failed to create Adam optimizer: %v", err)
	}

	params := []core.Tensor{
		newMockTensor(1, 1, []float64{2.0}), // Non-zero parameter for weight decay effect
	}
	grads := []core.Tensor{
		newMockTensor(1, 1, []float64{0.0}), // Zero gradient to isolate weight decay effect
	}

	originalParam := params[0].At(0, 0)
	err = adam.Update(params, grads)
	if err != nil {
		t.Fatalf("Failed to update parameters: %v", err)
	}
	updatedParam := params[0].At(0, 0)

	// With weight decay, parameter should decrease even with zero gradient
	if updatedParam >= originalParam {
		t.Errorf("Weight decay should decrease parameter: %f -> %f", originalParam, updatedParam)
	}
}

func TestAdamConfig_Interface(t *testing.T) {
	adam, err := NewAdamWithDefaults(0.01)
	if err != nil {
		t.Fatalf("Failed to create Adam optimizer: %v", err)
	}

	config := adam.Config()
	if config.Name != "adam" {
		t.Errorf("Expected config name 'adam', got '%s'", config.Name)
	}
	if config.LearningRate != 0.01 {
		t.Errorf("Expected config learning rate 0.01, got %f", config.LearningRate)
	}

	// Check that parameters are included
	if _, ok := config.Parameters["beta1"]; !ok {
		t.Error("Expected beta1 in config parameters")
	}
	if _, ok := config.Parameters["beta2"]; !ok {
		t.Error("Expected beta2 in config parameters")
	}
}

func TestAdamReset(t *testing.T) {
	adam, err := NewAdamWithDefaults(0.01)
	if err != nil {
		t.Fatalf("Failed to create Adam optimizer: %v", err)
	}

	params := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}
	grads := []core.Tensor{
		newMockTensor(1, 1, []float64{0.1}),
	}

	// Perform update to initialize state
	err = adam.Update(params, grads)
	if err != nil {
		t.Fatalf("Failed to update parameters: %v", err)
	}

	if !adam.initialized {
		t.Error("Adam should be initialized after update")
	}
	if adam.stepCount != 1 {
		t.Errorf("Expected step count 1, got %d", adam.stepCount)
	}

	// Reset and check state
	adam.Reset()

	if adam.initialized {
		t.Error("Adam should not be initialized after reset")
	}
	if adam.stepCount != 0 {
		t.Errorf("Expected step count 0 after reset, got %d", adam.stepCount)
	}
	if adam.m != nil {
		t.Error("Momentum buffers should be nil after reset")
	}
	if adam.v != nil {
		t.Error("Velocity buffers should be nil after reset")
	}
}

func TestAdamInvalidUpdate(t *testing.T) {
	adam, err := NewAdamWithDefaults(0.01)
	if err != nil {
		t.Fatalf("Failed to create Adam optimizer: %v", err)
	}

	params := []core.Tensor{
		newMockTensor(2, 2, []float64{1, 2, 3, 4}),
	}
	grads := []core.Tensor{
		newMockTensor(2, 3, []float64{1, 2, 3, 4, 5, 6}), // Wrong dimensions
	}

	// Should return error on dimension mismatch
	err = adam.Update(params, grads)
	if err == nil {
		t.Error("Expected error for dimension mismatch")
	}
}
