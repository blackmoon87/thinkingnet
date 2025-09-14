package optimizers

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestRMSpropConfig(t *testing.T) {
	// Test default config
	config := DefaultRMSpropConfig()
	if config.LearningRate != 0.01 {
		t.Errorf("Expected default learning rate 0.01, got %f", config.LearningRate)
	}
	if config.Alpha != 0.99 {
		t.Errorf("Expected default alpha 0.99, got %f", config.Alpha)
	}
	if config.Epsilon != 1e-8 {
		t.Errorf("Expected default epsilon 1e-8, got %f", config.Epsilon)
	}
	if config.WeightDecay != 0.0 {
		t.Errorf("Expected default weight decay 0.0, got %f", config.WeightDecay)
	}
	if config.Momentum != 0.0 {
		t.Errorf("Expected default momentum 0.0, got %f", config.Momentum)
	}
	if config.Centered != false {
		t.Errorf("Expected default centered false, got %t", config.Centered)
	}

	// Test config validation
	validConfig := RMSpropConfig{
		LearningRate: 0.01,
		Alpha:        0.99,
		Epsilon:      1e-8,
		WeightDecay:  0.0001,
		Momentum:     0.9,
		Centered:     true,
		MaxGradNorm:  1.0,
	}
	if err := validateRMSpropConfig(validConfig); err != nil {
		t.Errorf("Valid config should not produce error: %v", err)
	}

	// Test invalid learning rate
	invalidConfig := validConfig
	invalidConfig.LearningRate = -0.01
	if err := validateRMSpropConfig(invalidConfig); err == nil {
		t.Error("Expected error for negative learning rate")
	}

	// Test invalid alpha
	invalidConfig = validConfig
	invalidConfig.Alpha = 1.0
	if err := validateRMSpropConfig(invalidConfig); err == nil {
		t.Error("Expected error for alpha >= 1")
	}

	// Test invalid epsilon
	invalidConfig = validConfig
	invalidConfig.Epsilon = 0.0
	if err := validateRMSpropConfig(invalidConfig); err == nil {
		t.Error("Expected error for zero epsilon")
	}

	// Test invalid weight decay
	invalidConfig = validConfig
	invalidConfig.WeightDecay = -0.1
	if err := validateRMSpropConfig(invalidConfig); err == nil {
		t.Error("Expected error for negative weight decay")
	}

	// Test invalid momentum
	invalidConfig = validConfig
	invalidConfig.Momentum = 1.0
	if err := validateRMSpropConfig(invalidConfig); err == nil {
		t.Error("Expected error for momentum >= 1")
	}
}

func TestRMSpropCreation(t *testing.T) {
	// Test creation with valid config
	config := DefaultRMSpropConfig()
	config.LearningRate = 0.02
	rmsprop := NewRMSprop(config)

	if rmsprop.Name() != "rmsprop" {
		t.Errorf("Expected name 'rmsprop', got '%s'", rmsprop.Name())
	}
	if rmsprop.LearningRate() != 0.02 {
		t.Errorf("Expected learning rate 0.02, got %f", rmsprop.LearningRate())
	}

	// Test creation with defaults
	rmspropDefault := NewRMSpropWithDefaults(0.03)
	if rmspropDefault.LearningRate() != 0.03 {
		t.Errorf("Expected learning rate 0.03, got %f", rmspropDefault.LearningRate())
	}

	// Test creation of centered variant
	rmspropCentered := NewRMSpropCentered(0.01, 0.95)
	if rmspropCentered.LearningRate() != 0.01 {
		t.Errorf("Expected learning rate 0.01, got %f", rmspropCentered.LearningRate())
	}
	if rmspropCentered.config.Alpha != 0.95 {
		t.Errorf("Expected alpha 0.95, got %f", rmspropCentered.config.Alpha)
	}
	if !rmspropCentered.config.Centered {
		t.Error("Expected centered to be true")
	}

	// Test creation with invalid config
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for invalid config")
		}
	}()
	invalidConfig := config
	invalidConfig.LearningRate = -0.01
	NewRMSprop(invalidConfig)
}

func TestRMSpropBasicUpdate(t *testing.T) {
	// Test basic RMSprop without momentum or centering
	config := RMSpropConfig{
		LearningRate: 0.1,
		Alpha:        0.9,
		Epsilon:      1e-8,
		WeightDecay:  0.0,
		Momentum:     0.0,
		Centered:     false,
		MaxGradNorm:  0.0,
	}
	rmsprop := NewRMSprop(config)

	params := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}
	grads := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}

	// First update
	// squareAvg = 0.9 * 0 + 0.1 * 1^2 = 0.1
	// update = lr * grad / sqrt(squareAvg + eps) = 0.1 * 1.0 / sqrt(0.1 + 1e-8)
	originalParam := params[0].At(0, 0)
	rmsprop.Update(params, grads)

	expectedUpdate := 0.1 * 1.0 / math.Sqrt(0.1+1e-8)
	expectedParam := originalParam - expectedUpdate

	tolerance := 1e-6
	if !almostEqual(params[0].At(0, 0), expectedParam, tolerance) {
		t.Errorf("First update: expected %f, got %f", expectedParam, params[0].At(0, 0))
	}
}

func TestRMSpropAdaptiveLearning(t *testing.T) {
	config := RMSpropConfig{
		LearningRate: 0.1,
		Alpha:        0.9,
		Epsilon:      1e-8,
		WeightDecay:  0.0,
		Momentum:     0.0,
		Centered:     false,
		MaxGradNorm:  0.0,
	}
	rmsprop := NewRMSprop(config)

	params := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}

	// First update with large gradient
	grads1 := []core.Tensor{
		newMockTensor(1, 1, []float64{10.0}),
	}

	originalParam := params[0].At(0, 0)
	rmsprop.Update(params, grads1)
	firstUpdate := originalParam - params[0].At(0, 0)

	// Second update with same large gradient
	secondOriginal := params[0].At(0, 0)
	rmsprop.Update(params, grads1)
	secondUpdate := secondOriginal - params[0].At(0, 0)

	// Second update should be smaller due to accumulated squared gradients
	if secondUpdate >= firstUpdate {
		t.Errorf("RMSprop should reduce step size with repeated large gradients: first=%f, second=%f",
			firstUpdate, secondUpdate)
	}
}

func TestRMSpropCentered(t *testing.T) {
	// Test centered RMSprop variant
	config := RMSpropConfig{
		LearningRate: 0.1,
		Alpha:        0.9,
		Epsilon:      1e-8,
		WeightDecay:  0.0,
		Momentum:     0.0,
		Centered:     true, // Enable centered variant
		MaxGradNorm:  0.0,
	}
	rmsprop := NewRMSprop(config)

	params := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}
	grads := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}

	// Perform update
	originalParam := params[0].At(0, 0)
	rmsprop.Update(params, grads)

	// Check that gradAvg was initialized (indirect test)
	if len(rmsprop.gradAvg) != 1 {
		t.Errorf("Expected gradAvg to be initialized with 1 tensor, got %d", len(rmsprop.gradAvg))
	}

	// Parameter should still be updated
	if params[0].At(0, 0) >= originalParam {
		t.Errorf("Parameter should decrease: %f -> %f", originalParam, params[0].At(0, 0))
	}
}

func TestRMSpropMomentum(t *testing.T) {
	// Test RMSprop with momentum
	config := RMSpropConfig{
		LearningRate: 0.1,
		Alpha:        0.9,
		Epsilon:      1e-8,
		WeightDecay:  0.0,
		Momentum:     0.9, // Add momentum
		Centered:     false,
		MaxGradNorm:  0.0,
	}
	rmsprop := NewRMSprop(config)

	params := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}
	grads := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}

	// First update
	rmsprop.Update(params, grads)

	// Second update with same gradient
	rmsprop.Update(params, grads)

	// With momentum, the second update should be influenced by momentum accumulation
	// The exact comparison depends on the balance between momentum and adaptive learning rate
	// We just check that momentum buffers were initialized
	if len(rmsprop.momentum) != 1 {
		t.Errorf("Expected momentum to be initialized with 1 tensor, got %d", len(rmsprop.momentum))
	}
}

func TestRMSpropWeightDecay(t *testing.T) {
	config := RMSpropConfig{
		LearningRate: 0.1,
		Alpha:        0.9,
		Epsilon:      1e-8,
		WeightDecay:  0.1, // Add weight decay
		Momentum:     0.0,
		Centered:     false,
		MaxGradNorm:  0.0,
	}
	rmsprop := NewRMSprop(config)

	params := []core.Tensor{
		newMockTensor(1, 1, []float64{2.0}), // Non-zero parameter for weight decay effect
	}
	grads := []core.Tensor{
		newMockTensor(1, 1, []float64{0.0}), // Zero gradient to isolate weight decay effect
	}

	originalParam := params[0].At(0, 0)
	rmsprop.Update(params, grads)
	updatedParam := params[0].At(0, 0)

	// With weight decay, parameter should decrease even with zero gradient
	if updatedParam >= originalParam {
		t.Errorf("Weight decay should decrease parameter: %f -> %f", originalParam, updatedParam)
	}
}

func TestRMSpropConfig_Interface(t *testing.T) {
	rmsprop := NewRMSpropWithDefaults(0.01)

	config := rmsprop.Config()
	if config.Name != "rmsprop" {
		t.Errorf("Expected config name 'rmsprop', got '%s'", config.Name)
	}
	if config.LearningRate != 0.01 {
		t.Errorf("Expected config learning rate 0.01, got %f", config.LearningRate)
	}

	// Check that parameters are included
	if _, ok := config.Parameters["alpha"]; !ok {
		t.Error("Expected alpha in config parameters")
	}
	if _, ok := config.Parameters["epsilon"]; !ok {
		t.Error("Expected epsilon in config parameters")
	}
	if _, ok := config.Parameters["momentum"]; !ok {
		t.Error("Expected momentum in config parameters")
	}
	if _, ok := config.Parameters["centered"]; !ok {
		t.Error("Expected centered in config parameters")
	}
}

func TestRMSpropReset(t *testing.T) {
	rmsprop := NewRMSpropCentered(0.01, 0.9)

	params := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}
	grads := []core.Tensor{
		newMockTensor(1, 1, []float64{0.1}),
	}

	// Perform update to initialize state
	rmsprop.Update(params, grads)

	if !rmsprop.initialized {
		t.Error("RMSprop should be initialized after update")
	}
	if rmsprop.stepCount != 1 {
		t.Errorf("Expected step count 1, got %d", rmsprop.stepCount)
	}

	// Reset and check state
	rmsprop.Reset()

	if rmsprop.initialized {
		t.Error("RMSprop should not be initialized after reset")
	}
	if rmsprop.stepCount != 0 {
		t.Errorf("Expected step count 0 after reset, got %d", rmsprop.stepCount)
	}
	if rmsprop.squareAvg != nil {
		t.Error("Square average buffers should be nil after reset")
	}
	if rmsprop.gradAvg != nil {
		t.Error("Gradient average buffers should be nil after reset")
	}
	if rmsprop.momentum != nil {
		t.Error("Momentum buffers should be nil after reset")
	}
}

func TestRMSpropInvalidUpdate(t *testing.T) {
	rmsprop := NewRMSpropWithDefaults(0.01)

	params := []core.Tensor{
		newMockTensor(2, 2, []float64{1, 2, 3, 4}),
	}
	grads := []core.Tensor{
		newMockTensor(2, 3, []float64{1, 2, 3, 4, 5, 6}), // Wrong dimensions
	}

	// Should panic on dimension mismatch
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for dimension mismatch")
		}
	}()
	rmsprop.Update(params, grads)
}

func TestRMSpropNumericalStability(t *testing.T) {
	// Test with very small gradients to check numerical stability
	config := RMSpropConfig{
		LearningRate: 0.1,
		Alpha:        0.999, // High alpha for slow adaptation
		Epsilon:      1e-8,
		WeightDecay:  0.0,
		Momentum:     0.0,
		Centered:     false,
		MaxGradNorm:  0.0,
	}
	rmsprop := NewRMSprop(config)

	params := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}
	grads := []core.Tensor{
		newMockTensor(1, 1, []float64{1e-10}), // Very small gradient
	}

	// Should not panic or produce NaN/Inf
	originalParam := params[0].At(0, 0)
	rmsprop.Update(params, grads)
	updatedParam := params[0].At(0, 0)

	// Check for NaN or Inf
	if math.IsNaN(updatedParam) || math.IsInf(updatedParam, 0) {
		t.Errorf("Update produced invalid value: %f", updatedParam)
	}

	// Parameter should change slightly
	if updatedParam == originalParam {
		t.Error("Parameter should change with non-zero gradient")
	}
}
