package optimizers

import (
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestSGDConfig(t *testing.T) {
	// Test default config
	config := DefaultSGDConfig()
	if config.LearningRate != 0.01 {
		t.Errorf("Expected default learning rate 0.01, got %f", config.LearningRate)
	}
	if config.Momentum != 0.0 {
		t.Errorf("Expected default momentum 0.0, got %f", config.Momentum)
	}
	if config.Dampening != 0.0 {
		t.Errorf("Expected default dampening 0.0, got %f", config.Dampening)
	}
	if config.WeightDecay != 0.0 {
		t.Errorf("Expected default weight decay 0.0, got %f", config.WeightDecay)
	}
	if config.Nesterov != false {
		t.Errorf("Expected default nesterov false, got %t", config.Nesterov)
	}

	// Test config validation
	validConfig := SGDConfig{
		LearningRate: 0.01,
		Momentum:     0.9,
		Dampening:    0.1,
		WeightDecay:  0.0001,
		Nesterov:     true,
		MaxGradNorm:  1.0,
	}
	if err := validateSGDConfig(validConfig); err != nil {
		t.Errorf("Valid config should not produce error: %v", err)
	}

	// Test invalid learning rate
	invalidConfig := validConfig
	invalidConfig.LearningRate = -0.01
	if err := validateSGDConfig(invalidConfig); err == nil {
		t.Error("Expected error for negative learning rate")
	}

	// Test invalid momentum
	invalidConfig = validConfig
	invalidConfig.Momentum = 1.0
	if err := validateSGDConfig(invalidConfig); err == nil {
		t.Error("Expected error for momentum >= 1")
	}

	// Test invalid dampening
	invalidConfig = validConfig
	invalidConfig.Dampening = 1.5
	if err := validateSGDConfig(invalidConfig); err == nil {
		t.Error("Expected error for dampening > 1")
	}

	// Test invalid weight decay
	invalidConfig = validConfig
	invalidConfig.WeightDecay = -0.1
	if err := validateSGDConfig(invalidConfig); err == nil {
		t.Error("Expected error for negative weight decay")
	}

	// Test Nesterov without momentum
	invalidConfig = validConfig
	invalidConfig.Momentum = 0.0
	invalidConfig.Nesterov = true
	if err := validateSGDConfig(invalidConfig); err == nil {
		t.Error("Expected error for Nesterov without momentum")
	}
}

func TestSGDCreation(t *testing.T) {
	// Test creation with valid config
	config := DefaultSGDConfig()
	config.LearningRate = 0.02
	sgd := NewSGD(config)

	if sgd.Name() != "sgd" {
		t.Errorf("Expected name 'sgd', got '%s'", sgd.Name())
	}
	if sgd.LearningRate() != 0.02 {
		t.Errorf("Expected learning rate 0.02, got %f", sgd.LearningRate())
	}

	// Test creation with defaults
	sgdDefault := NewSGDWithDefaults(0.03)
	if sgdDefault.LearningRate() != 0.03 {
		t.Errorf("Expected learning rate 0.03, got %f", sgdDefault.LearningRate())
	}

	// Test creation with momentum
	sgdMomentum := NewSGDWithMomentum(0.01, 0.9)
	if sgdMomentum.LearningRate() != 0.01 {
		t.Errorf("Expected learning rate 0.01, got %f", sgdMomentum.LearningRate())
	}
	if sgdMomentum.config.Momentum != 0.9 {
		t.Errorf("Expected momentum 0.9, got %f", sgdMomentum.config.Momentum)
	}

	// Test creation with invalid config
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for invalid config")
		}
	}()
	invalidConfig := config
	invalidConfig.LearningRate = -0.01
	NewSGD(invalidConfig)
}

func TestSGDBasicUpdate(t *testing.T) {
	// Test basic SGD without momentum
	config := SGDConfig{
		LearningRate: 0.1,
		Momentum:     0.0,
		Dampening:    0.0,
		WeightDecay:  0.0,
		Nesterov:     false,
		MaxGradNorm:  0.0,
	}
	sgd := NewSGD(config)

	params := []core.Tensor{
		newMockTensor(2, 2, []float64{1.0, 2.0, 3.0, 4.0}),
	}
	grads := []core.Tensor{
		newMockTensor(2, 2, []float64{0.1, 0.2, 0.3, 0.4}),
	}

	// Store original parameter values
	originalValues := make([]float64, 4)
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			originalValues[i*2+j] = params[0].At(i, j)
		}
	}

	// Perform update
	sgd.Update(params, grads)

	// Check that parameters were updated correctly: param = param - lr * grad
	tolerance := 1e-6
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			expected := originalValues[i*2+j] - 0.1*grads[0].At(i, j)
			actual := params[0].At(i, j)
			if !almostEqual(expected, actual, tolerance) {
				t.Errorf("Parameter at (%d, %d): expected %f, got %f", i, j, expected, actual)
			}
		}
	}
}

func TestSGDMomentum(t *testing.T) {
	// Test SGD with momentum
	config := SGDConfig{
		LearningRate: 0.1,
		Momentum:     0.9,
		Dampening:    0.0,
		WeightDecay:  0.0,
		Nesterov:     false,
		MaxGradNorm:  0.0,
	}
	sgd := NewSGD(config)

	params := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}
	grads := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}

	// First update: velocity = 0.9 * 0 + 1.0 * 1.0 = 1.0
	// param = 1.0 - 0.1 * 1.0 = 0.9
	originalParam := params[0].At(0, 0)
	sgd.Update(params, grads)
	firstUpdate := originalParam - params[0].At(0, 0)

	// Second update with same gradient
	// velocity = 0.9 * 1.0 + 1.0 * 1.0 = 1.9
	// param = 0.9 - 0.1 * 1.9 = 0.71
	secondOriginal := params[0].At(0, 0)
	sgd.Update(params, grads)
	secondUpdate := secondOriginal - params[0].At(0, 0)

	// Second update should be larger due to momentum accumulation
	if secondUpdate <= firstUpdate {
		t.Errorf("Momentum should increase update size: first=%f, second=%f", firstUpdate, secondUpdate)
	}

	tolerance := 1e-6
	expectedFirstUpdate := 0.1 * 1.0 // lr * velocity
	if !almostEqual(firstUpdate, expectedFirstUpdate, tolerance) {
		t.Errorf("First update: expected %f, got %f", expectedFirstUpdate, firstUpdate)
	}

	expectedSecondUpdate := 0.1 * 1.9 // lr * accumulated_velocity
	if !almostEqual(secondUpdate, expectedSecondUpdate, tolerance) {
		t.Errorf("Second update: expected %f, got %f", expectedSecondUpdate, secondUpdate)
	}
}

func TestSGDNesterov(t *testing.T) {
	// Test SGD with Nesterov momentum
	config := SGDConfig{
		LearningRate: 0.1,
		Momentum:     0.9,
		Dampening:    0.0,
		WeightDecay:  0.0,
		Nesterov:     true,
		MaxGradNorm:  0.0,
	}
	sgd := NewSGD(config)

	params := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}
	grads := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}

	// Perform update
	sgd.Update(params, grads)

	// For Nesterov: update = momentum * velocity + grad
	// First iteration: velocity = 1.0, update = 0.9 * 1.0 + 1.0 = 1.9
	// param = 1.0 - 0.1 * 1.9 = 0.81
	expectedParam := 1.0 - 0.1*1.9
	tolerance := 1e-6
	if !almostEqual(params[0].At(0, 0), expectedParam, tolerance) {
		t.Errorf("Nesterov update: expected %f, got %f", expectedParam, params[0].At(0, 0))
	}
}

func TestSGDWeightDecay(t *testing.T) {
	config := SGDConfig{
		LearningRate: 0.1,
		Momentum:     0.0,
		Dampening:    0.0,
		WeightDecay:  0.1, // Add weight decay
		Nesterov:     false,
		MaxGradNorm:  0.0,
	}
	sgd := NewSGD(config)

	params := []core.Tensor{
		newMockTensor(1, 1, []float64{2.0}), // Non-zero parameter for weight decay effect
	}
	grads := []core.Tensor{
		newMockTensor(1, 1, []float64{0.0}), // Zero gradient to isolate weight decay effect
	}

	sgd.Update(params, grads)
	updatedParam := params[0].At(0, 0)

	// With weight decay: effective_grad = 0.0 + 0.1 * 2.0 = 0.2
	// param = 2.0 - 0.1 * 0.2 = 1.98
	expectedParam := 2.0 - 0.1*0.2
	tolerance := 1e-6
	if !almostEqual(updatedParam, expectedParam, tolerance) {
		t.Errorf("Weight decay update: expected %f, got %f", expectedParam, updatedParam)
	}
}

func TestSGDDampening(t *testing.T) {
	config := SGDConfig{
		LearningRate: 0.1,
		Momentum:     0.9,
		Dampening:    0.5, // Add dampening
		WeightDecay:  0.0,
		Nesterov:     false,
		MaxGradNorm:  0.0,
	}
	sgd := NewSGD(config)

	params := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}
	grads := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}

	// First update: velocity = 0.9 * 0 + (1 - 0.5) * 1.0 = 0.5
	sgd.Update(params, grads)

	// Second update: velocity = 0.9 * 0.5 + (1 - 0.5) * 1.0 = 0.45 + 0.5 = 0.95
	firstParam := params[0].At(0, 0)
	sgd.Update(params, grads)
	secondParam := params[0].At(0, 0)

	secondUpdate := firstParam - secondParam
	expectedSecondUpdate := 0.1 * 0.95 // lr * velocity_with_dampening

	tolerance := 1e-6
	if !almostEqual(secondUpdate, expectedSecondUpdate, tolerance) {
		t.Errorf("Dampening update: expected %f, got %f", expectedSecondUpdate, secondUpdate)
	}
}

func TestSGDConfig_Interface(t *testing.T) {
	sgd := NewSGDWithDefaults(0.01)

	config := sgd.Config()
	if config.Name != "sgd" {
		t.Errorf("Expected config name 'sgd', got '%s'", config.Name)
	}
	if config.LearningRate != 0.01 {
		t.Errorf("Expected config learning rate 0.01, got %f", config.LearningRate)
	}

	// Check that parameters are included
	if _, ok := config.Parameters["momentum"]; !ok {
		t.Error("Expected momentum in config parameters")
	}
	if _, ok := config.Parameters["dampening"]; !ok {
		t.Error("Expected dampening in config parameters")
	}
}

func TestSGDReset(t *testing.T) {
	sgd := NewSGDWithMomentum(0.01, 0.9)

	params := []core.Tensor{
		newMockTensor(1, 1, []float64{1.0}),
	}
	grads := []core.Tensor{
		newMockTensor(1, 1, []float64{0.1}),
	}

	// Perform update to initialize state
	sgd.Update(params, grads)

	if !sgd.initialized {
		t.Error("SGD should be initialized after update")
	}
	if sgd.stepCount != 1 {
		t.Errorf("Expected step count 1, got %d", sgd.stepCount)
	}

	// Reset and check state
	sgd.Reset()

	if sgd.initialized {
		t.Error("SGD should not be initialized after reset")
	}
	if sgd.stepCount != 0 {
		t.Errorf("Expected step count 0 after reset, got %d", sgd.stepCount)
	}
	if sgd.velocity != nil {
		t.Error("Velocity buffers should be nil after reset")
	}
}

func TestSGDInvalidUpdate(t *testing.T) {
	sgd := NewSGDWithDefaults(0.01)

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
	sgd.Update(params, grads)
}
