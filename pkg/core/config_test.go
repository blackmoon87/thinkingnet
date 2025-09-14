package core

import (
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig()

	if config.Epsilon <= 0 {
		t.Errorf("Expected positive epsilon, got %f", config.Epsilon)
	}

	if config.MaxFloat <= 0 {
		t.Errorf("Expected positive MaxFloat, got %f", config.MaxFloat)
	}

	if config.MinFloat >= 0 {
		t.Errorf("Expected negative MinFloat, got %f", config.MinFloat)
	}
}

func TestGetSetConfig(t *testing.T) {
	originalConfig := GetConfig()

	newConfig := DefaultConfig()
	newConfig.Epsilon = 1e-10
	newConfig.EnablePooling = false

	SetConfig(newConfig)

	retrievedConfig := GetConfig()
	if retrievedConfig.Epsilon != 1e-10 {
		t.Errorf("Expected epsilon 1e-10, got %f", retrievedConfig.Epsilon)
	}

	if retrievedConfig.EnablePooling {
		t.Error("Expected pooling to be disabled")
	}

	// Restore original config
	SetConfig(originalConfig)
}

func TestConfigGetters(t *testing.T) {
	originalConfig := GetConfig()

	config := NewConfig(
		WithEpsilon(1e-9),
		WithPooling(true),
		WithParallel(false),
		WithDebug(true),
		WithGlobalSeed(54321),
	)

	SetConfig(config)

	if GetEpsilon() != 1e-9 {
		t.Errorf("Expected epsilon 1e-9, got %f", GetEpsilon())
	}

	if GetMaxFloat() <= 0 {
		t.Errorf("Expected positive MaxFloat, got %f", GetMaxFloat())
	}

	if GetMinFloat() >= 0 {
		t.Errorf("Expected negative MinFloat, got %f", GetMinFloat())
	}

	if !IsPoolingEnabled() {
		t.Error("Expected pooling to be enabled")
	}

	if IsParallelEnabled() {
		t.Error("Expected parallel to be disabled")
	}

	if !IsDebugMode() {
		t.Error("Expected debug mode to be enabled")
	}

	if GetGlobalSeed() != 54321 {
		t.Errorf("Expected global seed 54321, got %d", GetGlobalSeed())
	}

	// Restore original config
	SetConfig(originalConfig)
}

func TestPrintConfig(t *testing.T) {
	// This test just ensures PrintConfig doesn't panic
	PrintConfig()
}
