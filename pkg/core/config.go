package core

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config represents the global library configuration.
type Config struct {
	// Numerical precision settings
	Epsilon  float64 `json:"epsilon"`
	MaxFloat float64 `json:"max_float"`
	MinFloat float64 `json:"min_float"`

	// Memory management
	EnablePooling bool `json:"enable_pooling"`
	PoolSize      int  `json:"pool_size"`

	// Parallel processing
	MaxWorkers     int  `json:"max_workers"`
	EnableParallel bool `json:"enable_parallel"`

	// Debugging and logging
	DebugMode bool   `json:"debug_mode"`
	LogLevel  string `json:"log_level"`

	// Random seed for reproducibility
	GlobalSeed int64 `json:"global_seed"`

	// Performance settings
	UseBLAS        bool `json:"use_blas"`
	OptimizeMemory bool `json:"optimize_memory"`
}

// DefaultConfig returns the default configuration.
func DefaultConfig() *Config {
	return &Config{
		Epsilon:        1e-8,
		MaxFloat:       1e10,
		MinFloat:       -1e10,
		EnablePooling:  true,
		PoolSize:       100,
		MaxWorkers:     4,
		EnableParallel: true,
		DebugMode:      false,
		LogLevel:       "INFO",
		GlobalSeed:     42,
		UseBLAS:        true,
		OptimizeMemory: true,
	}
}

// globalConfig holds the current global configuration.
var globalConfig *Config = DefaultConfig()

// GetConfig returns the current global configuration.
func GetConfig() *Config {
	return globalConfig
}

// SetConfig sets the global configuration.
func SetConfig(config *Config) {
	globalConfig = config
}

// UpdateConfig updates specific fields in the global configuration.
func UpdateConfig(updates map[string]any) error {
	for key, value := range updates {
		switch key {
		case "epsilon":
			if v, ok := value.(float64); ok {
				globalConfig.Epsilon = v
			} else {
				return NewError(ErrConfigurationError, fmt.Sprintf("epsilon must be float64, got %T", value))
			}
		case "max_float":
			if v, ok := value.(float64); ok {
				globalConfig.MaxFloat = v
			} else {
				return NewError(ErrConfigurationError, fmt.Sprintf("max_float must be float64, got %T", value))
			}
		case "min_float":
			if v, ok := value.(float64); ok {
				globalConfig.MinFloat = v
			} else {
				return NewError(ErrConfigurationError, fmt.Sprintf("min_float must be float64, got %T", value))
			}
		case "enable_pooling":
			if v, ok := value.(bool); ok {
				globalConfig.EnablePooling = v
			} else {
				return NewError(ErrConfigurationError, fmt.Sprintf("enable_pooling must be bool, got %T", value))
			}
		case "pool_size":
			if v, ok := value.(int); ok {
				globalConfig.PoolSize = v
			} else {
				return NewError(ErrConfigurationError, fmt.Sprintf("pool_size must be int, got %T", value))
			}
		case "max_workers":
			if v, ok := value.(int); ok {
				globalConfig.MaxWorkers = v
			} else {
				return NewError(ErrConfigurationError, fmt.Sprintf("max_workers must be int, got %T", value))
			}
		case "enable_parallel":
			if v, ok := value.(bool); ok {
				globalConfig.EnableParallel = v
			} else {
				return NewError(ErrConfigurationError, fmt.Sprintf("enable_parallel must be bool, got %T", value))
			}
		case "debug_mode":
			if v, ok := value.(bool); ok {
				globalConfig.DebugMode = v
			} else {
				return NewError(ErrConfigurationError, fmt.Sprintf("debug_mode must be bool, got %T", value))
			}
		case "log_level":
			if v, ok := value.(string); ok {
				globalConfig.LogLevel = v
			} else {
				return NewError(ErrConfigurationError, fmt.Sprintf("log_level must be string, got %T", value))
			}
		case "global_seed":
			if v, ok := value.(int64); ok {
				globalConfig.GlobalSeed = v
			} else {
				return NewError(ErrConfigurationError, fmt.Sprintf("global_seed must be int64, got %T", value))
			}
		case "use_blas":
			if v, ok := value.(bool); ok {
				globalConfig.UseBLAS = v
			} else {
				return NewError(ErrConfigurationError, fmt.Sprintf("use_blas must be bool, got %T", value))
			}
		case "optimize_memory":
			if v, ok := value.(bool); ok {
				globalConfig.OptimizeMemory = v
			} else {
				return NewError(ErrConfigurationError, fmt.Sprintf("optimize_memory must be bool, got %T", value))
			}
		default:
			return NewError(ErrConfigurationError, fmt.Sprintf("unknown configuration key: %s", key))
		}
	}
	return nil
}

// LoadConfig loads configuration from a JSON file.
func LoadConfig(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return NewErrorWithCause(ErrFileIO, fmt.Sprintf("failed to read config file %s", filename), err)
	}

	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return NewErrorWithCause(ErrConfigurationError, "failed to parse config JSON", err)
	}

	// Validate configuration
	if err := validateConfig(&config); err != nil {
		return err
	}

	globalConfig = &config
	return nil
}

// SaveConfig saves the current configuration to a JSON file.
func SaveConfig(filename string) error {
	data, err := json.MarshalIndent(globalConfig, "", "  ")
	if err != nil {
		return NewErrorWithCause(ErrConfigurationError, "failed to marshal config to JSON", err)
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		return NewErrorWithCause(ErrFileIO, fmt.Sprintf("failed to write config file %s", filename), err)
	}

	return nil
}

// validateConfig validates the configuration values.
func validateConfig(config *Config) error {
	if config.Epsilon <= 0 {
		return NewError(ErrConfigurationError, "epsilon must be positive")
	}

	if config.MaxFloat <= config.MinFloat {
		return NewError(ErrConfigurationError, "max_float must be greater than min_float")
	}

	if config.PoolSize <= 0 {
		return NewError(ErrConfigurationError, "pool_size must be positive")
	}

	if config.MaxWorkers <= 0 {
		return NewError(ErrConfigurationError, "max_workers must be positive")
	}

	validLogLevels := map[string]bool{
		"DEBUG": true,
		"INFO":  true,
		"WARN":  true,
		"ERROR": true,
	}

	if !validLogLevels[config.LogLevel] {
		return NewError(ErrConfigurationError, fmt.Sprintf("invalid log_level: %s", config.LogLevel))
	}

	return nil
}

// ConfigOption represents a functional option for configuration.
type ConfigOption func(*Config)

// WithEpsilon sets the numerical epsilon.
func WithEpsilon(epsilon float64) ConfigOption {
	return func(c *Config) { c.Epsilon = epsilon }
}

// WithPooling enables or disables memory pooling.
func WithPooling(enabled bool) ConfigOption {
	return func(c *Config) { c.EnablePooling = enabled }
}

// WithParallel enables or disables parallel processing.
func WithParallel(enabled bool) ConfigOption {
	return func(c *Config) { c.EnableParallel = enabled }
}

// WithDebug enables or disables debug mode.
func WithDebug(enabled bool) ConfigOption {
	return func(c *Config) { c.DebugMode = enabled }
}

// WithLogLevel sets the log level.
func WithLogLevel(level string) ConfigOption {
	return func(c *Config) { c.LogLevel = level }
}

// WithGlobalSeed sets the global random seed.
func WithGlobalSeed(seed int64) ConfigOption {
	return func(c *Config) { c.GlobalSeed = seed }
}

// NewConfig creates a new configuration with options.
func NewConfig(options ...ConfigOption) *Config {
	config := DefaultConfig()

	for _, option := range options {
		option(config)
	}

	return config
}

// GetEpsilon returns the current epsilon value.
func GetEpsilon() float64 {
	return globalConfig.Epsilon
}

// GetMaxFloat returns the maximum float value.
func GetMaxFloat() float64 {
	return globalConfig.MaxFloat
}

// GetMinFloat returns the minimum float value.
func GetMinFloat() float64 {
	return globalConfig.MinFloat
}

// IsPoolingEnabled returns true if memory pooling is enabled.
func IsPoolingEnabled() bool {
	return globalConfig.EnablePooling
}

// IsParallelEnabled returns true if parallel processing is enabled.
func IsParallelEnabled() bool {
	return globalConfig.EnableParallel
}

// IsDebugMode returns true if debug mode is enabled.
func IsDebugMode() bool {
	return globalConfig.DebugMode
}

// GetGlobalSeed returns the global random seed.
func GetGlobalSeed() int64 {
	return globalConfig.GlobalSeed
}

// PrintConfig prints the current configuration to stdout.
func PrintConfig() {
	data, _ := json.MarshalIndent(globalConfig, "", "  ")
	fmt.Println("ThinkingNet Configuration:")
	fmt.Println(string(data))
}

// ConfigExists checks if a configuration file exists.
func ConfigExists(filename string) bool {
	_, err := os.Stat(filename)
	return !os.IsNotExist(err)
}
