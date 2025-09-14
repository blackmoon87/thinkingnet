package core

import (
	"time"
)

// TrainingConfig holds configuration for model training.
type TrainingConfig struct {
	Epochs          int                 `json:"epochs"`
	BatchSize       int                 `json:"batch_size"`
	LearningRate    float64             `json:"learning_rate"`
	ValidationSplit float64             `json:"validation_split"`
	EarlyStopping   EarlyStoppingConfig `json:"early_stopping"`
	Callbacks       []Callback          `json:"-"` // Not serializable
	Metrics         []string            `json:"metrics"`
	Verbose         int                 `json:"verbose"`
	Shuffle         bool                `json:"shuffle"`
	Seed            int64               `json:"seed"`
}

// EarlyStoppingConfig configures early stopping behavior.
type EarlyStoppingConfig struct {
	Enabled  bool    `json:"enabled"`
	Monitor  string  `json:"monitor"` // "loss", "val_loss", "accuracy", etc.
	Patience int     `json:"patience"`
	MinDelta float64 `json:"min_delta"`
	Mode     string  `json:"mode"` // "min" or "max"
}

// OptimizerConfig holds optimizer configuration.
type OptimizerConfig struct {
	Name         string         `json:"name"`
	LearningRate float64        `json:"learning_rate"`
	Parameters   map[string]any `json:"parameters"`
}

// History tracks training progress.
type History struct {
	Epoch      []int                `json:"epoch"`
	Loss       []float64            `json:"loss"`
	Metrics    map[string][]float64 `json:"metrics"`
	ValLoss    []float64            `json:"val_loss"`
	ValMetrics map[string][]float64 `json:"val_metrics"`
	Duration   time.Duration        `json:"duration"`
	BestEpoch  int                  `json:"best_epoch"`
	BestScore  float64              `json:"best_score"`
}

// Metrics holds evaluation metrics.
type Metrics struct {
	Accuracy  float64 `json:"accuracy,omitempty"`
	Precision float64 `json:"precision,omitempty"`
	Recall    float64 `json:"recall,omitempty"`
	F1Score   float64 `json:"f1_score,omitempty"`
	ROCAUC    float64 `json:"roc_auc,omitempty"`
	MSE       float64 `json:"mse,omitempty"`
	RMSE      float64 `json:"rmse,omitempty"`
	MAE       float64 `json:"mae,omitempty"`
	R2Score   float64 `json:"r2_score,omitempty"`
}

// ModelState represents the complete state of a model for persistence.
type ModelState struct {
	Architecture ModelArchitecture `json:"architecture"`
	Weights      [][]float64       `json:"weights"`
	Config       ModelConfig       `json:"config"`
	History      *History          `json:"history,omitempty"`
	Metadata     map[string]any    `json:"metadata"`
	Version      string            `json:"version"`
	Timestamp    time.Time         `json:"timestamp"`
}

// ModelArchitecture describes the model structure.
type ModelArchitecture struct {
	Type   string        `json:"type"`
	Layers []LayerConfig `json:"layers"`
}

// LayerConfig describes a layer configuration.
type LayerConfig struct {
	Type       string         `json:"type"`
	Name       string         `json:"name"`
	Parameters map[string]any `json:"parameters"`
}

// ModelConfig holds model configuration.
type ModelConfig struct {
	Name       string           `json:"name"`
	Seed       int64            `json:"seed"`
	Verbose    bool             `json:"verbose"`
	Validation ValidationConfig `json:"validation"`
}

// ValidationConfig holds validation configuration.
type ValidationConfig struct {
	Split   float64 `json:"split"`
	Method  string  `json:"method"` // "split", "kfold", "stratified"
	Folds   int     `json:"folds"`
	Shuffle bool    `json:"shuffle"`
	Seed    int64   `json:"seed"`
}

// Callback interface for training callbacks.
type Callback interface {
	OnTrainBegin()
	OnTrainEnd()
	OnEpochBegin(epoch int)
	OnEpochEnd(epoch, maxEpochs int, loss float64, metrics map[string]float64)
	OnBatchBegin(batch int)
	OnBatchEnd(batch int, loss float64)
}

// BaseCallback provides default implementations for Callback interface.
type BaseCallback struct{}

func (cb *BaseCallback) OnTrainBegin()                                                             {}
func (cb *BaseCallback) OnTrainEnd()                                                               {}
func (cb *BaseCallback) OnEpochBegin(epoch int)                                                    {}
func (cb *BaseCallback) OnEpochEnd(epoch, maxEpochs int, loss float64, metrics map[string]float64) {}
func (cb *BaseCallback) OnBatchBegin(batch int)                                                    {}
func (cb *BaseCallback) OnBatchEnd(batch int, loss float64)                                        {}

// DataSplit represents a train/validation/test split.
type DataSplit struct {
	XTrain Tensor
	XVal   Tensor
	XTest  Tensor
	YTrain Tensor
	YVal   Tensor
	YTest  Tensor
}

// Dataset represents a machine learning dataset.
type Dataset struct {
	X            Tensor
	Y            Tensor
	FeatureNames []string
	TargetNames  []string
	Description  string
}

// ModelOption represents a functional option for model configuration.
type ModelOption func(*ModelConfig)

// WithSeed sets the random seed.
func WithSeed(seed int64) ModelOption {
	return func(c *ModelConfig) { c.Seed = seed }
}

// WithValidation sets validation configuration.
func WithValidation(split float64) ModelOption {
	return func(c *ModelConfig) {
		c.Validation = ValidationConfig{Split: split, Shuffle: true}
	}
}

// WithName sets the model name.
func WithName(name string) ModelOption {
	return func(c *ModelConfig) { c.Name = name }
}

// WithVerbose sets verbose mode.
func WithVerbose(verbose bool) ModelOption {
	return func(c *ModelConfig) { c.Verbose = verbose }
}

// NewModelConfig creates a new model configuration with options.
func NewModelConfig(options ...ModelOption) *ModelConfig {
	config := &ModelConfig{
		Name:    "model",
		Seed:    42,
		Verbose: false,
		Validation: ValidationConfig{
			Split:   0.2,
			Method:  "split",
			Shuffle: true,
			Seed:    42,
		},
	}

	for _, option := range options {
		option(config)
	}

	return config
}

// NewTrainingConfig creates a default training configuration.
func NewTrainingConfig() *TrainingConfig {
	return &TrainingConfig{
		Epochs:          100,
		BatchSize:       32,
		LearningRate:    0.001,
		ValidationSplit: 0.2,
		EarlyStopping: EarlyStoppingConfig{
			Enabled:  false,
			Monitor:  "val_loss",
			Patience: 10,
			MinDelta: 0.0001,
			Mode:     "min",
		},
		Metrics: []string{"accuracy"},
		Verbose: 1,
		Shuffle: true,
		Seed:    42,
	}
}
