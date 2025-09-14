// Package preprocessing provides data splitting and validation utilities.
package preprocessing

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// SplitConfig holds configuration for data splitting.
type SplitConfig struct {
	TestSize    float64 `json:"test_size"`
	TrainSize   float64 `json:"train_size"`
	Shuffle     bool    `json:"shuffle"`
	Stratify    bool    `json:"stratify"`
	RandomState int64   `json:"random_state"`
}

// NewSplitConfig creates a default split configuration.
func NewSplitConfig() *SplitConfig {
	return &SplitConfig{
		TestSize:    0.25,
		TrainSize:   0.0, // Auto-calculated as 1 - TestSize
		Shuffle:     true,
		Stratify:    false,
		RandomState: 42,
	}
}

// SplitOption represents a functional option for split configuration.
type SplitOption func(*SplitConfig)

// WithTestSize sets the test size proportion.
func WithTestSize(size float64) SplitOption {
	return func(c *SplitConfig) { c.TestSize = size }
}

// WithTrainSize sets the train size proportion.
func WithTrainSize(size float64) SplitOption {
	return func(c *SplitConfig) { c.TrainSize = size }
}

// WithShuffle enables or disables shuffling.
func WithShuffle(shuffle bool) SplitOption {
	return func(c *SplitConfig) { c.Shuffle = shuffle }
}

// WithStratify enables or disables stratified splitting.
func WithStratify(stratify bool) SplitOption {
	return func(c *SplitConfig) { c.Stratify = stratify }
}

// WithRandomState sets the random seed.
func WithRandomState(seed int64) SplitOption {
	return func(c *SplitConfig) { c.RandomState = seed }
}

// TrainTestSplit splits data into training and testing sets.
func TrainTestSplit(X, y core.Tensor, options ...SplitOption) (core.Tensor, core.Tensor, core.Tensor, core.Tensor, error) {
	config := NewSplitConfig()
	for _, option := range options {
		option(config)
	}

	if err := validateSplitInputs(X, y, config); err != nil {
		return nil, nil, nil, nil, err
	}

	if config.Stratify {
		return stratifiedSplit(X, y, config)
	}

	return randomSplit(X, y, config)
}

// TrainValTestSplit splits data into training, validation, and testing sets.
func TrainValTestSplit(X, y core.Tensor, valSize float64, options ...SplitOption) (core.Tensor, core.Tensor, core.Tensor, core.Tensor, core.Tensor, core.Tensor, error) {
	config := NewSplitConfig()
	for _, option := range options {
		option(config)
	}

	if err := core.ValidateRange(valSize, 0.0, 1.0, "validation size"); err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}

	if err := validateSplitInputs(X, y, config); err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}

	// Adjust sizes for three-way split
	totalTestVal := config.TestSize + valSize
	if totalTestVal >= 1.0 {
		return nil, nil, nil, nil, nil, nil, core.NewError(core.ErrInvalidInput,
			fmt.Sprintf("test_size + val_size must be < 1.0, got %f", totalTestVal))
	}

	// First split: separate training from test+validation
	tempTestSize := totalTestVal
	XTrain, XTemp, yTrain, yTemp, err := TrainTestSplit(X, y,
		WithTestSize(tempTestSize),
		WithShuffle(config.Shuffle),
		WithStratify(config.Stratify),
		WithRandomState(config.RandomState))
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}

	// Second split: separate validation from test
	valProportion := valSize / totalTestVal
	XVal, XTest, yVal, yTest, err := TrainTestSplit(XTemp, yTemp,
		WithTestSize(1.0-valProportion),
		WithShuffle(false), // Already shuffled
		WithStratify(config.Stratify),
		WithRandomState(config.RandomState+1))
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}

	return XTrain, XVal, XTest, yTrain, yVal, yTest, nil
}

// randomSplit performs random train-test split.
func randomSplit(X, y core.Tensor, config *SplitConfig) (core.Tensor, core.Tensor, core.Tensor, core.Tensor, error) {
	nSamples, _ := X.Dims()
	indices := make([]int, nSamples)
	for i := range indices {
		indices[i] = i
	}

	if config.Shuffle {
		rng := rand.New(rand.NewSource(config.RandomState))
		rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}

	testSamples := int(math.Round(float64(nSamples) * config.TestSize))
	trainSamples := nSamples - testSamples

	trainIndices := indices[:trainSamples]
	testIndices := indices[trainSamples:]

	XTrain, err := selectRows(X, trainIndices)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	XTest, err := selectRows(X, testIndices)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	yTrain, err := selectRows(y, trainIndices)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	yTest, err := selectRows(y, testIndices)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	return XTrain, XTest, yTrain, yTest, nil
}

// stratifiedSplit performs stratified train-test split for classification.
func stratifiedSplit(X, y core.Tensor, config *SplitConfig) (core.Tensor, core.Tensor, core.Tensor, core.Tensor, error) {
	nSamples, _ := X.Dims()
	yRows, yCols := y.Dims()

	if yCols != 1 {
		return nil, nil, nil, nil, core.NewError(core.ErrDimensionMismatch,
			fmt.Sprintf("stratified split requires y to be a column vector, got shape (%d,%d)", yRows, yCols))
	}

	// Extract labels and create class groups
	labels := make([]int, nSamples)
	for i := range nSamples {
		labels[i] = int(math.Round(y.At(i, 0)))
	}

	classGroups := make(map[int][]int)
	for i, label := range labels {
		classGroups[label] = append(classGroups[label], i)
	}

	// Validate minimum samples per class
	for class, indices := range classGroups {
		if len(indices) < 2 {
			return nil, nil, nil, nil, core.NewError(core.ErrInvalidInput,
				fmt.Sprintf("class %d has only %d sample(s), need at least 2 for stratified split", class, len(indices)))
		}
	}

	rng := rand.New(rand.NewSource(config.RandomState))
	var trainIndices, testIndices []int

	// Split each class proportionally
	for _, indices := range classGroups {
		classIndices := make([]int, len(indices))
		copy(classIndices, indices)

		if config.Shuffle {
			rng.Shuffle(len(classIndices), func(i, j int) {
				classIndices[i], classIndices[j] = classIndices[j], classIndices[i]
			})
		}

		testSamples := int(math.Round(float64(len(classIndices)) * config.TestSize))
		if testSamples == 0 && config.TestSize > 0 {
			testSamples = 1 // Ensure at least one test sample per class
		}
		if testSamples >= len(classIndices) {
			testSamples = len(classIndices) - 1 // Ensure at least one train sample per class
		}

		trainIndices = append(trainIndices, classIndices[:len(classIndices)-testSamples]...)
		testIndices = append(testIndices, classIndices[len(classIndices)-testSamples:]...)
	}

	// Shuffle the final indices if requested
	if config.Shuffle {
		rng.Shuffle(len(trainIndices), func(i, j int) {
			trainIndices[i], trainIndices[j] = trainIndices[j], trainIndices[i]
		})
		rng.Shuffle(len(testIndices), func(i, j int) {
			testIndices[i], testIndices[j] = testIndices[j], testIndices[i]
		})
	}

	XTrain, err := selectRows(X, trainIndices)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	XTest, err := selectRows(X, testIndices)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	yTrain, err := selectRows(y, trainIndices)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	yTest, err := selectRows(y, testIndices)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	return XTrain, XTest, yTrain, yTest, nil
}

// selectRows creates a new tensor with selected rows.
func selectRows(tensor core.Tensor, indices []int) (core.Tensor, error) {
	if len(indices) == 0 {
		return nil, core.NewError(core.ErrInvalidInput, "cannot select zero rows")
	}

	_, cols := tensor.Dims()
	result := core.NewZerosTensor(len(indices), cols)

	for i, idx := range indices {
		for j := range cols {
			result.Set(i, j, tensor.At(idx, j))
		}
	}

	return result, nil
}

// validateSplitInputs validates inputs for data splitting.
func validateSplitInputs(X, y core.Tensor, config *SplitConfig) error {
	if err := core.ValidateTrainingData(X, y); err != nil {
		return err
	}

	if err := core.ValidateRange(config.TestSize, 0.0, 1.0, "test_size"); err != nil {
		return err
	}

	if config.TrainSize > 0 {
		if err := core.ValidateRange(config.TrainSize, 0.0, 1.0, "train_size"); err != nil {
			return err
		}

		if config.TestSize+config.TrainSize > 1.0 {
			return core.NewError(core.ErrInvalidInput,
				fmt.Sprintf("test_size + train_size cannot exceed 1.0, got %f", config.TestSize+config.TrainSize))
		}
	}

	nSamples, _ := X.Dims()
	minSamples := 2
	if config.Stratify {
		minSamples = 4 // Need at least 2 samples per class for train and test
	}

	if nSamples < minSamples {
		return core.NewError(core.ErrInvalidInput,
			fmt.Sprintf("need at least %d samples for splitting, got %d", minSamples, nSamples))
	}

	return nil
}

// Data Validation Utilities

// DataValidator provides comprehensive data validation functionality.
type DataValidator struct {
	name string
}

// NewDataValidator creates a new data validator.
func NewDataValidator() *DataValidator {
	return &DataValidator{
		name: "DataValidator",
	}
}

// ValidationResult holds the results of data validation.
type ValidationResult struct {
	IsValid      bool                   `json:"is_valid"`
	Errors       []string               `json:"errors"`
	Warnings     []string               `json:"warnings"`
	Statistics   map[string]interface{} `json:"statistics"`
	MissingCount int                    `json:"missing_count"`
	InfCount     int                    `json:"inf_count"`
	NaNCount     int                    `json:"nan_count"`
}

// ValidateData performs comprehensive validation of input data.
func (dv *DataValidator) ValidateData(X core.Tensor, y core.Tensor) *ValidationResult {
	result := &ValidationResult{
		IsValid:    true,
		Errors:     []string{},
		Warnings:   []string{},
		Statistics: make(map[string]interface{}),
	}

	// Validate X tensor
	if X != nil {
		dv.validateTensor(X, "X", result)
		dv.computeStatistics(X, "X", result)
	} else {
		result.Errors = append(result.Errors, "X tensor is nil")
		result.IsValid = false
	}

	// Validate y tensor if provided
	if y != nil {
		dv.validateTensor(y, "y", result)
		dv.computeStatistics(y, "y", result)

		// Check X and y compatibility
		if X != nil {
			if err := core.ValidateTrainingData(X, y); err != nil {
				result.Errors = append(result.Errors, fmt.Sprintf("X and y incompatible: %v", err))
				result.IsValid = false
			}
		}
	}

	return result
}

// ValidateFeatures validates feature data specifically.
func (dv *DataValidator) ValidateFeatures(X core.Tensor) *ValidationResult {
	result := &ValidationResult{
		IsValid:    true,
		Errors:     []string{},
		Warnings:   []string{},
		Statistics: make(map[string]interface{}),
	}

	if X == nil {
		result.Errors = append(result.Errors, "feature tensor is nil")
		result.IsValid = false
		return result
	}

	dv.validateTensor(X, "features", result)
	dv.computeStatistics(X, "features", result)

	// Feature-specific validations
	rows, cols := X.Dims()

	// Check for constant features
	constantFeatures := dv.findConstantFeatures(X)
	if len(constantFeatures) > 0 {
		result.Warnings = append(result.Warnings,
			fmt.Sprintf("found %d constant features: %v", len(constantFeatures), constantFeatures))
	}

	// Check for highly correlated features
	if cols > 1 && rows > cols {
		correlatedPairs := dv.findHighlyCorrelatedFeatures(X, 0.95)
		if len(correlatedPairs) > 0 {
			result.Warnings = append(result.Warnings,
				fmt.Sprintf("found %d highly correlated feature pairs", len(correlatedPairs)))
		}
	}

	return result
}

// ValidateLabels validates label data for classification tasks.
func (dv *DataValidator) ValidateLabels(y core.Tensor) *ValidationResult {
	result := &ValidationResult{
		IsValid:    true,
		Errors:     []string{},
		Warnings:   []string{},
		Statistics: make(map[string]interface{}),
	}

	if y == nil {
		result.Errors = append(result.Errors, "label tensor is nil")
		result.IsValid = false
		return result
	}

	dv.validateTensor(y, "labels", result)

	// Label-specific validations
	_, cols := y.Dims()

	if cols != 1 {
		result.Warnings = append(result.Warnings,
			fmt.Sprintf("labels have %d columns, expected 1 for classification", cols))
	}

	// Check label distribution
	if cols == 1 {
		labelCounts := dv.computeLabelDistribution(y)
		result.Statistics["label_distribution"] = labelCounts

		// Check for class imbalance
		minCount, maxCount := dv.getMinMaxCounts(labelCounts)
		if maxCount > 0 && float64(minCount)/float64(maxCount) < 0.1 {
			result.Warnings = append(result.Warnings, "severe class imbalance detected")
		}

		// Check for single class
		if len(labelCounts) == 1 {
			result.Errors = append(result.Errors, "only one class found in labels")
			result.IsValid = false
		}
	}

	return result
}

// validateTensor performs basic tensor validation.
func (dv *DataValidator) validateTensor(tensor core.Tensor, name string, result *ValidationResult) {
	// Check dimensions
	rows, cols := tensor.Dims()
	if rows == 0 || cols == 0 {
		result.Errors = append(result.Errors, fmt.Sprintf("%s has zero dimensions: (%d, %d)", name, rows, cols))
		result.IsValid = false
		return
	}

	// Count problematic values
	nanCount := 0
	infCount := 0

	for i := range rows {
		for j := range cols {
			val := tensor.At(i, j)
			if math.IsNaN(val) {
				nanCount++
			} else if math.IsInf(val, 0) {
				infCount++
			}
		}
	}

	result.NaNCount += nanCount
	result.InfCount += infCount

	if nanCount > 0 {
		result.Errors = append(result.Errors, fmt.Sprintf("%s contains %d NaN values", name, nanCount))
		result.IsValid = false
	}

	if infCount > 0 {
		result.Errors = append(result.Errors, fmt.Sprintf("%s contains %d infinite values", name, infCount))
		result.IsValid = false
	}
}

// computeStatistics computes basic statistics for a tensor.
func (dv *DataValidator) computeStatistics(tensor core.Tensor, name string, result *ValidationResult) {
	rows, cols := tensor.Dims()

	stats := map[string]interface{}{
		"shape": []int{rows, cols},
		"size":  rows * cols,
	}

	if !tensor.HasNaN() && !tensor.HasInf() {
		stats["mean"] = tensor.Mean()
		stats["std"] = tensor.Std()
		stats["min"] = tensor.Min()
		stats["max"] = tensor.Max()
		stats["sum"] = tensor.Sum()
	}

	result.Statistics[name] = stats
}

// findConstantFeatures identifies features with constant values.
func (dv *DataValidator) findConstantFeatures(X core.Tensor) []int {
	rows, cols := X.Dims()
	var constantFeatures []int

	for j := range cols {
		if rows == 0 {
			continue
		}

		firstVal := X.At(0, j)
		isConstant := true

		for i := 1; i < rows; i++ {
			if math.Abs(X.At(i, j)-firstVal) > core.GetEpsilon() {
				isConstant = false
				break
			}
		}

		if isConstant {
			constantFeatures = append(constantFeatures, j)
		}
	}

	return constantFeatures
}

// findHighlyCorrelatedFeatures finds pairs of features with high correlation.
func (dv *DataValidator) findHighlyCorrelatedFeatures(X core.Tensor, threshold float64) [][]int {
	rows, cols := X.Dims()
	var correlatedPairs [][]int

	for i := range cols {
		for j := i + 1; j < cols; j++ {
			corr := dv.computeCorrelation(X, i, j, rows)
			if math.Abs(corr) > threshold {
				correlatedPairs = append(correlatedPairs, []int{i, j})
			}
		}
	}

	return correlatedPairs
}

// computeCorrelation computes Pearson correlation between two features.
func (dv *DataValidator) computeCorrelation(X core.Tensor, col1, col2, rows int) float64 {
	// Compute means
	mean1, mean2 := 0.0, 0.0
	for i := range rows {
		mean1 += X.At(i, col1)
		mean2 += X.At(i, col2)
	}
	mean1 /= float64(rows)
	mean2 /= float64(rows)

	// Compute correlation
	numerator := 0.0
	sum1Sq := 0.0
	sum2Sq := 0.0

	for i := range rows {
		diff1 := X.At(i, col1) - mean1
		diff2 := X.At(i, col2) - mean2
		numerator += diff1 * diff2
		sum1Sq += diff1 * diff1
		sum2Sq += diff2 * diff2
	}

	denominator := math.Sqrt(sum1Sq * sum2Sq)
	if denominator < core.GetEpsilon() {
		return 0.0
	}

	return numerator / denominator
}

// computeLabelDistribution computes the distribution of labels.
func (dv *DataValidator) computeLabelDistribution(y core.Tensor) map[int]int {
	rows, _ := y.Dims()
	distribution := make(map[int]int)

	for i := range rows {
		label := int(math.Round(y.At(i, 0)))
		distribution[label]++
	}

	return distribution
}

// getMinMaxCounts returns the minimum and maximum counts from a distribution.
func (dv *DataValidator) getMinMaxCounts(distribution map[int]int) (int, int) {
	if len(distribution) == 0 {
		return 0, 0
	}

	var counts []int
	for _, count := range distribution {
		counts = append(counts, count)
	}

	sort.Ints(counts)
	return counts[0], counts[len(counts)-1]
}

// CheckDataQuality performs a comprehensive data quality assessment.
func CheckDataQuality(X, y core.Tensor) *ValidationResult {
	validator := NewDataValidator()
	return validator.ValidateData(X, y)
}

// CheckFeatureQuality performs feature-specific quality assessment.
func CheckFeatureQuality(X core.Tensor) *ValidationResult {
	validator := NewDataValidator()
	return validator.ValidateFeatures(X)
}

// CheckLabelQuality performs label-specific quality assessment.
func CheckLabelQuality(y core.Tensor) *ValidationResult {
	validator := NewDataValidator()
	return validator.ValidateLabels(y)
}

// Name returns the validator name.
func (dv *DataValidator) Name() string {
	return dv.name
}

// Helper Functions for Easy Usage

// EasySplit performs train-test split with sensible defaults.
// This is a convenience function that splits data with commonly used settings:
// - 80/20 train/test split
// - Shuffling enabled
// - Random seed of 42 for reproducibility
func EasySplit(X, y core.Tensor, testSize float64) (core.Tensor, core.Tensor, core.Tensor, core.Tensor, error) {
	if err := core.ValidateTrainingData(X, y); err != nil {
		return nil, nil, nil, nil, err
	}

	if err := core.ValidateRange(testSize, 0.0, 1.0, "test_size"); err != nil {
		return nil, nil, nil, nil, err
	}

	return TrainTestSplit(X, y,
		WithTestSize(testSize),
		WithShuffle(true),
		WithRandomState(42))
}
