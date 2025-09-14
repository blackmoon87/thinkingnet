// Package preprocessing provides data preprocessing utilities for machine learning.
package preprocessing

import (
	"fmt"
	"math"
	"sort"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// StandardScaler standardizes features by removing the mean and scaling to unit variance.
// The standard score of a sample x is calculated as: z = (x - u) / s
// where u is the mean of the training samples and s is the standard deviation.
type StandardScaler struct {
	name     string
	fitted   bool
	mean     core.Tensor
	std      core.Tensor
	withMean bool
	withStd  bool
}

// NewStandardScaler creates a new StandardScaler.
func NewStandardScaler(options ...StandardScalerOption) *StandardScaler {
	scaler := &StandardScaler{
		name:     "StandardScaler",
		fitted:   false,
		withMean: true,
		withStd:  true,
	}

	for _, option := range options {
		option(scaler)
	}

	return scaler
}

// StandardScalerOption represents a functional option for StandardScaler.
type StandardScalerOption func(*StandardScaler)

// WithMean controls whether to center the data before scaling.
func WithMean(withMean bool) StandardScalerOption {
	return func(s *StandardScaler) { s.withMean = withMean }
}

// WithStd controls whether to scale the data to unit variance.
func WithStd(withStd bool) StandardScalerOption {
	return func(s *StandardScaler) { s.withStd = withStd }
}

// Fit learns the mean and standard deviation from the training data.
func (s *StandardScaler) Fit(data core.Tensor) error {
	if err := core.ValidateNonEmpty(data, "input data"); err != nil {
		return err
	}

	rows, cols := data.Dims()

	// Calculate mean for each feature
	if s.withMean {
		meanData := make([]float64, cols)
		for j := range cols {
			var sum float64
			for i := range rows {
				sum += data.At(i, j)
			}
			meanData[j] = sum / float64(rows)
		}
		s.mean = core.NewTensorFromData(1, cols, meanData)
	}

	// Calculate standard deviation for each feature
	if s.withStd {
		stdData := make([]float64, cols)
		for j := range cols {
			var sumSq float64
			mean := 0.0
			if s.withMean {
				mean = s.mean.At(0, j)
			}

			for i := range rows {
				diff := data.At(i, j) - mean
				sumSq += diff * diff
			}

			variance := sumSq / float64(rows)
			std := math.Sqrt(variance)

			// Prevent division by zero
			if std < core.GetEpsilon() {
				std = 1.0
			}
			stdData[j] = std
		}
		s.std = core.NewTensorFromData(1, cols, stdData)
	}

	s.fitted = true
	return nil
}

// Transform applies the standardization to the data.
func (s *StandardScaler) Transform(data core.Tensor) (core.Tensor, error) {
	if err := core.ValidateNotFitted(s.fitted, s.name); err != nil {
		return nil, err
	}

	if err := core.ValidateNonEmpty(data, "input data"); err != nil {
		return nil, err
	}

	rows, cols := data.Dims()
	result := data.Copy()

	// Center the data
	if s.withMean && s.mean != nil {
		_, meanCols := s.mean.Dims()
		if cols != meanCols {
			return nil, core.NewError(core.ErrDimensionMismatch,
				fmt.Sprintf("input has %d features, but scaler was fitted with %d features", cols, meanCols))
		}

		for i := range rows {
			for j := range cols {
				result.Set(i, j, result.At(i, j)-s.mean.At(0, j))
			}
		}
	}

	// Scale the data
	if s.withStd && s.std != nil {
		_, stdCols := s.std.Dims()
		if cols != stdCols {
			return nil, core.NewError(core.ErrDimensionMismatch,
				fmt.Sprintf("input has %d features, but scaler was fitted with %d features", cols, stdCols))
		}

		for i := range rows {
			for j := range cols {
				result.Set(i, j, result.At(i, j)/s.std.At(0, j))
			}
		}
	}

	return result, nil
}

// FitTransform fits the scaler and transforms the data in one step.
func (s *StandardScaler) FitTransform(data core.Tensor) (core.Tensor, error) {
	if err := s.Fit(data); err != nil {
		return nil, err
	}
	return s.Transform(data)
}

// InverseTransform reverses the standardization.
func (s *StandardScaler) InverseTransform(data core.Tensor) (core.Tensor, error) {
	if err := core.ValidateNotFitted(s.fitted, s.name); err != nil {
		return nil, err
	}

	if err := core.ValidateNonEmpty(data, "input data"); err != nil {
		return nil, err
	}

	rows, cols := data.Dims()
	result := data.Copy()

	// Reverse scaling
	if s.withStd && s.std != nil {
		_, stdCols := s.std.Dims()
		if cols != stdCols {
			return nil, core.NewError(core.ErrDimensionMismatch,
				fmt.Sprintf("input has %d features, but scaler was fitted with %d features", cols, stdCols))
		}

		for i := range rows {
			for j := range cols {
				result.Set(i, j, result.At(i, j)*s.std.At(0, j))
			}
		}
	}

	// Reverse centering
	if s.withMean && s.mean != nil {
		_, meanCols := s.mean.Dims()
		if cols != meanCols {
			return nil, core.NewError(core.ErrDimensionMismatch,
				fmt.Sprintf("input has %d features, but scaler was fitted with %d features", cols, meanCols))
		}

		for i := range rows {
			for j := range cols {
				result.Set(i, j, result.At(i, j)+s.mean.At(0, j))
			}
		}
	}

	return result, nil
}

// IsFitted returns true if the scaler has been fitted.
func (s *StandardScaler) IsFitted() bool {
	return s.fitted
}

// Name returns the scaler name.
func (s *StandardScaler) Name() string {
	return s.name
}

// GetMean returns the learned mean values.
func (s *StandardScaler) GetMean() core.Tensor {
	if s.mean == nil {
		return nil
	}
	return s.mean.Copy()
}

// GetStd returns the learned standard deviation values.
func (s *StandardScaler) GetStd() core.Tensor {
	if s.std == nil {
		return nil
	}
	return s.std.Copy()
}

// MinMaxScaler scales features to a given range, typically [0, 1].
// The transformation is given by: X_scaled = (X - X.min) / (X.max - X.min) * (max - min) + min
type MinMaxScaler struct {
	name       string
	fitted     bool
	dataMin    core.Tensor
	dataMax    core.Tensor
	featureMin float64
	featureMax float64
}

// NewMinMaxScaler creates a new MinMaxScaler.
func NewMinMaxScaler(options ...MinMaxScalerOption) *MinMaxScaler {
	scaler := &MinMaxScaler{
		name:       "MinMaxScaler",
		fitted:     false,
		featureMin: 0.0,
		featureMax: 1.0,
	}

	for _, option := range options {
		option(scaler)
	}

	return scaler
}

// MinMaxScalerOption represents a functional option for MinMaxScaler.
type MinMaxScalerOption func(*MinMaxScaler)

// WithFeatureRange sets the desired range for the scaled features.
func WithFeatureRange(min, max float64) MinMaxScalerOption {
	return func(s *MinMaxScaler) {
		s.featureMin = min
		s.featureMax = max
	}
}

// Fit learns the minimum and maximum values from the training data.
func (m *MinMaxScaler) Fit(data core.Tensor) error {
	if err := core.ValidateNonEmpty(data, "input data"); err != nil {
		return err
	}

	if m.featureMax <= m.featureMin {
		return core.NewError(core.ErrInvalidInput,
			fmt.Sprintf("feature_max (%f) must be greater than feature_min (%f)", m.featureMax, m.featureMin))
	}

	rows, cols := data.Dims()

	minData := make([]float64, cols)
	maxData := make([]float64, cols)

	// Find min and max for each feature
	for j := range cols {
		min := data.At(0, j)
		max := data.At(0, j)

		for i := 1; i < rows; i++ {
			val := data.At(i, j)
			if val < min {
				min = val
			}
			if val > max {
				max = val
			}
		}

		minData[j] = min
		maxData[j] = max
	}

	m.dataMin = core.NewTensorFromData(1, cols, minData)
	m.dataMax = core.NewTensorFromData(1, cols, maxData)
	m.fitted = true

	return nil
}

// Transform applies the min-max scaling to the data.
func (m *MinMaxScaler) Transform(data core.Tensor) (core.Tensor, error) {
	if err := core.ValidateNotFitted(m.fitted, m.name); err != nil {
		return nil, err
	}

	if err := core.ValidateNonEmpty(data, "input data"); err != nil {
		return nil, err
	}

	rows, cols := data.Dims()
	_, minCols := m.dataMin.Dims()

	if cols != minCols {
		return nil, core.NewError(core.ErrDimensionMismatch,
			fmt.Sprintf("input has %d features, but scaler was fitted with %d features", cols, minCols))
	}

	result := core.NewZerosTensor(rows, cols)
	scale := m.featureMax - m.featureMin

	for i := range rows {
		for j := range cols {
			dataRange := m.dataMax.At(0, j) - m.dataMin.At(0, j)

			// Handle constant features (where min == max)
			if math.Abs(dataRange) < core.GetEpsilon() {
				result.Set(i, j, m.featureMin)
			} else {
				scaled := (data.At(i, j)-m.dataMin.At(0, j))/dataRange*scale + m.featureMin
				result.Set(i, j, scaled)
			}
		}
	}

	return result, nil
}

// FitTransform fits the scaler and transforms the data in one step.
func (m *MinMaxScaler) FitTransform(data core.Tensor) (core.Tensor, error) {
	if err := m.Fit(data); err != nil {
		return nil, err
	}
	return m.Transform(data)
}

// InverseTransform reverses the min-max scaling.
func (m *MinMaxScaler) InverseTransform(data core.Tensor) (core.Tensor, error) {
	if err := core.ValidateNotFitted(m.fitted, m.name); err != nil {
		return nil, err
	}

	if err := core.ValidateNonEmpty(data, "input data"); err != nil {
		return nil, err
	}

	rows, cols := data.Dims()
	_, minCols := m.dataMin.Dims()

	if cols != minCols {
		return nil, core.NewError(core.ErrDimensionMismatch,
			fmt.Sprintf("input has %d features, but scaler was fitted with %d features", cols, minCols))
	}

	result := core.NewZerosTensor(rows, cols)
	scale := m.featureMax - m.featureMin

	for i := range rows {
		for j := range cols {
			dataRange := m.dataMax.At(0, j) - m.dataMin.At(0, j)

			// Handle constant features
			if math.Abs(dataRange) < core.GetEpsilon() {
				result.Set(i, j, m.dataMin.At(0, j))
			} else {
				original := (data.At(i, j)-m.featureMin)/scale*dataRange + m.dataMin.At(0, j)
				result.Set(i, j, original)
			}
		}
	}

	return result, nil
}

// IsFitted returns true if the scaler has been fitted.
func (m *MinMaxScaler) IsFitted() bool {
	return m.fitted
}

// Name returns the scaler name.
func (m *MinMaxScaler) Name() string {
	return m.name
}

// GetDataMin returns the learned minimum values.
func (m *MinMaxScaler) GetDataMin() core.Tensor {
	if m.dataMin == nil {
		return nil
	}
	return m.dataMin.Copy()
}

// GetDataMax returns the learned maximum values.
func (m *MinMaxScaler) GetDataMax() core.Tensor {
	if m.dataMax == nil {
		return nil
	}
	return m.dataMax.Copy()
}

// GetFeatureRange returns the configured feature range.
func (m *MinMaxScaler) GetFeatureRange() (float64, float64) {
	return m.featureMin, m.featureMax
}

// OneHotEncoder encodes categorical features as a one-hot numeric array.
// Each categorical feature with n possible values becomes n binary features.
type OneHotEncoder struct {
	name     string
	fitted   bool
	classes  []string
	classMap map[string]int
}

// NewOneHotEncoder creates a new OneHotEncoder.
func NewOneHotEncoder() *OneHotEncoder {
	return &OneHotEncoder{
		name:     "OneHotEncoder",
		fitted:   false,
		classMap: make(map[string]int),
	}
}

// Fit learns the unique categories from the categorical data.
func (o *OneHotEncoder) Fit(data []string) error {
	if len(data) == 0 {
		return core.NewError(core.ErrInvalidInput, "input data cannot be empty")
	}

	// Find unique classes
	uniqueClasses := make(map[string]bool)
	for _, class := range data {
		uniqueClasses[class] = true
	}

	// Convert to sorted slice for consistent ordering
	o.classes = make([]string, 0, len(uniqueClasses))
	for class := range uniqueClasses {
		o.classes = append(o.classes, class)
	}
	sort.Strings(o.classes)

	// Create class to index mapping
	o.classMap = make(map[string]int)
	for i, class := range o.classes {
		o.classMap[class] = i
	}

	o.fitted = true
	return nil
}

// Transform encodes categorical data to one-hot representation.
func (o *OneHotEncoder) Transform(data []string) (core.Tensor, error) {
	if err := core.ValidateNotFitted(o.fitted, o.name); err != nil {
		return nil, err
	}

	if len(data) == 0 {
		return nil, core.NewError(core.ErrInvalidInput, "input data cannot be empty")
	}

	rows := len(data)
	cols := len(o.classes)
	result := core.NewZerosTensor(rows, cols)

	for i, class := range data {
		if idx, exists := o.classMap[class]; exists {
			result.Set(i, idx, 1.0)
		} else {
			return nil, core.NewError(core.ErrInvalidInput,
				fmt.Sprintf("unknown category '%s' encountered during transform", class))
		}
	}

	return result, nil
}

// FitTransform fits the encoder and transforms the data in one step.
func (o *OneHotEncoder) FitTransform(data []string) (core.Tensor, error) {
	if err := o.Fit(data); err != nil {
		return nil, err
	}
	return o.Transform(data)
}

// Classes returns the learned classes.
func (o *OneHotEncoder) Classes() []string {
	result := make([]string, len(o.classes))
	copy(result, o.classes)
	return result
}

// IsFitted returns true if the encoder has been fitted.
func (o *OneHotEncoder) IsFitted() bool {
	return o.fitted
}

// Name returns the encoder name.
func (o *OneHotEncoder) Name() string {
	return o.name
}

// GetNumClasses returns the number of unique classes.
func (o *OneHotEncoder) GetNumClasses() int {
	return len(o.classes)
}

// LabelEncoder encodes categorical labels as integers.
// Each unique category is assigned an integer from 0 to n_classes-1.
type LabelEncoder struct {
	name     string
	fitted   bool
	classes  []string
	classMap map[string]int
}

// NewLabelEncoder creates a new LabelEncoder.
func NewLabelEncoder() *LabelEncoder {
	return &LabelEncoder{
		name:     "LabelEncoder",
		fitted:   false,
		classMap: make(map[string]int),
	}
}

// Fit learns the unique categories from the categorical data.
func (l *LabelEncoder) Fit(data []string) error {
	if len(data) == 0 {
		return core.NewError(core.ErrInvalidInput, "input data cannot be empty")
	}

	// Find unique classes
	uniqueClasses := make(map[string]bool)
	for _, class := range data {
		uniqueClasses[class] = true
	}

	// Convert to sorted slice for consistent ordering
	l.classes = make([]string, 0, len(uniqueClasses))
	for class := range uniqueClasses {
		l.classes = append(l.classes, class)
	}
	sort.Strings(l.classes)

	// Create class to index mapping
	l.classMap = make(map[string]int)
	for i, class := range l.classes {
		l.classMap[class] = i
	}

	l.fitted = true
	return nil
}

// Transform encodes categorical data to integer labels.
func (l *LabelEncoder) Transform(data []string) (core.Tensor, error) {
	if err := core.ValidateNotFitted(l.fitted, l.name); err != nil {
		return nil, err
	}

	if len(data) == 0 {
		return nil, core.NewError(core.ErrInvalidInput, "input data cannot be empty")
	}

	rows := len(data)
	result := core.NewZerosTensor(rows, 1)

	for i, class := range data {
		if idx, exists := l.classMap[class]; exists {
			result.Set(i, 0, float64(idx))
		} else {
			return nil, core.NewError(core.ErrInvalidInput,
				fmt.Sprintf("unknown category '%s' encountered during transform", class))
		}
	}

	return result, nil
}

// FitTransform fits the encoder and transforms the data in one step.
func (l *LabelEncoder) FitTransform(data []string) (core.Tensor, error) {
	if err := l.Fit(data); err != nil {
		return nil, err
	}
	return l.Transform(data)
}

// InverseTransform converts integer labels back to original categories.
func (l *LabelEncoder) InverseTransform(data core.Tensor) ([]string, error) {
	if err := core.ValidateNotFitted(l.fitted, l.name); err != nil {
		return nil, err
	}

	if err := core.ValidateNonEmpty(data, "input data"); err != nil {
		return nil, err
	}

	rows, cols := data.Dims()
	if cols != 1 {
		return nil, core.NewError(core.ErrDimensionMismatch,
			fmt.Sprintf("input must be a column vector, got shape (%d, %d)", rows, cols))
	}

	result := make([]string, rows)
	numClasses := len(l.classes)

	for i := range rows {
		label := int(math.Round(data.At(i, 0)))
		if label < 0 || label >= numClasses {
			return nil, core.NewError(core.ErrInvalidInput,
				fmt.Sprintf("invalid label %d at position %d, must be in range [0, %d)", label, i, numClasses))
		}
		result[i] = l.classes[label]
	}

	return result, nil
}

// Classes returns the learned classes.
func (l *LabelEncoder) Classes() []string {
	result := make([]string, len(l.classes))
	copy(result, l.classes)
	return result
}

// IsFitted returns true if the encoder has been fitted.
func (l *LabelEncoder) IsFitted() bool {
	return l.fitted
}

// Name returns the encoder name.
func (l *LabelEncoder) Name() string {
	return l.name
}

// GetNumClasses returns the number of unique classes.
func (l *LabelEncoder) GetNumClasses() int {
	return len(l.classes)
}

// Helper Functions for Easy Usage

// EasyStandardScale applies standard scaling to data with default settings.
// This is a convenience function that creates a StandardScaler, fits it to the data,
// and transforms the data in one step with sensible defaults.
func EasyStandardScale(X core.Tensor) (core.Tensor, error) {
	if err := core.ValidateNonEmpty(X, "input data"); err != nil {
		return nil, err
	}

	scaler := NewStandardScaler()
	return scaler.FitTransform(X)
}

// EasyMinMaxScale applies min-max scaling to data with default settings (0-1 range).
// This is a convenience function that creates a MinMaxScaler, fits it to the data,
// and transforms the data in one step with sensible defaults.
func EasyMinMaxScale(X core.Tensor) (core.Tensor, error) {
	if err := core.ValidateNonEmpty(X, "input data"); err != nil {
		return nil, err
	}

	scaler := NewMinMaxScaler()
	return scaler.FitTransform(X)
}
