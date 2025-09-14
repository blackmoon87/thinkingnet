package metrics

import (
	"math"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// RegressionMetrics holds comprehensive regression evaluation metrics.
type RegressionMetrics struct {
	MSE     float64 `json:"mse"`      // Mean Squared Error
	RMSE    float64 `json:"rmse"`     // Root Mean Squared Error
	MAE     float64 `json:"mae"`      // Mean Absolute Error
	R2Score float64 `json:"r2_score"` // R² (coefficient of determination)
	MAPE    float64 `json:"mape"`     // Mean Absolute Percentage Error
	EVS     float64 `json:"evs"`      // Explained Variance Score
}

// CalculateMSE computes the Mean Squared Error.
func CalculateMSE(yTrue, yPred core.Tensor) float64 {
	nSamples, _ := yTrue.Dims()
	if nSamples == 0 {
		return 0.0
	}

	var mse float64
	for i := range nSamples {
		diff := yTrue.At(i, 0) - yPred.At(i, 0)
		mse += diff * diff
	}

	return mse / float64(nSamples)
}

// CalculateRMSE computes the Root Mean Squared Error.
func CalculateRMSE(yTrue, yPred core.Tensor) float64 {
	return math.Sqrt(CalculateMSE(yTrue, yPred))
}

// CalculateMAE computes the Mean Absolute Error.
func CalculateMAE(yTrue, yPred core.Tensor) float64 {
	nSamples, _ := yTrue.Dims()
	if nSamples == 0 {
		return 0.0
	}

	var mae float64
	for i := range nSamples {
		diff := math.Abs(yTrue.At(i, 0) - yPred.At(i, 0))
		mae += diff
	}

	return mae / float64(nSamples)
}

// CalculateR2Score computes the R² (coefficient of determination) score.
func CalculateR2Score(yTrue, yPred core.Tensor) float64 {
	nSamples, _ := yTrue.Dims()
	if nSamples == 0 {
		return 0.0
	}

	// Calculate mean of true values
	var yMean float64
	for i := range nSamples {
		yMean += yTrue.At(i, 0)
	}
	yMean /= float64(nSamples)

	// Calculate sum of squares
	var ssRes, ssTot float64
	for i := range nSamples {
		yTrueVal := yTrue.At(i, 0)
		yPredVal := yPred.At(i, 0)

		// Residual sum of squares
		ssRes += (yTrueVal - yPredVal) * (yTrueVal - yPredVal)

		// Total sum of squares
		ssTot += (yTrueVal - yMean) * (yTrueVal - yMean)
	}

	// Avoid division by zero
	if ssTot == 0 {
		return 1.0 // Perfect prediction when all y values are the same
	}

	return 1.0 - (ssRes / ssTot)
}

// CalculateMAPE computes the Mean Absolute Percentage Error.
func CalculateMAPE(yTrue, yPred core.Tensor) float64 {
	nSamples, _ := yTrue.Dims()
	if nSamples == 0 {
		return 0.0
	}

	var mape float64
	validSamples := 0

	for i := range nSamples {
		yTrueVal := yTrue.At(i, 0)
		yPredVal := yPred.At(i, 0)

		// Skip samples where true value is zero to avoid division by zero
		if math.Abs(yTrueVal) > core.GetEpsilon() {
			percentError := math.Abs((yTrueVal - yPredVal) / yTrueVal)
			mape += percentError
			validSamples++
		}
	}

	if validSamples == 0 {
		return 0.0
	}

	return (mape / float64(validSamples)) * 100.0
}

// CalculateExplainedVarianceScore computes the Explained Variance Score.
func CalculateExplainedVarianceScore(yTrue, yPred core.Tensor) float64 {
	nSamples, _ := yTrue.Dims()
	if nSamples == 0 {
		return 0.0
	}

	// Calculate means
	var yTrueMean, yPredMean float64
	for i := range nSamples {
		yTrueMean += yTrue.At(i, 0)
		yPredMean += yPred.At(i, 0)
	}
	yTrueMean /= float64(nSamples)
	yPredMean /= float64(nSamples)

	// Calculate variances and covariance
	var varTrue, varResidual float64
	for i := range nSamples {
		yTrueVal := yTrue.At(i, 0)
		yPredVal := yPred.At(i, 0)

		varTrue += (yTrueVal - yTrueMean) * (yTrueVal - yTrueMean)
		varResidual += (yTrueVal - yPredVal) * (yTrueVal - yPredVal)
	}

	if varTrue == 0 {
		return 1.0 // Perfect prediction when all y values are the same
	}

	return 1.0 - (varResidual / varTrue)
}

// CalculateRegressionMetrics computes all regression metrics.
func CalculateRegressionMetrics(yTrue, yPred core.Tensor) *RegressionMetrics {
	return &RegressionMetrics{
		MSE:     CalculateMSE(yTrue, yPred),
		RMSE:    CalculateRMSE(yTrue, yPred),
		MAE:     CalculateMAE(yTrue, yPred),
		R2Score: CalculateR2Score(yTrue, yPred),
		MAPE:    CalculateMAPE(yTrue, yPred),
		EVS:     CalculateExplainedVarianceScore(yTrue, yPred),
	}
}

// ResidualAnalysis provides detailed residual analysis for regression models.
type ResidualAnalysis struct {
	Residuals    core.Tensor `json:"-"` // Raw residuals (y_true - y_pred)
	MeanResidual float64     `json:"mean_residual"`
	StdResidual  float64     `json:"std_residual"`
	MinResidual  float64     `json:"min_residual"`
	MaxResidual  float64     `json:"max_residual"`
	Percentiles  []float64   `json:"percentiles"` // 25th, 50th, 75th percentiles
}

// CalculateResidualAnalysis performs comprehensive residual analysis.
func CalculateResidualAnalysis(yTrue, yPred core.Tensor) *ResidualAnalysis {
	nSamples, _ := yTrue.Dims()
	if nSamples == 0 {
		return &ResidualAnalysis{}
	}

	// Calculate residuals
	residuals := core.NewZerosTensor(nSamples, 1)
	for i := range nSamples {
		residual := yTrue.At(i, 0) - yPred.At(i, 0)
		residuals.Set(i, 0, residual)
	}

	// Calculate statistics
	meanResidual := residuals.Mean()
	stdResidual := residuals.Std()
	minResidual := residuals.Min()
	maxResidual := residuals.Max()

	// Calculate percentiles
	residualSlice := make([]float64, nSamples)
	for i := range nSamples {
		residualSlice[i] = residuals.At(i, 0)
	}

	percentiles := calculatePercentiles(residualSlice, []float64{25, 50, 75})

	return &ResidualAnalysis{
		Residuals:    residuals,
		MeanResidual: meanResidual,
		StdResidual:  stdResidual,
		MinResidual:  minResidual,
		MaxResidual:  maxResidual,
		Percentiles:  percentiles,
	}
}

// calculatePercentiles computes specified percentiles from a slice of values.
func calculatePercentiles(values []float64, percentiles []float64) []float64 {
	if len(values) == 0 {
		return make([]float64, len(percentiles))
	}

	// Sort values
	sorted := make([]float64, len(values))
	copy(sorted, values)

	// Simple insertion sort for small arrays, or use sort package for larger ones
	if len(sorted) < 50 {
		for i := 1; i < len(sorted); i++ {
			key := sorted[i]
			j := i - 1
			for j >= 0 && sorted[j] > key {
				sorted[j+1] = sorted[j]
				j--
			}
			sorted[j+1] = key
		}
	} else {
		// For larger arrays, we'd typically use sort.Float64s(sorted)
		// but to avoid additional imports, we'll use the simple sort above
		for i := 1; i < len(sorted); i++ {
			key := sorted[i]
			j := i - 1
			for j >= 0 && sorted[j] > key {
				sorted[j+1] = sorted[j]
				j--
			}
			sorted[j+1] = key
		}
	}

	result := make([]float64, len(percentiles))
	n := len(sorted)

	for i, p := range percentiles {
		if p <= 0 {
			result[i] = sorted[0]
		} else if p >= 100 {
			result[i] = sorted[n-1]
		} else {
			// Linear interpolation
			index := (p / 100.0) * float64(n-1)
			lower := int(index)
			upper := lower + 1

			if upper >= n {
				result[i] = sorted[n-1]
			} else {
				weight := index - float64(lower)
				result[i] = sorted[lower]*(1-weight) + sorted[upper]*weight
			}
		}
	}

	return result
}

// CrossValidationMetrics holds metrics from cross-validation.
type CrossValidationMetrics struct {
	MeanScore  float64   `json:"mean_score"`
	StdScore   float64   `json:"std_score"`
	Scores     []float64 `json:"scores"`
	FoldCount  int       `json:"fold_count"`
	MetricName string    `json:"metric_name"`
}

// CalculateCrossValidationMetrics computes cross-validation statistics.
func CalculateCrossValidationMetrics(scores []float64, metricName string) *CrossValidationMetrics {
	if len(scores) == 0 {
		return &CrossValidationMetrics{
			MetricName: metricName,
		}
	}

	// Calculate mean
	var sum float64
	for _, score := range scores {
		sum += score
	}
	mean := sum / float64(len(scores))

	// Calculate standard deviation
	var sumSq float64
	for _, score := range scores {
		diff := score - mean
		sumSq += diff * diff
	}
	std := math.Sqrt(sumSq / float64(len(scores)))

	return &CrossValidationMetrics{
		MeanScore:  mean,
		StdScore:   std,
		Scores:     scores,
		FoldCount:  len(scores),
		MetricName: metricName,
	}
}

// RegressionReport provides a comprehensive regression evaluation report.
type RegressionReport struct {
	Metrics          *RegressionMetrics `json:"metrics"`
	ResidualAnalysis *ResidualAnalysis  `json:"residual_analysis"`
	NumSamples       int                `json:"num_samples"`
	NumFeatures      int                `json:"num_features"`
}

// GenerateRegressionReport creates a comprehensive regression evaluation report.
func GenerateRegressionReport(yTrue, yPred core.Tensor) *RegressionReport {
	nSamples, _ := yTrue.Dims()

	return &RegressionReport{
		Metrics:          CalculateRegressionMetrics(yTrue, yPred),
		ResidualAnalysis: CalculateResidualAnalysis(yTrue, yPred),
		NumSamples:       nSamples,
		NumFeatures:      1, // Single target variable
	}
}
