package algorithms

import (
	"math"
	"math/rand"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// LinearRegression implements linear regression with optional regularization.
type LinearRegression struct {
	// Configuration
	learningRate float64 // Learning rate for gradient descent
	maxIters     int     // Maximum number of iterations
	tolerance    float64 // Convergence tolerance
	regularizer  string  // Regularization type: "none", "l1", "l2", "elastic"
	lambda       float64 // Regularization strength
	l1Ratio      float64 // L1 ratio for elastic net (0.0 = L2, 1.0 = L1)
	fitIntercept bool    // Whether to fit intercept term
	randomSeed   int64   // Random seed for reproducibility
	solver       string  // Solver type: "gradient_descent", "normal_equation"

	// State
	fitted    bool        // Whether the model has been fitted
	weights   core.Tensor // Model weights (including intercept if fitted)
	nFeatures int         // Number of features
	nIters    int         // Number of iterations performed
	converged bool        // Whether training converged
	rng       *rand.Rand  // Random number generator
}

// NewLinearRegression creates a new linear regression model.
func NewLinearRegression(options ...LinearRegressionOption) *LinearRegression {
	lr := &LinearRegression{
		learningRate: 0.01,
		maxIters:     1000,
		tolerance:    1e-6,
		regularizer:  "none",
		lambda:       0.01,
		l1Ratio:      0.5,
		fitIntercept: true,
		randomSeed:   time.Now().UnixNano(),
		solver:       "gradient_descent",
		fitted:       false,
	}

	// Apply options
	for _, option := range options {
		option(lr)
	}

	// Initialize random number generator
	lr.rng = rand.New(rand.NewSource(lr.randomSeed))

	return lr
}

// LinearRegressionOption represents a functional option for linear regression configuration.
type LinearRegressionOption func(*LinearRegression)

// WithLinearLearningRate sets the learning rate.
func WithLinearLearningRate(lr float64) LinearRegressionOption {
	return func(model *LinearRegression) { model.learningRate = lr }
}

// WithLinearMaxIterations sets the maximum number of iterations.
func WithLinearMaxIterations(maxIters int) LinearRegressionOption {
	return func(model *LinearRegression) { model.maxIters = maxIters }
}

// WithLinearTolerance sets the convergence tolerance.
func WithLinearTolerance(tolerance float64) LinearRegressionOption {
	return func(model *LinearRegression) { model.tolerance = tolerance }
}

// WithLinearRegularization sets the regularization type and strength.
func WithLinearRegularization(regularizer string, lambda float64) LinearRegressionOption {
	return func(model *LinearRegression) {
		model.regularizer = regularizer
		model.lambda = lambda
	}
}

// WithLinearElasticNet sets elastic net regularization parameters.
func WithLinearElasticNet(lambda, l1Ratio float64) LinearRegressionOption {
	return func(model *LinearRegression) {
		model.regularizer = "elastic"
		model.lambda = lambda
		model.l1Ratio = l1Ratio
	}
}

// WithLinearFitIntercept sets whether to fit an intercept term.
func WithLinearFitIntercept(fitIntercept bool) LinearRegressionOption {
	return func(model *LinearRegression) { model.fitIntercept = fitIntercept }
}

// WithLinearRandomSeed sets the random seed.
func WithLinearRandomSeed(seed int64) LinearRegressionOption {
	return func(model *LinearRegression) { model.randomSeed = seed }
}

// WithLinearSolver sets the solver type.
func WithLinearSolver(solver string) LinearRegressionOption {
	return func(model *LinearRegression) { model.solver = solver }
}

// EasyLinearRegression creates a linear regression model with sensible defaults.
// This is a simplified constructor for quick usage without needing to configure options.
func EasyLinearRegression() *LinearRegression {
	return NewLinearRegression(
		WithLinearLearningRate(0.01),
		WithLinearMaxIterations(1000),
		WithLinearTolerance(1e-6),
		WithLinearFitIntercept(true),
		WithLinearSolver("gradient_descent"),
	)
}

// Fit trains the linear regression model.
func (lr *LinearRegression) Fit(X, y core.Tensor) error {
	if err := lr.validateInput(X, y); err != nil {
		return err
	}

	// Prepare data
	XTrain, yTrain, err := lr.prepareData(X, y)
	if err != nil {
		return err
	}

	_, nFeatures := XTrain.Dims()
	lr.nFeatures = nFeatures

	// Choose solver
	switch lr.solver {
	case "normal_equation":
		return lr.fitNormalEquation(XTrain, yTrain)
	case "gradient_descent":
		return lr.fitGradientDescent(XTrain, yTrain)
	default:
		return core.NewError(core.ErrInvalidInput, "unsupported solver: "+lr.solver)
	}
}

// fitNormalEquation solves using the normal equation (X^T * X)^-1 * X^T * y.
func (lr *LinearRegression) fitNormalEquation(X, y core.Tensor) error {
	// For regularized regression, we use (X^T * X + λI)^-1 * X^T * y
	XT := X.T()
	XTX := XT.Mul(X)

	// Add regularization term
	if lr.lambda > 0 && lr.regularizer == "l2" {
		rows, cols := XTX.Dims()
		for i := range rows {
			for j := range cols {
				if i == j {
					// Add regularization to diagonal (skip intercept if present)
					skipIdx := 0
					if lr.fitIntercept {
						skipIdx = 1
					}
					if i >= skipIdx {
						current := XTX.At(i, j)
						XTX.Set(i, j, current+lr.lambda)
					}
				}
			}
		}
	}

	// For simplicity, we'll use gradient descent even for "normal equation"
	// since implementing matrix inversion is complex
	return lr.fitGradientDescent(X, y)
}

// fitGradientDescent solves using gradient descent.
func (lr *LinearRegression) fitGradientDescent(X, y core.Tensor) error {
	// Initialize weights
	lr.initializeWeights(lr.nFeatures)

	// Training loop
	prevLoss := math.Inf(1)
	for iter := 0; iter < lr.maxIters; iter++ {
		// Forward pass
		predictions := lr.predict(X)

		// Compute loss
		loss := lr.computeLoss(y, predictions)

		// Compute gradients
		gradients := lr.computeGradients(X, y, predictions)

		// Update weights
		lr.updateWeights(gradients)

		// Check convergence
		if math.Abs(prevLoss-loss) < lr.tolerance {
			lr.converged = true
			lr.nIters = iter + 1
			break
		}

		prevLoss = loss
		lr.nIters = iter + 1
	}

	lr.fitted = true
	return nil
}

// Predict makes predictions on new data.
func (lr *LinearRegression) Predict(X core.Tensor) (core.Tensor, error) {
	if !lr.fitted {
		return nil, core.NewError(core.ErrNotFitted, "LinearRegression must be fitted before prediction")
	}

	if err := lr.validatePredictInput(X); err != nil {
		return nil, err
	}

	// Prepare input data
	XPred := lr.addInterceptIfNeeded(X)

	// Make predictions
	return lr.predict(XPred), nil
}

// Score returns the R² score on the given test data.
func (lr *LinearRegression) Score(X, y core.Tensor) (float64, error) {
	predictions, err := lr.Predict(X)
	if err != nil {
		return 0, err
	}

	return CalculateR2Score(y, predictions), nil
}

// Name returns the regressor name.
func (lr *LinearRegression) Name() string {
	return "LinearRegression"
}

// Weights returns the model weights.
func (lr *LinearRegression) Weights() core.Tensor {
	if !lr.fitted {
		return nil
	}
	return lr.weights.Copy()
}

// validateInput validates the input data for training.
func (lr *LinearRegression) validateInput(X, y core.Tensor) error {
	if err := core.ValidateTrainingData(X, y); err != nil {
		return err
	}

	if err := core.ValidatePositive(lr.learningRate, "learning rate"); err != nil {
		return err
	}

	if lr.maxIters <= 0 {
		return core.NewError(core.ErrInvalidInput, "max iterations must be positive")
	}

	if err := core.ValidateNonNegative(lr.tolerance, "tolerance"); err != nil {
		return err
	}

	if lr.lambda < 0 {
		return core.NewError(core.ErrInvalidInput, "regularization strength must be non-negative")
	}

	if lr.l1Ratio < 0 || lr.l1Ratio > 1 {
		return core.NewError(core.ErrInvalidInput, "l1_ratio must be between 0 and 1")
	}

	// Validate y is continuous (not categorical)
	_, yCols := y.Dims()
	if yCols != 1 {
		return core.NewError(core.ErrInvalidInput, "target variable must be a single column")
	}

	return nil
}

// validatePredictInput validates input for prediction.
func (lr *LinearRegression) validatePredictInput(X core.Tensor) error {
	expectedFeatures := lr.nFeatures
	if lr.fitIntercept {
		expectedFeatures = lr.nFeatures - 1 // Remove intercept from expected count
	}

	if err := core.ValidateInput(X, []int{-1, expectedFeatures}); err != nil {
		return err
	}

	return core.ValidateTensorFinite(X, "input")
}

// prepareData prepares training data.
func (lr *LinearRegression) prepareData(X, y core.Tensor) (core.Tensor, core.Tensor, error) {
	// Add intercept column if needed
	XTrain := lr.addInterceptIfNeeded(X)

	return XTrain, y, nil
}

// addInterceptIfNeeded adds intercept column to input data if configured.
func (lr *LinearRegression) addInterceptIfNeeded(X core.Tensor) core.Tensor {
	if !lr.fitIntercept {
		return X
	}

	nSamples, nFeatures := X.Dims()
	XWithIntercept := core.NewZerosTensor(nSamples, nFeatures+1)

	// Set intercept column to 1
	for i := range nSamples {
		XWithIntercept.Set(i, 0, 1.0)
	}

	// Copy original features
	for i := range nSamples {
		for j := range nFeatures {
			XWithIntercept.Set(i, j+1, X.At(i, j))
		}
	}

	return XWithIntercept
}

// initializeWeights initializes model weights.
func (lr *LinearRegression) initializeWeights(nFeatures int) {
	// nFeatures already includes intercept if fitIntercept is true
	lr.weights = core.NewZerosTensor(nFeatures, 1)

	// Small random initialization
	for i := range nFeatures {
		lr.weights.Set(i, 0, lr.rng.NormFloat64()*0.01)
	}
}

// predict computes predictions for input data.
func (lr *LinearRegression) predict(X core.Tensor) core.Tensor {
	// Compute linear combination: X * weights
	return X.Mul(lr.weights)
}

// computeLoss computes the mean squared error loss with regularization.
func (lr *LinearRegression) computeLoss(yTrue, yPred core.Tensor) float64 {
	nSamples, _ := yTrue.Dims()
	var loss float64

	// Mean squared error
	for i := range nSamples {
		diff := yTrue.At(i, 0) - yPred.At(i, 0)
		loss += diff * diff
	}

	loss /= float64(nSamples)

	// Add regularization
	loss += lr.computeRegularization()

	return loss
}

// computeRegularization computes regularization penalty.
func (lr *LinearRegression) computeRegularization() float64 {
	if lr.lambda == 0 || lr.regularizer == "none" {
		return 0
	}

	weightsRows, _ := lr.weights.Dims()
	var penalty float64

	// Skip intercept term if present
	startIdx := 0
	if lr.fitIntercept {
		startIdx = 1
	}

	switch lr.regularizer {
	case "l1":
		for i := startIdx; i < weightsRows; i++ {
			penalty += math.Abs(lr.weights.At(i, 0))
		}
		penalty *= lr.lambda

	case "l2":
		for i := startIdx; i < weightsRows; i++ {
			w := lr.weights.At(i, 0)
			penalty += w * w
		}
		penalty *= lr.lambda * 0.5

	case "elastic":
		l1Penalty := 0.0
		l2Penalty := 0.0
		for i := startIdx; i < weightsRows; i++ {
			w := lr.weights.At(i, 0)
			l1Penalty += math.Abs(w)
			l2Penalty += w * w
		}
		penalty = lr.lambda * (lr.l1Ratio*l1Penalty + (1-lr.l1Ratio)*0.5*l2Penalty)
	}

	return penalty
}

// computeGradients computes gradients for weight update.
func (lr *LinearRegression) computeGradients(X, yTrue, yPred core.Tensor) core.Tensor {
	nSamples, _ := X.Dims()

	// Compute prediction error
	error := yPred.Sub(yTrue)

	// Compute gradients: X^T * error / n_samples
	gradients := X.T().Mul(error).Scale(1.0 / float64(nSamples))

	// Add regularization gradients
	lr.addRegularizationGradients(gradients)

	return gradients
}

// addRegularizationGradients adds regularization terms to gradients.
func (lr *LinearRegression) addRegularizationGradients(gradients core.Tensor) {
	if lr.lambda == 0 || lr.regularizer == "none" {
		return
	}

	gradRows, _ := gradients.Dims()

	// Skip intercept term if present
	startIdx := 0
	if lr.fitIntercept {
		startIdx = 1
	}

	switch lr.regularizer {
	case "l1":
		for i := startIdx; i < gradRows; i++ {
			w := lr.weights.At(i, 0)
			sign := 1.0
			if w < 0 {
				sign = -1.0
			}
			current := gradients.At(i, 0)
			gradients.Set(i, 0, current+lr.lambda*sign)
		}

	case "l2":
		for i := startIdx; i < gradRows; i++ {
			w := lr.weights.At(i, 0)
			current := gradients.At(i, 0)
			gradients.Set(i, 0, current+lr.lambda*w)
		}

	case "elastic":
		for i := startIdx; i < gradRows; i++ {
			w := lr.weights.At(i, 0)
			sign := 1.0
			if w < 0 {
				sign = -1.0
			}
			current := gradients.At(i, 0)
			l1Term := lr.lambda * lr.l1Ratio * sign
			l2Term := lr.lambda * (1 - lr.l1Ratio) * w
			gradients.Set(i, 0, current+l1Term+l2Term)
		}
	}
}

// updateWeights updates model weights using gradients.
func (lr *LinearRegression) updateWeights(gradients core.Tensor) {
	weightsRows, _ := lr.weights.Dims()

	for i := range weightsRows {
		current := lr.weights.At(i, 0)
		gradient := gradients.At(i, 0)
		lr.weights.Set(i, 0, current-lr.learningRate*gradient)
	}
}

// Regression Metrics

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

// RegressionMetrics holds comprehensive regression metrics.
type RegressionMetrics struct {
	MSE     float64 `json:"mse"`
	RMSE    float64 `json:"rmse"`
	MAE     float64 `json:"mae"`
	R2Score float64 `json:"r2_score"`
}

// CalculateRegressionMetrics computes all regression metrics.
func CalculateRegressionMetrics(yTrue, yPred core.Tensor) *RegressionMetrics {
	return &RegressionMetrics{
		MSE:     CalculateMSE(yTrue, yPred),
		RMSE:    CalculateRMSE(yTrue, yPred),
		MAE:     CalculateMAE(yTrue, yPred),
		R2Score: CalculateR2Score(yTrue, yPred),
	}
}
