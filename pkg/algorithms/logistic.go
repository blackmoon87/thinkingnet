package algorithms

import (
	"math"
	"math/rand"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// LogisticRegression implements logistic regression for binary and multiclass classification.
type LogisticRegression struct {
	// Configuration
	learningRate float64 // Learning rate for gradient descent
	maxIters     int     // Maximum number of iterations
	tolerance    float64 // Convergence tolerance
	regularizer  string  // Regularization type: "none", "l1", "l2", "elastic"
	lambda       float64 // Regularization strength
	l1Ratio      float64 // L1 ratio for elastic net (0.0 = L2, 1.0 = L1)
	fitIntercept bool    // Whether to fit intercept term
	randomSeed   int64   // Random seed for reproducibility

	// State
	fitted    bool        // Whether the model has been fitted
	weights   core.Tensor // Model weights (including intercept if fitted)
	classes   []int       // Unique class labels
	nClasses  int         // Number of classes
	nFeatures int         // Number of features
	nIters    int         // Number of iterations performed
	converged bool        // Whether training converged
	rng       *rand.Rand  // Random number generator
}

// NewLogisticRegression creates a new logistic regression classifier.
func NewLogisticRegression(options ...LogisticRegressionOption) *LogisticRegression {
	lr := &LogisticRegression{
		learningRate: 0.01,
		maxIters:     1000,
		tolerance:    1e-6,
		regularizer:  "none",
		lambda:       0.01,
		l1Ratio:      0.5,
		fitIntercept: true,
		randomSeed:   time.Now().UnixNano(),
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

// LogisticRegressionOption represents a functional option for logistic regression configuration.
type LogisticRegressionOption func(*LogisticRegression)

// WithLearningRate sets the learning rate.
func WithLearningRate(lr float64) LogisticRegressionOption {
	return func(model *LogisticRegression) { model.learningRate = lr }
}

// WithLRMaxIters sets the maximum number of iterations.
func WithLRMaxIters(maxIters int) LogisticRegressionOption {
	return func(model *LogisticRegression) { model.maxIters = maxIters }
}

// WithLRTolerance sets the convergence tolerance.
func WithLRTolerance(tolerance float64) LogisticRegressionOption {
	return func(model *LogisticRegression) { model.tolerance = tolerance }
}

// WithRegularization sets the regularization type and strength.
func WithRegularization(regularizer string, lambda float64) LogisticRegressionOption {
	return func(model *LogisticRegression) {
		model.regularizer = regularizer
		model.lambda = lambda
	}
}

// WithElasticNet sets elastic net regularization parameters.
func WithElasticNet(lambda, l1Ratio float64) LogisticRegressionOption {
	return func(model *LogisticRegression) {
		model.regularizer = "elastic"
		model.lambda = lambda
		model.l1Ratio = l1Ratio
	}
}

// WithFitIntercept sets whether to fit an intercept term.
func WithFitIntercept(fitIntercept bool) LogisticRegressionOption {
	return func(model *LogisticRegression) { model.fitIntercept = fitIntercept }
}

// WithLRRandomSeed sets the random seed.
func WithLRRandomSeed(seed int64) LogisticRegressionOption {
	return func(model *LogisticRegression) { model.randomSeed = seed }
}

// EasyLogisticRegression creates a logistic regression model with sensible defaults.
// This is a simplified constructor for quick usage without needing to configure options.
func EasyLogisticRegression() *LogisticRegression {
	return NewLogisticRegression(
		WithLearningRate(0.01),
		WithLRMaxIters(1000),
		WithLRTolerance(1e-6),
		WithFitIntercept(true),
	)
}

// Fit trains the logistic regression model.
func (lr *LogisticRegression) Fit(X, y core.Tensor) error {
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

	// Initialize weights
	lr.initializeWeights(nFeatures)

	// Training loop
	prevLoss := math.Inf(1)
	for iter := 0; iter < lr.maxIters; iter++ {
		// Forward pass
		predictions := lr.predict(XTrain)

		// Compute loss
		loss := lr.computeLoss(yTrain, predictions)

		// Compute gradients
		gradients := lr.computeGradients(XTrain, yTrain, predictions)

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

// Predict makes class predictions.
func (lr *LogisticRegression) Predict(X core.Tensor) (core.Tensor, error) {
	if !lr.fitted {
		return nil, core.NewError(core.ErrNotFitted, "LogisticRegression must be fitted before prediction")
	}

	if err := lr.validatePredictInput(X); err != nil {
		return nil, err
	}

	// Prepare input data
	XPred := lr.addInterceptIfNeeded(X)

	// Get probabilities
	probabilities := lr.predict(XPred)

	// Convert to class predictions
	nSamples, _ := probabilities.Dims()
	predictions := core.NewZerosTensor(nSamples, 1)

	if lr.nClasses == 2 {
		// Binary classification
		for i := 0; i < nSamples; i++ {
			if probabilities.At(i, 0) >= 0.5 {
				predictions.Set(i, 0, float64(lr.classes[1]))
			} else {
				predictions.Set(i, 0, float64(lr.classes[0]))
			}
		}
	} else {
		// Multiclass classification
		for i := 0; i < nSamples; i++ {
			maxProb := -1.0
			maxClass := 0
			for j := 0; j < lr.nClasses; j++ {
				if probabilities.At(i, j) > maxProb {
					maxProb = probabilities.At(i, j)
					maxClass = j
				}
			}
			predictions.Set(i, 0, float64(lr.classes[maxClass]))
		}
	}

	return predictions, nil
}

// PredictProba predicts class probabilities.
func (lr *LogisticRegression) PredictProba(X core.Tensor) (core.Tensor, error) {
	if !lr.fitted {
		return nil, core.NewError(core.ErrNotFitted, "LogisticRegression must be fitted before prediction")
	}

	if err := lr.validatePredictInput(X); err != nil {
		return nil, err
	}

	// Prepare input data
	XPred := lr.addInterceptIfNeeded(X)

	// Get probabilities
	return lr.predict(XPred), nil
}

// Score returns the accuracy score on the given test data.
func (lr *LogisticRegression) Score(X, y core.Tensor) (float64, error) {
	predictions, err := lr.Predict(X)
	if err != nil {
		return 0, err
	}

	return CalculateAccuracy(y, predictions), nil
}

// Name returns the classifier name.
func (lr *LogisticRegression) Name() string {
	return "LogisticRegression"
}

// Classes returns the unique class labels.
func (lr *LogisticRegression) Classes() []int {
	if !lr.fitted {
		return nil
	}
	result := make([]int, len(lr.classes))
	copy(result, lr.classes)
	return result
}

// Weights returns the model weights.
func (lr *LogisticRegression) Weights() core.Tensor {
	if !lr.fitted {
		return nil
	}
	return lr.weights.Copy()
}

// validateInput validates the input data for training.
func (lr *LogisticRegression) validateInput(X, y core.Tensor) error {
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

	return nil
}

// validatePredictInput validates input for prediction.
func (lr *LogisticRegression) validatePredictInput(X core.Tensor) error {
	expectedFeatures := lr.nFeatures
	if lr.fitIntercept {
		expectedFeatures = lr.nFeatures - 1 // Remove intercept from expected count
	}

	if err := core.ValidateInput(X, []int{-1, expectedFeatures}); err != nil {
		return err
	}

	return core.ValidateTensorFinite(X, "input")
}

// prepareData prepares training data and extracts class information.
func (lr *LogisticRegression) prepareData(X, y core.Tensor) (core.Tensor, core.Tensor, error) {
	// Extract unique classes
	lr.classes = lr.extractClasses(y)
	lr.nClasses = len(lr.classes)

	if lr.nClasses < 2 {
		return nil, nil, core.NewError(core.ErrInvalidInput, "need at least 2 classes for classification")
	}

	// Add intercept column if needed
	XTrain := lr.addInterceptIfNeeded(X)

	// Convert labels to appropriate format
	yTrain := lr.encodeLabels(y)

	return XTrain, yTrain, nil
}

// extractClasses extracts unique class labels from target vector.
func (lr *LogisticRegression) extractClasses(y core.Tensor) []int {
	nSamples, _ := y.Dims()
	classSet := make(map[int]bool)

	for i := 0; i < nSamples; i++ {
		class := int(y.At(i, 0))
		classSet[class] = true
	}

	classes := make([]int, 0, len(classSet))
	for class := range classSet {
		classes = append(classes, class)
	}

	// Sort classes for consistency
	for i := 0; i < len(classes)-1; i++ {
		for j := i + 1; j < len(classes); j++ {
			if classes[i] > classes[j] {
				classes[i], classes[j] = classes[j], classes[i]
			}
		}
	}

	return classes
}

// encodeLabels encodes class labels for training.
func (lr *LogisticRegression) encodeLabels(y core.Tensor) core.Tensor {
	nSamples, _ := y.Dims()

	if lr.nClasses == 2 {
		// Binary classification: encode as 0/1
		encoded := core.NewZerosTensor(nSamples, 1)
		for i := 0; i < nSamples; i++ {
			class := int(y.At(i, 0))
			if class == lr.classes[1] {
				encoded.Set(i, 0, 1.0)
			}
		}
		return encoded
	} else {
		// Multiclass: one-hot encoding
		encoded := core.NewZerosTensor(nSamples, lr.nClasses)
		for i := 0; i < nSamples; i++ {
			class := int(y.At(i, 0))
			for j, c := range lr.classes {
				if class == c {
					encoded.Set(i, j, 1.0)
					break
				}
			}
		}
		return encoded
	}
}

// addInterceptIfNeeded adds intercept column to input data if configured.
func (lr *LogisticRegression) addInterceptIfNeeded(X core.Tensor) core.Tensor {
	if !lr.fitIntercept {
		return X
	}

	nSamples, nFeatures := X.Dims()
	XWithIntercept := core.NewZerosTensor(nSamples, nFeatures+1)

	// Set intercept column to 1
	for i := 0; i < nSamples; i++ {
		XWithIntercept.Set(i, 0, 1.0)
	}

	// Copy original features
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			XWithIntercept.Set(i, j+1, X.At(i, j))
		}
	}

	return XWithIntercept
}

// initializeWeights initializes model weights.
func (lr *LogisticRegression) initializeWeights(nFeatures int) {
	// nFeatures already includes intercept if fitIntercept is true
	if lr.nClasses == 2 {
		// Binary classification: single weight vector
		lr.weights = core.NewZerosTensor(nFeatures, 1)

		// Small random initialization
		for i := 0; i < nFeatures; i++ {
			lr.weights.Set(i, 0, lr.rng.NormFloat64()*0.01)
		}
	} else {
		// Multiclass: weight matrix
		lr.weights = core.NewZerosTensor(nFeatures, lr.nClasses)

		// Small random initialization
		for i := 0; i < nFeatures; i++ {
			for j := 0; j < lr.nClasses; j++ {
				lr.weights.Set(i, j, lr.rng.NormFloat64()*0.01)
			}
		}
	}
}

// predict computes predictions (probabilities) for input data.
func (lr *LogisticRegression) predict(X core.Tensor) core.Tensor {
	// Compute linear combination: X * weights
	logits := X.Mul(lr.weights)

	if lr.nClasses == 2 {
		// Binary classification: sigmoid activation
		return lr.sigmoid(logits)
	} else {
		// Multiclass: softmax activation
		return lr.softmax(logits)
	}
}

// sigmoid applies sigmoid activation function.
func (lr *LogisticRegression) sigmoid(x core.Tensor) core.Tensor {
	nSamples, nCols := x.Dims()
	result := core.NewZerosTensor(nSamples, nCols)

	for i := 0; i < nSamples; i++ {
		for j := 0; j < nCols; j++ {
			val := x.At(i, j)
			// Clamp to prevent overflow
			val = math.Max(-250, math.Min(250, val))
			result.Set(i, j, 1.0/(1.0+math.Exp(-val)))
		}
	}

	return result
}

// softmax applies softmax activation function.
func (lr *LogisticRegression) softmax(x core.Tensor) core.Tensor {
	nSamples, nClasses := x.Dims()
	result := core.NewZerosTensor(nSamples, nClasses)

	for i := 0; i < nSamples; i++ {
		// Find max for numerical stability
		maxVal := x.At(i, 0)
		for j := 1; j < nClasses; j++ {
			if x.At(i, j) > maxVal {
				maxVal = x.At(i, j)
			}
		}

		// Compute exponentials and sum
		var sum float64
		for j := 0; j < nClasses; j++ {
			val := math.Exp(x.At(i, j) - maxVal)
			result.Set(i, j, val)
			sum += val
		}

		// Normalize
		for j := 0; j < nClasses; j++ {
			result.Set(i, j, result.At(i, j)/sum)
		}
	}

	return result
}

// computeLoss computes the logistic loss.
func (lr *LogisticRegression) computeLoss(yTrue, yPred core.Tensor) float64 {
	nSamples, _ := yTrue.Dims()
	var loss float64

	if lr.nClasses == 2 {
		// Binary cross-entropy
		for i := 0; i < nSamples; i++ {
			y := yTrue.At(i, 0)
			p := math.Max(1e-15, math.Min(1-1e-15, yPred.At(i, 0))) // Clamp for numerical stability
			loss += -(y*math.Log(p) + (1-y)*math.Log(1-p))
		}
	} else {
		// Categorical cross-entropy
		for i := 0; i < nSamples; i++ {
			for j := 0; j < lr.nClasses; j++ {
				y := yTrue.At(i, j)
				p := math.Max(1e-15, yPred.At(i, j)) // Clamp for numerical stability
				loss += -y * math.Log(p)
			}
		}
	}

	loss /= float64(nSamples)

	// Add regularization
	loss += lr.computeRegularization()

	return loss
}

// computeRegularization computes regularization penalty.
func (lr *LogisticRegression) computeRegularization() float64 {
	if lr.lambda == 0 || lr.regularizer == "none" {
		return 0
	}

	weightsRows, weightsCols := lr.weights.Dims()
	var penalty float64

	// Skip intercept term if present
	startIdx := 0
	if lr.fitIntercept {
		startIdx = 1
	}

	switch lr.regularizer {
	case "l1":
		for i := startIdx; i < weightsRows; i++ {
			for j := 0; j < weightsCols; j++ {
				penalty += math.Abs(lr.weights.At(i, j))
			}
		}
		penalty *= lr.lambda

	case "l2":
		for i := startIdx; i < weightsRows; i++ {
			for j := 0; j < weightsCols; j++ {
				w := lr.weights.At(i, j)
				penalty += w * w
			}
		}
		penalty *= lr.lambda * 0.5

	case "elastic":
		l1Penalty := 0.0
		l2Penalty := 0.0
		for i := startIdx; i < weightsRows; i++ {
			for j := 0; j < weightsCols; j++ {
				w := lr.weights.At(i, j)
				l1Penalty += math.Abs(w)
				l2Penalty += w * w
			}
		}
		penalty = lr.lambda * (lr.l1Ratio*l1Penalty + (1-lr.l1Ratio)*0.5*l2Penalty)
	}

	return penalty
}

// computeGradients computes gradients for weight update.
func (lr *LogisticRegression) computeGradients(X, yTrue, yPred core.Tensor) core.Tensor {
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
func (lr *LogisticRegression) addRegularizationGradients(gradients core.Tensor) {
	if lr.lambda == 0 || lr.regularizer == "none" {
		return
	}

	gradRows, gradCols := gradients.Dims()

	// Skip intercept term if present
	startIdx := 0
	if lr.fitIntercept {
		startIdx = 1
	}

	switch lr.regularizer {
	case "l1":
		for i := startIdx; i < gradRows; i++ {
			for j := 0; j < gradCols; j++ {
				w := lr.weights.At(i, j)
				sign := 1.0
				if w < 0 {
					sign = -1.0
				}
				current := gradients.At(i, j)
				gradients.Set(i, j, current+lr.lambda*sign)
			}
		}

	case "l2":
		for i := startIdx; i < gradRows; i++ {
			for j := 0; j < gradCols; j++ {
				w := lr.weights.At(i, j)
				current := gradients.At(i, j)
				gradients.Set(i, j, current+lr.lambda*w)
			}
		}

	case "elastic":
		for i := startIdx; i < gradRows; i++ {
			for j := 0; j < gradCols; j++ {
				w := lr.weights.At(i, j)
				sign := 1.0
				if w < 0 {
					sign = -1.0
				}
				current := gradients.At(i, j)
				l1Term := lr.lambda * lr.l1Ratio * sign
				l2Term := lr.lambda * (1 - lr.l1Ratio) * w
				gradients.Set(i, j, current+l1Term+l2Term)
			}
		}
	}
}

// updateWeights updates model weights using gradients.
func (lr *LogisticRegression) updateWeights(gradients core.Tensor) {
	weightsRows, weightsCols := lr.weights.Dims()

	for i := 0; i < weightsRows; i++ {
		for j := 0; j < weightsCols; j++ {
			current := lr.weights.At(i, j)
			gradient := gradients.At(i, j)
			lr.weights.Set(i, j, current-lr.learningRate*gradient)
		}
	}
}

// Classification Metrics

// CalculateAccuracy computes the accuracy score.
func CalculateAccuracy(yTrue, yPred core.Tensor) float64 {
	nSamples, _ := yTrue.Dims()
	if nSamples == 0 {
		return 0.0
	}

	correct := 0
	for i := 0; i < nSamples; i++ {
		if math.Abs(yTrue.At(i, 0)-yPred.At(i, 0)) < 1e-10 {
			correct++
		}
	}

	return float64(correct) / float64(nSamples)
}

// CalculatePrecision computes the precision score for binary classification.
func CalculatePrecision(yTrue, yPred core.Tensor, positiveClass float64) float64 {
	truePositives := 0
	falsePositives := 0

	nSamples, _ := yTrue.Dims()
	for i := 0; i < nSamples; i++ {
		true_val := yTrue.At(i, 0)
		pred_val := yPred.At(i, 0)

		if math.Abs(pred_val-positiveClass) < 1e-10 {
			if math.Abs(true_val-positiveClass) < 1e-10 {
				truePositives++
			} else {
				falsePositives++
			}
		}
	}

	if truePositives+falsePositives == 0 {
		return 0.0
	}

	return float64(truePositives) / float64(truePositives+falsePositives)
}

// CalculateRecall computes the recall score for binary classification.
func CalculateRecall(yTrue, yPred core.Tensor, positiveClass float64) float64 {
	truePositives := 0
	falseNegatives := 0

	nSamples, _ := yTrue.Dims()
	for i := 0; i < nSamples; i++ {
		true_val := yTrue.At(i, 0)
		pred_val := yPred.At(i, 0)

		if math.Abs(true_val-positiveClass) < 1e-10 {
			if math.Abs(pred_val-positiveClass) < 1e-10 {
				truePositives++
			} else {
				falseNegatives++
			}
		}
	}

	if truePositives+falseNegatives == 0 {
		return 0.0
	}

	return float64(truePositives) / float64(truePositives+falseNegatives)
}

// CalculateF1Score computes the F1 score for binary classification.
func CalculateF1Score(yTrue, yPred core.Tensor, positiveClass float64) float64 {
	precision := CalculatePrecision(yTrue, yPred, positiveClass)
	recall := CalculateRecall(yTrue, yPred, positiveClass)

	if precision+recall == 0 {
		return 0.0
	}

	return 2 * (precision * recall) / (precision + recall)
}

// ConfusionMatrix computes the confusion matrix for binary classification.
type ConfusionMatrix struct {
	TruePositives  int
	TrueNegatives  int
	FalsePositives int
	FalseNegatives int
}

// CalculateConfusionMatrix computes the confusion matrix.
func CalculateConfusionMatrix(yTrue, yPred core.Tensor, positiveClass float64) *ConfusionMatrix {
	cm := &ConfusionMatrix{}

	nSamples, _ := yTrue.Dims()
	for i := 0; i < nSamples; i++ {
		true_val := yTrue.At(i, 0)
		pred_val := yPred.At(i, 0)

		trueIsPositive := math.Abs(true_val-positiveClass) < 1e-10
		predIsPositive := math.Abs(pred_val-positiveClass) < 1e-10

		if trueIsPositive && predIsPositive {
			cm.TruePositives++
		} else if trueIsPositive && !predIsPositive {
			cm.FalseNegatives++
		} else if !trueIsPositive && predIsPositive {
			cm.FalsePositives++
		} else {
			cm.TrueNegatives++
		}
	}

	return cm
}

// ClassificationMetrics holds comprehensive classification metrics.
type ClassificationMetrics struct {
	Accuracy        float64
	Precision       float64
	Recall          float64
	F1Score         float64
	ConfusionMatrix *ConfusionMatrix
}

// CalculateClassificationMetrics computes all classification metrics.
func CalculateClassificationMetrics(yTrue, yPred core.Tensor, positiveClass float64) *ClassificationMetrics {
	return &ClassificationMetrics{
		Accuracy:        CalculateAccuracy(yTrue, yPred),
		Precision:       CalculatePrecision(yTrue, yPred, positiveClass),
		Recall:          CalculateRecall(yTrue, yPred, positiveClass),
		F1Score:         CalculateF1Score(yTrue, yPred, positiveClass),
		ConfusionMatrix: CalculateConfusionMatrix(yTrue, yPred, positiveClass),
	}
}
