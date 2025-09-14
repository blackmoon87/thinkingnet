package algorithms

import (
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// RandomForest implements a random forest classifier.
type RandomForest struct {
	// Configuration
	nEstimators     int    // Number of trees in the forest
	maxDepth        int    // Maximum depth of trees (-1 for unlimited)
	minSamplesSplit int    // Minimum samples required to split a node
	minSamplesLeaf  int    // Minimum samples required at a leaf node
	maxFeatures     string // Number of features to consider: "sqrt", "log2", "all", or fraction
	bootstrap       bool   // Whether to use bootstrap sampling
	randomSeed      int64  // Random seed for reproducibility

	// State
	fitted    bool            // Whether the model has been fitted
	trees     []*DecisionTree // Collection of decision trees
	classes   []int           // Unique class labels
	nClasses  int             // Number of classes
	nFeatures int             // Number of features
	rng       *rand.Rand      // Random number generator
}

// DecisionTree represents a single decision tree in the forest.
type DecisionTree struct {
	root            *TreeNode
	maxDepth        int
	minSamplesSplit int
	minSamplesLeaf  int
	maxFeatures     int
	rng             *rand.Rand
}

// TreeNode represents a node in the decision tree.
type TreeNode struct {
	// Split criteria
	featureIndex int     // Feature index for split
	threshold    float64 // Threshold value for split

	// Node properties
	isLeaf     bool    // Whether this is a leaf node
	prediction int     // Class prediction for leaf nodes
	samples    int     // Number of samples in this node
	impurity   float64 // Gini impurity of this node

	// Child nodes
	left  *TreeNode
	right *TreeNode
}

// NewRandomForest creates a new random forest classifier.
func NewRandomForest(options ...RandomForestOption) *RandomForest {
	rf := &RandomForest{
		nEstimators:     100,
		maxDepth:        -1,
		minSamplesSplit: 2,
		minSamplesLeaf:  1,
		maxFeatures:     "sqrt",
		bootstrap:       true,
		randomSeed:      time.Now().UnixNano(),
		fitted:          false,
	}

	// Apply options
	for _, option := range options {
		option(rf)
	}

	// Initialize random number generator
	rf.rng = rand.New(rand.NewSource(rf.randomSeed))

	return rf
}

// RandomForestOption represents a functional option for random forest configuration.
type RandomForestOption func(*RandomForest)

// WithNEstimators sets the number of trees.
func WithNEstimators(nEstimators int) RandomForestOption {
	return func(rf *RandomForest) { rf.nEstimators = nEstimators }
}

// WithMaxDepth sets the maximum depth of trees.
func WithMaxDepth(maxDepth int) RandomForestOption {
	return func(rf *RandomForest) { rf.maxDepth = maxDepth }
}

// WithMinSamplesSplit sets the minimum samples required to split.
func WithMinSamplesSplit(minSamples int) RandomForestOption {
	return func(rf *RandomForest) { rf.minSamplesSplit = minSamples }
}

// WithMinSamplesLeaf sets the minimum samples required at leaf.
func WithMinSamplesLeaf(minSamples int) RandomForestOption {
	return func(rf *RandomForest) { rf.minSamplesLeaf = minSamples }
}

// WithMaxFeatures sets the number of features to consider.
func WithMaxFeatures(maxFeatures string) RandomForestOption {
	return func(rf *RandomForest) { rf.maxFeatures = maxFeatures }
}

// WithBootstrap sets whether to use bootstrap sampling.
func WithBootstrap(bootstrap bool) RandomForestOption {
	return func(rf *RandomForest) { rf.bootstrap = bootstrap }
}

// WithRFRandomSeed sets the random seed.
func WithRFRandomSeed(seed int64) RandomForestOption {
	return func(rf *RandomForest) { rf.randomSeed = seed }
}

// Fit trains the random forest model.
func (rf *RandomForest) Fit(X, y core.Tensor) error {
	if err := rf.validateInput(X, y); err != nil {
		return err
	}

	_, nFeatures := X.Dims()
	rf.nFeatures = nFeatures

	// Extract unique classes
	rf.classes = rf.extractClasses(y)
	rf.nClasses = len(rf.classes)

	if rf.nClasses < 2 {
		return core.NewError(core.ErrInvalidInput, "need at least 2 classes for classification")
	}

	// Calculate max features per tree
	maxFeaturesPerTree := rf.calculateMaxFeatures(nFeatures)

	// Initialize trees
	rf.trees = make([]*DecisionTree, rf.nEstimators)

	// Train each tree
	for i := 0; i < rf.nEstimators; i++ {
		// Create bootstrap sample if enabled
		var XTrain, yTrain core.Tensor
		if rf.bootstrap {
			XTrain, yTrain = rf.createBootstrapSample(X, y)
		} else {
			XTrain, yTrain = X, y
		}

		// Create and train tree
		tree := &DecisionTree{
			maxDepth:        rf.maxDepth,
			minSamplesSplit: rf.minSamplesSplit,
			minSamplesLeaf:  rf.minSamplesLeaf,
			maxFeatures:     maxFeaturesPerTree,
			rng:             rand.New(rand.NewSource(rf.rng.Int63())),
		}

		tree.fit(XTrain, yTrain)
		rf.trees[i] = tree
	}

	rf.fitted = true
	return nil
}

// Predict makes class predictions.
func (rf *RandomForest) Predict(X core.Tensor) (core.Tensor, error) {
	if !rf.fitted {
		return nil, core.NewError(core.ErrNotFitted, "RandomForest must be fitted before prediction")
	}

	if err := rf.validatePredictInput(X); err != nil {
		return nil, err
	}

	nSamples, _ := X.Dims()
	predictions := core.NewZerosTensor(nSamples, 1)

	// Get predictions from all trees and vote
	for i := 0; i < nSamples; i++ {
		sample := X.Row(i)
		votes := make(map[int]int)

		// Collect votes from all trees
		for _, tree := range rf.trees {
			prediction := tree.predict(sample)
			votes[prediction]++
		}

		// Find majority vote
		maxVotes := 0
		majorityClass := rf.classes[0]
		for class, count := range votes {
			if count > maxVotes {
				maxVotes = count
				majorityClass = class
			}
		}

		predictions.Set(i, 0, float64(majorityClass))
	}

	return predictions, nil
}

// PredictProba predicts class probabilities.
func (rf *RandomForest) PredictProba(X core.Tensor) (core.Tensor, error) {
	if !rf.fitted {
		return nil, core.NewError(core.ErrNotFitted, "RandomForest must be fitted before prediction")
	}

	if err := rf.validatePredictInput(X); err != nil {
		return nil, err
	}

	nSamples, _ := X.Dims()
	probabilities := core.NewZerosTensor(nSamples, rf.nClasses)

	// Get predictions from all trees and average
	for i := 0; i < nSamples; i++ {
		sample := X.Row(i)
		votes := make(map[int]int)

		// Collect votes from all trees
		for _, tree := range rf.trees {
			prediction := tree.predict(sample)
			votes[prediction]++
		}

		// Convert votes to probabilities
		for j, class := range rf.classes {
			prob := float64(votes[class]) / float64(rf.nEstimators)
			probabilities.Set(i, j, prob)
		}
	}

	return probabilities, nil
}

// Score returns the accuracy score on the given test data.
func (rf *RandomForest) Score(X, y core.Tensor) (float64, error) {
	predictions, err := rf.Predict(X)
	if err != nil {
		return 0, err
	}

	return CalculateAccuracy(y, predictions), nil
}

// Name returns the classifier name.
func (rf *RandomForest) Name() string {
	return "RandomForest"
}

// validateInput validates the input data for training.
func (rf *RandomForest) validateInput(X, y core.Tensor) error {
	if err := core.ValidateTrainingData(X, y); err != nil {
		return err
	}

	if rf.nEstimators <= 0 {
		return core.NewError(core.ErrInvalidInput, "number of estimators must be positive")
	}

	if rf.minSamplesSplit < 2 {
		return core.NewError(core.ErrInvalidInput, "min samples split must be at least 2")
	}

	if rf.minSamplesLeaf < 1 {
		return core.NewError(core.ErrInvalidInput, "min samples leaf must be at least 1")
	}

	return nil
}

// validatePredictInput validates input for prediction.
func (rf *RandomForest) validatePredictInput(X core.Tensor) error {
	if err := core.ValidateInput(X, []int{-1, rf.nFeatures}); err != nil {
		return err
	}

	return core.ValidateTensorFinite(X, "input")
}

// extractClasses extracts unique class labels from target vector.
func (rf *RandomForest) extractClasses(y core.Tensor) []int {
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
	sort.Ints(classes)

	return classes
}

// calculateMaxFeatures calculates the number of features to consider per tree.
func (rf *RandomForest) calculateMaxFeatures(nFeatures int) int {
	switch rf.maxFeatures {
	case "sqrt":
		return int(math.Sqrt(float64(nFeatures)))
	case "log2":
		return int(math.Log2(float64(nFeatures)))
	case "all":
		return nFeatures
	default:
		// Default to sqrt
		return int(math.Sqrt(float64(nFeatures)))
	}
}

// createBootstrapSample creates a bootstrap sample of the training data.
func (rf *RandomForest) createBootstrapSample(X, y core.Tensor) (core.Tensor, core.Tensor) {
	nSamples, nFeatures := X.Dims()

	// Sample with replacement
	XBootstrap := core.NewZerosTensor(nSamples, nFeatures)
	yBootstrap := core.NewZerosTensor(nSamples, 1)

	for i := 0; i < nSamples; i++ {
		idx := rf.rng.Intn(nSamples)

		// Copy row from original data
		for j := 0; j < nFeatures; j++ {
			XBootstrap.Set(i, j, X.At(idx, j))
		}
		yBootstrap.Set(i, 0, y.At(idx, 0))
	}

	return XBootstrap, yBootstrap
}

// DecisionTree methods

// fit trains a single decision tree.
func (dt *DecisionTree) fit(X, y core.Tensor) {
	nSamples, _ := X.Dims()
	indices := make([]int, nSamples)
	for i := range indices {
		indices[i] = i
	}

	dt.root = dt.buildTree(X, y, indices, 0)
}

// predict makes a prediction for a single sample.
func (dt *DecisionTree) predict(sample core.Tensor) int {
	return dt.predictNode(dt.root, sample)
}

// predictNode traverses the tree to make a prediction.
func (dt *DecisionTree) predictNode(node *TreeNode, sample core.Tensor) int {
	if node.isLeaf {
		return node.prediction
	}

	if sample.At(0, node.featureIndex) <= node.threshold {
		return dt.predictNode(node.left, sample)
	} else {
		return dt.predictNode(node.right, sample)
	}
}

// buildTree recursively builds the decision tree.
func (dt *DecisionTree) buildTree(X, y core.Tensor, indices []int, depth int) *TreeNode {
	node := &TreeNode{
		samples: len(indices),
	}

	// Calculate impurity
	node.impurity = dt.calculateGiniImpurity(y, indices)

	// Check stopping criteria
	if dt.shouldStop(indices, depth, node.impurity) {
		node.isLeaf = true
		node.prediction = dt.majorityClass(y, indices)
		return node
	}

	// Find best split
	bestFeature, bestThreshold, bestGain := dt.findBestSplit(X, y, indices)

	if bestGain <= 0 {
		node.isLeaf = true
		node.prediction = dt.majorityClass(y, indices)
		return node
	}

	// Split data
	leftIndices, rightIndices := dt.splitData(X, indices, bestFeature, bestThreshold)

	if len(leftIndices) < dt.minSamplesLeaf || len(rightIndices) < dt.minSamplesLeaf {
		node.isLeaf = true
		node.prediction = dt.majorityClass(y, indices)
		return node
	}

	// Create child nodes
	node.featureIndex = bestFeature
	node.threshold = bestThreshold
	node.left = dt.buildTree(X, y, leftIndices, depth+1)
	node.right = dt.buildTree(X, y, rightIndices, depth+1)

	return node
}

// shouldStop checks if we should stop splitting.
func (dt *DecisionTree) shouldStop(indices []int, depth int, impurity float64) bool {
	if len(indices) < dt.minSamplesSplit {
		return true
	}

	if dt.maxDepth != -1 && depth >= dt.maxDepth {
		return true
	}

	if impurity == 0 {
		return true
	}

	return false
}

// findBestSplit finds the best feature and threshold to split on.
func (dt *DecisionTree) findBestSplit(X, y core.Tensor, indices []int) (int, float64, float64) {
	_, nFeatures := X.Dims()

	// Select random subset of features
	features := dt.selectRandomFeatures(nFeatures)

	bestFeature := -1
	bestThreshold := 0.0
	bestGain := 0.0

	for _, feature := range features {
		// Get unique values for this feature
		values := dt.getUniqueValues(X, indices, feature)

		for _, threshold := range values {
			gain := dt.calculateInformationGain(X, y, indices, feature, threshold)
			if gain > bestGain {
				bestGain = gain
				bestFeature = feature
				bestThreshold = threshold
			}
		}
	}

	return bestFeature, bestThreshold, bestGain
}

// selectRandomFeatures selects a random subset of features.
func (dt *DecisionTree) selectRandomFeatures(nFeatures int) []int {
	maxFeatures := dt.maxFeatures
	if maxFeatures > nFeatures {
		maxFeatures = nFeatures
	}

	// Create list of all features
	allFeatures := make([]int, nFeatures)
	for i := range allFeatures {
		allFeatures[i] = i
	}

	// Shuffle and select subset
	dt.rng.Shuffle(len(allFeatures), func(i, j int) {
		allFeatures[i], allFeatures[j] = allFeatures[j], allFeatures[i]
	})

	return allFeatures[:maxFeatures]
}

// getUniqueValues gets unique values for a feature.
func (dt *DecisionTree) getUniqueValues(X core.Tensor, indices []int, feature int) []float64 {
	valueSet := make(map[float64]bool)

	for _, idx := range indices {
		value := X.At(idx, feature)
		valueSet[value] = true
	}

	values := make([]float64, 0, len(valueSet))
	for value := range valueSet {
		values = append(values, value)
	}

	sort.Float64s(values)
	return values
}

// calculateInformationGain calculates the information gain for a split.
func (dt *DecisionTree) calculateInformationGain(X, y core.Tensor, indices []int, feature int, threshold float64) float64 {
	// Calculate parent impurity
	parentImpurity := dt.calculateGiniImpurity(y, indices)

	// Split data
	leftIndices, rightIndices := dt.splitData(X, indices, feature, threshold)

	if len(leftIndices) == 0 || len(rightIndices) == 0 {
		return 0
	}

	// Calculate weighted impurity of children
	nSamples := len(indices)
	leftWeight := float64(len(leftIndices)) / float64(nSamples)
	rightWeight := float64(len(rightIndices)) / float64(nSamples)

	leftImpurity := dt.calculateGiniImpurity(y, leftIndices)
	rightImpurity := dt.calculateGiniImpurity(y, rightIndices)

	weightedImpurity := leftWeight*leftImpurity + rightWeight*rightImpurity

	return parentImpurity - weightedImpurity
}

// splitData splits indices based on feature and threshold.
func (dt *DecisionTree) splitData(X core.Tensor, indices []int, feature int, threshold float64) ([]int, []int) {
	var leftIndices, rightIndices []int

	for _, idx := range indices {
		if X.At(idx, feature) <= threshold {
			leftIndices = append(leftIndices, idx)
		} else {
			rightIndices = append(rightIndices, idx)
		}
	}

	return leftIndices, rightIndices
}

// calculateGiniImpurity calculates the Gini impurity.
func (dt *DecisionTree) calculateGiniImpurity(y core.Tensor, indices []int) float64 {
	if len(indices) == 0 {
		return 0
	}

	// Count class frequencies
	classCounts := make(map[int]int)
	for _, idx := range indices {
		class := int(y.At(idx, 0))
		classCounts[class]++
	}

	// Calculate Gini impurity
	impurity := 1.0
	nSamples := len(indices)

	for _, count := range classCounts {
		prob := float64(count) / float64(nSamples)
		impurity -= prob * prob
	}

	return impurity
}

// majorityClass returns the majority class in the given indices.
func (dt *DecisionTree) majorityClass(y core.Tensor, indices []int) int {
	classCounts := make(map[int]int)

	for _, idx := range indices {
		class := int(y.At(idx, 0))
		classCounts[class]++
	}

	maxCount := 0
	majorityClass := 0
	for class, count := range classCounts {
		if count > maxCount {
			maxCount = count
			majorityClass = class
		}
	}

	return majorityClass
}
