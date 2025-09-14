package algorithms

import (
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestRandomForest_BinaryClassification(t *testing.T) {
	// Create simple binary classification dataset
	XData := [][]float64{
		{0, 0}, {0.1, 0.1}, {0.2, 0.0}, {0.0, 0.2}, {0.1, 0.0}, {0.0, 0.1},
		{2, 2}, {2.1, 1.9}, {1.9, 2.1}, {2.0, 2.0}, {1.8, 2.0}, {2.0, 1.8},
	}
	yData := [][]float64{
		{0}, {0}, {0}, {0}, {0}, {0},
		{1}, {1}, {1}, {1}, {1}, {1},
	}

	X := core.NewTensorFromSlice(XData)
	y := core.NewTensorFromSlice(yData)

	// Create and train model
	model := NewRandomForest(
		WithNEstimators(10),
		WithMaxDepth(5),
		WithMinSamplesSplit(2),
		WithMinSamplesLeaf(1),
		WithRFRandomSeed(42),
	)

	err := model.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	if !model.fitted {
		t.Error("Model should be fitted after training")
	}

	// Check that trees were created
	if len(model.trees) != 10 {
		t.Errorf("Expected 10 trees, got %d", len(model.trees))
	}

	// Test predictions
	predictions, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// Check prediction shape
	predRows, predCols := predictions.Dims()
	if predRows != 12 || predCols != 1 {
		t.Errorf("Expected predictions shape (12, 1), got (%d, %d)", predRows, predCols)
	}

	// Test probability predictions
	probabilities, err := model.PredictProba(X)
	if err != nil {
		t.Fatalf("Failed to predict probabilities: %v", err)
	}

	probRows, probCols := probabilities.Dims()
	if probRows != 12 || probCols != 2 {
		t.Errorf("Expected probabilities shape (12, 2), got (%d, %d)", probRows, probCols)
	}

	// Check that probabilities are between 0 and 1
	for i := 0; i < probRows; i++ {
		for j := 0; j < probCols; j++ {
			prob := probabilities.At(i, j)
			if prob < 0 || prob > 1 {
				t.Errorf("Probability should be between 0 and 1, got %f", prob)
			}
		}
	}

	// Test accuracy
	accuracy, err := model.Score(X, y)
	if err != nil {
		t.Fatalf("Failed to calculate score: %v", err)
	}

	if accuracy < 0.5 { // Should be better than random
		t.Errorf("Accuracy should be > 0.5, got %f", accuracy)
	}
}

func TestRandomForest_MulticlassClassification(t *testing.T) {
	// Create simple 3-class dataset
	XData := [][]float64{
		{0, 0}, {0.1, 0.1}, {-0.1, 0.1}, {0.1, -0.1}, // Class 0
		{2, 2}, {2.1, 1.9}, {1.9, 2.1}, {2.0, 2.0}, // Class 1
		{-2, 2}, {-1.9, 2.1}, {-2.1, 1.9}, {-2.0, 2.0}, // Class 2
	}
	yData := [][]float64{
		{0}, {0}, {0}, {0},
		{1}, {1}, {1}, {1},
		{2}, {2}, {2}, {2},
	}

	X := core.NewTensorFromSlice(XData)
	y := core.NewTensorFromSlice(yData)

	// Create and train model
	model := NewRandomForest(
		WithNEstimators(20),
		WithMaxDepth(10),
		WithRFRandomSeed(42),
	)

	err := model.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	// Check classes
	if model.nClasses != 3 {
		t.Errorf("Expected 3 classes, got %d", model.nClasses)
	}

	// Test predictions
	_, err = model.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// Test probability predictions
	probabilities, err := model.PredictProba(X)
	if err != nil {
		t.Fatalf("Failed to predict probabilities: %v", err)
	}

	probRows, probCols := probabilities.Dims()
	if probRows != 12 || probCols != 3 {
		t.Errorf("Expected probabilities shape (12, 3), got (%d, %d)", probRows, probCols)
	}

	// Test accuracy
	accuracy, err := model.Score(X, y)
	if err != nil {
		t.Fatalf("Failed to calculate score: %v", err)
	}

	if accuracy < 0.3 { // Should be better than random for 3 classes
		t.Errorf("Accuracy should be > 0.3, got %f", accuracy)
	}
}

func TestRandomForest_Options(t *testing.T) {
	// Test all configuration options
	model := NewRandomForest(
		WithNEstimators(50),
		WithMaxDepth(8),
		WithMinSamplesSplit(5),
		WithMinSamplesLeaf(2),
		WithMaxFeatures("log2"),
		WithBootstrap(false),
		WithRFRandomSeed(123),
	)

	if model.nEstimators != 50 {
		t.Errorf("Expected 50 estimators, got %d", model.nEstimators)
	}
	if model.maxDepth != 8 {
		t.Errorf("Expected max depth 8, got %d", model.maxDepth)
	}
	if model.minSamplesSplit != 5 {
		t.Errorf("Expected min samples split 5, got %d", model.minSamplesSplit)
	}
	if model.minSamplesLeaf != 2 {
		t.Errorf("Expected min samples leaf 2, got %d", model.minSamplesLeaf)
	}
	if model.maxFeatures != "log2" {
		t.Errorf("Expected max features 'log2', got %s", model.maxFeatures)
	}
	if model.bootstrap != false {
		t.Errorf("Expected bootstrap false, got %t", model.bootstrap)
	}
	if model.randomSeed != 123 {
		t.Errorf("Expected random seed 123, got %d", model.randomSeed)
	}
}

func TestRandomForest_ValidationErrors(t *testing.T) {
	model := NewRandomForest()

	// Test prediction before fitting
	X := core.NewTensorFromSlice([][]float64{{1, 2}})
	_, err := model.Predict(X)
	if err == nil {
		t.Error("Should fail when predicting before fitting")
	}

	// Test invalid input dimensions
	XInvalid := core.NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	yInvalid := core.NewTensorFromSlice([][]float64{{1}}) // Mismatched dimensions

	err = model.Fit(XInvalid, yInvalid)
	if err == nil {
		t.Error("Should fail with mismatched X and y dimensions")
	}

	// Test single class
	XSingle := core.NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	ySingle := core.NewTensorFromSlice([][]float64{{1}, {1}}) // Only one class

	err = model.Fit(XSingle, ySingle)
	if err == nil {
		t.Error("Should fail with only one class")
	}

	// Test invalid configuration
	invalidModel := NewRandomForest(WithNEstimators(0))
	XValid := core.NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	yValid := core.NewTensorFromSlice([][]float64{{0}, {1}})

	err = invalidModel.Fit(XValid, yValid)
	if err == nil {
		t.Error("Should fail with 0 estimators")
	}
}

func TestRandomForest_MaxFeaturesCalculation(t *testing.T) {
	model := NewRandomForest()

	// Test sqrt
	model.maxFeatures = "sqrt"
	maxFeatures := model.calculateMaxFeatures(16)
	if maxFeatures != 4 {
		t.Errorf("Expected 4 features for sqrt(16), got %d", maxFeatures)
	}

	// Test log2
	model.maxFeatures = "log2"
	maxFeatures = model.calculateMaxFeatures(16)
	if maxFeatures != 4 {
		t.Errorf("Expected 4 features for log2(16), got %d", maxFeatures)
	}

	// Test all
	model.maxFeatures = "all"
	maxFeatures = model.calculateMaxFeatures(10)
	if maxFeatures != 10 {
		t.Errorf("Expected 10 features for 'all', got %d", maxFeatures)
	}

	// Test default (should be sqrt)
	model.maxFeatures = "unknown"
	maxFeatures = model.calculateMaxFeatures(9)
	if maxFeatures != 3 {
		t.Errorf("Expected 3 features for default sqrt(9), got %d", maxFeatures)
	}
}

func TestRandomForest_WithoutBootstrap(t *testing.T) {
	// Create dataset
	XData := [][]float64{
		{0, 0}, {0.1, 0.1}, {0.2, 0.0}, {0.0, 0.2},
		{2, 2}, {2.1, 1.9}, {1.9, 2.1}, {2.0, 2.0},
	}
	yData := [][]float64{
		{0}, {0}, {0}, {0},
		{1}, {1}, {1}, {1},
	}

	X := core.NewTensorFromSlice(XData)
	y := core.NewTensorFromSlice(yData)

	// Train without bootstrap
	model := NewRandomForest(
		WithNEstimators(5),
		WithBootstrap(false),
		WithRFRandomSeed(42),
	)

	err := model.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model without bootstrap: %v", err)
	}

	// Should still be able to make predictions
	predictions, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// Check prediction shape
	predRows, predCols := predictions.Dims()
	if predRows != 8 || predCols != 1 {
		t.Errorf("Expected predictions shape (8, 1), got (%d, %d)", predRows, predCols)
	}
}

func TestDecisionTree_GiniImpurity(t *testing.T) {
	// Create a simple decision tree for testing
	tree := &DecisionTree{
		maxDepth:        5,
		minSamplesSplit: 2,
		minSamplesLeaf:  1,
		maxFeatures:     2,
	}

	// Test pure node (all same class)
	yPure := core.NewTensorFromSlice([][]float64{{1}, {1}, {1}, {1}})
	indices := []int{0, 1, 2, 3}
	impurity := tree.calculateGiniImpurity(yPure, indices)
	if impurity != 0.0 {
		t.Errorf("Expected 0 impurity for pure node, got %f", impurity)
	}

	// Test mixed node (50-50 split)
	yMixed := core.NewTensorFromSlice([][]float64{{0}, {0}, {1}, {1}})
	impurity = tree.calculateGiniImpurity(yMixed, indices)
	expected := 0.5 // 1 - (0.5^2 + 0.5^2) = 0.5
	if impurity != expected {
		t.Errorf("Expected %f impurity for 50-50 split, got %f", expected, impurity)
	}

	// Test empty node
	emptyIndices := []int{}
	impurity = tree.calculateGiniImpurity(yMixed, emptyIndices)
	if impurity != 0.0 {
		t.Errorf("Expected 0 impurity for empty node, got %f", impurity)
	}
}

func TestDecisionTree_MajorityClass(t *testing.T) {
	tree := &DecisionTree{}

	// Test majority class
	y := core.NewTensorFromSlice([][]float64{{0}, {1}, {1}, {0}, {1}})
	indices := []int{0, 1, 2, 3, 4}

	majorityClass := tree.majorityClass(y, indices)
	if majorityClass != 1 {
		t.Errorf("Expected majority class 1, got %d", majorityClass)
	}

	// Test tie (should return one of them consistently)
	yTie := core.NewTensorFromSlice([][]float64{{0}, {1}})
	indicesTie := []int{0, 1}

	majorityClassTie := tree.majorityClass(yTie, indicesTie)
	if majorityClassTie != 0 && majorityClassTie != 1 {
		t.Errorf("Expected majority class 0 or 1, got %d", majorityClassTie)
	}
}
