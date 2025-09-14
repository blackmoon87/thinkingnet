package algorithms

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestLinearRegression_BasicFit(t *testing.T) {
	// Create simple linear data: y = 2x + 1
	X := core.NewTensorFromSlice([][]float64{
		{1.0}, {2.0}, {3.0}, {4.0}, {5.0},
	})
	y := core.NewTensorFromSlice([][]float64{
		{3.0}, {5.0}, {7.0}, {9.0}, {11.0},
	})

	lr := NewLinearRegression(
		WithLinearLearningRate(0.01),
		WithLinearMaxIterations(1000),
		WithLinearTolerance(1e-6),
	)

	err := lr.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	if !lr.fitted {
		t.Error("Model should be marked as fitted")
	}

	// Test predictions
	predictions, err := lr.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// Check that predictions are close to actual values
	for i := 0; i < 5; i++ {
		pred := predictions.At(i, 0)
		actual := y.At(i, 0)
		if math.Abs(pred-actual) > 0.5 {
			t.Errorf("Prediction %d: expected ~%.2f, got %.2f", i, actual, pred)
		}
	}
}

func TestEasyLinearRegression(t *testing.T) {
	// Create simple linear data: y = 2x + 1
	X := core.NewTensorFromSlice([][]float64{
		{1.0}, {2.0}, {3.0}, {4.0}, {5.0},
	})
	y := core.NewTensorFromSlice([][]float64{
		{3.0}, {5.0}, {7.0}, {9.0}, {11.0},
	})

	// Test the easy constructor
	lr := EasyLinearRegression()

	// Verify default parameters are set correctly
	if lr.learningRate != 0.01 {
		t.Errorf("Expected learning rate 0.01, got %f", lr.learningRate)
	}
	if lr.maxIters != 1000 {
		t.Errorf("Expected max iterations 1000, got %d", lr.maxIters)
	}
	if lr.tolerance != 1e-6 {
		t.Errorf("Expected tolerance 1e-6, got %f", lr.tolerance)
	}
	if !lr.fitIntercept {
		t.Error("Expected fitIntercept to be true")
	}
	if lr.solver != "gradient_descent" {
		t.Errorf("Expected solver 'gradient_descent', got %s", lr.solver)
	}

	// Test that it works for training and prediction
	err := lr.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	predictions, err := lr.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// Check that predictions are reasonable
	for i := 0; i < 5; i++ {
		pred := predictions.At(i, 0)
		actual := y.At(i, 0)
		if math.Abs(pred-actual) > 1.0 {
			t.Errorf("Prediction %d: expected ~%.2f, got %.2f", i, actual, pred)
		}
	}
}

func TestLinearRegression_WithoutIntercept(t *testing.T) {
	// Create data: y = 2x (no intercept)
	X := core.NewTensorFromSlice([][]float64{
		{1.0}, {2.0}, {3.0}, {4.0}, {5.0},
	})
	y := core.NewTensorFromSlice([][]float64{
		{2.0}, {4.0}, {6.0}, {8.0}, {10.0},
	})

	lr := NewLinearRegression(
		WithLinearFitIntercept(false),
		WithLinearLearningRate(0.01),
		WithLinearMaxIterations(1000),
	)

	err := lr.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	// Check that the weight is approximately 2
	weights := lr.Weights()
	if weights == nil {
		t.Fatal("Weights should not be nil after fitting")
	}

	weight := weights.At(0, 0)
	if math.Abs(weight-2.0) > 0.1 {
		t.Errorf("Expected weight ~2.0, got %.3f", weight)
	}
}

func TestLinearRegression_L2Regularization(t *testing.T) {
	// Create data with some noise
	X := core.NewTensorFromSlice([][]float64{
		{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}, {5.0, 6.0},
	})
	y := core.NewTensorFromSlice([][]float64{
		{5.0}, {8.0}, {11.0}, {14.0}, {17.0},
	})

	lr := NewLinearRegression(
		WithLinearRegularization("l2", 0.1),
		WithLinearLearningRate(0.01),
		WithLinearMaxIterations(1000),
	)

	err := lr.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model with L2 regularization: %v", err)
	}

	// Test that model can make predictions
	predictions, err := lr.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict with regularized model: %v", err)
	}

	if predictions == nil {
		t.Error("Predictions should not be nil")
	}
}

func TestLinearRegression_L1Regularization(t *testing.T) {
	// Create data
	X := core.NewTensorFromSlice([][]float64{
		{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
	})
	y := core.NewTensorFromSlice([][]float64{
		{3.0}, {5.0}, {7.0}, {9.0},
	})

	lr := NewLinearRegression(
		WithLinearRegularization("l1", 0.1),
		WithLinearLearningRate(0.01),
		WithLinearMaxIterations(1000),
	)

	err := lr.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model with L1 regularization: %v", err)
	}

	// Test predictions
	predictions, err := lr.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict with L1 regularized model: %v", err)
	}

	if predictions == nil {
		t.Error("Predictions should not be nil")
	}
}

func TestLinearRegression_ElasticNet(t *testing.T) {
	// Create data
	X := core.NewTensorFromSlice([][]float64{
		{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
	})
	y := core.NewTensorFromSlice([][]float64{
		{3.0}, {5.0}, {7.0}, {9.0},
	})

	lr := NewLinearRegression(
		WithLinearElasticNet(0.1, 0.5),
		WithLinearLearningRate(0.01),
		WithLinearMaxIterations(1000),
	)

	err := lr.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model with Elastic Net: %v", err)
	}

	// Test predictions
	predictions, err := lr.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict with Elastic Net model: %v", err)
	}

	if predictions == nil {
		t.Error("Predictions should not be nil")
	}
}

func TestLinearRegression_Score(t *testing.T) {
	// Create perfect linear data
	X := core.NewTensorFromSlice([][]float64{
		{1.0}, {2.0}, {3.0}, {4.0}, {5.0},
	})
	y := core.NewTensorFromSlice([][]float64{
		{2.0}, {4.0}, {6.0}, {8.0}, {10.0},
	})

	lr := NewLinearRegression(
		WithLinearFitIntercept(false),
		WithLinearLearningRate(0.01),
		WithLinearMaxIterations(1000),
	)

	err := lr.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	score, err := lr.Score(X, y)
	if err != nil {
		t.Fatalf("Failed to compute score: %v", err)
	}

	// R² should be close to 1 for perfect linear data
	if score < 0.9 {
		t.Errorf("Expected R² score > 0.9, got %.3f", score)
	}
}

func TestLinearRegression_ValidationErrors(t *testing.T) {
	lr := NewLinearRegression()

	// Test prediction before fitting
	X := core.NewTensorFromSlice([][]float64{{1.0}, {2.0}})
	_, err := lr.Predict(X)
	if err == nil {
		t.Error("Should return error when predicting before fitting")
	}

	// Test invalid input dimensions
	X = core.NewTensorFromSlice([][]float64{{1.0}, {2.0}})
	y := core.NewTensorFromSlice([][]float64{{1.0}, {2.0}, {3.0}}) // Mismatched dimensions

	err = lr.Fit(X, y)
	if err == nil {
		t.Error("Should return error for mismatched input dimensions")
	}

	// Test invalid regularization parameters
	lr = NewLinearRegression(WithLinearRegularization("l1", -0.1))
	X = core.NewTensorFromSlice([][]float64{{1.0}, {2.0}})
	y = core.NewTensorFromSlice([][]float64{{1.0}, {2.0}})

	err = lr.Fit(X, y)
	if err == nil {
		t.Error("Should return error for negative regularization strength")
	}
}

func TestLinearRegression_Name(t *testing.T) {
	lr := NewLinearRegression()
	if lr.Name() != "LinearRegression" {
		t.Errorf("Expected name 'LinearRegression', got '%s'", lr.Name())
	}
}

func TestCalculateMSE(t *testing.T) {
	yTrue := core.NewTensorFromSlice([][]float64{{1.0}, {2.0}, {3.0}})
	yPred := core.NewTensorFromSlice([][]float64{{1.1}, {1.9}, {3.1}})

	mse := CalculateMSE(yTrue, yPred)
	expected := (0.01 + 0.01 + 0.01) / 3.0 // (0.1² + 0.1² + 0.1²) / 3

	if math.Abs(mse-expected) > 1e-10 {
		t.Errorf("Expected MSE %.10f, got %.10f", expected, mse)
	}
}

func TestCalculateRMSE(t *testing.T) {
	yTrue := core.NewTensorFromSlice([][]float64{{1.0}, {2.0}, {3.0}})
	yPred := core.NewTensorFromSlice([][]float64{{1.1}, {1.9}, {3.1}})

	rmse := CalculateRMSE(yTrue, yPred)
	expectedMSE := (0.01 + 0.01 + 0.01) / 3.0
	expected := math.Sqrt(expectedMSE)

	if math.Abs(rmse-expected) > 1e-10 {
		t.Errorf("Expected RMSE %.10f, got %.10f", expected, rmse)
	}
}

func TestCalculateMAE(t *testing.T) {
	yTrue := core.NewTensorFromSlice([][]float64{{1.0}, {2.0}, {3.0}})
	yPred := core.NewTensorFromSlice([][]float64{{1.1}, {1.9}, {3.1}})

	mae := CalculateMAE(yTrue, yPred)
	expected := (0.1 + 0.1 + 0.1) / 3.0

	if math.Abs(mae-expected) > 1e-10 {
		t.Errorf("Expected MAE %.10f, got %.10f", expected, mae)
	}
}

func TestCalculateR2Score(t *testing.T) {
	// Perfect predictions should give R² = 1
	yTrue := core.NewTensorFromSlice([][]float64{{1.0}, {2.0}, {3.0}})
	yPred := core.NewTensorFromSlice([][]float64{{1.0}, {2.0}, {3.0}})

	r2 := CalculateR2Score(yTrue, yPred)
	if math.Abs(r2-1.0) > 1e-10 {
		t.Errorf("Expected R² = 1.0 for perfect predictions, got %.10f", r2)
	}

	// Test with some error
	yPred = core.NewTensorFromSlice([][]float64{{1.1}, {1.9}, {3.1}})
	r2 = CalculateR2Score(yTrue, yPred)
	if r2 >= 1.0 || r2 <= 0.0 {
		t.Errorf("Expected 0 < R² < 1 for imperfect predictions, got %.3f", r2)
	}
}

func TestCalculateRegressionMetrics(t *testing.T) {
	yTrue := core.NewTensorFromSlice([][]float64{{1.0}, {2.0}, {3.0}})
	yPred := core.NewTensorFromSlice([][]float64{{1.1}, {1.9}, {3.1}})

	metrics := CalculateRegressionMetrics(yTrue, yPred)

	if metrics == nil {
		t.Fatal("Metrics should not be nil")
	}

	// Check that all metrics are computed
	if metrics.MSE <= 0 {
		t.Error("MSE should be positive")
	}

	if metrics.RMSE <= 0 {
		t.Error("RMSE should be positive")
	}

	if metrics.MAE <= 0 {
		t.Error("MAE should be positive")
	}

	if metrics.R2Score <= 0 || metrics.R2Score >= 1 {
		t.Errorf("R² should be between 0 and 1, got %.3f", metrics.R2Score)
	}

	// Check consistency between MSE and RMSE
	expectedRMSE := math.Sqrt(metrics.MSE)
	if math.Abs(metrics.RMSE-expectedRMSE) > 1e-10 {
		t.Errorf("RMSE should equal sqrt(MSE): expected %.10f, got %.10f", expectedRMSE, metrics.RMSE)
	}
}

func TestLinearRegression_EmptyData(t *testing.T) {
	lr := NewLinearRegression()

	// Test with nil tensors
	err := lr.Fit(nil, nil)
	if err == nil {
		t.Error("Should return error for nil training data")
	}
}

func TestLinearRegression_MultipleFeatures(t *testing.T) {
	// Create data with multiple features: y = 2*x1 + 3*x2 + 1
	X := core.NewTensorFromSlice([][]float64{
		{1.0, 1.0}, {2.0, 1.0}, {1.0, 2.0}, {2.0, 2.0}, {3.0, 2.0},
	})
	y := core.NewTensorFromSlice([][]float64{
		{6.0}, {8.0}, {9.0}, {11.0}, {13.0},
	})

	lr := NewLinearRegression(
		WithLinearLearningRate(0.01),
		WithLinearMaxIterations(2000),
		WithLinearTolerance(1e-6),
	)

	err := lr.Fit(X, y)
	if err != nil {
		t.Fatalf("Failed to fit model with multiple features: %v", err)
	}

	// Test predictions
	predictions, err := lr.Predict(X)
	if err != nil {
		t.Fatalf("Failed to predict with multiple features: %v", err)
	}

	// Check that predictions are reasonably close
	for i := 0; i < 5; i++ {
		pred := predictions.At(i, 0)
		actual := y.At(i, 0)
		if math.Abs(pred-actual) > 1.0 {
			t.Errorf("Prediction %d: expected ~%.2f, got %.2f", i, actual, pred)
		}
	}

	// Test R² score
	score, err := lr.Score(X, y)
	if err != nil {
		t.Fatalf("Failed to compute score: %v", err)
	}

	if score < 0.8 {
		t.Errorf("Expected R² score > 0.8, got %.3f", score)
	}
}

func BenchmarkLinearRegression_Fit(b *testing.B) {
	// Create benchmark data
	nSamples := 1000
	nFeatures := 10

	X := core.NewZerosTensor(nSamples, nFeatures)
	y := core.NewZerosTensor(nSamples, 1)

	// Fill with random data
	for i := 0; i < nSamples; i++ {
		var target float64
		for j := 0; j < nFeatures; j++ {
			val := float64(i*nFeatures + j)
			X.Set(i, j, val)
			target += val * 0.1
		}
		y.Set(i, 0, target)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		lr := NewLinearRegression(
			WithLinearLearningRate(0.001),
			WithLinearMaxIterations(100),
		)
		lr.Fit(X, y)
	}
}

func BenchmarkLinearRegression_Predict(b *testing.B) {
	// Setup
	X := core.NewTensorFromSlice([][]float64{
		{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
	})
	y := core.NewTensorFromSlice([][]float64{
		{3.0}, {5.0}, {7.0}, {9.0},
	})

	lr := NewLinearRegression()
	lr.Fit(X, y)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		lr.Predict(X)
	}
}
