package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/algorithms"
	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func main() {
	fmt.Println("Linear Regression Demo")
	fmt.Println("=====================")

	// Set random seed for reproducibility
	rand.Seed(42)

	// Demo 1: Simple linear regression
	fmt.Println("\n1. Simple Linear Regression (y = 2x + 1)")
	simpleLinearDemo()

	// Demo 2: Multiple features
	fmt.Println("\n2. Multiple Features Regression")
	multipleFeatureDemo()

	// Demo 3: Regularization comparison
	fmt.Println("\n3. Regularization Comparison")
	regularizationDemo()

	// Demo 4: Regression metrics
	fmt.Println("\n4. Regression Metrics")
	metricsDemo()
}

func simpleLinearDemo() {
	// Create simple linear data: y = 2x + 1 + noise
	nSamples := 100
	X := core.NewZerosTensor(nSamples, 1)
	y := core.NewZerosTensor(nSamples, 1)

	for i := 0; i < nSamples; i++ {
		x := float64(i) / 10.0
		noise := rand.NormFloat64() * 0.1
		target := 2.0*x + 1.0 + noise

		X.Set(i, 0, x)
		y.Set(i, 0, target)
	}

	// Train linear regression
	lr := algorithms.NewLinearRegression(
		algorithms.WithLinearLearningRate(0.01),
		algorithms.WithLinearMaxIterations(1000),
		algorithms.WithLinearTolerance(1e-6),
	)

	start := time.Now()
	err := lr.Fit(X, y)
	duration := time.Since(start)

	if err != nil {
		fmt.Printf("Error training model: %v\n", err)
		return
	}

	// Get model weights
	weights := lr.Weights()
	intercept := weights.At(0, 0)
	slope := weights.At(1, 0)

	fmt.Printf("Training completed in %v\n", duration)
	fmt.Printf("Learned parameters: y = %.3fx + %.3f\n", slope, intercept)
	fmt.Printf("True parameters:    y = 2.000x + 1.000\n")

	// Evaluate model
	score, _ := lr.Score(X, y)
	fmt.Printf("R² Score: %.4f\n", score)

	// Make some predictions
	testX := core.NewTensorFromSlice([][]float64{{5.0}, {7.5}, {10.0}})
	predictions, _ := lr.Predict(testX)

	fmt.Println("Predictions:")
	for i := 0; i < 3; i++ {
		x := testX.At(i, 0)
		pred := predictions.At(i, 0)
		expected := 2.0*x + 1.0
		fmt.Printf("  x=%.1f: predicted=%.3f, expected=%.3f\n", x, pred, expected)
	}
}

func multipleFeatureDemo() {
	// Create data with multiple features: y = 2*x1 + 3*x2 - 1*x3 + 5 + noise
	nSamples := 200
	nFeatures := 3
	X := core.NewZerosTensor(nSamples, nFeatures)
	y := core.NewZerosTensor(nSamples, 1)

	trueWeights := []float64{2.0, 3.0, -1.0}
	trueBias := 5.0

	for i := 0; i < nSamples; i++ {
		var target float64 = trueBias
		for j := 0; j < nFeatures; j++ {
			feature := rand.NormFloat64()
			X.Set(i, j, feature)
			target += trueWeights[j] * feature
		}
		target += rand.NormFloat64() * 0.1 // Add noise
		y.Set(i, 0, target)
	}

	// Train model
	lr := algorithms.NewLinearRegression(
		algorithms.WithLinearLearningRate(0.01),
		algorithms.WithLinearMaxIterations(2000),
	)

	err := lr.Fit(X, y)
	if err != nil {
		fmt.Printf("Error training model: %v\n", err)
		return
	}

	// Display results
	weights := lr.Weights()
	fmt.Printf("Learned weights: [%.3f, %.3f, %.3f, %.3f] (last is bias)\n",
		weights.At(1, 0), weights.At(2, 0), weights.At(3, 0), weights.At(0, 0))
	fmt.Printf("True weights:    [%.3f, %.3f, %.3f, %.3f]\n",
		trueWeights[0], trueWeights[1], trueWeights[2], trueBias)

	score, _ := lr.Score(X, y)
	fmt.Printf("R² Score: %.4f\n", score)
}

func regularizationDemo() {
	// Create data with some multicollinearity
	nSamples := 100
	X := core.NewZerosTensor(nSamples, 4)
	y := core.NewZerosTensor(nSamples, 1)

	for i := 0; i < nSamples; i++ {
		x1 := rand.NormFloat64()
		x2 := rand.NormFloat64()
		x3 := x1 + rand.NormFloat64()*0.1 // Correlated with x1
		x4 := x2 + rand.NormFloat64()*0.1 // Correlated with x2

		X.Set(i, 0, x1)
		X.Set(i, 1, x2)
		X.Set(i, 2, x3)
		X.Set(i, 3, x4)

		target := 2.0*x1 + 3.0*x2 + rand.NormFloat64()*0.1
		y.Set(i, 0, target)
	}

	// Train models with different regularization
	models := map[string]*algorithms.LinearRegression{
		"No Regularization": algorithms.NewLinearRegression(
			algorithms.WithLinearLearningRate(0.01),
			algorithms.WithLinearMaxIterations(1000),
		),
		"L1 (Lasso)": algorithms.NewLinearRegression(
			algorithms.WithLinearRegularization("l1", 0.1),
			algorithms.WithLinearLearningRate(0.01),
			algorithms.WithLinearMaxIterations(1000),
		),
		"L2 (Ridge)": algorithms.NewLinearRegression(
			algorithms.WithLinearRegularization("l2", 0.1),
			algorithms.WithLinearLearningRate(0.01),
			algorithms.WithLinearMaxIterations(1000),
		),
		"Elastic Net": algorithms.NewLinearRegression(
			algorithms.WithLinearElasticNet(0.1, 0.5),
			algorithms.WithLinearLearningRate(0.01),
			algorithms.WithLinearMaxIterations(1000),
		),
	}

	for name, model := range models {
		err := model.Fit(X, y)
		if err != nil {
			fmt.Printf("Error training %s: %v\n", name, err)
			continue
		}

		score, _ := model.Score(X, y)
		weights := model.Weights()

		fmt.Printf("%s:\n", name)
		fmt.Printf("  R² Score: %.4f\n", score)
		fmt.Printf("  Weights: [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
			weights.At(1, 0), weights.At(2, 0), weights.At(3, 0), weights.At(4, 0), weights.At(0, 0))
	}
}

func metricsDemo() {
	// Create test data
	yTrue := core.NewTensorFromSlice([][]float64{
		{1.0}, {2.0}, {3.0}, {4.0}, {5.0},
	})
	yPred := core.NewTensorFromSlice([][]float64{
		{1.1}, {1.9}, {3.2}, {3.8}, {5.1},
	})

	// Calculate individual metrics
	mse := algorithms.CalculateMSE(yTrue, yPred)
	rmse := algorithms.CalculateRMSE(yTrue, yPred)
	mae := algorithms.CalculateMAE(yTrue, yPred)
	r2 := algorithms.CalculateR2Score(yTrue, yPred)

	fmt.Printf("Individual Metrics:\n")
	fmt.Printf("  MSE:  %.6f\n", mse)
	fmt.Printf("  RMSE: %.6f\n", rmse)
	fmt.Printf("  MAE:  %.6f\n", mae)
	fmt.Printf("  R²:   %.6f\n", r2)

	// Calculate all metrics at once
	metrics := algorithms.CalculateRegressionMetrics(yTrue, yPred)
	fmt.Printf("\nAll Metrics (struct):\n")
	fmt.Printf("  MSE:  %.6f\n", metrics.MSE)
	fmt.Printf("  RMSE: %.6f\n", metrics.RMSE)
	fmt.Printf("  MAE:  %.6f\n", metrics.MAE)
	fmt.Printf("  R²:   %.6f\n", metrics.R2Score)
}
