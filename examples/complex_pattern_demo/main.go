// Package main demonstrates complex pattern learning with detailed generalization report.
// This example tests the neural network's ability to learn and generalize complex patterns.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"strings"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/activations"
	"github.com/blackmoon87/thinkingnet/pkg/core"
	"github.com/blackmoon87/thinkingnet/pkg/layers"
	"github.com/blackmoon87/thinkingnet/pkg/losses"
	"github.com/blackmoon87/thinkingnet/pkg/models"
	"github.com/blackmoon87/thinkingnet/pkg/optimizers"
	"github.com/blackmoon87/thinkingnet/pkg/preprocessing"
	"gonum.org/v1/gonum/mat"
)

// ═══════════════════════════════════════════════════════════════════════════════
// COMPLEX PATTERN GENERATORS
// ═══════════════════════════════════════════════════════════════════════════════

// Pattern 1: Spiral Classification (2 interleaved spirals)
func generateSpiralData(samplesPerClass int) (core.Tensor, core.Tensor) {
	totalSamples := samplesPerClass * 2
	XData := make([]float64, totalSamples*2)
	yData := make([]float64, totalSamples*2) // One-hot encoded

	for i := 0; i < samplesPerClass; i++ {
		// Class 0: First spiral
		r := float64(i) / float64(samplesPerClass) * 5
		t := 1.75*float64(i)/float64(samplesPerClass)*2*math.Pi + rand.Float64()*0.2
		XData[i*2] = r * math.Sin(t)
		XData[i*2+1] = r * math.Cos(t)
		yData[i*2] = 1.0   // Class 0
		yData[i*2+1] = 0.0

		// Class 1: Second spiral (offset by pi)
		idx := (samplesPerClass + i) * 2
		t2 := 1.75*float64(i)/float64(samplesPerClass)*2*math.Pi + math.Pi + rand.Float64()*0.2
		XData[idx] = r * math.Sin(t2)
		XData[idx+1] = r * math.Cos(t2)
		yData[(samplesPerClass+i)*2] = 0.0
		yData[(samplesPerClass+i)*2+1] = 1.0 // Class 1
	}

	X := core.NewTensor(mat.NewDense(totalSamples, 2, XData))
	y := core.NewTensor(mat.NewDense(totalSamples, 2, yData))
	return X, y
}

// Pattern 2: Concentric Circles (XOR-like in polar coordinates)
func generateCirclesData(samplesPerClass int) (core.Tensor, core.Tensor) {
	totalSamples := samplesPerClass * 3
	XData := make([]float64, totalSamples*2)
	yData := make([]float64, totalSamples*3) // 3 classes

	for i := 0; i < samplesPerClass; i++ {
		theta := rand.Float64() * 2 * math.Pi

		// Inner circle (class 0)
		r1 := rand.Float64()*0.5 + 0.5
		XData[i*2] = r1 * math.Cos(theta)
		XData[i*2+1] = r1 * math.Sin(theta)
		yData[i*3] = 1.0
		yData[i*3+1] = 0.0
		yData[i*3+2] = 0.0

		// Middle ring (class 1)
		idx := (samplesPerClass + i) * 2
		r2 := rand.Float64()*0.5 + 2.0
		XData[idx] = r2 * math.Cos(theta)
		XData[idx+1] = r2 * math.Sin(theta)
		yData[(samplesPerClass+i)*3] = 0.0
		yData[(samplesPerClass+i)*3+1] = 1.0
		yData[(samplesPerClass+i)*3+2] = 0.0

		// Outer ring (class 2)
		idx = (2*samplesPerClass + i) * 2
		r3 := rand.Float64()*0.5 + 4.0
		XData[idx] = r3 * math.Cos(theta)
		XData[idx+1] = r3 * math.Sin(theta)
		yData[(2*samplesPerClass+i)*3] = 0.0
		yData[(2*samplesPerClass+i)*3+1] = 0.0
		yData[(2*samplesPerClass+i)*3+2] = 1.0
	}

	X := core.NewTensor(mat.NewDense(totalSamples, 2, XData))
	y := core.NewTensor(mat.NewDense(totalSamples, 3, yData))
	return X, y
}

// Pattern 3: Checkerboard Pattern (highly non-linear)
func generateCheckerboardData(gridSize, samplesPerCell int) (core.Tensor, core.Tensor) {
	totalSamples := gridSize * gridSize * samplesPerCell
	XData := make([]float64, totalSamples*2)
	yData := make([]float64, totalSamples*2)

	idx := 0
	for i := 0; i < gridSize; i++ {
		for j := 0; j < gridSize; j++ {
			isWhite := (i+j)%2 == 0
			for k := 0; k < samplesPerCell; k++ {
				x := float64(i) + rand.Float64()
				y := float64(j) + rand.Float64()
				XData[idx*2] = x
				XData[idx*2+1] = y
				if isWhite {
					yData[idx*2] = 1.0
					yData[idx*2+1] = 0.0
				} else {
					yData[idx*2] = 0.0
					yData[idx*2+1] = 1.0
				}
				idx++
			}
		}
	}

	X := core.NewTensor(mat.NewDense(totalSamples, 2, XData))
	Y := core.NewTensor(mat.NewDense(totalSamples, 2, yData))
	return X, Y
}

// Pattern 4: Sinusoidal Wave Classification
func generateWaveData(samples int) (core.Tensor, core.Tensor) {
	XData := make([]float64, samples*2)
	yData := make([]float64, samples*2)

	for i := 0; i < samples; i++ {
		x := rand.Float64()*10 - 5
		y := rand.Float64()*6 - 3

		// Complex wave boundary: y = sin(x) + 0.5*sin(2x) + 0.3*cos(3x)
		boundary := math.Sin(x) + 0.5*math.Sin(2*x) + 0.3*math.Cos(3*x)

		XData[i*2] = x
		XData[i*2+1] = y

		if y > boundary {
			yData[i*2] = 1.0
			yData[i*2+1] = 0.0
		} else {
			yData[i*2] = 0.0
			yData[i*2+1] = 1.0
		}
	}

	X := core.NewTensor(mat.NewDense(samples, 2, XData))
	Y := core.NewTensor(mat.NewDense(samples, 2, yData))
	return X, Y
}

// Pattern 5: Multi-frequency Regression (complex function approximation)
func generateComplexRegressionData(samples int) (core.Tensor, core.Tensor) {
	XData := make([]float64, samples)
	yData := make([]float64, samples)

	for i := 0; i < samples; i++ {
		x := float64(i) / float64(samples) * 4 * math.Pi

		// Complex function: combination of multiple frequencies and non-linearities
		y := math.Sin(x) +
			0.5*math.Sin(2*x) +
			0.3*math.Cos(3*x) +
			0.2*math.Sin(5*x)*math.Cos(x) +
			0.1*math.Exp(-0.1*x)*math.Sin(x) +
			rand.Float64()*0.05 // Small noise

		XData[i] = x
		yData[i] = y
	}

	X := core.NewTensor(mat.NewDense(samples, 1, XData))
	Y := core.NewTensor(mat.NewDense(samples, 1, yData))
	return X, Y
}

// Pattern 6: XOR with Gaussian Noise
func generateNoisyXORData(samples int, noiseLevel float64) (core.Tensor, core.Tensor) {
	XData := make([]float64, samples*2)
	yData := make([]float64, samples*2)

	for i := 0; i < samples; i++ {
		x1 := rand.Float64()*2 - 1 + rand.NormFloat64()*noiseLevel
		x2 := rand.Float64()*2 - 1 + rand.NormFloat64()*noiseLevel

		XData[i*2] = x1
		XData[i*2+1] = x2

		// XOR pattern
		if (x1 > 0 && x2 > 0) || (x1 < 0 && x2 < 0) {
			yData[i*2] = 1.0
			yData[i*2+1] = 0.0
		} else {
			yData[i*2] = 0.0
			yData[i*2+1] = 1.0
		}
	}

	X := core.NewTensor(mat.NewDense(samples, 2, XData))
	Y := core.NewTensor(mat.NewDense(samples, 2, yData))
	return X, Y
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEARNING AND GENERALIZATION REPORT STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

type PatternResult struct {
	PatternName       string
	Complexity        string
	TrainSamples      int
	TestSamples       int
	InputFeatures     int
	OutputClasses     int
	Architecture      []int
	TrainingTime      time.Duration
	Epochs            int
	InitialLoss       float64
	FinalLoss         float64
	TrainAccuracy     float64
	TestAccuracy      float64
	GeneralizationGap float64
	Overfitting       bool
	LearningCurve     []float64
	ConfusionMatrix   [][]int
	Converged         bool
}

type GeneralizationReport struct {
	Patterns          []PatternResult
	TotalPatterns     int
	SuccessfulLearns  int
	GoodGeneralizers  int
	OverfitCount      int
	AvgTrainAccuracy  float64
	AvgTestAccuracy   float64
	AvgGenGap         float64
	BestPattern       string
	WorstPattern      string
	TotalTrainingTime time.Duration
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRAINING AND EVALUATION
// ═══════════════════════════════════════════════════════════════════════════════

func trainAndEvaluate(
	name string,
	complexity string,
	X, y core.Tensor,
	architecture []int,
	epochs int,
	learningRate float64,
	isClassification bool,
) PatternResult {

	result := PatternResult{
		PatternName:  name,
		Complexity:   complexity,
		Architecture: architecture,
	}

	_, inputFeatures := X.Dims()
	_, outputSize := y.Dims()
	result.InputFeatures = inputFeatures
	result.OutputClasses = outputSize

	// Split data
	XTrain, XTest, yTrain, yTest, err := preprocessing.EasySplit(X, y, 0.2)
	if err != nil {
		fmt.Printf("  ✗ Split error: %v\n", err)
		return result
	}

	trainSamples, _ := XTrain.Dims()
	testSamples, _ := XTest.Dims()
	result.TrainSamples = trainSamples
	result.TestSamples = testSamples

	// Scale data
	XTrainScaled, err := preprocessing.EasyStandardScale(XTrain)
	if err != nil {
		XTrainScaled = XTrain
	}

	// Build model
	model := models.NewSequential()
	prevSize := inputFeatures

	for i, size := range architecture {
		var activation core.Activation
		if i < len(architecture)-1 {
			activation = activations.NewReLU()
		} else {
			if isClassification {
				activation = activations.NewSoftmax()
			} else {
				activation = activations.NewLinear()
			}
		}
		model.AddLayer(layers.NewDense(size, &layers.DenseConfig{
			Activation: activation,
		}))
		prevSize = size
	}
	_ = prevSize

	// Compile
	optimizer, _ := optimizers.NewAdamWithDefaults(learningRate)
	var lossFunc core.Loss
	if isClassification {
		lossFunc = losses.NewCategoricalCrossEntropy()
	} else {
		lossFunc = losses.NewMeanSquaredError()
	}
	model.Compile(optimizer, lossFunc)

	// Train
	config := core.TrainingConfig{
		Epochs:    epochs,
		BatchSize: 32,
		Verbose:   0,
	}

	startTime := time.Now()
	history, err := model.Fit(XTrainScaled, yTrain, config)
	result.TrainingTime = time.Since(startTime)
	result.Epochs = epochs

	if err != nil {
		fmt.Printf("  ✗ Training error: %v\n", err)
		return result
	}

	// Record learning curve
	if history != nil && len(history.Loss) > 0 {
		result.LearningCurve = history.Loss
		result.InitialLoss = history.Loss[0]
		result.FinalLoss = history.Loss[len(history.Loss)-1]
		result.Converged = result.FinalLoss < result.InitialLoss*0.5
	}

	// Evaluate on training set
	predTrain, err := model.Forward(XTrainScaled)
	if err == nil {
		result.TrainAccuracy = calculateAccuracy(predTrain, yTrain, isClassification)
	}

	// Evaluate on test set (scale test data the same way)
	XTestScaled, err := preprocessing.EasyStandardScale(XTest)
	if err != nil {
		XTestScaled = XTest
	}

	predTest, err := model.Forward(XTestScaled)
	if err == nil {
		result.TestAccuracy = calculateAccuracy(predTest, yTest, isClassification)
		if isClassification {
			result.ConfusionMatrix = calculateConfusionMatrix(predTest, yTest)
		}
	}

	// Calculate generalization gap
	result.GeneralizationGap = result.TrainAccuracy - result.TestAccuracy
	result.Overfitting = result.GeneralizationGap > 0.15 // >15% gap indicates overfitting

	return result
}

func calculateAccuracy(predictions, targets core.Tensor, isClassification bool) float64 {
	predRows, predCols := predictions.Dims()
	targRows, _ := targets.Dims()

	if predRows != targRows {
		return 0
	}

	if isClassification {
		correct := 0
		for i := 0; i < predRows; i++ {
			predClass := 0
			targClass := 0
			maxPred := predictions.At(i, 0)
			maxTarg := targets.At(i, 0)

			for j := 1; j < predCols; j++ {
				if predictions.At(i, j) > maxPred {
					maxPred = predictions.At(i, j)
					predClass = j
				}
				if targets.At(i, j) > maxTarg {
					maxTarg = targets.At(i, j)
					targClass = j
				}
			}

			if predClass == targClass {
				correct++
			}
		}
		return float64(correct) / float64(predRows)
	} else {
		// For regression, use R² score
		var ssRes, ssTot float64
		var mean float64
		for i := 0; i < predRows; i++ {
			mean += targets.At(i, 0)
		}
		mean /= float64(predRows)

		for i := 0; i < predRows; i++ {
			diff := targets.At(i, 0) - predictions.At(i, 0)
			ssRes += diff * diff
			diffMean := targets.At(i, 0) - mean
			ssTot += diffMean * diffMean
		}

		if ssTot == 0 {
			return 1.0
		}
		r2 := 1 - ssRes/ssTot
		if r2 < 0 {
			r2 = 0
		}
		return r2
	}
}

func calculateConfusionMatrix(predictions, targets core.Tensor) [][]int {
	_, numClasses := predictions.Dims()
	rows, _ := predictions.Dims()

	matrix := make([][]int, numClasses)
	for i := range matrix {
		matrix[i] = make([]int, numClasses)
	}

	for i := 0; i < rows; i++ {
		predClass := 0
		targClass := 0
		maxPred := predictions.At(i, 0)
		maxTarg := targets.At(i, 0)

		for j := 1; j < numClasses; j++ {
			if predictions.At(i, j) > maxPred {
				maxPred = predictions.At(i, j)
				predClass = j
			}
			if targets.At(i, j) > maxTarg {
				maxTarg = targets.At(i, j)
				targClass = j
			}
		}

		matrix[targClass][predClass]++
	}

	return matrix
}

// ═══════════════════════════════════════════════════════════════════════════════
// REPORT PRINTING
// ═══════════════════════════════════════════════════════════════════════════════

func printPatternResult(r PatternResult) {
	fmt.Printf("\n  ┌─────────────────────────────────────────────────────────────────────────┐\n")
	fmt.Printf("  │ Pattern: %-63s │\n", r.PatternName)
	fmt.Printf("  │ Complexity: %-60s │\n", r.Complexity)
	fmt.Printf("  ├─────────────────────────────────────────────────────────────────────────┤\n")
	fmt.Printf("  │ Data:       Train=%d, Test=%d, Features=%d, Classes=%d %17s│\n",
		r.TrainSamples, r.TestSamples, r.InputFeatures, r.OutputClasses, "")
	fmt.Printf("  │ Architecture: %-58v │\n", r.Architecture)
	fmt.Printf("  │ Training Time: %-56s │\n", formatDuration(r.TrainingTime))
	fmt.Printf("  ├─────────────────────────────────────────────────────────────────────────┤\n")
	fmt.Printf("  │                         LEARNING METRICS                               │\n")
	fmt.Printf("  ├─────────────────────────────────────────────────────────────────────────┤\n")
	fmt.Printf("  │ Initial Loss:    %-54.6f │\n", r.InitialLoss)
	fmt.Printf("  │ Final Loss:      %-54.6f │\n", r.FinalLoss)
	lossReduction := 0.0
	if r.InitialLoss > 0 {
		lossReduction = (1 - r.FinalLoss/r.InitialLoss) * 100
	}
	fmt.Printf("  │ Loss Reduction:  %-54.2f%% │\n", lossReduction)

	convergedStr := "❌ No"
	if r.Converged {
		convergedStr = "✅ Yes"
	}
	fmt.Printf("  │ Converged:       %-54s │\n", convergedStr)

	fmt.Printf("  ├─────────────────────────────────────────────────────────────────────────┤\n")
	fmt.Printf("  │                      GENERALIZATION METRICS                            │\n")
	fmt.Printf("  ├─────────────────────────────────────────────────────────────────────────┤\n")
	fmt.Printf("  │ Train Accuracy:  %-54.2f%% │\n", r.TrainAccuracy*100)
	fmt.Printf("  │ Test Accuracy:   %-54.2f%% │\n", r.TestAccuracy*100)
	fmt.Printf("  │ Gen. Gap:        %-54.2f%% │\n", r.GeneralizationGap*100)

	overfitStr := "✅ Good Generalization"
	if r.Overfitting {
		overfitStr = "⚠️  Overfitting Detected"
	}
	fmt.Printf("  │ Status:          %-54s │\n", overfitStr)

	// Learning curve visualization (ASCII)
	if len(r.LearningCurve) > 0 {
		fmt.Printf("  ├─────────────────────────────────────────────────────────────────────────┤\n")
		fmt.Printf("  │                         LEARNING CURVE                                 │\n")
		fmt.Printf("  ├─────────────────────────────────────────────────────────────────────────┤\n")
		printLearningCurve(r.LearningCurve, 10)
	}

	// Confusion Matrix
	if r.ConfusionMatrix != nil && len(r.ConfusionMatrix) > 0 {
		fmt.Printf("  ├─────────────────────────────────────────────────────────────────────────┤\n")
		fmt.Printf("  │                        CONFUSION MATRIX                                │\n")
		fmt.Printf("  ├─────────────────────────────────────────────────────────────────────────┤\n")
		printConfusionMatrix(r.ConfusionMatrix)
	}

	fmt.Printf("  └─────────────────────────────────────────────────────────────────────────┘\n")
}

func printLearningCurve(losses []float64, numPoints int) {
	if len(losses) < numPoints {
		numPoints = len(losses)
	}

	step := len(losses) / numPoints
	if step < 1 {
		step = 1
	}

	maxLoss := losses[0]
	minLoss := losses[0]
	for _, l := range losses {
		if l > maxLoss {
			maxLoss = l
		}
		if l < minLoss {
			minLoss = l
		}
	}

	height := 5
	for h := height; h >= 0; h-- {
		threshold := minLoss + (maxLoss-minLoss)*float64(h)/float64(height)
		line := "  │ "
		for i := 0; i < numPoints; i++ {
			idx := i * step
			if idx >= len(losses) {
				idx = len(losses) - 1
			}
			if losses[idx] >= threshold {
				line += "█"
			} else {
				line += " "
			}
		}
		if h == height {
			line += fmt.Sprintf(" %.4f", maxLoss)
		} else if h == 0 {
			line += fmt.Sprintf(" %.4f", minLoss)
		}
		fmt.Printf("%s%s│\n", line, strings.Repeat(" ", 60-len(line)))
	}
	fmt.Printf("  │   Epoch: 0%s→ %d %30s│\n", strings.Repeat("─", 8), len(losses), "")
}

func printConfusionMatrix(cm [][]int) {
	numClasses := len(cm)
	fmt.Printf("  │         ")
	for i := 0; i < numClasses; i++ {
		fmt.Printf("Pred_%d ", i)
	}
	fmt.Printf("%*s│\n", 71-9-numClasses*7, "")

	for i := 0; i < numClasses; i++ {
		fmt.Printf("  │ True_%d: ", i)
		for j := 0; j < numClasses; j++ {
			fmt.Printf("%5d  ", cm[i][j])
		}
		fmt.Printf("%*s│\n", 71-9-numClasses*7, "")
	}
}

func printGeneralizationReport(report GeneralizationReport) {
	fmt.Println("\n")
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║             LEARNING & GENERALIZATION SUMMARY REPORT                        ║")
	fmt.Println("║                 تقرير ملخص التعلم والتعميم                                  ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════╣")

	fmt.Printf("║  Total Patterns Tested:    %-50d ║\n", report.TotalPatterns)
	fmt.Printf("║  Successful Learners:      %-50d ║\n", report.SuccessfulLearns)
	fmt.Printf("║  Good Generalizers:        %-50d ║\n", report.GoodGeneralizers)
	fmt.Printf("║  Overfitting Cases:        %-50d ║\n", report.OverfitCount)

	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                           ACCURACY METRICS                                  ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════╣")

	fmt.Printf("║  Average Train Accuracy:   %-50.2f%% ║\n", report.AvgTrainAccuracy*100)
	fmt.Printf("║  Average Test Accuracy:    %-50.2f%% ║\n", report.AvgTestAccuracy*100)
	fmt.Printf("║  Average Gen. Gap:         %-50.2f%% ║\n", report.AvgGenGap*100)

	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                          PATTERN RANKINGS                                   ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════╣")

	fmt.Printf("║  🏆 Best Pattern:          %-50s ║\n", report.BestPattern)
	fmt.Printf("║  📉 Hardest Pattern:       %-50s ║\n", report.WorstPattern)

	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  Total Training Time:      %-50s ║\n", formatDuration(report.TotalTrainingTime))
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════╝")

	// Pattern comparison table
	fmt.Println("\n┌───────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                        PATTERN COMPARISON TABLE                              │")
	fmt.Println("├─────────────────────────┬────────────┬───────────┬──────────┬────────────────┤")
	fmt.Println("│ Pattern                 │ Train Acc  │ Test Acc  │ Gen Gap  │ Status         │")
	fmt.Println("├─────────────────────────┼────────────┼───────────┼──────────┼────────────────┤")

	for _, p := range report.Patterns {
		status := "✅ Good"
		if p.Overfitting {
			status = "⚠️  Overfit"
		}
		if p.TestAccuracy < 0.6 {
			status = "❌ Poor"
		}

		name := p.PatternName
		if len(name) > 23 {
			name = name[:20] + "..."
		}

		fmt.Printf("│ %-23s │ %8.2f%% │ %7.2f%% │ %6.2f%% │ %-14s │\n",
			name,
			p.TrainAccuracy*100,
			p.TestAccuracy*100,
			p.GeneralizationGap*100,
			status)
	}
	fmt.Println("└─────────────────────────┴────────────┴───────────┴──────────┴────────────────┘")

	// Recommendations
	fmt.Println("\n┌───────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                            RECOMMENDATIONS                                   │")
	fmt.Println("│                               التوصيات                                       │")
	fmt.Println("├───────────────────────────────────────────────────────────────────────────────┤")

	if report.OverfitCount > 0 {
		fmt.Println("│  ⚠️  Overfitting detected in some patterns. Consider:                       │")
		fmt.Println("│      • Adding dropout layers                                                │")
		fmt.Println("│      • Reducing model complexity                                            │")
		fmt.Println("│      • Using early stopping                                                 │")
		fmt.Println("│      • Increasing training data                                             │")
	}

	if report.AvgTestAccuracy < 0.7 {
		fmt.Println("│  📊 Test accuracy could be improved. Try:                                   │")
		fmt.Println("│      • Deeper architectures for complex patterns                            │")
		fmt.Println("│      • Different activation functions                                       │")
		fmt.Println("│      • Learning rate scheduling                                             │")
		fmt.Println("│      • More training epochs                                                 │")
	}

	if report.AvgTestAccuracy >= 0.8 && report.OverfitCount == 0 {
		fmt.Println("│  ✅ Excellent performance! The model shows:                                 │")
		fmt.Println("│      • Good learning capability                                             │")
		fmt.Println("│      • Strong generalization                                                │")
		fmt.Println("│      • Appropriate model complexity                                         │")
	}

	fmt.Println("└───────────────────────────────────────────────────────────────────────────────┘")
}

func formatDuration(d time.Duration) string {
	if d < time.Millisecond {
		return fmt.Sprintf("%.2fµs", float64(d.Nanoseconds())/1000)
	} else if d < time.Second {
		return fmt.Sprintf("%.2fms", float64(d.Nanoseconds())/1e6)
	} else if d < time.Minute {
		return fmt.Sprintf("%.2fs", d.Seconds())
	}
	return fmt.Sprintf("%.2fm", d.Minutes())
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN FUNCTION
// ═══════════════════════════════════════════════════════════════════════════════

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║       THINKINGNET-GO COMPLEX PATTERN LEARNING DEMONSTRATION                 ║")
	fmt.Println("║               اختبار تعلم الأنماط المعقدة والتعميم                           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Printf("System: %s/%s with %d CPU cores | Go %s\n",
		runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), runtime.Version())
	fmt.Println()

	var results []PatternResult
	totalStart := time.Now()

	// ═══════════════════════════════════════════════════════════════════════════
	// PATTERN 1: Spiral Classification
	// ═══════════════════════════════════════════════════════════════════════════
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  🌀 Testing: SPIRAL CLASSIFICATION (Two interleaved spirals)")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	X1, y1 := generateSpiralData(500)
	result1 := trainAndEvaluate(
		"Spiral Classification",
		"High (Non-linear, interleaved)",
		X1, y1,
		[]int{32, 64, 32, 2},
		100,
		0.001,
		true,
	)
	results = append(results, result1)
	printPatternResult(result1)

	// ═══════════════════════════════════════════════════════════════════════════
	// PATTERN 2: Concentric Circles
	// ═══════════════════════════════════════════════════════════════════════════
	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  ⭕ Testing: CONCENTRIC CIRCLES (3-class radial classification)")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	X2, y2 := generateCirclesData(300)
	result2 := trainAndEvaluate(
		"Concentric Circles",
		"Medium (Radial separation)",
		X2, y2,
		[]int{16, 32, 16, 3},
		80,
		0.002,
		true,
	)
	results = append(results, result2)
	printPatternResult(result2)

	// ═══════════════════════════════════════════════════════════════════════════
	// PATTERN 3: Checkerboard
	// ═══════════════════════════════════════════════════════════════════════════
	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  ♟️  Testing: CHECKERBOARD PATTERN (Highly non-linear)")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	X3, y3 := generateCheckerboardData(4, 100)
	result3 := trainAndEvaluate(
		"Checkerboard 4x4",
		"Very High (Multiple XOR-like regions)",
		X3, y3,
		[]int{64, 128, 64, 2},
		150,
		0.001,
		true,
	)
	results = append(results, result3)
	printPatternResult(result3)

	// ═══════════════════════════════════════════════════════════════════════════
	// PATTERN 4: Sinusoidal Wave
	// ═══════════════════════════════════════════════════════════════════════════
	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  〰️  Testing: SINUSOIDAL WAVE BOUNDARY (Multi-frequency)")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	X4, y4 := generateWaveData(1000)
	result4 := trainAndEvaluate(
		"Sinusoidal Wave",
		"High (Multi-frequency boundary)",
		X4, y4,
		[]int{32, 64, 32, 2},
		100,
		0.001,
		true,
	)
	results = append(results, result4)
	printPatternResult(result4)

	// ═══════════════════════════════════════════════════════════════════════════
	// PATTERN 5: Complex Regression
	// ═══════════════════════════════════════════════════════════════════════════
	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  📈 Testing: COMPLEX FUNCTION REGRESSION (Multi-frequency)")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	X5, y5 := generateComplexRegressionData(1000)
	result5 := trainAndEvaluate(
		"Complex Regression",
		"Very High (Multi-frequency function)",
		X5, y5,
		[]int{64, 128, 64, 32, 1},
		200,
		0.001,
		false,
	)
	results = append(results, result5)
	printPatternResult(result5)

	// ═══════════════════════════════════════════════════════════════════════════
	// PATTERN 6: Noisy XOR
	// ═══════════════════════════════════════════════════════════════════════════
	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  🔀 Testing: NOISY XOR (Classic with Gaussian noise)")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	X6, y6 := generateNoisyXORData(800, 0.1)
	result6 := trainAndEvaluate(
		"Noisy XOR",
		"Medium (XOR + noise)",
		X6, y6,
		[]int{16, 32, 16, 2},
		80,
		0.002,
		true,
	)
	results = append(results, result6)
	printPatternResult(result6)

	// ═══════════════════════════════════════════════════════════════════════════
	// GENERATE SUMMARY REPORT
	// ═══════════════════════════════════════════════════════════════════════════

	report := GeneralizationReport{
		Patterns:          results,
		TotalPatterns:     len(results),
		TotalTrainingTime: time.Since(totalStart),
	}

	var totalTrainAcc, totalTestAcc, totalGenGap float64
	bestAcc := 0.0
	worstAcc := 1.0

	for _, r := range results {
		totalTrainAcc += r.TrainAccuracy
		totalTestAcc += r.TestAccuracy
		totalGenGap += r.GeneralizationGap

		if r.TestAccuracy > 0.6 {
			report.SuccessfulLearns++
		}
		if !r.Overfitting && r.TestAccuracy > 0.6 {
			report.GoodGeneralizers++
		}
		if r.Overfitting {
			report.OverfitCount++
		}

		if r.TestAccuracy > bestAcc {
			bestAcc = r.TestAccuracy
			report.BestPattern = r.PatternName
		}
		if r.TestAccuracy < worstAcc {
			worstAcc = r.TestAccuracy
			report.WorstPattern = r.PatternName
		}
	}

	report.AvgTrainAccuracy = totalTrainAcc / float64(len(results))
	report.AvgTestAccuracy = totalTestAcc / float64(len(results))
	report.AvgGenGap = totalGenGap / float64(len(results))

	printGeneralizationReport(report)
}
