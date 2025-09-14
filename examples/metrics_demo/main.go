package main

import (
	"fmt"

	"github.com/blackmoon87/thinkingnet/pkg/core"
	"github.com/blackmoon87/thinkingnet/pkg/metrics"
)

func main() {
	fmt.Println("ThinkingNet Metrics Demo")
	fmt.Println("========================")

	// Classification Metrics Demo
	fmt.Println("\n1. Classification Metrics")
	fmt.Println("--------------------------")

	// Sample binary classification data
	yTrueClass := core.NewTensorFromSlice([][]float64{{1}, {0}, {1}, {1}, {0}, {1}, {0}, {0}})
	yPredClass := core.NewTensorFromSlice([][]float64{{1}, {0}, {0}, {1}, {0}, {1}, {1}, {0}})
	yProbaClass := core.NewTensorFromSlice([][]float64{{0.9}, {0.1}, {0.4}, {0.8}, {0.2}, {0.95}, {0.6}, {0.15}})

	// Calculate classification metrics
	classMetrics := metrics.CalculateClassificationMetricsWithProba(yTrueClass, yPredClass, yProbaClass)
	fmt.Printf("Accuracy:  %.3f\n", classMetrics.Accuracy)
	fmt.Printf("Precision: %.3f\n", classMetrics.Precision)
	fmt.Printf("Recall:    %.3f\n", classMetrics.Recall)
	fmt.Printf("F1-Score:  %.3f\n", classMetrics.F1Score)
	fmt.Printf("ROC-AUC:   %.3f\n", classMetrics.ROCAUC)

	// Confusion Matrix
	fmt.Println("\nConfusion Matrix:")
	cm := metrics.NewConfusionMatrix(yTrueClass, yPredClass)
	fmt.Printf("Classes: %v\n", cm.Classes)
	for i, row := range cm.Matrix {
		fmt.Printf("Class %d: %v\n", cm.Classes[i], row)
	}

	// ROC Curve
	fmt.Println("\nROC Curve Analysis:")
	roc := metrics.CalculateROCCurve(yTrueClass, yProbaClass)
	fmt.Printf("AUC: %.3f\n", roc.AUC)
	fmt.Printf("ROC Points (first 5): ")
	for i := 0; i < 5 && i < len(roc.FPR); i++ {
		fmt.Printf("(%.3f, %.3f) ", roc.FPR[i], roc.TPR[i])
	}
	fmt.Println()

	// Regression Metrics Demo
	fmt.Println("\n2. Regression Metrics")
	fmt.Println("---------------------")

	// Sample regression data
	yTrueReg := core.NewTensorFromSlice([][]float64{{2.5}, {3.2}, {1.8}, {4.1}, {2.9}, {3.7}, {1.5}, {4.3}})
	yPredReg := core.NewTensorFromSlice([][]float64{{2.3}, {3.1}, {1.9}, {4.0}, {3.1}, {3.5}, {1.7}, {4.1}})

	// Calculate regression metrics
	regMetrics := metrics.CalculateRegressionMetrics(yTrueReg, yPredReg)
	fmt.Printf("MSE:      %.4f\n", regMetrics.MSE)
	fmt.Printf("RMSE:     %.4f\n", regMetrics.RMSE)
	fmt.Printf("MAE:      %.4f\n", regMetrics.MAE)
	fmt.Printf("R²:       %.4f\n", regMetrics.R2Score)
	fmt.Printf("MAPE:     %.2f%%\n", regMetrics.MAPE)
	fmt.Printf("EVS:      %.4f\n", regMetrics.EVS)

	// Residual Analysis
	fmt.Println("\nResidual Analysis:")
	residualAnalysis := metrics.CalculateResidualAnalysis(yTrueReg, yPredReg)
	fmt.Printf("Mean Residual: %.4f\n", residualAnalysis.MeanResidual)
	fmt.Printf("Std Residual:  %.4f\n", residualAnalysis.StdResidual)
	fmt.Printf("Min Residual:  %.4f\n", residualAnalysis.MinResidual)
	fmt.Printf("Max Residual:  %.4f\n", residualAnalysis.MaxResidual)
	fmt.Printf("Percentiles (25th, 50th, 75th): %.4f, %.4f, %.4f\n",
		residualAnalysis.Percentiles[0], residualAnalysis.Percentiles[1], residualAnalysis.Percentiles[2])

	// Cross-Validation Metrics Demo
	fmt.Println("\n3. Cross-Validation Metrics")
	fmt.Println("----------------------------")

	// Sample cross-validation scores
	cvScores := []float64{0.85, 0.82, 0.88, 0.79, 0.86, 0.84, 0.87, 0.81, 0.83, 0.89}
	cvMetrics := metrics.CalculateCrossValidationMetrics(cvScores, "accuracy")
	fmt.Printf("Metric: %s\n", cvMetrics.MetricName)
	fmt.Printf("Mean Score: %.4f\n", cvMetrics.MeanScore)
	fmt.Printf("Std Score:  %.4f\n", cvMetrics.StdScore)
	fmt.Printf("Fold Count: %d\n", cvMetrics.FoldCount)
	fmt.Printf("All Scores: %v\n", cvMetrics.Scores)

	// Comprehensive Reports
	fmt.Println("\n4. Comprehensive Reports")
	fmt.Println("------------------------")

	// Classification Report
	fmt.Println("\nClassification Report:")
	classReport := cm.GetClassificationReport()
	fmt.Printf("Overall Accuracy: %.3f\n", classReport["accuracy"])
	fmt.Printf("Number of Samples: %d\n", classReport["num_samples"])
	fmt.Printf("Number of Classes: %d\n", classReport["num_classes"])

	// Regression Report
	fmt.Println("\nRegression Report:")
	regReport := metrics.GenerateRegressionReport(yTrueReg, yPredReg)
	fmt.Printf("Number of Samples: %d\n", regReport.NumSamples)
	fmt.Printf("Number of Features: %d\n", regReport.NumFeatures)
	fmt.Printf("Model Performance Summary:\n")
	fmt.Printf("  - RMSE: %.4f\n", regReport.Metrics.RMSE)
	fmt.Printf("  - R²:   %.4f\n", regReport.Metrics.R2Score)
	fmt.Printf("  - MAE:  %.4f\n", regReport.Metrics.MAE)

	fmt.Println("\nDemo completed successfully!")
}
