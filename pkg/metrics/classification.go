package metrics

import (
	"fmt"
	"math"
	"sort"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// ClassificationMetrics holds comprehensive classification evaluation metrics.
type ClassificationMetrics struct {
	Accuracy  float64 `json:"accuracy"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1Score   float64 `json:"f1_score"`
	ROCAUC    float64 `json:"roc_auc,omitempty"`
}

// ConfusionMatrix represents a confusion matrix for classification evaluation.
type ConfusionMatrix struct {
	Matrix     [][]int `json:"matrix"`
	Classes    []int   `json:"classes"`
	TruePos    []int   `json:"true_positives"`
	FalsePos   []int   `json:"false_positives"`
	TrueNeg    []int   `json:"true_negatives"`
	FalseNeg   []int   `json:"false_negatives"`
	NumClasses int     `json:"num_classes"`
	NumSamples int     `json:"num_samples"`
}

// ROCCurve represents ROC curve data for binary classification.
type ROCCurve struct {
	FPR       []float64 `json:"fpr"`        // False Positive Rate
	TPR       []float64 `json:"tpr"`        // True Positive Rate (Sensitivity)
	AUC       float64   `json:"auc"`        // Area Under Curve
	Threshold []float64 `json:"thresholds"` // Decision thresholds
}

// CalculateAccuracy computes the accuracy score.
func CalculateAccuracy(yTrue, yPred core.Tensor) float64 {
	nSamples, _ := yTrue.Dims()
	if nSamples == 0 {
		return 0.0
	}

	correct := 0
	for i := range nSamples {
		if math.Abs(yTrue.At(i, 0)-yPred.At(i, 0)) < core.GetEpsilon() {
			correct++
		}
	}

	return float64(correct) / float64(nSamples)
}

// CalculatePrecision computes precision for binary classification.
func CalculatePrecision(yTrue, yPred core.Tensor) float64 {
	cm := NewConfusionMatrix(yTrue, yPred)
	if len(cm.TruePos) == 0 {
		return 0.0
	}

	// For binary classification, use positive class (class 1)
	if cm.NumClasses == 2 {
		tp := float64(cm.TruePos[1])
		fp := float64(cm.FalsePos[1])
		if tp+fp == 0 {
			return 0.0
		}
		return tp / (tp + fp)
	}

	// For multiclass, calculate macro-averaged precision
	var totalPrecision float64
	validClasses := 0

	for i := range cm.NumClasses {
		tp := float64(cm.TruePos[i])
		fp := float64(cm.FalsePos[i])
		if tp+fp > 0 {
			totalPrecision += tp / (tp + fp)
			validClasses++
		}
	}

	if validClasses == 0 {
		return 0.0
	}

	return totalPrecision / float64(validClasses)
}

// CalculateRecall computes recall (sensitivity) for binary classification.
func CalculateRecall(yTrue, yPred core.Tensor) float64 {
	cm := NewConfusionMatrix(yTrue, yPred)
	if len(cm.TruePos) == 0 {
		return 0.0
	}

	// For binary classification, use positive class (class 1)
	if cm.NumClasses == 2 {
		tp := float64(cm.TruePos[1])
		fn := float64(cm.FalseNeg[1])
		if tp+fn == 0 {
			return 0.0
		}
		return tp / (tp + fn)
	}

	// For multiclass, calculate macro-averaged recall
	var totalRecall float64
	validClasses := 0

	for i := range cm.NumClasses {
		tp := float64(cm.TruePos[i])
		fn := float64(cm.FalseNeg[i])
		if tp+fn > 0 {
			totalRecall += tp / (tp + fn)
			validClasses++
		}
	}

	if validClasses == 0 {
		return 0.0
	}

	return totalRecall / float64(validClasses)
}

// CalculateF1Score computes the F1 score.
func CalculateF1Score(yTrue, yPred core.Tensor) float64 {
	precision := CalculatePrecision(yTrue, yPred)
	recall := CalculateRecall(yTrue, yPred)

	if precision+recall == 0 {
		return 0.0
	}

	return 2 * (precision * recall) / (precision + recall)
}

// CalculateClassificationMetrics computes all classification metrics.
func CalculateClassificationMetrics(yTrue, yPred core.Tensor) *ClassificationMetrics {
	return &ClassificationMetrics{
		Accuracy:  CalculateAccuracy(yTrue, yPred),
		Precision: CalculatePrecision(yTrue, yPred),
		Recall:    CalculateRecall(yTrue, yPred),
		F1Score:   CalculateF1Score(yTrue, yPred),
	}
}

// CalculateClassificationMetricsWithProba computes classification metrics including ROC-AUC.
func CalculateClassificationMetricsWithProba(yTrue, yPred, yProba core.Tensor) *ClassificationMetrics {
	metrics := CalculateClassificationMetrics(yTrue, yPred)

	// Calculate ROC-AUC for binary classification
	if isBinaryClassification(yTrue) {
		roc := CalculateROCCurve(yTrue, yProba)
		metrics.ROCAUC = roc.AUC
	}

	return metrics
}

// NewConfusionMatrix creates a confusion matrix from true and predicted labels.
func NewConfusionMatrix(yTrue, yPred core.Tensor) *ConfusionMatrix {
	nSamples, _ := yTrue.Dims()
	if nSamples == 0 {
		return &ConfusionMatrix{}
	}

	// Find unique classes
	classSet := make(map[int]bool)
	for i := range nSamples {
		classSet[int(yTrue.At(i, 0))] = true
		classSet[int(yPred.At(i, 0))] = true
	}

	// Convert to sorted slice
	classes := make([]int, 0, len(classSet))
	for class := range classSet {
		classes = append(classes, class)
	}
	sort.Ints(classes)

	numClasses := len(classes)
	classToIndex := make(map[int]int)
	for i, class := range classes {
		classToIndex[class] = i
	}

	// Initialize confusion matrix
	matrix := make([][]int, numClasses)
	for i := range matrix {
		matrix[i] = make([]int, numClasses)
	}

	// Fill confusion matrix
	for i := range nSamples {
		trueClass := int(yTrue.At(i, 0))
		predClass := int(yPred.At(i, 0))
		trueIdx := classToIndex[trueClass]
		predIdx := classToIndex[predClass]
		matrix[trueIdx][predIdx]++
	}

	// Calculate per-class metrics
	truePos := make([]int, numClasses)
	falsePos := make([]int, numClasses)
	trueNeg := make([]int, numClasses)
	falseNeg := make([]int, numClasses)

	for i := range numClasses {
		for j := range numClasses {
			if i == j {
				truePos[i] = matrix[i][j]
			} else {
				falsePos[j] += matrix[i][j] // Predicted as j but actually i
				falseNeg[i] += matrix[i][j] // Actually i but predicted as j
			}
		}

		// Calculate true negatives
		for k := range numClasses {
			for l := range numClasses {
				if k != i && l != i {
					trueNeg[i] += matrix[k][l]
				}
			}
		}
	}

	return &ConfusionMatrix{
		Matrix:     matrix,
		Classes:    classes,
		TruePos:    truePos,
		FalsePos:   falsePos,
		TrueNeg:    trueNeg,
		FalseNeg:   falseNeg,
		NumClasses: numClasses,
		NumSamples: nSamples,
	}
}

// CalculateROCCurve computes ROC curve for binary classification.
func CalculateROCCurve(yTrue, yProba core.Tensor) *ROCCurve {
	nSamples, _ := yTrue.Dims()
	if nSamples == 0 {
		return &ROCCurve{}
	}

	// Validate binary classification
	if !isBinaryClassification(yTrue) {
		return &ROCCurve{}
	}

	// Create pairs of (probability, true_label) and sort by probability descending
	type probLabel struct {
		prob  float64
		label int
	}

	pairs := make([]probLabel, nSamples)
	for i := range nSamples {
		pairs[i] = probLabel{
			prob:  yProba.At(i, 0),
			label: int(yTrue.At(i, 0)),
		}
	}

	// Sort by probability descending
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].prob > pairs[j].prob
	})

	// Count total positives and negatives
	totalPos := 0
	totalNeg := 0
	for _, pair := range pairs {
		if pair.label == 1 {
			totalPos++
		} else {
			totalNeg++
		}
	}

	if totalPos == 0 || totalNeg == 0 {
		return &ROCCurve{AUC: 0.0}
	}

	// Calculate ROC points
	var fpr, tpr, thresholds []float64
	var tp, fp int

	// Add initial point (0, 0)
	fpr = append(fpr, 0.0)
	tpr = append(tpr, 0.0)
	thresholds = append(thresholds, math.Inf(1))

	// Process each threshold
	for i, pair := range pairs {
		if pair.label == 1 {
			tp++
		} else {
			fp++
		}

		// Add point when threshold changes or at the end
		if i == len(pairs)-1 || pairs[i+1].prob != pair.prob {
			fprVal := float64(fp) / float64(totalNeg)
			tprVal := float64(tp) / float64(totalPos)

			fpr = append(fpr, fprVal)
			tpr = append(tpr, tprVal)
			thresholds = append(thresholds, pair.prob)
		}
	}

	// Calculate AUC using trapezoidal rule
	auc := 0.0
	for i := 1; i < len(fpr); i++ {
		auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2.0
	}

	return &ROCCurve{
		FPR:       fpr,
		TPR:       tpr,
		AUC:       auc,
		Threshold: thresholds,
	}
}

// isBinaryClassification checks if the problem is binary classification.
func isBinaryClassification(yTrue core.Tensor) bool {
	nSamples, _ := yTrue.Dims()
	classSet := make(map[int]bool)

	for i := range nSamples {
		classSet[int(yTrue.At(i, 0))] = true
		if len(classSet) > 2 {
			return false
		}
	}

	return len(classSet) == 2
}

// GetClassificationReport returns a detailed classification report.
func (cm *ConfusionMatrix) GetClassificationReport() map[string]interface{} {
	report := make(map[string]interface{})

	// Overall metrics
	report["accuracy"] = cm.GetAccuracy()
	report["num_samples"] = cm.NumSamples
	report["num_classes"] = cm.NumClasses

	// Per-class metrics
	classMetrics := make(map[string]interface{})
	for i, class := range cm.Classes {
		classMetrics[fmt.Sprintf("class_%d", class)] = map[string]float64{
			"precision": cm.GetPrecision(i),
			"recall":    cm.GetRecall(i),
			"f1_score":  cm.GetF1Score(i),
			"support":   float64(cm.TruePos[i] + cm.FalseNeg[i]),
		}
	}
	report["per_class"] = classMetrics

	// Macro averages
	report["macro_avg"] = map[string]float64{
		"precision": cm.GetMacroPrecision(),
		"recall":    cm.GetMacroRecall(),
		"f1_score":  cm.GetMacroF1Score(),
	}

	return report
}

// GetAccuracy returns the overall accuracy from the confusion matrix.
func (cm *ConfusionMatrix) GetAccuracy() float64 {
	if cm.NumSamples == 0 {
		return 0.0
	}

	correct := 0
	for i := range cm.NumClasses {
		correct += cm.TruePos[i]
	}

	return float64(correct) / float64(cm.NumSamples)
}

// GetPrecision returns precision for a specific class.
func (cm *ConfusionMatrix) GetPrecision(classIdx int) float64 {
	if classIdx >= len(cm.TruePos) {
		return 0.0
	}

	tp := float64(cm.TruePos[classIdx])
	fp := float64(cm.FalsePos[classIdx])

	if tp+fp == 0 {
		return 0.0
	}

	return tp / (tp + fp)
}

// GetRecall returns recall for a specific class.
func (cm *ConfusionMatrix) GetRecall(classIdx int) float64 {
	if classIdx >= len(cm.TruePos) {
		return 0.0
	}

	tp := float64(cm.TruePos[classIdx])
	fn := float64(cm.FalseNeg[classIdx])

	if tp+fn == 0 {
		return 0.0
	}

	return tp / (tp + fn)
}

// GetF1Score returns F1 score for a specific class.
func (cm *ConfusionMatrix) GetF1Score(classIdx int) float64 {
	precision := cm.GetPrecision(classIdx)
	recall := cm.GetRecall(classIdx)

	if precision+recall == 0 {
		return 0.0
	}

	return 2 * (precision * recall) / (precision + recall)
}

// GetMacroPrecision returns macro-averaged precision.
func (cm *ConfusionMatrix) GetMacroPrecision() float64 {
	var total float64
	validClasses := 0

	for i := range cm.NumClasses {
		precision := cm.GetPrecision(i)
		if !math.IsNaN(precision) {
			total += precision
			validClasses++
		}
	}

	if validClasses == 0 {
		return 0.0
	}

	return total / float64(validClasses)
}

// GetMacroRecall returns macro-averaged recall.
func (cm *ConfusionMatrix) GetMacroRecall() float64 {
	var total float64
	validClasses := 0

	for i := range cm.NumClasses {
		recall := cm.GetRecall(i)
		if !math.IsNaN(recall) {
			total += recall
			validClasses++
		}
	}

	if validClasses == 0 {
		return 0.0
	}

	return total / float64(validClasses)
}

// GetMacroF1Score returns macro-averaged F1 score.
func (cm *ConfusionMatrix) GetMacroF1Score() float64 {
	var total float64
	validClasses := 0

	for i := range cm.NumClasses {
		f1 := cm.GetF1Score(i)
		if !math.IsNaN(f1) {
			total += f1
			validClasses++
		}
	}

	if validClasses == 0 {
		return 0.0
	}

	return total / float64(validClasses)
}
