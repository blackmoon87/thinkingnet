package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/blackmoon87/thinkingnet/pkg/algorithms"
	"github.com/blackmoon87/thinkingnet/pkg/core"
	"github.com/blackmoon87/thinkingnet/pkg/preprocessing"
)

func main() {
	fmt.Println("=== ThinkingNet-Go Easy Classification Demo ===")
	fmt.Println("مثال شامل على التصنيف باستخدام مكتبة ThinkingNet-Go")
	fmt.Println()

	// الخطوة 1: تحميل البيانات
	// Step 1: Load data
	fmt.Println("الخطوة 1: تحميل البيانات من ملف CSV")
	fmt.Println("Step 1: Loading data from CSV file")

	X, y, err := loadMoonsData("../data/moons_train.csv")
	if err != nil {
		log.Fatalf("خطأ في تحميل البيانات / Error loading data: %v", err)
	}

	fmt.Printf("تم تحميل البيانات بنجاح: %d عينة، %d خاصية\n", X.Shape()[0], X.Shape()[1])
	fmt.Printf("Data loaded successfully: %d samples, %d features\n", X.Shape()[0], X.Shape()[1])
	fmt.Println()

	// الخطوة 2: استكشاف البيانات
	// Step 2: Data exploration
	fmt.Println("الخطوة 2: استكشاف البيانات")
	fmt.Println("Step 2: Data exploration")

	exploreData(X, y)
	fmt.Println()

	// الخطوة 3: تقسيم البيانات إلى تدريب واختبار
	// Step 3: Split data into training and testing sets
	fmt.Println("الخطوة 3: تقسيم البيانات (80% تدريب، 20% اختبار)")
	fmt.Println("Step 3: Split data (80% training, 20% testing)")

	XTrain, XTest, yTrain, yTest, err := preprocessing.EasySplit(X, y, 0.2)
	if err != nil {
		log.Fatalf("خطأ في تقسيم البيانات / Error splitting data: %v", err)
	}

	fmt.Printf("بيانات التدريب: %d عينة\n", XTrain.Shape()[0])
	fmt.Printf("بيانات الاختبار: %d عينة\n", XTest.Shape()[0])
	fmt.Printf("Training data: %d samples\n", XTrain.Shape()[0])
	fmt.Printf("Testing data: %d samples\n", XTest.Shape()[0])
	fmt.Println()

	// الخطوة 4: معالجة البيانات (التطبيع)
	// Step 4: Data preprocessing (normalization)
	fmt.Println("الخطوة 4: تطبيع البيانات باستخدام StandardScaler")
	fmt.Println("Step 4: Normalize data using StandardScaler")

	XTrainScaled, err := preprocessing.EasyStandardScale(XTrain)
	if err != nil {
		log.Fatalf("خطأ في تطبيع بيانات التدريب / Error scaling training data: %v", err)
	}

	XTestScaled, err := preprocessing.EasyStandardScale(XTest)
	if err != nil {
		log.Fatalf("خطأ في تطبيع بيانات الاختبار / Error scaling test data: %v", err)
	}

	fmt.Println("تم تطبيع البيانات بنجاح")
	fmt.Println("Data normalized successfully")
	fmt.Println()

	// الخطوة 5: تدريب نموذج الانحدار اللوجستي
	// Step 5: Train Logistic Regression model
	fmt.Println("الخطوة 5: تدريب نموذج الانحدار اللوجستي")
	fmt.Println("Step 5: Training Logistic Regression model")

	// استخدام الدالة المساعدة لإنشاء النموذج
	// Use helper function to create the model
	logisticModel := algorithms.EasyLogisticRegression()

	err = logisticModel.Fit(XTrainScaled, yTrain)
	if err != nil {
		log.Fatalf("خطأ في تدريب النموذج / Error training model: %v", err)
	}

	fmt.Println("تم تدريب نموذج الانحدار اللوجستي بنجاح")
	fmt.Println("Logistic Regression model trained successfully")
	fmt.Println()

	// الخطوة 6: تقييم النموذج
	// Step 6: Model evaluation
	fmt.Println("الخطوة 6: تقييم أداء النموذج")
	fmt.Println("Step 6: Model evaluation")

	evaluateModel(logisticModel, XTrainScaled, yTrain, XTestScaled, yTest, "Logistic Regression")
	fmt.Println()

	// الخطوة 7: تدريب نموذج Random Forest للمقارنة
	// Step 7: Train Random Forest model for comparison
	fmt.Println("الخطوة 7: تدريب نموذج Random Forest للمقارنة")
	fmt.Println("Step 7: Training Random Forest model for comparison")

	// إنشاء نموذج Random Forest مع إعدادات افتراضية جيدة
	// Create Random Forest model with good default settings
	rfModel := algorithms.NewRandomForest(
		algorithms.WithNEstimators(50),
		algorithms.WithMaxDepth(10),
		algorithms.WithMinSamplesSplit(2),
		algorithms.WithMinSamplesLeaf(1),
		algorithms.WithRFRandomSeed(42),
	)

	err = rfModel.Fit(XTrainScaled, yTrain)
	if err != nil {
		log.Fatalf("خطأ في تدريب Random Forest / Error training Random Forest: %v", err)
	}

	fmt.Println("تم تدريب نموذج Random Forest بنجاح")
	fmt.Println("Random Forest model trained successfully")
	fmt.Println()

	// الخطوة 8: تقييم نموذج Random Forest
	// Step 8: Evaluate Random Forest model
	fmt.Println("الخطوة 8: تقييم أداء نموذج Random Forest")
	fmt.Println("Step 8: Random Forest model evaluation")

	evaluateModel(rfModel, XTrainScaled, yTrain, XTestScaled, yTest, "Random Forest")
	fmt.Println()

	// الخطوة 9: مقارنة النماذج وعرض أمثلة على التنبؤات
	// Step 9: Compare models and show prediction examples
	fmt.Println("الخطوة 9: مقارنة النماذج وأمثلة على التنبؤات")
	fmt.Println("Step 9: Model comparison and prediction examples")

	compareModels(logisticModel, rfModel, XTestScaled, yTest)
	fmt.Println()

	fmt.Println("=== اكتمل المثال بنجاح! ===")
	fmt.Println("=== Demo completed successfully! ===")
}

// loadMoonsData تحمل بيانات moons من ملف CSV
// loadMoonsData loads moons data from CSV file
func loadMoonsData(filename string) (core.Tensor, core.Tensor, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, fmt.Errorf("فشل في فتح الملف / Failed to open file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("فشل في قراءة CSV / Failed to read CSV: %v", err)
	}

	// تخطي الرأس (header)
	// Skip header
	if len(records) < 2 {
		return nil, nil, fmt.Errorf("ملف فارغ أو لا يحتوي على بيانات / Empty file or no data")
	}

	records = records[1:] // تخطي الرأس / Skip header

	var XData [][]float64
	var yData [][]float64

	for i, record := range records {
		if len(record) != 3 {
			return nil, nil, fmt.Errorf("صف غير صحيح في السطر %d / Invalid row at line %d", i+2)
		}

		// قراءة الخصائص (feature_1, feature_2)
		// Read features (feature_1, feature_2)
		feature1, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			return nil, nil, fmt.Errorf("خطأ في تحويل feature_1 في السطر %d / Error parsing feature_1 at line %d: %v", i+2, err)
		}

		feature2, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			return nil, nil, fmt.Errorf("خطأ في تحويل feature_2 في السطر %d / Error parsing feature_2 at line %d: %v", i+2, err)
		}

		// قراءة التصنيف (label)
		// Read label
		label, err := strconv.ParseFloat(record[2], 64)
		if err != nil {
			return nil, nil, fmt.Errorf("خطأ في تحويل label في السطر %d / Error parsing label at line %d: %v", i+2, err)
		}

		XData = append(XData, []float64{feature1, feature2})
		yData = append(yData, []float64{label})
	}

	X := core.NewTensorFromSlice(XData)
	y := core.NewTensorFromSlice(yData)

	return X, y, nil
}

// exploreData يستكشف البيانات ويعرض إحصائيات أساسية
// exploreData explores the data and shows basic statistics
func exploreData(X, y core.Tensor) {
	rows, cols := X.Dims()

	fmt.Printf("شكل البيانات: (%d, %d)\n", rows, cols)
	fmt.Printf("Data shape: (%d, %d)\n", rows, cols)

	// حساب إحصائيات الخصائص
	// Calculate feature statistics
	fmt.Println("إحصائيات الخصائص / Feature statistics:")
	for j := 0; j < cols; j++ {
		var sum, min, max float64
		min = X.At(0, j)
		max = X.At(0, j)

		for i := 0; i < rows; i++ {
			val := X.At(i, j)
			sum += val
			if val < min {
				min = val
			}
			if val > max {
				max = val
			}
		}

		mean := sum / float64(rows)
		fmt.Printf("  الخاصية %d / Feature %d: متوسط/Mean=%.3f, أدنى/Min=%.3f, أعلى/Max=%.3f\n",
			j+1, j+1, mean, min, max)
	}

	// حساب توزيع التصنيفات
	// Calculate label distribution
	labelCounts := make(map[int]int)
	yRows, _ := y.Dims()
	for i := 0; i < yRows; i++ {
		label := int(y.At(i, 0))
		labelCounts[label]++
	}

	fmt.Println("توزيع التصنيفات / Label distribution:")
	for label, count := range labelCounts {
		percentage := float64(count) / float64(yRows) * 100
		fmt.Printf("  التصنيف %d / Class %d: %d عينة / samples (%.1f%%)\n",
			label, label, count, percentage)
	}
}

// evaluateModel يقيم أداء النموذج ويعرض المقاييس
// evaluateModel evaluates model performance and shows metrics
func evaluateModel(model interface{}, XTrain, yTrain, XTest, yTest core.Tensor, modelName string) {
	fmt.Printf("تقييم نموذج %s / Evaluating %s model:\n", modelName, modelName)

	// تقييم على بيانات التدريب
	// Evaluate on training data
	var trainAccuracy, testAccuracy float64
	var err error

	switch m := model.(type) {
	case *algorithms.LogisticRegression:
		trainAccuracy, err = m.Score(XTrain, yTrain)
		if err != nil {
			log.Printf("خطأ في حساب دقة التدريب / Error calculating training accuracy: %v", err)
			return
		}

		testAccuracy, err = m.Score(XTest, yTest)
		if err != nil {
			log.Printf("خطأ في حساب دقة الاختبار / Error calculating test accuracy: %v", err)
			return
		}

		// الحصول على التنبؤات لحساب المقاييس التفصيلية
		// Get predictions for detailed metrics
		predictions, err := m.Predict(XTest)
		if err != nil {
			log.Printf("خطأ في التنبؤ / Error making predictions: %v", err)
			return
		}

		// حساب المقاييس التفصيلية
		// Calculate detailed metrics
		metrics := algorithms.CalculateClassificationMetrics(yTest, predictions, 1.0)

		fmt.Printf("  دقة التدريب / Training Accuracy: %.3f\n", trainAccuracy)
		fmt.Printf("  دقة الاختبار / Test Accuracy: %.3f\n", testAccuracy)
		fmt.Printf("  الدقة / Precision: %.3f\n", metrics.Precision)
		fmt.Printf("  الاستدعاء / Recall: %.3f\n", metrics.Recall)
		fmt.Printf("  F1-Score: %.3f\n", metrics.F1Score)

	case *algorithms.RandomForest:
		trainAccuracy, err = m.Score(XTrain, yTrain)
		if err != nil {
			log.Printf("خطأ في حساب دقة التدريب / Error calculating training accuracy: %v", err)
			return
		}

		testAccuracy, err = m.Score(XTest, yTest)
		if err != nil {
			log.Printf("خطأ في حساب دقة الاختبار / Error calculating test accuracy: %v", err)
			return
		}

		// الحصول على التنبؤات لحساب المقاييس التفصيلية
		// Get predictions for detailed metrics
		predictions, err := m.Predict(XTest)
		if err != nil {
			log.Printf("خطأ في التنبؤ / Error making predictions: %v", err)
			return
		}

		// حساب المقاييس التفصيلية
		// Calculate detailed metrics
		metrics := algorithms.CalculateClassificationMetrics(yTest, predictions, 1.0)

		fmt.Printf("  دقة التدريب / Training Accuracy: %.3f\n", trainAccuracy)
		fmt.Printf("  دقة الاختبار / Test Accuracy: %.3f\n", testAccuracy)
		fmt.Printf("  الدقة / Precision: %.3f\n", metrics.Precision)
		fmt.Printf("  الاستدعاء / Recall: %.3f\n", metrics.Recall)
		fmt.Printf("  F1-Score: %.3f\n", metrics.F1Score)
	}
}

// compareModels يقارن بين النماذج ويعرض أمثلة على التنبؤات
// compareModels compares models and shows prediction examples
func compareModels(logisticModel *algorithms.LogisticRegression, rfModel *algorithms.RandomForest, XTest, yTest core.Tensor) {
	fmt.Println("مقارنة النماذج على عينات من بيانات الاختبار:")
	fmt.Println("Model comparison on test data samples:")

	// الحصول على التنبؤات من كلا النموذجين
	// Get predictions from both models
	lrPredictions, err := logisticModel.Predict(XTest)
	if err != nil {
		log.Printf("خطأ في تنبؤات الانحدار اللوجستي / Error in logistic regression predictions: %v", err)
		return
	}

	rfPredictions, err := rfModel.Predict(XTest)
	if err != nil {
		log.Printf("خطأ في تنبؤات Random Forest / Error in Random Forest predictions: %v", err)
		return
	}

	// الحصول على الاحتماليات
	// Get probabilities
	lrProbs, err := logisticModel.PredictProba(XTest)
	if err != nil {
		log.Printf("خطأ في احتماليات الانحدار اللوجستي / Error in logistic regression probabilities: %v", err)
		return
	}

	rfProbs, err := rfModel.PredictProba(XTest)
	if err != nil {
		log.Printf("خطأ في احتماليات Random Forest / Error in Random Forest probabilities: %v", err)
		return
	}

	fmt.Println()
	fmt.Printf("%-15s %-15s %-10s %-15s %-15s %-15s %-15s\n",
		"Feature1", "Feature2", "True", "LR_Pred", "LR_Prob", "RF_Pred", "RF_Prob")
	fmt.Printf("%-15s %-15s %-10s %-15s %-15s %-15s %-15s\n",
		"الخاصية1", "الخاصية2", "الحقيقي", "تنبؤ_LR", "احتمال_LR", "تنبؤ_RF", "احتمال_RF")
	fmt.Println(strings.Repeat("-", 105))

	// عرض أول 10 عينات
	// Show first 10 samples
	rows, _ := XTest.Dims()
	maxSamples := 10
	if rows < maxSamples {
		maxSamples = rows
	}

	for i := 0; i < maxSamples; i++ {
		feature1 := XTest.At(i, 0)
		feature2 := XTest.At(i, 1)
		trueLabel := int(yTest.At(i, 0))
		lrPred := int(lrPredictions.At(i, 0))
		rfPred := int(rfPredictions.At(i, 0))
		lrProb := lrProbs.At(i, 0)
		rfProb := rfProbs.At(i, 0)

		fmt.Printf("%-15.3f %-15.3f %-10d %-15d %-15.3f %-15d %-15.3f\n",
			feature1, feature2, trueLabel, lrPred, lrProb, rfPred, rfProb)
	}

	// حساب عدد التنبؤات المتطابقة
	// Calculate agreement between models
	agreement := 0
	for i := 0; i < rows; i++ {
		if int(lrPredictions.At(i, 0)) == int(rfPredictions.At(i, 0)) {
			agreement++
		}
	}

	agreementPercentage := float64(agreement) / float64(rows) * 100
	fmt.Printf("\nاتفاق النماذج / Model Agreement: %d/%d (%.1f%%)\n",
		agreement, rows, agreementPercentage)
}
