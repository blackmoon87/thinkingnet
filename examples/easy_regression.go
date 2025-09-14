package main

import (
	"fmt"
	"log"
	"math"
	"strings"

	"github.com/blackmoon87/thinkingnet/pkg/algorithms"
	"github.com/blackmoon87/thinkingnet/pkg/core"
	"github.com/blackmoon87/thinkingnet/pkg/preprocessing"
)

func main() {
	fmt.Println("=== ThinkingNet-Go Easy Regression Demo ===")
	fmt.Println("مثال شامل على الانحدار باستخدام مكتبة ThinkingNet-Go")
	fmt.Println()

	// الخطوة 1: إنشاء بيانات انحدار تجريبية
	// Step 1: Create synthetic regression data
	fmt.Println("الخطوة 1: إنشاء بيانات انحدار تجريبية")
	fmt.Println("Step 1: Creating synthetic regression data")

	X, y := createRegressionData(200, 2, 0.1) // 200 samples, 2 features, noise level 0.1

	fmt.Printf("تم إنشاء البيانات: %d عينة، %d خاصية\n", X.Shape()[0], X.Shape()[1])
	fmt.Printf("Data created: %d samples, %d features\n", X.Shape()[0], X.Shape()[1])
	fmt.Println()

	// الخطوة 2: استكشاف البيانات
	// Step 2: Data exploration
	fmt.Println("الخطوة 2: استكشاف البيانات")
	fmt.Println("Step 2: Data exploration")

	exploreRegressionData(X, y)
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

	// الخطوة 5: تدريب نموذج الانحدار الخطي البسيط
	// Step 5: Train simple Linear Regression model
	fmt.Println("الخطوة 5: تدريب نموذج الانحدار الخطي البسيط")
	fmt.Println("Step 5: Training simple Linear Regression model")

	// استخدام الدالة المساعدة لإنشاء النموذج
	// Use helper function to create the model
	linearModel := algorithms.EasyLinearRegression()

	err = linearModel.Fit(XTrainScaled, yTrain)
	if err != nil {
		log.Fatalf("خطأ في تدريب النموذج / Error training model: %v", err)
	}

	fmt.Println("تم تدريب نموذج الانحدار الخطي بنجاح")
	fmt.Println("Linear Regression model trained successfully")
	fmt.Println()

	// الخطوة 6: تقييم النموذج البسيط
	// Step 6: Evaluate simple model
	fmt.Println("الخطوة 6: تقييم أداء النموذج البسيط")
	fmt.Println("Step 6: Simple model evaluation")

	evaluateRegressionModel(linearModel, XTrainScaled, yTrain, XTestScaled, yTest, "Linear Regression (Simple)")
	fmt.Println()

	// الخطوة 7: تدريب نموذج انحدار خطي مع تنظيم L2
	// Step 7: Train Linear Regression with L2 regularization
	fmt.Println("الخطوة 7: تدريب نموذج انحدار خطي مع تنظيم L2")
	fmt.Println("Step 7: Training Linear Regression with L2 regularization")

	// إنشاء نموذج مع تنظيم L2
	// Create model with L2 regularization
	l2Model := algorithms.NewLinearRegression(
		algorithms.WithLinearLearningRate(0.01),
		algorithms.WithLinearMaxIterations(1000),
		algorithms.WithLinearTolerance(1e-6),
		algorithms.WithLinearRegularization("l2", 0.1),
		algorithms.WithLinearFitIntercept(true),
	)

	err = l2Model.Fit(XTrainScaled, yTrain)
	if err != nil {
		log.Fatalf("خطأ في تدريب نموذج L2 / Error training L2 model: %v", err)
	}

	fmt.Println("تم تدريب نموذج الانحدار مع تنظيم L2 بنجاح")
	fmt.Println("L2 regularized Linear Regression model trained successfully")
	fmt.Println()

	// الخطوة 8: تقييم نموذج L2
	// Step 8: Evaluate L2 model
	fmt.Println("الخطوة 8: تقييم أداء نموذج L2")
	fmt.Println("Step 8: L2 model evaluation")

	evaluateRegressionModel(l2Model, XTrainScaled, yTrain, XTestScaled, yTest, "Linear Regression (L2)")
	fmt.Println()

	// الخطوة 9: تدريب نموذج انحدار خطي مع تنظيم Elastic Net
	// Step 9: Train Linear Regression with Elastic Net regularization
	fmt.Println("الخطوة 9: تدريب نموذج انحدار خطي مع تنظيم Elastic Net")
	fmt.Println("Step 9: Training Linear Regression with Elastic Net regularization")

	// إنشاء نموذج مع تنظيم Elastic Net
	// Create model with Elastic Net regularization
	elasticModel := algorithms.NewLinearRegression(
		algorithms.WithLinearLearningRate(0.01),
		algorithms.WithLinearMaxIterations(1000),
		algorithms.WithLinearTolerance(1e-6),
		algorithms.WithLinearElasticNet(0.1, 0.5), // lambda=0.1, l1_ratio=0.5
		algorithms.WithLinearFitIntercept(true),
	)

	err = elasticModel.Fit(XTrainScaled, yTrain)
	if err != nil {
		log.Fatalf("خطأ في تدريب نموذج Elastic Net / Error training Elastic Net model: %v", err)
	}

	fmt.Println("تم تدريب نموذج الانحدار مع تنظيم Elastic Net بنجاح")
	fmt.Println("Elastic Net Linear Regression model trained successfully")
	fmt.Println()

	// الخطوة 10: تقييم نموذج Elastic Net
	// Step 10: Evaluate Elastic Net model
	fmt.Println("الخطوة 10: تقييم أداء نموذج Elastic Net")
	fmt.Println("Step 10: Elastic Net model evaluation")

	evaluateRegressionModel(elasticModel, XTrainScaled, yTrain, XTestScaled, yTest, "Linear Regression (Elastic Net)")
	fmt.Println()

	// الخطوة 11: مقارنة النماذج وعرض أمثلة على التنبؤات
	// Step 11: Compare models and show prediction examples
	fmt.Println("الخطوة 11: مقارنة النماذج وأمثلة على التنبؤات")
	fmt.Println("Step 11: Model comparison and prediction examples")

	compareRegressionModels(linearModel, l2Model, elasticModel, XTestScaled, yTest)
	fmt.Println()

	// الخطوة 12: نصائح للتحسين والتصور
	// Step 12: Tips for improvement and visualization
	fmt.Println("الخطوة 12: نصائح للتحسين والتصور")
	fmt.Println("Step 12: Tips for improvement and visualization")

	showImprovementTips()
	fmt.Println()

	fmt.Println("=== اكتمل المثال بنجاح! ===")
	fmt.Println("=== Demo completed successfully! ===")
}

// createRegressionData ينشئ بيانات انحدار تجريبية
// createRegressionData creates synthetic regression data
func createRegressionData(nSamples, nFeatures int, noiseLevel float64) (core.Tensor, core.Tensor) {
	// إنشاء بيانات الميزات العشوائية
	// Create random feature data
	XData := make([][]float64, nSamples)
	yData := make([][]float64, nSamples)

	// معاملات حقيقية للانحدار
	// True regression coefficients
	trueCoeffs := []float64{2.5, -1.8} // للميزتين
	intercept := 1.2

	for i := 0; i < nSamples; i++ {
		features := make([]float64, nFeatures)

		// إنشاء ميزات عشوائية
		// Generate random features
		for j := 0; j < nFeatures; j++ {
			features[j] = (float64(i%100)/50.0 - 1.0) + 0.5*math.Sin(float64(i)*0.1) // نمط منتظم مع تنوع
		}

		// حساب القيمة المستهدفة
		// Calculate target value
		target := intercept
		for j := 0; j < nFeatures; j++ {
			target += trueCoeffs[j] * features[j]
		}

		// إضافة ضوضاء
		// Add noise
		noise := (math.Sin(float64(i)*0.05) + math.Cos(float64(i)*0.03)) * noiseLevel
		target += noise

		XData[i] = features
		yData[i] = []float64{target}
	}

	X := core.NewTensorFromSlice(XData)
	y := core.NewTensorFromSlice(yData)

	return X, y
}

// exploreRegressionData يستكشف بيانات الانحدار ويعرض إحصائيات أساسية
// exploreRegressionData explores regression data and shows basic statistics
func exploreRegressionData(X, y core.Tensor) {
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

	// حساب إحصائيات المتغير المستهدف
	// Calculate target variable statistics
	var ySum, yMin, yMax float64
	yRows, _ := y.Dims()
	yMin = y.At(0, 0)
	yMax = y.At(0, 0)

	for i := 0; i < yRows; i++ {
		val := y.At(i, 0)
		ySum += val
		if val < yMin {
			yMin = val
		}
		if val > yMax {
			yMax = val
		}
	}

	yMean := ySum / float64(yRows)
	fmt.Printf("المتغير المستهدف / Target variable: متوسط/Mean=%.3f, أدنى/Min=%.3f, أعلى/Max=%.3f\n",
		yMean, yMin, yMax)

	// حساب الانحراف المعياري للمتغير المستهدف
	// Calculate standard deviation of target variable
	var yVariance float64
	for i := 0; i < yRows; i++ {
		diff := y.At(i, 0) - yMean
		yVariance += diff * diff
	}
	yStd := math.Sqrt(yVariance / float64(yRows))
	fmt.Printf("الانحراف المعياري للمتغير المستهدف / Target std: %.3f\n", yStd)
}

// evaluateRegressionModel يقيم أداء نموذج الانحدار ويعرض المقاييس
// evaluateRegressionModel evaluates regression model performance and shows metrics
func evaluateRegressionModel(model *algorithms.LinearRegression, XTrain, yTrain, XTest, yTest core.Tensor, modelName string) {
	fmt.Printf("تقييم نموذج %s / Evaluating %s model:\n", modelName, modelName)

	// تقييم على بيانات التدريب
	// Evaluate on training data
	trainR2, err := model.Score(XTrain, yTrain)
	if err != nil {
		log.Printf("خطأ في حساب R² للتدريب / Error calculating training R²: %v", err)
		return
	}

	// تقييم على بيانات الاختبار
	// Evaluate on test data
	testR2, err := model.Score(XTest, yTest)
	if err != nil {
		log.Printf("خطأ في حساب R² للاختبار / Error calculating test R²: %v", err)
		return
	}

	// الحصول على التنبؤات لحساب المقاييس التفصيلية
	// Get predictions for detailed metrics
	trainPredictions, err := model.Predict(XTrain)
	if err != nil {
		log.Printf("خطأ في تنبؤات التدريب / Error making training predictions: %v", err)
		return
	}

	testPredictions, err := model.Predict(XTest)
	if err != nil {
		log.Printf("خطأ في تنبؤات الاختبار / Error making test predictions: %v", err)
		return
	}

	// حساب المقاييس التفصيلية
	// Calculate detailed metrics
	trainMetrics := algorithms.CalculateRegressionMetrics(yTrain, trainPredictions)
	testMetrics := algorithms.CalculateRegressionMetrics(yTest, testPredictions)

	fmt.Printf("  أداء التدريب / Training Performance:\n")
	fmt.Printf("    R² Score: %.4f\n", trainR2)
	fmt.Printf("    MSE: %.4f\n", trainMetrics.MSE)
	fmt.Printf("    RMSE: %.4f\n", trainMetrics.RMSE)
	fmt.Printf("    MAE: %.4f\n", trainMetrics.MAE)

	fmt.Printf("  أداء الاختبار / Test Performance:\n")
	fmt.Printf("    R² Score: %.4f\n", testR2)
	fmt.Printf("    MSE: %.4f\n", testMetrics.MSE)
	fmt.Printf("    RMSE: %.4f\n", testMetrics.RMSE)
	fmt.Printf("    MAE: %.4f\n", testMetrics.MAE)

	// عرض معاملات النموذج
	// Show model coefficients
	weights := model.Weights()
	if weights != nil {
		fmt.Printf("  معاملات النموذج / Model Coefficients:\n")
		weightsRows, _ := weights.Dims()
		fmt.Printf("    المقطع / Intercept: %.4f\n", weights.At(0, 0))
		for i := 1; i < weightsRows; i++ {
			fmt.Printf("    الميزة %d / Feature %d: %.4f\n", i, i, weights.At(i, 0))
		}
	}

	// تحليل الإفراط في التدريب
	// Analyze overfitting
	overfit := trainR2 - testR2
	if overfit > 0.1 {
		fmt.Printf("  ⚠️  تحذير: قد يكون هناك إفراط في التدريب (فرق R²: %.4f)\n", overfit)
		fmt.Printf("  ⚠️  Warning: Possible overfitting (R² difference: %.4f)\n", overfit)
	} else if overfit < -0.05 {
		fmt.Printf("  ℹ️  ملاحظة: النموذج قد يحتاج لمزيد من التدريب (فرق R²: %.4f)\n", overfit)
		fmt.Printf("  ℹ️  Note: Model might need more training (R² difference: %.4f)\n", overfit)
	} else {
		fmt.Printf("  ✅ النموذج متوازن جيداً (فرق R²: %.4f)\n", overfit)
		fmt.Printf("  ✅ Model is well balanced (R² difference: %.4f)\n", overfit)
	}
}

// compareRegressionModels يقارن بين نماذج الانحدار ويعرض أمثلة على التنبؤات
// compareRegressionModels compares regression models and shows prediction examples
func compareRegressionModels(simpleModel, l2Model, elasticModel *algorithms.LinearRegression, XTest, yTest core.Tensor) {
	fmt.Println("مقارنة النماذج على عينات من بيانات الاختبار:")
	fmt.Println("Model comparison on test data samples:")

	// الحصول على التنبؤات من جميع النماذج
	// Get predictions from all models
	simplePreds, err := simpleModel.Predict(XTest)
	if err != nil {
		log.Printf("خطأ في تنبؤات النموذج البسيط / Error in simple model predictions: %v", err)
		return
	}

	l2Preds, err := l2Model.Predict(XTest)
	if err != nil {
		log.Printf("خطأ في تنبؤات نموذج L2 / Error in L2 model predictions: %v", err)
		return
	}

	elasticPreds, err := elasticModel.Predict(XTest)
	if err != nil {
		log.Printf("خطأ في تنبؤات نموذج Elastic Net / Error in Elastic Net model predictions: %v", err)
		return
	}

	fmt.Println()
	fmt.Printf("%-10s %-10s %-12s %-12s %-12s %-12s %-12s %-12s\n",
		"Feature1", "Feature2", "True", "Simple", "L2", "Elastic", "Simple_Err", "L2_Err")
	fmt.Printf("%-10s %-10s %-12s %-12s %-12s %-12s %-12s %-12s\n",
		"الخاصية1", "الخاصية2", "الحقيقي", "البسيط", "L2", "Elastic", "خطأ_البسيط", "خطأ_L2")
	fmt.Println(strings.Repeat("-", 100))

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
		trueVal := yTest.At(i, 0)
		simplePred := simplePreds.At(i, 0)
		l2Pred := l2Preds.At(i, 0)
		elasticPred := elasticPreds.At(i, 0)

		simpleErr := math.Abs(trueVal - simplePred)
		l2Err := math.Abs(trueVal - l2Pred)

		fmt.Printf("%-10.3f %-10.3f %-12.3f %-12.3f %-12.3f %-12.3f %-12.3f %-12.3f\n",
			feature1, feature2, trueVal, simplePred, l2Pred, elasticPred, simpleErr, l2Err)
	}

	// حساب المقاييس الإجمالية للمقارنة
	// Calculate overall metrics for comparison
	fmt.Println("\nمقارنة المقاييس الإجمالية / Overall Metrics Comparison:")
	fmt.Println(strings.Repeat("-", 60))

	simpleMetrics := algorithms.CalculateRegressionMetrics(yTest, simplePreds)
	l2Metrics := algorithms.CalculateRegressionMetrics(yTest, l2Preds)
	elasticMetrics := algorithms.CalculateRegressionMetrics(yTest, elasticPreds)

	fmt.Printf("%-20s %-12s %-12s %-12s\n", "Metric", "Simple", "L2", "Elastic")
	fmt.Printf("%-20s %-12s %-12s %-12s\n", "المقياس", "البسيط", "L2", "Elastic")
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("%-20s %-12.4f %-12.4f %-12.4f\n", "R² Score", simpleMetrics.R2Score, l2Metrics.R2Score, elasticMetrics.R2Score)
	fmt.Printf("%-20s %-12.4f %-12.4f %-12.4f\n", "MSE", simpleMetrics.MSE, l2Metrics.MSE, elasticMetrics.MSE)
	fmt.Printf("%-20s %-12.4f %-12.4f %-12.4f\n", "RMSE", simpleMetrics.RMSE, l2Metrics.RMSE, elasticMetrics.RMSE)
	fmt.Printf("%-20s %-12.4f %-12.4f %-12.4f\n", "MAE", simpleMetrics.MAE, l2Metrics.MAE, elasticMetrics.MAE)

	// تحديد أفضل نموذج
	// Determine best model
	bestR2 := simpleMetrics.R2Score
	bestModel := "Simple"

	if l2Metrics.R2Score > bestR2 {
		bestR2 = l2Metrics.R2Score
		bestModel = "L2"
	}

	if elasticMetrics.R2Score > bestR2 {
		bestR2 = elasticMetrics.R2Score
		bestModel = "Elastic Net"
	}

	fmt.Printf("\n🏆 أفضل نموذج / Best Model: %s (R² = %.4f)\n", bestModel, bestR2)
}

// showImprovementTips يعرض نصائح لتحسين الأداء والتصور
// showImprovementTips shows tips for performance improvement and visualization
func showImprovementTips() {
	fmt.Println("نصائح لتحسين نماذج الانحدار:")
	fmt.Println("Tips for improving regression models:")
	fmt.Println()

	fmt.Println("1. تحسين البيانات / Data Improvement:")
	fmt.Println("   - جمع المزيد من البيانات / Collect more data")
	fmt.Println("   - إضافة ميزات جديدة ذات صلة / Add relevant new features")
	fmt.Println("   - معالجة القيم المفقودة / Handle missing values")
	fmt.Println("   - إزالة القيم الشاذة / Remove outliers")
	fmt.Println()

	fmt.Println("2. هندسة الميزات / Feature Engineering:")
	fmt.Println("   - إنشاء ميزات تفاعلية / Create interaction features")
	fmt.Println("   - تحويل الميزات (log, sqrt, polynomial) / Transform features")
	fmt.Println("   - تطبيع الميزات / Normalize features")
	fmt.Println("   - اختيار الميزات المهمة / Select important features")
	fmt.Println()

	fmt.Println("3. ضبط النموذج / Model Tuning:")
	fmt.Println("   - تجربة قيم مختلفة لمعامل التعلم / Try different learning rates")
	fmt.Println("   - ضبط قوة التنظيم / Tune regularization strength")
	fmt.Println("   - زيادة عدد التكرارات / Increase iterations")
	fmt.Println("   - تجربة أنواع تنظيم مختلفة / Try different regularization types")
	fmt.Println()

	fmt.Println("4. التحقق من صحة النموذج / Model Validation:")
	fmt.Println("   - استخدام التحقق المتقاطع / Use cross-validation")
	fmt.Println("   - تقسيم البيانات إلى تدريب/تحقق/اختبار / Split into train/val/test")
	fmt.Println("   - مراقبة منحنيات التعلم / Monitor learning curves")
	fmt.Println("   - تحليل البواقي / Analyze residuals")
	fmt.Println()

	fmt.Println("5. التصور والتحليل / Visualization & Analysis:")
	fmt.Println("   - رسم البيانات الأصلية / Plot original data")
	fmt.Println("   - رسم التنبؤات مقابل القيم الحقيقية / Plot predictions vs actual")
	fmt.Println("   - رسم البواقي / Plot residuals")
	fmt.Println("   - تحليل توزيع الأخطاء / Analyze error distribution")
	fmt.Println()

	fmt.Println("6. نماذج متقدمة / Advanced Models:")
	fmt.Println("   - تجربة Random Forest للانحدار / Try Random Forest regression")
	fmt.Println("   - استخدام الشبكات العصبية / Use neural networks")
	fmt.Println("   - تجربة نماذج التجميع / Try ensemble models")
	fmt.Println("   - استخدام نماذج غير خطية / Use non-linear models")
	fmt.Println()

	fmt.Println("مثال على كود التصور (يتطلب مكتبة رسم):")
	fmt.Println("Example visualization code (requires plotting library):")
	fmt.Println(`
// رسم التنبؤات مقابل القيم الحقيقية
// Plot predictions vs actual values
func plotPredictionsVsActual(yTrue, yPred core.Tensor) {
    // استخدم مكتبة مثل gonum/plot أو go-echarts
    // Use a library like gonum/plot or go-echarts
    
    // إنشاء scatter plot
    // Create scatter plot
    // x-axis: القيم الحقيقية / True values
    // y-axis: التنبؤات / Predictions
    // خط مثالي: y = x / Perfect line: y = x
}

// رسم البواقي
// Plot residuals
func plotResiduals(yTrue, yPred core.Tensor) {
    // حساب البواقي / Calculate residuals
    // residuals = yTrue - yPred
    
    // رسم البواقي مقابل التنبؤات
    // Plot residuals vs predictions
    // يجب أن تكون عشوائية حول الصفر
    // Should be random around zero
}`)
}
