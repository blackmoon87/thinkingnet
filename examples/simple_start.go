package main

import (
	"fmt"
	"log"

	"github.com/blackmoon87/thinkingnet/pkg/algorithms"
	"github.com/blackmoon87/thinkingnet/pkg/core"
	"github.com/blackmoon87/thinkingnet/pkg/preprocessing"
)

func main() {
	fmt.Println("=== مرحباً بك في ThinkingNet-Go - البداية السريعة ===")
	fmt.Println("=== Welcome to ThinkingNet-Go - Quick Start ===")
	fmt.Println()

	// إنشاء بيانات تجريبية بسيطة
	// Create simple sample data
	fmt.Println("1. إنشاء البيانات / Creating Data")
	fmt.Println("--------------------------------")

	// بيانات الميزات (Features)
	بيانات_الميزات := [][]float64{
		{1.0, 2.0},
		{2.0, 3.0},
		{3.0, 4.0},
		{4.0, 5.0},
		{5.0, 6.0},
		{6.0, 7.0},
		{7.0, 8.0},
		{8.0, 9.0},
	}

	// التصنيفات (Labels) - بيانات تصنيف ثنائي
	التصنيفات := [][]float64{
		{0}, {0}, {0}, {0},
		{1}, {1}, {1}, {1},
	}

	// استخدام دالة EasyTensor لإنشاء tensors بسهولة
	// Using EasyTensor function to create tensors easily
	X := core.EasyTensor(بيانات_الميزات)
	y := core.EasyTensor(التصنيفات)

	rows, cols := X.Dims()
	fmt.Printf("تم إنشاء البيانات: %d عينة، %d ميزة\n", rows, cols)
	fmt.Printf("Data created: %d samples, %d features\n", rows, cols)
	fmt.Println()

	// تقسيم البيانات باستخدام EasySplit
	// Split data using EasySplit
	fmt.Println("2. تقسيم البيانات / Data Splitting")
	fmt.Println("----------------------------------")

	X_تدريب, X_اختبار, y_تدريب, y_اختبار, err := preprocessing.EasySplit(X, y, 0.3)
	if err != nil {
		log.Fatalf("خطأ في تقسيم البيانات / Data splitting error: %v", err)
	}

	trainRows, _ := X_تدريب.Dims()
	testRows, _ := X_اختبار.Dims()
	fmt.Printf("بيانات التدريب: %d عينة\n", trainRows)
	fmt.Printf("بيانات الاختبار: %d عينة\n", testRows)
	fmt.Printf("Training data: %d samples\n", trainRows)
	fmt.Printf("Test data: %d samples\n", testRows)
	fmt.Println()

	// تطبيع البيانات باستخدام EasyStandardScale
	// Normalize data using EasyStandardScale
	fmt.Println("3. تطبيع البيانات / Data Normalization")
	fmt.Println("-------------------------------------")

	X_تدريب_مطبع, err := preprocessing.EasyStandardScale(X_تدريب)
	if err != nil {
		log.Fatalf("خطأ في تطبيع بيانات التدريب / Training data normalization error: %v", err)
	}

	X_اختبار_مطبع, err := preprocessing.EasyStandardScale(X_اختبار)
	if err != nil {
		log.Fatalf("خطأ في تطبيع بيانات الاختبار / Test data normalization error: %v", err)
	}

	fmt.Println("تم تطبيع البيانات بنجاح")
	fmt.Println("Data normalized successfully")
	fmt.Println()

	// استخدام الانحدار اللوجستي السهل
	// Using Easy Logistic Regression
	fmt.Println("4. التصنيف باستخدام الانحدار اللوجستي / Classification with Logistic Regression")
	fmt.Println("--------------------------------------------------------------------------")

	نموذج_التصنيف := algorithms.EasyLogisticRegression()

	err = نموذج_التصنيف.Fit(X_تدريب_مطبع, y_تدريب)
	if err != nil {
		log.Fatalf("خطأ في تدريب النموذج / Model training error: %v", err)
	}

	// حساب الدقة
	دقة_التصنيف, err := نموذج_التصنيف.Score(X_اختبار_مطبع, y_اختبار)
	if err != nil {
		log.Fatalf("خطأ في حساب الدقة / Accuracy calculation error: %v", err)
	}

	fmt.Printf("دقة التصنيف: %.3f\n", دقة_التصنيف)
	fmt.Printf("Classification accuracy: %.3f\n", دقة_التصنيف)
	fmt.Println()

	// استخدام الانحدار الخطي السهل
	// Using Easy Linear Regression
	fmt.Println("5. الانحدار الخطي / Linear Regression")
	fmt.Println("------------------------------------")

	// إنشاء بيانات انحدار بسيطة
	بيانات_انحدار := [][]float64{
		{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0},
	}
	قيم_الهدف := [][]float64{
		{2.1}, {4.2}, {6.1}, {8.0}, {10.1}, {12.0},
	}

	X_انحدار := core.EasyTensor(بيانات_انحدار)
	y_انحدار := core.EasyTensor(قيم_الهدف)

	نموذج_الانحدار := algorithms.EasyLinearRegression()

	err = نموذج_الانحدار.Fit(X_انحدار, y_انحدار)
	if err != nil {
		log.Fatalf("خطأ في تدريب نموذج الانحدار / Regression model training error: %v", err)
	}

	// التنبؤ بقيم جديدة
	بيانات_جديدة := core.EasyTensor([][]float64{{7.0}, {8.0}})
	تنبؤات_الانحدار, err := نموذج_الانحدار.Predict(بيانات_جديدة)
	if err != nil {
		log.Fatalf("خطأ في تنبؤ الانحدار / Regression prediction error: %v", err)
	}

	fmt.Printf("التنبؤ للقيمة 7.0: %.2f\n", تنبؤات_الانحدار.At(0, 0))
	fmt.Printf("التنبؤ للقيمة 8.0: %.2f\n", تنبؤات_الانحدار.At(1, 0))
	fmt.Printf("Prediction for 7.0: %.2f\n", تنبؤات_الانحدار.At(0, 0))
	fmt.Printf("Prediction for 8.0: %.2f\n", تنبؤات_الانحدار.At(1, 0))
	fmt.Println()

	// استخدام التجميع السهل (K-Means)
	// Using Easy K-Means Clustering
	fmt.Println("6. التجميع باستخدام K-Means / K-Means Clustering")
	fmt.Println("-----------------------------------------------")

	// إنشاء بيانات تجميع
	بيانات_التجميع := [][]float64{
		{1.0, 1.0}, {1.5, 2.0}, {2.0, 1.0},
		{8.0, 8.0}, {8.5, 9.0}, {9.0, 8.0},
		{1.0, 8.0}, {2.0, 9.0}, {1.5, 8.5},
	}

	X_تجميع := core.EasyTensor(بيانات_التجميع)

	نموذج_التجميع := algorithms.EasyKMeans(3) // 3 مجموعات

	err = نموذج_التجميع.Fit(X_تجميع)
	if err != nil {
		log.Fatalf("خطأ في تدريب نموذج التجميع / Clustering model training error: %v", err)
	}

	// الحصول على التصنيفات
	تصنيفات_التجميع, err := نموذج_التجميع.Predict(X_تجميع)
	if err != nil {
		log.Fatalf("خطأ في تصنيف التجميع / Clustering prediction error: %v", err)
	}

	fmt.Println("تصنيفات التجميع:")
	fmt.Println("Cluster assignments:")
	for i := 0; i < len(بيانات_التجميع); i++ {
		cluster := تصنيفات_التجميع[i]
		fmt.Printf("النقطة (%.1f, %.1f) -> المجموعة %d\n",
			بيانات_التجميع[i][0], بيانات_التجميع[i][1], cluster)
		fmt.Printf("Point (%.1f, %.1f) -> Cluster %d\n",
			بيانات_التجميع[i][0], بيانات_التجميع[i][1], cluster)
	}
	fmt.Println()

	// الخلاصة
	// Summary
	fmt.Println("=== الخلاصة / Summary ===")
	fmt.Println("تم تشغيل جميع الأمثلة بنجاح!")
	fmt.Println("All examples completed successfully!")
	fmt.Println()
	fmt.Println("الدوال المساعدة المستخدمة:")
	fmt.Println("Helper functions used:")
	fmt.Println("- core.EasyTensor() - إنشاء tensors بسهولة")
	fmt.Println("- preprocessing.EasySplit() - تقسيم البيانات")
	fmt.Println("- preprocessing.EasyStandardScale() - تطبيع البيانات")
	fmt.Println("- algorithms.EasyLogisticRegression() - التصنيف")
	fmt.Println("- algorithms.EasyLinearRegression() - الانحدار")
	fmt.Println("- algorithms.EasyKMeans() - التجميع")
	fmt.Println()
	fmt.Println("=== مرحباً بك في عالم التعلم الآلي مع Go! ===")
	fmt.Println("=== Welcome to Machine Learning with Go! ===")
}
