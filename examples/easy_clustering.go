package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"strings"

	"github.com/blackmoon87/thinkingnet/pkg/algorithms"
	"github.com/blackmoon87/thinkingnet/pkg/core"
	"github.com/blackmoon87/thinkingnet/pkg/preprocessing"
)

func main() {
	fmt.Println("=== ThinkingNet-Go Easy Clustering Demo ===")
	fmt.Println("مثال شامل على التجميع باستخدام مكتبة ThinkingNet-Go")
	fmt.Println()

	// الخطوة 1: إنشاء بيانات تجميع تجريبية
	// Step 1: Create synthetic clustering data
	fmt.Println("الخطوة 1: إنشاء بيانات تجميع تجريبية")
	fmt.Println("Step 1: Creating synthetic clustering data")

	X := createClusteringData(300, 3) // 300 samples, 3 clusters

	fmt.Printf("تم إنشاء البيانات: %d عينة، %d خاصية\n", X.Shape()[0], X.Shape()[1])
	fmt.Printf("Data created: %d samples, %d features\n", X.Shape()[0], X.Shape()[1])
	fmt.Println()

	// الخطوة 2: استكشاف البيانات
	// Step 2: Data exploration
	fmt.Println("الخطوة 2: استكشاف البيانات")
	fmt.Println("Step 2: Data exploration")

	exploreClusteringData(X)
	fmt.Println()

	// الخطوة 3: معالجة البيانات (التطبيع)
	// Step 3: Data preprocessing (normalization)
	fmt.Println("الخطوة 3: تطبيع البيانات باستخدام StandardScaler")
	fmt.Println("Step 3: Normalize data using StandardScaler")

	XScaled, err := preprocessing.EasyStandardScale(X)
	if err != nil {
		log.Fatalf("خطأ في تطبيع البيانات / Error scaling data: %v", err)
	}

	fmt.Println("تم تطبيع البيانات بنجاح")
	fmt.Println("Data normalized successfully")
	fmt.Println()

	// الخطوة 4: تحديد العدد الأمثل للمجموعات باستخدام Elbow Method
	// Step 4: Determine optimal number of clusters using Elbow Method
	fmt.Println("الخطوة 4: تحديد العدد الأمثل للمجموعات باستخدام Elbow Method")
	fmt.Println("Step 4: Determine optimal number of clusters using Elbow Method")

	optimalK := findOptimalClusters(XScaled, 1, 8)
	fmt.Printf("العدد الأمثل المقترح للمجموعات: %d\n", optimalK)
	fmt.Printf("Suggested optimal number of clusters: %d\n", optimalK)
	fmt.Println()

	// الخطوة 5: تطبيق K-means باستخدام الدالة المساعدة
	// Step 5: Apply K-means using helper function
	fmt.Println("الخطوة 5: تطبيق K-means باستخدام الدالة المساعدة EasyKMeans")
	fmt.Println("Step 5: Apply K-means using EasyKMeans helper function")

	// استخدام الدالة المساعدة لإنشاء نموذج K-means
	// Use helper function to create K-means model
	kmeans := algorithms.EasyKMeans(3) // نعلم أن لدينا 3 مجموعات / We know we have 3 clusters

	err = kmeans.Fit(XScaled)
	if err != nil {
		log.Fatalf("خطأ في تدريب نموذج K-means / Error training K-means model: %v", err)
	}

	fmt.Println("تم تدريب نموذج K-means بنجاح")
	fmt.Println("K-means model trained successfully")
	fmt.Printf("عدد التكرارات: %d\n", kmeans.NIters())
	fmt.Printf("Number of iterations: %d\n", kmeans.NIters())
	fmt.Printf("القصور الذاتي النهائي: %.4f\n", kmeans.Inertia())
	fmt.Printf("Final inertia: %.4f\n", kmeans.Inertia())
	fmt.Println()

	// الخطوة 6: الحصول على التنبؤات وتحليل النتائج
	// Step 6: Get predictions and analyze results
	fmt.Println("الخطوة 6: الحصول على التنبؤات وتحليل النتائج")
	fmt.Println("Step 6: Get predictions and analyze results")

	labels, err := kmeans.Predict(XScaled)
	if err != nil {
		log.Fatalf("خطأ في التنبؤ / Error making predictions: %v", err)
	}

	analyzeClusteringResults(XScaled, labels, kmeans)
	fmt.Println()

	// الخطوة 7: مقارنة مع عدد مختلف من المجموعات
	// Step 7: Compare with different number of clusters
	fmt.Println("الخطوة 7: مقارنة مع عدد مختلف من المجموعات")
	fmt.Println("Step 7: Compare with different number of clusters")

	compareClusterNumbers(XScaled, []int{2, 3, 4, 5})
	fmt.Println()

	// الخطوة 8: تقييم جودة التجميع باستخدام المقاييس
	// Step 8: Evaluate clustering quality using metrics
	fmt.Println("الخطوة 8: تقييم جودة التجميع باستخدام المقاييس")
	fmt.Println("Step 8: Evaluate clustering quality using metrics")

	evaluateClusteringQuality(XScaled, labels)
	fmt.Println()

	// الخطوة 9: عرض أمثلة على النقاط المجمعة
	// Step 9: Show examples of clustered points
	fmt.Println("الخطوة 9: عرض أمثلة على النقاط المجمعة")
	fmt.Println("Step 9: Show examples of clustered points")

	showClusterExamples(X, XScaled, labels, kmeans)
	fmt.Println()

	// الخطوة 10: نصائح للتحسين والاستخدام المتقدم
	// Step 10: Tips for improvement and advanced usage
	fmt.Println("الخطوة 10: نصائح للتحسين والاستخدام المتقدم")
	fmt.Println("Step 10: Tips for improvement and advanced usage")

	showClusteringTips()
	fmt.Println()

	fmt.Println("=== اكتمل المثال بنجاح! ===")
	fmt.Println("=== Demo completed successfully! ===")
}

// createClusteringData ينشئ بيانات تجميع تجريبية مع مجموعات واضحة
// createClusteringData creates synthetic clustering data with clear clusters
func createClusteringData(nSamples, nClusters int) core.Tensor {
	rand.Seed(42) // للحصول على نتائج قابلة للتكرار / For reproducible results

	samplesPerCluster := nSamples / nClusters
	var data [][]float64

	// مراكز المجموعات المختلفة
	// Different cluster centers
	clusterCenters := [][]float64{
		{2.0, 2.0}, // المجموعة الأولى / First cluster
		{8.0, 8.0}, // المجموعة الثانية / Second cluster
		{2.0, 8.0}, // المجموعة الثالثة / Third cluster
		{8.0, 2.0}, // المجموعة الرابعة / Fourth cluster
		{5.0, 5.0}, // المجموعة الخامسة / Fifth cluster
	}

	for cluster := 0; cluster < nClusters && cluster < len(clusterCenters); cluster++ {
		center := clusterCenters[cluster]

		for i := 0; i < samplesPerCluster; i++ {
			// إضافة ضوضاء عشوائية حول مركز المجموعة
			// Add random noise around cluster center
			x := center[0] + rand.NormFloat64()*0.8
			y := center[1] + rand.NormFloat64()*0.8

			data = append(data, []float64{x, y})
		}
	}

	// إضافة العينات المتبقية للمجموعة الأخيرة
	// Add remaining samples to the last cluster
	remainingSamples := nSamples - len(data)
	if remainingSamples > 0 && nClusters > 0 {
		lastCenter := clusterCenters[(nClusters-1)%len(clusterCenters)]
		for i := 0; i < remainingSamples; i++ {
			x := lastCenter[0] + rand.NormFloat64()*0.8
			y := lastCenter[1] + rand.NormFloat64()*0.8
			data = append(data, []float64{x, y})
		}
	}

	return core.NewTensorFromSlice(data)
}

// exploreClusteringData يستكشف بيانات التجميع ويعرض إحصائيات أساسية
// exploreClusteringData explores clustering data and shows basic statistics
func exploreClusteringData(X core.Tensor) {
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

		// حساب الانحراف المعياري
		// Calculate standard deviation
		var variance float64
		for i := 0; i < rows; i++ {
			diff := X.At(i, j) - mean
			variance += diff * diff
		}
		std := math.Sqrt(variance / float64(rows))

		fmt.Printf("  الخاصية %d / Feature %d: متوسط/Mean=%.3f, انحراف/Std=%.3f, أدنى/Min=%.3f, أعلى/Max=%.3f\n",
			j+1, j+1, mean, std, min, max)
	}

	// حساب المسافة بين أقرب وأبعد النقاط
	// Calculate distance between closest and farthest points
	minDist, maxDist := calculateDistanceRange(X)
	fmt.Printf("نطاق المسافات / Distance range: أدنى/Min=%.3f, أعلى/Max=%.3f\n", minDist, maxDist)
}

// calculateDistanceRange يحسب أدنى وأعلى مسافة بين النقاط
// calculateDistanceRange calculates minimum and maximum distance between points
func calculateDistanceRange(X core.Tensor) (float64, float64) {
	rows, _ := X.Dims()
	if rows < 2 {
		return 0, 0
	}

	minDist := math.Inf(1)
	maxDist := 0.0

	// عينة من النقاط لتجنب الحساب المكثف
	// Sample points to avoid intensive computation
	sampleSize := 100
	if rows < sampleSize {
		sampleSize = rows
	}

	for i := 0; i < sampleSize; i++ {
		for j := i + 1; j < sampleSize; j++ {
			dist := euclideanDistance(X, i, j)
			if dist < minDist {
				minDist = dist
			}
			if dist > maxDist {
				maxDist = dist
			}
		}
	}

	return minDist, maxDist
}

// euclideanDistance يحسب المسافة الإقليدية بين نقطتين
// euclideanDistance calculates Euclidean distance between two points
func euclideanDistance(X core.Tensor, i, j int) float64 {
	_, cols := X.Dims()
	sum := 0.0
	for k := 0; k < cols; k++ {
		diff := X.At(i, k) - X.At(j, k)
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// findOptimalClusters يجد العدد الأمثل للمجموعات باستخدام Elbow Method
// findOptimalClusters finds optimal number of clusters using Elbow Method
func findOptimalClusters(X core.Tensor, minK, maxK int) int {
	fmt.Println("تجربة أعداد مختلفة من المجموعات:")
	fmt.Println("Testing different numbers of clusters:")
	fmt.Printf("%-10s %-15s %-15s %-15s\n", "K", "Inertia", "Improvement", "Ratio")
	fmt.Printf("%-10s %-15s %-15s %-15s\n", "عدد_المجموعات", "القصور", "التحسن", "النسبة")
	fmt.Println(strings.Repeat("-", 60))

	var inertias []float64
	var improvements []float64

	for k := minK; k <= maxK; k++ {
		kmeans := algorithms.EasyKMeans(k)
		err := kmeans.Fit(X)
		if err != nil {
			log.Printf("خطأ في K=%d / Error at K=%d: %v", k, k, err)
			continue
		}

		inertia := kmeans.Inertia()
		inertias = append(inertias, inertia)

		var improvement, ratio float64
		if len(inertias) > 1 {
			improvement = inertias[len(inertias)-2] - inertia
			if inertia > 0 {
				ratio = improvement / inertia
			}
		}
		improvements = append(improvements, improvement)

		fmt.Printf("%-10d %-15.4f %-15.4f %-15.4f\n", k, inertia, improvement, ratio)
	}

	// العثور على "الكوع" - النقطة التي يقل فيها التحسن بشكل كبير
	// Find the "elbow" - point where improvement decreases significantly
	optimalK := minK
	if len(improvements) > 2 {
		maxImprovement := 0.0
		for i := 1; i < len(improvements)-1; i++ {
			if improvements[i] > maxImprovement {
				maxImprovement = improvements[i]
				optimalK = minK + i
			}
		}
	}

	return optimalK
}

// analyzeClusteringResults يحلل نتائج التجميع ويعرض الإحصائيات
// analyzeClusteringResults analyzes clustering results and shows statistics
func analyzeClusteringResults(X core.Tensor, labels []int, kmeans *algorithms.KMeans) {
	// حساب توزيع النقاط على المجموعات
	// Calculate point distribution across clusters
	clusterCounts := make(map[int]int)
	for _, label := range labels {
		clusterCounts[label]++
	}

	fmt.Println("توزيع النقاط على المجموعات / Point distribution across clusters:")
	totalPoints := len(labels)
	for cluster := 0; cluster < len(clusterCounts); cluster++ {
		count := clusterCounts[cluster]
		percentage := float64(count) / float64(totalPoints) * 100
		fmt.Printf("  المجموعة %d / Cluster %d: %d نقطة / points (%.1f%%)\n",
			cluster, cluster, count, percentage)
	}

	// عرض مراكز المجموعات
	// Show cluster centers
	centers := kmeans.ClusterCenters()
	if centers != nil {
		fmt.Println("\nمراكز المجموعات / Cluster centers:")
		rows, cols := centers.Dims()
		for i := 0; i < rows; i++ {
			fmt.Printf("  المجموعة %d / Cluster %d: (", i, i)
			for j := 0; j < cols; j++ {
				fmt.Printf("%.3f", centers.At(i, j))
				if j < cols-1 {
					fmt.Print(", ")
				}
			}
			fmt.Println(")")
		}
	}

	// حساب المسافات داخل المجموعات
	// Calculate within-cluster distances
	fmt.Println("\nإحصائيات المسافات داخل المجموعات / Within-cluster distance statistics:")
	calculateWithinClusterStats(X, labels, clusterCounts)
}

// calculateWithinClusterStats يحسب إحصائيات المسافات داخل كل مجموعة
// calculateWithinClusterStats calculates distance statistics within each cluster
func calculateWithinClusterStats(X core.Tensor, labels []int, clusterCounts map[int]int) {
	rows, _ := X.Dims()

	for cluster := 0; cluster < len(clusterCounts); cluster++ {
		var clusterPoints []int
		for i := 0; i < rows; i++ {
			if labels[i] == cluster {
				clusterPoints = append(clusterPoints, i)
			}
		}

		if len(clusterPoints) < 2 {
			fmt.Printf("  المجموعة %d / Cluster %d: نقطة واحدة فقط / Only one point\n", cluster, cluster)
			continue
		}

		// حساب متوسط المسافات داخل المجموعة
		// Calculate average within-cluster distances
		var totalDistance float64
		var pairCount int

		for i := 0; i < len(clusterPoints); i++ {
			for j := i + 1; j < len(clusterPoints); j++ {
				dist := euclideanDistance(X, clusterPoints[i], clusterPoints[j])
				totalDistance += dist
				pairCount++
			}
		}

		avgDistance := totalDistance / float64(pairCount)
		fmt.Printf("  المجموعة %d / Cluster %d: متوسط المسافة الداخلية / Avg internal distance = %.3f\n",
			cluster, cluster, avgDistance)
	}
}

// compareClusterNumbers يقارن بين أعداد مختلفة من المجموعات
// compareClusterNumbers compares different numbers of clusters
func compareClusterNumbers(X core.Tensor, kValues []int) {
	fmt.Printf("%-10s %-15s %-15s %-15s %-15s\n",
		"K", "Inertia", "Iterations", "Silhouette", "CH_Score")
	fmt.Printf("%-10s %-15s %-15s %-15s %-15s\n",
		"عدد_المجموعات", "القصور", "التكرارات", "الصورة_الظلية", "نقاط_CH")
	fmt.Println(strings.Repeat("-", 75))

	for _, k := range kValues {
		kmeans := algorithms.EasyKMeans(k)
		err := kmeans.Fit(X)
		if err != nil {
			log.Printf("خطأ في K=%d / Error at K=%d: %v", k, k, err)
			continue
		}

		labels, err := kmeans.Predict(X)
		if err != nil {
			log.Printf("خطأ في التنبؤ لـ K=%d / Prediction error for K=%d: %v", k, k, err)
			continue
		}

		// حساب المقاييس
		// Calculate metrics
		inertia := kmeans.Inertia()
		iterations := kmeans.NIters()

		// حساب Silhouette Score
		// Calculate Silhouette Score
		metrics := algorithms.NewClusteringMetrics()
		silhouette, err := metrics.SilhouetteScore(X, labels)
		if err != nil {
			silhouette = -999 // قيمة تشير للخطأ / Error indicator value
		}

		// حساب Calinski-Harabasz Score
		// Calculate Calinski-Harabasz Score
		chScore, err := metrics.CalinskiHarabaszScore(X, labels)
		if err != nil {
			chScore = -999 // قيمة تشير للخطأ / Error indicator value
		}

		fmt.Printf("%-10d %-15.4f %-15d %-15.4f %-15.4f\n",
			k, inertia, iterations, silhouette, chScore)
	}
}

// evaluateClusteringQuality يقيم جودة التجميع باستخدام مقاييس مختلفة
// evaluateClusteringQuality evaluates clustering quality using different metrics
func evaluateClusteringQuality(X core.Tensor, labels []int) {
	metrics := algorithms.NewClusteringMetrics()

	fmt.Println("مقاييس جودة التجميع / Clustering quality metrics:")

	// Silhouette Score
	silhouette, err := metrics.SilhouetteScore(X, labels)
	if err != nil {
		fmt.Printf("خطأ في حساب Silhouette Score / Error calculating Silhouette Score: %v\n", err)
	} else {
		fmt.Printf("Silhouette Score: %.4f (النطاق: -1 إلى 1، الأعلى أفضل / Range: -1 to 1, higher is better)\n", silhouette)
		interpretSilhouetteScore(silhouette)
	}

	// Calinski-Harabasz Score
	chScore, err := metrics.CalinskiHarabaszScore(X, labels)
	if err != nil {
		fmt.Printf("خطأ في حساب Calinski-Harabasz Score / Error calculating Calinski-Harabasz Score: %v\n", err)
	} else {
		fmt.Printf("Calinski-Harabasz Score: %.4f (الأعلى أفضل / Higher is better)\n", chScore)
		interpretCHScore(chScore)
	}

	// Davies-Bouldin Score
	dbScore, err := metrics.DaviesBouldinScore(X, labels)
	if err != nil {
		fmt.Printf("خطأ في حساب Davies-Bouldin Score / Error calculating Davies-Bouldin Score: %v\n", err)
	} else {
		fmt.Printf("Davies-Bouldin Score: %.4f (الأقل أفضل / Lower is better)\n", dbScore)
		interpretDBScore(dbScore)
	}

	// Inertia
	inertia, err := metrics.Inertia(X, labels)
	if err != nil {
		fmt.Printf("خطأ في حساب Inertia / Error calculating Inertia: %v\n", err)
	} else {
		fmt.Printf("Inertia (WCSS): %.4f (الأقل أفضل لنفس عدد المجموعات / Lower is better for same K)\n", inertia)
	}
}

// interpretSilhouetteScore يفسر قيمة Silhouette Score
// interpretSilhouetteScore interprets Silhouette Score value
func interpretSilhouetteScore(score float64) {
	if score > 0.7 {
		fmt.Println("  تفسير / Interpretation: تجميع قوي وواضح / Strong, clear clustering")
	} else if score > 0.5 {
		fmt.Println("  تفسير / Interpretation: تجميع معقول / Reasonable clustering")
	} else if score > 0.25 {
		fmt.Println("  تفسير / Interpretation: تجميع ضعيف، قد تتداخل المجموعات / Weak clustering, clusters may overlap")
	} else {
		fmt.Println("  تفسير / Interpretation: تجميع ضعيف جداً أو غير مناسب / Very weak or inappropriate clustering")
	}
}

// interpretCHScore يفسر قيمة Calinski-Harabasz Score
// interpretCHScore interprets Calinski-Harabasz Score value
func interpretCHScore(score float64) {
	if score > 100 {
		fmt.Println("  تفسير / Interpretation: تجميع جيد جداً / Very good clustering")
	} else if score > 50 {
		fmt.Println("  تفسير / Interpretation: تجميع جيد / Good clustering")
	} else if score > 20 {
		fmt.Println("  تفسير / Interpretation: تجميع متوسط / Average clustering")
	} else {
		fmt.Println("  تفسير / Interpretation: تجميع ضعيف / Poor clustering")
	}
}

// interpretDBScore يفسر قيمة Davies-Bouldin Score
// interpretDBScore interprets Davies-Bouldin Score value
func interpretDBScore(score float64) {
	if score < 0.5 {
		fmt.Println("  تفسير / Interpretation: تجميع ممتاز / Excellent clustering")
	} else if score < 1.0 {
		fmt.Println("  تفسير / Interpretation: تجميع جيد / Good clustering")
	} else if score < 1.5 {
		fmt.Println("  تفسير / Interpretation: تجميع متوسط / Average clustering")
	} else {
		fmt.Println("  تفسير / Interpretation: تجميع ضعيف / Poor clustering")
	}
}

// showClusterExamples يعرض أمثلة على النقاط في كل مجموعة
// showClusterExamples shows examples of points in each cluster
func showClusterExamples(XOriginal, XScaled core.Tensor, labels []int, kmeans *algorithms.KMeans) {
	rows, _ := XOriginal.Dims()

	// تجميع النقاط حسب المجموعة
	// Group points by cluster
	clusterPoints := make(map[int][]int)
	for i := 0; i < rows; i++ {
		cluster := labels[i]
		clusterPoints[cluster] = append(clusterPoints[cluster], i)
	}

	fmt.Printf("%-10s %-15s %-15s %-15s %-15s %-15s\n",
		"Cluster", "Original_X", "Original_Y", "Scaled_X", "Scaled_Y", "Distance_to_Center")
	fmt.Printf("%-10s %-15s %-15s %-15s %-15s %-15s\n",
		"المجموعة", "الأصلي_X", "الأصلي_Y", "المطبع_X", "المطبع_Y", "المسافة_للمركز")
	fmt.Println(strings.Repeat("-", 90))

	centers := kmeans.ClusterCenters()

	for cluster := 0; cluster < len(clusterPoints); cluster++ {
		points := clusterPoints[cluster]

		// عرض أول 3 نقاط من كل مجموعة
		// Show first 3 points from each cluster
		maxExamples := 3
		if len(points) < maxExamples {
			maxExamples = len(points)
		}

		for i := 0; i < maxExamples; i++ {
			pointIdx := points[i]

			origX := XOriginal.At(pointIdx, 0)
			origY := XOriginal.At(pointIdx, 1)
			scaledX := XScaled.At(pointIdx, 0)
			scaledY := XScaled.At(pointIdx, 1)

			// حساب المسافة إلى مركز المجموعة
			// Calculate distance to cluster center
			var distToCenter float64
			if centers != nil {
				centerX := centers.At(cluster, 0)
				centerY := centers.At(cluster, 1)
				distToCenter = math.Sqrt((scaledX-centerX)*(scaledX-centerX) + (scaledY-centerY)*(scaledY-centerY))
			}

			fmt.Printf("%-10d %-15.3f %-15.3f %-15.3f %-15.3f %-15.3f\n",
				cluster, origX, origY, scaledX, scaledY, distToCenter)
		}

		if len(points) > maxExamples {
			fmt.Printf("%-10s ... و %d نقطة أخرى / ... and %d more points\n", "", len(points)-maxExamples, len(points)-maxExamples)
		}
		fmt.Println()
	}
}

// showClusteringTips يعرض نصائح لتحسين التجميع والاستخدام المتقدم
// showClusteringTips shows tips for improving clustering and advanced usage
func showClusteringTips() {
	fmt.Println("نصائح لتحسين التجميع والاستخدام المتقدم:")
	fmt.Println("Tips for improving clustering and advanced usage:")
	fmt.Println()

	fmt.Println("1. اختيار عدد المجموعات / Choosing Number of Clusters:")
	fmt.Println("   - استخدم Elbow Method لإيجاد العدد الأمثل / Use Elbow Method to find optimal number")
	fmt.Println("   - جرب Silhouette Analysis للتحقق من الجودة / Try Silhouette Analysis to verify quality")
	fmt.Println("   - فكر في السياق التجاري للمشكلة / Consider business context of the problem")
	fmt.Println("   - استخدم Gap Statistic للتحليل الإحصائي / Use Gap Statistic for statistical analysis")
	fmt.Println()

	fmt.Println("2. معالجة البيانات / Data Preprocessing:")
	fmt.Println("   - طبع البيانات دائماً قبل التجميع / Always normalize data before clustering")
	fmt.Println("   - تعامل مع القيم المفقودة / Handle missing values")
	fmt.Println("   - أزل القيم الشاذة إذا لزم الأمر / Remove outliers if necessary")
	fmt.Println("   - فكر في تقليل الأبعاد للبيانات عالية الأبعاد / Consider dimensionality reduction for high-dimensional data")
	fmt.Println()

	fmt.Println("3. تحسين خوارزمية K-means / Improving K-means Algorithm:")
	fmt.Println("   - جرب طرق تهيئة مختلفة (kmeans++, random) / Try different initialization methods")
	fmt.Println("   - اضبط عدد التكرارات القصوى / Adjust maximum iterations")
	fmt.Println("   - غير قيمة التسامح للتقارب / Change tolerance for convergence")
	fmt.Println("   - استخدم بذور عشوائية مختلفة وقارن النتائج / Use different random seeds and compare results")
	fmt.Println()

	fmt.Println("4. خوارزميات تجميع بديلة / Alternative Clustering Algorithms:")
	fmt.Println("   - DBSCAN للمجموعات ذات الكثافة المتغيرة / DBSCAN for density-based clustering")
	fmt.Println("   - Hierarchical Clustering للتجميع الهرمي / Hierarchical Clustering for hierarchical structure")
	fmt.Println("   - Gaussian Mixture Models للتوزيعات المعقدة / GMM for complex distributions")
	fmt.Println("   - Mean Shift للمجموعات غير الكروية / Mean Shift for non-spherical clusters")
	fmt.Println()

	fmt.Println("5. تقييم وتفسير النتائج / Evaluation and Interpretation:")
	fmt.Println("   - استخدم مقاييس متعددة للتقييم / Use multiple metrics for evaluation")
	fmt.Println("   - ارسم البيانات إذا كانت ثنائية الأبعاد / Visualize data if 2D")
	fmt.Println("   - احسب إحصائيات كل مجموعة / Calculate statistics for each cluster")
	fmt.Println("   - تحقق من التوزيع المتوازن للمجموعات / Check for balanced cluster distribution")
	fmt.Println()

	fmt.Println("6. الاستخدام العملي / Practical Usage:")
	fmt.Println("   - احفظ النموذج المدرب لاستخدامه لاحقاً / Save trained model for later use")
	fmt.Println("   - اختبر على بيانات جديدة / Test on new data")
	fmt.Println("   - راقب أداء النموذج مع الوقت / Monitor model performance over time")
	fmt.Println("   - وثق المعايير المستخدمة / Document the criteria used")
	fmt.Println()

	fmt.Println("مثال على كود متقدم:")
	fmt.Println("Example of advanced code:")
	fmt.Println(`
// تجربة إعدادات مختلفة لـ K-means
// Try different K-means settings
func experimentWithKMeans(X core.Tensor) {
    // تجربة طرق تهيئة مختلفة
    // Try different initialization methods
    for _, initMethod := range []string{"random", "kmeans++"} {
        kmeans := algorithms.NewKMeans(3,
            algorithms.WithInitMethod(initMethod),
            algorithms.WithMaxIters(500),
            algorithms.WithTolerance(1e-6),
            algorithms.WithRandomSeed(42))
        
        err := kmeans.Fit(X)
        if err == nil {
            fmt.Printf("طريقة التهيئة %s: القصور = %.4f\n", 
                initMethod, kmeans.Inertia())
        }
    }
}

// مقارنة خوارزميات مختلفة
// Compare different algorithms
func compareAlgorithms(X core.Tensor) {
    // K-means
    kmeans := algorithms.EasyKMeans(3)
    kmeans.Fit(X)
    
    // DBSCAN
    dbscan := algorithms.NewDBSCAN(1.0, 5)
    dbscanLabels, _ := dbscan.FitPredict(X)
    
    // قارن النتائج / Compare results
    fmt.Printf("K-means clusters: 3\n")
    fmt.Printf("DBSCAN clusters: %d\n", dbscan.NClusters())
}`)
}
