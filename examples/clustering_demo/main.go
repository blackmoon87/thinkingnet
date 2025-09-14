package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/algorithms"
	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func main() {
	fmt.Println("ThinkingNet Clustering Algorithms Demo")
	fmt.Println("=====================================")

	// Generate sample data with clear clusters
	data := generateClusterData()
	X := core.NewTensorFromSlice(data)

	fmt.Printf("Generated %d data points with 2 features\n\n", len(data))

	// Demonstrate K-means clustering
	demonstrateKMeans(X)

	// Demonstrate DBSCAN clustering
	demonstrateDBSCAN(X)

	// Demonstrate clustering metrics
	demonstrateClusteringMetrics(X)
}

// generateClusterData creates sample data with 3 distinct clusters
func generateClusterData() [][]float64 {
	rand.Seed(42)
	data := make([][]float64, 0, 150)

	// Cluster 1: around (2, 2)
	for i := 0; i < 50; i++ {
		x := 2.0 + rand.NormFloat64()*0.5
		y := 2.0 + rand.NormFloat64()*0.5
		data = append(data, []float64{x, y})
	}

	// Cluster 2: around (8, 8)
	for i := 0; i < 50; i++ {
		x := 8.0 + rand.NormFloat64()*0.5
		y := 8.0 + rand.NormFloat64()*0.5
		data = append(data, []float64{x, y})
	}

	// Cluster 3: around (2, 8)
	for i := 0; i < 50; i++ {
		x := 2.0 + rand.NormFloat64()*0.5
		y := 8.0 + rand.NormFloat64()*0.5
		data = append(data, []float64{x, y})
	}

	return data
}

// demonstrateKMeans shows K-means clustering in action
func demonstrateKMeans(X core.Tensor) {
	fmt.Println("K-Means Clustering Demo")
	fmt.Println("-----------------------")

	// Create K-means clusterer with k=3
	kmeans := algorithms.NewKMeans(3,
		algorithms.WithRandomSeed(42),
		algorithms.WithMaxIters(100),
		algorithms.WithTolerance(1e-4),
		algorithms.WithInitMethod("kmeans++"),
	)

	// Measure training time
	start := time.Now()
	err := kmeans.Fit(X)
	duration := time.Since(start)

	if err != nil {
		fmt.Printf("Error fitting K-means: %v\n", err)
		return
	}

	fmt.Printf("Training completed in %v\n", duration)
	fmt.Printf("Number of iterations: %d\n", kmeans.NIters())
	fmt.Printf("Final inertia: %.4f\n", kmeans.Inertia())

	// Get cluster centers
	centers := kmeans.ClusterCenters()
	if centers != nil {
		fmt.Println("Cluster centers:")
		rows, cols := centers.Dims()
		for i := 0; i < rows; i++ {
			fmt.Printf("  Cluster %d: (", i)
			for j := 0; j < cols; j++ {
				fmt.Printf("%.2f", centers.At(i, j))
				if j < cols-1 {
					fmt.Print(", ")
				}
			}
			fmt.Println(")")
		}
	}

	// Predict cluster labels
	labels, err := kmeans.Predict(X)
	if err != nil {
		fmt.Printf("Error predicting: %v\n", err)
		return
	}

	// Count points in each cluster
	clusterCounts := make(map[int]int)
	for _, label := range labels {
		clusterCounts[label]++
	}

	fmt.Println("Cluster assignments:")
	for cluster, count := range clusterCounts {
		fmt.Printf("  Cluster %d: %d points\n", cluster, count)
	}

	fmt.Println()
}

// demonstrateDBSCAN shows DBSCAN clustering in action
func demonstrateDBSCAN(X core.Tensor) {
	fmt.Println("DBSCAN Clustering Demo")
	fmt.Println("----------------------")

	// Create DBSCAN clusterer
	dbscan := algorithms.NewDBSCAN(1.0, 5)

	// Measure training time
	start := time.Now()
	labels, err := dbscan.FitPredict(X)
	duration := time.Since(start)

	if err != nil {
		fmt.Printf("Error fitting DBSCAN: %v\n", err)
		return
	}

	fmt.Printf("Training completed in %v\n", duration)
	fmt.Printf("Number of clusters found: %d\n", dbscan.NClusters())
	fmt.Printf("Number of noise points: %d\n", dbscan.NNoise())
	fmt.Printf("Number of core points: %d\n", len(dbscan.CoreIndices()))

	// Count points in each cluster
	clusterCounts := make(map[int]int)
	for _, label := range labels {
		clusterCounts[label]++
	}

	fmt.Println("Cluster assignments:")
	for cluster, count := range clusterCounts {
		if cluster == -1 {
			fmt.Printf("  Noise: %d points\n", count)
		} else {
			fmt.Printf("  Cluster %d: %d points\n", cluster, count)
		}
	}

	fmt.Println()
}

// demonstrateClusteringMetrics shows how to evaluate clustering results
func demonstrateClusteringMetrics(X core.Tensor) {
	fmt.Println("Clustering Metrics Demo")
	fmt.Println("-----------------------")

	// Get clustering results from K-means
	kmeans := algorithms.NewKMeans(3, algorithms.WithRandomSeed(42))
	err := kmeans.Fit(X)
	if err != nil {
		fmt.Printf("Error fitting K-means for metrics: %v\n", err)
		return
	}

	labels, err := kmeans.Predict(X)
	if err != nil {
		fmt.Printf("Error predicting for metrics: %v\n", err)
		return
	}

	// Create metrics calculator
	metrics := algorithms.NewClusteringMetrics()

	// Calculate various metrics
	silhouette, err := metrics.SilhouetteScore(X, labels)
	if err != nil {
		fmt.Printf("Error calculating silhouette score: %v\n", err)
	} else {
		fmt.Printf("Silhouette Score: %.4f (higher is better, range: -1 to 1)\n", silhouette)
	}

	ch, err := metrics.CalinskiHarabaszScore(X, labels)
	if err != nil {
		fmt.Printf("Error calculating Calinski-Harabasz score: %v\n", err)
	} else {
		fmt.Printf("Calinski-Harabasz Score: %.4f (higher is better)\n", ch)
	}

	db, err := metrics.DaviesBouldinScore(X, labels)
	if err != nil {
		fmt.Printf("Error calculating Davies-Bouldin score: %v\n", err)
	} else {
		fmt.Printf("Davies-Bouldin Score: %.4f (lower is better)\n", db)
	}

	inertia, err := metrics.Inertia(X, labels)
	if err != nil {
		fmt.Printf("Error calculating inertia: %v\n", err)
	} else {
		fmt.Printf("Inertia (WCSS): %.4f (lower is better for same k)\n", inertia)
	}

	fmt.Println()
	fmt.Println("Demo completed successfully!")
}
