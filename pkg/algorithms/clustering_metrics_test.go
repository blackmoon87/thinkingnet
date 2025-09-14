package algorithms

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestClusteringMetrics_SilhouetteScore(t *testing.T) {
	// Create test data with clear clusters
	data := [][]float64{
		{0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2}, // Cluster 0
		{5.0, 5.0}, {5.1, 5.1}, {5.2, 5.2}, // Cluster 1
	}
	X := core.NewTensorFromSlice(data)
	labels := []int{0, 0, 0, 1, 1, 1}

	metrics := NewClusteringMetrics()
	score, err := metrics.SilhouetteScore(X, labels)
	if err != nil {
		t.Fatalf("Failed to calculate silhouette score: %v", err)
	}

	// Should be positive for well-separated clusters
	if score <= 0 {
		t.Errorf("Expected positive silhouette score for well-separated clusters, got %f", score)
	}

	// Should be between -1 and 1
	if score < -1 || score > 1 {
		t.Errorf("Silhouette score should be between -1 and 1, got %f", score)
	}
}

func TestClusteringMetrics_SilhouetteScore_WithNoise(t *testing.T) {
	// Test with noise points (label -1)
	data := [][]float64{
		{0.0, 0.0}, {0.1, 0.1}, // Cluster 0
		{5.0, 5.0}, {5.1, 5.1}, // Cluster 1
		{10.0, 10.0}, // Noise
	}
	X := core.NewTensorFromSlice(data)
	labels := []int{0, 0, 1, 1, -1}

	metrics := NewClusteringMetrics()
	score, err := metrics.SilhouetteScore(X, labels)
	if err != nil {
		t.Fatalf("Failed to calculate silhouette score with noise: %v", err)
	}

	// Should still be valid (noise points are excluded)
	if score < -1 || score > 1 {
		t.Errorf("Silhouette score should be between -1 and 1, got %f", score)
	}
}

func TestClusteringMetrics_CalinskiHarabaszScore(t *testing.T) {
	// Create test data with clear clusters
	data := [][]float64{
		{0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2}, // Cluster 0
		{5.0, 5.0}, {5.1, 5.1}, {5.2, 5.2}, // Cluster 1
	}
	X := core.NewTensorFromSlice(data)
	labels := []int{0, 0, 0, 1, 1, 1}

	metrics := NewClusteringMetrics()
	score, err := metrics.CalinskiHarabaszScore(X, labels)
	if err != nil {
		t.Fatalf("Failed to calculate Calinski-Harabasz score: %v", err)
	}

	// Should be positive for well-separated clusters
	if score <= 0 {
		t.Errorf("Expected positive Calinski-Harabasz score, got %f", score)
	}

	// Should be finite
	if math.IsNaN(score) || math.IsInf(score, 0) {
		t.Errorf("Calinski-Harabasz score should be finite, got %f", score)
	}
}

func TestClusteringMetrics_DaviesBouldinScore(t *testing.T) {
	// Create test data with clear clusters
	data := [][]float64{
		{0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2}, // Cluster 0
		{5.0, 5.0}, {5.1, 5.1}, {5.2, 5.2}, // Cluster 1
	}
	X := core.NewTensorFromSlice(data)
	labels := []int{0, 0, 0, 1, 1, 1}

	metrics := NewClusteringMetrics()
	score, err := metrics.DaviesBouldinScore(X, labels)
	if err != nil {
		t.Fatalf("Failed to calculate Davies-Bouldin score: %v", err)
	}

	// Should be positive
	if score < 0 {
		t.Errorf("Expected non-negative Davies-Bouldin score, got %f", score)
	}

	// Should be finite
	if math.IsNaN(score) || math.IsInf(score, 0) {
		t.Errorf("Davies-Bouldin score should be finite, got %f", score)
	}
}

func TestClusteringMetrics_Inertia(t *testing.T) {
	// Create test data
	data := [][]float64{
		{0.0, 0.0}, {1.0, 1.0}, // Cluster 0
		{5.0, 5.0}, {6.0, 6.0}, // Cluster 1
	}
	X := core.NewTensorFromSlice(data)
	labels := []int{0, 0, 1, 1}

	metrics := NewClusteringMetrics()
	inertia, err := metrics.Inertia(X, labels)
	if err != nil {
		t.Fatalf("Failed to calculate inertia: %v", err)
	}

	// Should be non-negative
	if inertia < 0 {
		t.Errorf("Expected non-negative inertia, got %f", inertia)
	}

	// Should be finite
	if math.IsNaN(inertia) || math.IsInf(inertia, 0) {
		t.Errorf("Inertia should be finite, got %f", inertia)
	}
}

func TestClusteringMetrics_EdgeCases(t *testing.T) {
	metrics := NewClusteringMetrics()

	// Test with nil input
	_, err := metrics.SilhouetteScore(nil, []int{0, 1})
	if err == nil {
		t.Error("Expected error for nil input")
	}

	// Test with empty data - skip this test as NewZerosTensor(0,0) panics
	// This is expected behavior as empty tensors are not valid

	// Test with mismatched dimensions
	data := [][]float64{{1.0, 1.0}, {2.0, 2.0}}
	X := core.NewTensorFromSlice(data)
	_, err = metrics.SilhouetteScore(X, []int{0}) // Wrong number of labels
	if err == nil {
		t.Error("Expected error for mismatched dimensions")
	}

	// Test with single cluster
	_, err = metrics.SilhouetteScore(X, []int{0, 0})
	if err == nil {
		t.Error("Expected error for single cluster in silhouette score")
	}

	// Test with too few samples
	singlePoint := [][]float64{{1.0, 1.0}}
	singleX := core.NewTensorFromSlice(singlePoint)
	_, err = metrics.SilhouetteScore(singleX, []int{0})
	if err == nil {
		t.Error("Expected error for too few samples")
	}
}

func TestClusteringMetrics_SingleClusterEdgeCases(t *testing.T) {
	data := [][]float64{
		{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0},
	}
	X := core.NewTensorFromSlice(data)
	labels := []int{0, 0, 0} // All same cluster

	metrics := NewClusteringMetrics()

	// Silhouette score should error for single cluster
	_, err := metrics.SilhouetteScore(X, labels)
	if err == nil {
		t.Error("Expected error for single cluster in silhouette score")
	}

	// Calinski-Harabasz should error for single cluster
	_, err = metrics.CalinskiHarabaszScore(X, labels)
	if err == nil {
		t.Error("Expected error for single cluster in Calinski-Harabasz score")
	}

	// Davies-Bouldin should error for single cluster
	_, err = metrics.DaviesBouldinScore(X, labels)
	if err == nil {
		t.Error("Expected error for single cluster in Davies-Bouldin score")
	}

	// Inertia should work for single cluster
	inertia, err := metrics.Inertia(X, labels)
	if err != nil {
		t.Fatalf("Inertia should work for single cluster: %v", err)
	}

	if inertia < 0 {
		t.Errorf("Expected non-negative inertia, got %f", inertia)
	}
}

func TestClusteringMetrics_AllNoise(t *testing.T) {
	data := [][]float64{
		{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0},
	}
	X := core.NewTensorFromSlice(data)
	labels := []int{-1, -1, -1} // All noise

	metrics := NewClusteringMetrics()

	// Should error when all points are noise
	_, err := metrics.SilhouetteScore(X, labels)
	if err == nil {
		t.Error("Expected error when all points are noise")
	}

	_, err = metrics.CalinskiHarabaszScore(X, labels)
	if err == nil {
		t.Error("Expected error when all points are noise")
	}

	_, err = metrics.DaviesBouldinScore(X, labels)
	if err == nil {
		t.Error("Expected error when all points are noise")
	}

	_, err = metrics.Inertia(X, labels)
	if err != nil {
		t.Errorf("Inertia should handle all noise points: %v", err)
	}
}

func TestClusteringMetrics_CompareGoodVsBadClustering(t *testing.T) {
	// Create data with clear clusters
	data := [][]float64{
		{0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2}, // Should be cluster 0
		{5.0, 5.0}, {5.1, 5.1}, {5.2, 5.2}, // Should be cluster 1
	}
	X := core.NewTensorFromSlice(data)

	goodLabels := []int{0, 0, 0, 1, 1, 1} // Correct clustering
	badLabels := []int{0, 1, 0, 1, 0, 1}  // Poor clustering

	metrics := NewClusteringMetrics()

	// Silhouette score: higher is better
	goodSilhouette, err := metrics.SilhouetteScore(X, goodLabels)
	if err != nil {
		t.Fatalf("Failed to calculate good silhouette: %v", err)
	}

	badSilhouette, err := metrics.SilhouetteScore(X, badLabels)
	if err != nil {
		t.Fatalf("Failed to calculate bad silhouette: %v", err)
	}

	if goodSilhouette <= badSilhouette {
		t.Errorf("Good clustering should have higher silhouette score: %f vs %f", goodSilhouette, badSilhouette)
	}

	// Calinski-Harabasz score: higher is better
	goodCH, err := metrics.CalinskiHarabaszScore(X, goodLabels)
	if err != nil {
		t.Fatalf("Failed to calculate good CH score: %v", err)
	}

	badCH, err := metrics.CalinskiHarabaszScore(X, badLabels)
	if err != nil {
		t.Fatalf("Failed to calculate bad CH score: %v", err)
	}

	if goodCH <= badCH {
		t.Errorf("Good clustering should have higher CH score: %f vs %f", goodCH, badCH)
	}

	// Davies-Bouldin score: lower is better
	goodDB, err := metrics.DaviesBouldinScore(X, goodLabels)
	if err != nil {
		t.Fatalf("Failed to calculate good DB score: %v", err)
	}

	badDB, err := metrics.DaviesBouldinScore(X, badLabels)
	if err != nil {
		t.Fatalf("Failed to calculate bad DB score: %v", err)
	}

	if goodDB >= badDB {
		t.Errorf("Good clustering should have lower DB score: %f vs %f", goodDB, badDB)
	}

	// Inertia: lower is better (for same number of clusters)
	goodInertia, err := metrics.Inertia(X, goodLabels)
	if err != nil {
		t.Fatalf("Failed to calculate good inertia: %v", err)
	}

	badInertia, err := metrics.Inertia(X, badLabels)
	if err != nil {
		t.Fatalf("Failed to calculate bad inertia: %v", err)
	}

	if goodInertia >= badInertia {
		t.Errorf("Good clustering should have lower inertia: %f vs %f", goodInertia, badInertia)
	}
}

func TestClusteringMetrics_IdenticalPoints(t *testing.T) {
	// Test with identical points in clusters
	data := [][]float64{
		{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, // Identical points in cluster 0
		{5.0, 5.0}, {5.0, 5.0}, {5.0, 5.0}, // Identical points in cluster 1
	}
	X := core.NewTensorFromSlice(data)
	labels := []int{0, 0, 0, 1, 1, 1}

	metrics := NewClusteringMetrics()

	// Should handle identical points gracefully
	silhouette, err := metrics.SilhouetteScore(X, labels)
	if err != nil {
		t.Fatalf("Failed to calculate silhouette with identical points: %v", err)
	}

	if math.IsNaN(silhouette) || math.IsInf(silhouette, 0) {
		t.Errorf("Silhouette score should be finite with identical points, got %f", silhouette)
	}

	inertia, err := metrics.Inertia(X, labels)
	if err != nil {
		t.Fatalf("Failed to calculate inertia with identical points: %v", err)
	}

	// Inertia should be 0 for identical points
	if inertia != 0 {
		t.Errorf("Expected inertia=0 for identical points, got %f", inertia)
	}
}

func BenchmarkClusteringMetrics_SilhouetteScore(b *testing.B) {
	// Create larger dataset for benchmarking
	data := make([][]float64, 1000)
	labels := make([]int, 1000)
	for i := 0; i < 1000; i++ {
		data[i] = []float64{float64(i % 10), float64((i * 2) % 10)}
		labels[i] = i % 5 // 5 clusters
	}
	X := core.NewTensorFromSlice(data)

	metrics := NewClusteringMetrics()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := metrics.SilhouetteScore(X, labels)
		if err != nil {
			b.Fatalf("Failed to calculate silhouette score: %v", err)
		}
	}
}

func BenchmarkClusteringMetrics_Inertia(b *testing.B) {
	// Create larger dataset for benchmarking
	data := make([][]float64, 1000)
	labels := make([]int, 1000)
	for i := 0; i < 1000; i++ {
		data[i] = []float64{float64(i % 10), float64((i * 2) % 10)}
		labels[i] = i % 5 // 5 clusters
	}
	X := core.NewTensorFromSlice(data)

	metrics := NewClusteringMetrics()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := metrics.Inertia(X, labels)
		if err != nil {
			b.Fatalf("Failed to calculate inertia: %v", err)
		}
	}
}
