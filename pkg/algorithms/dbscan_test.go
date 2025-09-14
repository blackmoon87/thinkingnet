package algorithms

import (
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestDBSCAN_Basic(t *testing.T) {
	// Create test data with clear clusters and noise
	data := [][]float64{
		// Cluster 1
		{1.0, 1.0}, {1.1, 1.1}, {1.2, 1.0}, {1.0, 1.2},
		// Cluster 2
		{5.0, 5.0}, {5.1, 5.1}, {5.2, 5.0}, {5.0, 5.2},
		// Noise points
		{10.0, 10.0}, {0.0, 10.0},
	}
	X := core.NewTensorFromSlice(data)

	// Test DBSCAN with eps=0.5, minSamples=3
	dbscan := NewDBSCAN(0.5, 3)

	err := dbscan.Fit(X)
	if err != nil {
		t.Fatalf("Failed to fit DBSCAN: %v", err)
	}

	if !dbscan.fitted {
		t.Error("DBSCAN should be marked as fitted after Fit()")
	}

	// Should find 2 clusters
	if dbscan.NClusters() != 2 {
		t.Errorf("Expected 2 clusters, got %d", dbscan.NClusters())
	}

	// Should have some noise points
	if dbscan.NNoise() == 0 {
		t.Error("Expected some noise points")
	}

	// Test FitPredict
	labels, err := dbscan.FitPredict(X)
	if err != nil {
		t.Fatalf("Failed to fit and predict: %v", err)
	}

	if len(labels) != 10 {
		t.Errorf("Expected 10 labels, got %d", len(labels))
	}

	// Count clusters and noise
	clusterCount := make(map[int]int)
	for _, label := range labels {
		clusterCount[label]++
	}

	// Should have exactly 2 non-noise clusters (0 and 1)
	nonNoiseClusters := 0
	for label := range clusterCount {
		if label >= 0 {
			nonNoiseClusters++
		}
	}

	if nonNoiseClusters != 2 {
		t.Errorf("Expected 2 non-noise clusters, got %d", nonNoiseClusters)
	}

	// Check that noise points are labeled as -1
	noiseCount := clusterCount[-1]
	if noiseCount != dbscan.NNoise() {
		t.Errorf("Noise count mismatch: expected %d, got %d", dbscan.NNoise(), noiseCount)
	}
}

func TestDBSCAN_AllNoise(t *testing.T) {
	// Create data where all points are too far apart
	data := [][]float64{
		{0.0, 0.0}, {10.0, 10.0}, {20.0, 20.0}, {30.0, 30.0},
	}
	X := core.NewTensorFromSlice(data)

	dbscan := NewDBSCAN(1.0, 3) // High minSamples, low eps
	err := dbscan.Fit(X)
	if err != nil {
		t.Fatalf("Failed to fit DBSCAN: %v", err)
	}

	// Should find no clusters
	if dbscan.NClusters() != 0 {
		t.Errorf("Expected 0 clusters, got %d", dbscan.NClusters())
	}

	// All points should be noise
	if dbscan.NNoise() != 4 {
		t.Errorf("Expected 4 noise points, got %d", dbscan.NNoise())
	}

	labels, err := dbscan.FitPredict(X)
	if err != nil {
		t.Fatalf("Failed to fit and predict: %v", err)
	}

	// All labels should be -1 (noise)
	for i, label := range labels {
		if label != -1 {
			t.Errorf("Expected noise label (-1) at index %d, got %d", i, label)
		}
	}
}

func TestDBSCAN_SingleCluster(t *testing.T) {
	// Create data that forms one dense cluster
	data := [][]float64{
		{1.0, 1.0}, {1.1, 1.0}, {1.0, 1.1}, {1.1, 1.1},
		{1.2, 1.0}, {1.0, 1.2}, {1.2, 1.2}, {1.1, 1.2},
	}
	X := core.NewTensorFromSlice(data)

	dbscan := NewDBSCAN(0.3, 3)
	err := dbscan.Fit(X)
	if err != nil {
		t.Fatalf("Failed to fit DBSCAN: %v", err)
	}

	// Should find exactly 1 cluster
	if dbscan.NClusters() != 1 {
		t.Errorf("Expected 1 cluster, got %d", dbscan.NClusters())
	}

	// Should have no noise points
	if dbscan.NNoise() != 0 {
		t.Errorf("Expected 0 noise points, got %d", dbscan.NNoise())
	}

	labels, err := dbscan.FitPredict(X)
	if err != nil {
		t.Fatalf("Failed to fit and predict: %v", err)
	}

	// All labels should be 0 (first cluster)
	for i, label := range labels {
		if label != 0 {
			t.Errorf("Expected cluster label 0 at index %d, got %d", i, label)
		}
	}
}

func TestDBSCAN_CoreIndices(t *testing.T) {
	data := [][]float64{
		{1.0, 1.0}, {1.1, 1.0}, {1.0, 1.1}, {1.1, 1.1}, // Dense cluster
		{5.0, 5.0}, // Isolated point
	}
	X := core.NewTensorFromSlice(data)

	dbscan := NewDBSCAN(0.2, 3)
	err := dbscan.Fit(X)
	if err != nil {
		t.Fatalf("Failed to fit DBSCAN: %v", err)
	}

	coreIndices := dbscan.CoreIndices()
	if coreIndices == nil {
		t.Error("Core indices should not be nil after fitting")
	}

	// Should have at least one core point
	if len(coreIndices) == 0 {
		t.Error("Expected at least one core point")
	}

	// Core indices should be valid
	for _, idx := range coreIndices {
		if idx < 0 || idx >= 5 {
			t.Errorf("Invalid core index: %d", idx)
		}
	}
}

func TestDBSCAN_EdgeCases(t *testing.T) {
	// Test with nil input
	dbscan := NewDBSCAN(1.0, 3)
	err := dbscan.Fit(nil)
	if err == nil {
		t.Error("Expected error for nil input")
	}

	// Test with empty data - skip this test as NewZerosTensor(0,0) panics
	// This is expected behavior as empty tensors are not valid

	// Test with invalid eps
	data := [][]float64{{1.0, 1.0}, {2.0, 2.0}}
	X := core.NewTensorFromSlice(data)
	dbscan = NewDBSCAN(-1.0, 3)
	err = dbscan.Fit(X)
	if err == nil {
		t.Error("Expected error for negative eps")
	}

	// Test with invalid minSamples
	dbscan = NewDBSCAN(1.0, -1)
	err = dbscan.Fit(X)
	if err == nil {
		t.Error("Expected error for negative minSamples")
	}

	// Test prediction before fitting
	dbscan = NewDBSCAN(1.0, 3)
	_, err = dbscan.Predict(X)
	if err == nil {
		t.Error("Expected error when predicting before fitting")
	}
}

func TestDBSCAN_Options(t *testing.T) {
	// Test with custom options
	dbscan := NewDBSCAN(1.5, 5,
		WithEps(2.0),
		WithMinSamples(4),
	)

	if dbscan.eps != 2.0 {
		t.Errorf("Expected eps=2.0, got %f", dbscan.eps)
	}

	if dbscan.minSamples != 4 {
		t.Errorf("Expected minSamples=4, got %d", dbscan.minSamples)
	}
}

func TestDBSCAN_Name(t *testing.T) {
	dbscan := NewDBSCAN(1.0, 3)
	if dbscan.Name() != "DBSCAN" {
		t.Errorf("Expected name 'DBSCAN', got '%s'", dbscan.Name())
	}
}

func TestDBSCAN_ClusterCenters(t *testing.T) {
	data := [][]float64{
		{1.0, 1.0}, {1.1, 1.0}, {1.0, 1.1},
	}
	X := core.NewTensorFromSlice(data)

	dbscan := NewDBSCAN(0.2, 2)
	err := dbscan.Fit(X)
	if err != nil {
		t.Fatalf("Failed to fit DBSCAN: %v", err)
	}

	// DBSCAN doesn't have explicit centroids like K-means
	// This implementation returns nil
	centers := dbscan.ClusterCenters()
	if centers != nil {
		t.Error("DBSCAN cluster centers should be nil (not implemented)")
	}
}

func TestDBSCAN_PredictLimitation(t *testing.T) {
	// Test the limitation of DBSCAN prediction on new data
	data := [][]float64{
		{1.0, 1.0}, {1.1, 1.0}, {1.0, 1.1}, {1.1, 1.1},
	}
	X := core.NewTensorFromSlice(data)

	dbscan := NewDBSCAN(0.2, 3)
	err := dbscan.Fit(X)
	if err != nil {
		t.Fatalf("Failed to fit DBSCAN: %v", err)
	}

	// Try to predict on new data
	newData := [][]float64{{2.0, 2.0}, {3.0, 3.0}}
	newX := core.NewTensorFromSlice(newData)

	labels, err := dbscan.Predict(newX)
	if err != nil {
		t.Fatalf("Predict should not error, but has limitations: %v", err)
	}

	// All new points should be marked as noise (-1) due to implementation limitation
	for i, label := range labels {
		if label != -1 {
			t.Errorf("Expected noise label (-1) for new data at index %d, got %d", i, label)
		}
	}
}

func TestDBSCAN_DifferentEpsValues(t *testing.T) {
	data := [][]float64{
		{0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},
		{5.0, 5.0}, {5.1, 5.1}, {5.2, 5.2},
	}
	X := core.NewTensorFromSlice(data)

	// Test with small eps - should find no clusters
	dbscan1 := NewDBSCAN(0.05, 2)
	err := dbscan1.Fit(X)
	if err != nil {
		t.Fatalf("Failed to fit DBSCAN with small eps: %v", err)
	}

	// Test with large eps - should find one cluster
	dbscan2 := NewDBSCAN(10.0, 2)
	err = dbscan2.Fit(X)
	if err != nil {
		t.Fatalf("Failed to fit DBSCAN with large eps: %v", err)
	}

	// Small eps should find fewer/no clusters
	// Large eps should find more clusters or merge them
	if dbscan2.NClusters() < dbscan1.NClusters() {
		t.Error("Larger eps should generally result in fewer, larger clusters")
	}
}

func BenchmarkDBSCAN_Fit(b *testing.B) {
	// Create larger dataset for benchmarking
	data := make([][]float64, 500)
	for i := 0; i < 500; i++ {
		data[i] = []float64{float64(i % 20), float64((i * 2) % 20)}
	}
	X := core.NewTensorFromSlice(data)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dbscan := NewDBSCAN(2.0, 5)
		err := dbscan.Fit(X)
		if err != nil {
			b.Fatalf("Failed to fit: %v", err)
		}
	}
}

func BenchmarkDBSCAN_FitPredict(b *testing.B) {
	// Setup
	data := make([][]float64, 100)
	for i := 0; i < 100; i++ {
		data[i] = []float64{float64(i % 10), float64((i * 2) % 10)}
	}
	X := core.NewTensorFromSlice(data)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dbscan := NewDBSCAN(2.0, 3)
		_, err := dbscan.FitPredict(X)
		if err != nil {
			b.Fatalf("Failed to fit and predict: %v", err)
		}
	}
}
