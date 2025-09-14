package algorithms

import (
	"math"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// ClusteringMetrics provides evaluation metrics for clustering algorithms.
type ClusteringMetrics struct{}

// NewClusteringMetrics creates a new clustering metrics calculator.
func NewClusteringMetrics() *ClusteringMetrics {
	return &ClusteringMetrics{}
}

// SilhouetteScore calculates the silhouette score for clustering results.
// Returns a value between -1 and 1, where higher values indicate better clustering.
func (cm *ClusteringMetrics) SilhouetteScore(X core.Tensor, labels []int) (float64, error) {
	if err := cm.validateClusteringInput(X, labels); err != nil {
		return 0, err
	}

	nSamples, _ := X.Dims()
	if nSamples < 2 {
		return 0, core.NewError(core.ErrInvalidInput, "silhouette score requires at least 2 samples")
	}

	// Get unique clusters (excluding noise points marked as -1)
	clusters := cm.getUniqueClusters(labels)
	if len(clusters) < 2 {
		return 0, core.NewError(core.ErrInvalidInput, "silhouette score requires at least 2 clusters")
	}

	silhouetteSum := 0.0
	validSamples := 0

	for i := 0; i < nSamples; i++ {
		if labels[i] == -1 {
			continue // Skip noise points
		}

		// Calculate silhouette coefficient for this sample
		s := cm.calculateSilhouetteCoefficient(X, labels, i, clusters)
		silhouetteSum += s
		validSamples++
	}

	if validSamples == 0 {
		return 0, core.NewError(core.ErrInvalidInput, "no valid samples for silhouette calculation")
	}

	return silhouetteSum / float64(validSamples), nil
}

// CalinskiHarabaszScore calculates the Calinski-Harabasz score (variance ratio criterion).
// Higher values indicate better clustering.
func (cm *ClusteringMetrics) CalinskiHarabaszScore(X core.Tensor, labels []int) (float64, error) {
	if err := cm.validateClusteringInput(X, labels); err != nil {
		return 0, err
	}

	nSamples, nFeatures := X.Dims()
	clusters := cm.getUniqueClusters(labels)

	if len(clusters) < 2 {
		return 0, core.NewError(core.ErrInvalidInput, "Calinski-Harabasz score requires at least 2 clusters")
	}

	// Calculate overall centroid
	overallCentroid := cm.calculateOverallCentroid(X, labels)

	// Calculate between-cluster sum of squares (BCSS)
	bcss := 0.0
	for _, cluster := range clusters {
		clusterIndices := cm.getClusterIndices(labels, cluster)
		if len(clusterIndices) == 0 {
			continue
		}

		clusterCentroid := cm.calculateClusterCentroid(X, clusterIndices)
		clusterSize := float64(len(clusterIndices))

		for j := 0; j < nFeatures; j++ {
			diff := clusterCentroid.At(0, j) - overallCentroid.At(0, j)
			bcss += clusterSize * diff * diff
		}
	}

	// Calculate within-cluster sum of squares (WCSS)
	wcss := 0.0
	for _, cluster := range clusters {
		clusterIndices := cm.getClusterIndices(labels, cluster)
		if len(clusterIndices) == 0 {
			continue
		}

		clusterCentroid := cm.calculateClusterCentroid(X, clusterIndices)

		for _, idx := range clusterIndices {
			for j := 0; j < nFeatures; j++ {
				diff := X.At(idx, j) - clusterCentroid.At(0, j)
				wcss += diff * diff
			}
		}
	}

	if wcss == 0 {
		return math.Inf(1), nil
	}

	// Calculate Calinski-Harabasz score
	k := float64(len(clusters))
	n := float64(nSamples)

	return (bcss / (k - 1)) / (wcss / (n - k)), nil
}

// DaviesBouldinScore calculates the Davies-Bouldin score.
// Lower values indicate better clustering.
func (cm *ClusteringMetrics) DaviesBouldinScore(X core.Tensor, labels []int) (float64, error) {
	if err := cm.validateClusteringInput(X, labels); err != nil {
		return 0, err
	}

	clusters := cm.getUniqueClusters(labels)
	if len(clusters) < 2 {
		return 0, core.NewError(core.ErrInvalidInput, "Davies-Bouldin score requires at least 2 clusters")
	}

	// Calculate cluster centroids and within-cluster distances
	centroids := make([]core.Tensor, len(clusters))
	withinClusterDists := make([]float64, len(clusters))

	for i, cluster := range clusters {
		clusterIndices := cm.getClusterIndices(labels, cluster)
		if len(clusterIndices) == 0 {
			continue
		}

		centroids[i] = cm.calculateClusterCentroid(X, clusterIndices)
		withinClusterDists[i] = cm.calculateWithinClusterDistance(X, clusterIndices, centroids[i])
	}

	// Calculate Davies-Bouldin score
	dbSum := 0.0
	for i := 0; i < len(clusters); i++ {
		maxRatio := 0.0

		for j := 0; j < len(clusters); j++ {
			if i == j {
				continue
			}

			betweenDist := cm.euclideanDistance(centroids[i], centroids[j])
			if betweenDist == 0 {
				continue
			}

			ratio := (withinClusterDists[i] + withinClusterDists[j]) / betweenDist
			if ratio > maxRatio {
				maxRatio = ratio
			}
		}

		dbSum += maxRatio
	}

	return dbSum / float64(len(clusters)), nil
}

// Inertia calculates the within-cluster sum of squares.
func (cm *ClusteringMetrics) Inertia(X core.Tensor, labels []int) (float64, error) {
	if err := cm.validateClusteringInput(X, labels); err != nil {
		return 0, err
	}

	clusters := cm.getUniqueClusters(labels)
	inertia := 0.0

	for _, cluster := range clusters {
		clusterIndices := cm.getClusterIndices(labels, cluster)
		if len(clusterIndices) == 0 {
			continue
		}

		clusterCentroid := cm.calculateClusterCentroid(X, clusterIndices)

		for _, idx := range clusterIndices {
			point := X.Row(idx)
			dist := cm.euclideanDistanceSquared(point, clusterCentroid)
			inertia += dist
		}
	}

	return inertia, nil
}

// validateClusteringInput validates input for clustering metrics.
func (cm *ClusteringMetrics) validateClusteringInput(X core.Tensor, labels []int) error {
	if X == nil {
		return core.NewError(core.ErrInvalidInput, "input data cannot be nil")
	}

	nSamples, nFeatures := X.Dims()
	if nSamples == 0 || nFeatures == 0 {
		return core.NewError(core.ErrInvalidInput, "input data cannot be empty")
	}

	if len(labels) != nSamples {
		return core.NewError(core.ErrDimensionMismatch, "number of labels must match number of samples")
	}

	if !X.IsFinite() {
		return core.NewError(core.ErrNumericalInstability, "input data contains NaN or infinite values")
	}

	return nil
}

// getUniqueClusters returns unique cluster labels (excluding -1 for noise).
func (cm *ClusteringMetrics) getUniqueClusters(labels []int) []int {
	clusterSet := make(map[int]bool)
	for _, label := range labels {
		if label >= 0 { // Exclude noise points (-1)
			clusterSet[label] = true
		}
	}

	clusters := make([]int, 0, len(clusterSet))
	for cluster := range clusterSet {
		clusters = append(clusters, cluster)
	}

	return clusters
}

// getClusterIndices returns indices of samples belonging to a specific cluster.
func (cm *ClusteringMetrics) getClusterIndices(labels []int, cluster int) []int {
	indices := make([]int, 0)
	for i, label := range labels {
		if label == cluster {
			indices = append(indices, i)
		}
	}
	return indices
}

// calculateClusterCentroid calculates the centroid of a cluster.
func (cm *ClusteringMetrics) calculateClusterCentroid(X core.Tensor, indices []int) core.Tensor {
	if len(indices) == 0 {
		return nil
	}

	_, nFeatures := X.Dims()
	centroid := core.NewZerosTensor(1, nFeatures)

	// Sum all points in the cluster
	for _, idx := range indices {
		for j := 0; j < nFeatures; j++ {
			current := centroid.At(0, j)
			centroid.Set(0, j, current+X.At(idx, j))
		}
	}

	// Average to get centroid
	clusterSize := float64(len(indices))
	for j := 0; j < nFeatures; j++ {
		avg := centroid.At(0, j) / clusterSize
		centroid.Set(0, j, avg)
	}

	return centroid
}

// calculateOverallCentroid calculates the centroid of all non-noise points.
func (cm *ClusteringMetrics) calculateOverallCentroid(X core.Tensor, labels []int) core.Tensor {
	nSamples, nFeatures := X.Dims()
	centroid := core.NewZerosTensor(1, nFeatures)
	validSamples := 0

	// Sum all non-noise points
	for i := 0; i < nSamples; i++ {
		if labels[i] >= 0 { // Exclude noise points
			for j := 0; j < nFeatures; j++ {
				current := centroid.At(0, j)
				centroid.Set(0, j, current+X.At(i, j))
			}
			validSamples++
		}
	}

	// Average to get overall centroid
	if validSamples > 0 {
		for j := 0; j < nFeatures; j++ {
			avg := centroid.At(0, j) / float64(validSamples)
			centroid.Set(0, j, avg)
		}
	}

	return centroid
}

// calculateWithinClusterDistance calculates average distance from cluster centroid.
func (cm *ClusteringMetrics) calculateWithinClusterDistance(X core.Tensor, indices []int, centroid core.Tensor) float64 {
	if len(indices) == 0 {
		return 0
	}

	totalDist := 0.0
	for _, idx := range indices {
		point := X.Row(idx)
		dist := cm.euclideanDistance(point, centroid)
		totalDist += dist
	}

	return totalDist / float64(len(indices))
}

// calculateSilhouetteCoefficient calculates silhouette coefficient for a single sample.
func (cm *ClusteringMetrics) calculateSilhouetteCoefficient(X core.Tensor, labels []int, sampleIdx int, clusters []int) float64 {
	sampleCluster := labels[sampleIdx]

	// Calculate a(i): mean distance to other points in the same cluster
	a := cm.calculateMeanIntraClusterDistance(X, labels, sampleIdx, sampleCluster)

	// Calculate b(i): mean distance to points in the nearest cluster
	b := math.Inf(1)
	for _, cluster := range clusters {
		if cluster == sampleCluster {
			continue
		}

		meanDist := cm.calculateMeanInterClusterDistance(X, labels, sampleIdx, cluster)
		if meanDist < b {
			b = meanDist
		}
	}

	// Calculate silhouette coefficient
	if a == 0 && b == 0 {
		return 0
	}

	maxAB := math.Max(a, b)
	if maxAB == 0 {
		return 0
	}

	return (b - a) / maxAB
}

// calculateMeanIntraClusterDistance calculates mean distance to other points in the same cluster.
func (cm *ClusteringMetrics) calculateMeanIntraClusterDistance(X core.Tensor, labels []int, sampleIdx, cluster int) float64 {
	samplePoint := X.Row(sampleIdx)
	totalDist := 0.0
	count := 0

	nSamples := len(labels)
	for i := 0; i < nSamples; i++ {
		if i != sampleIdx && labels[i] == cluster {
			otherPoint := X.Row(i)
			dist := cm.euclideanDistance(samplePoint, otherPoint)
			totalDist += dist
			count++
		}
	}

	if count == 0 {
		return 0
	}

	return totalDist / float64(count)
}

// calculateMeanInterClusterDistance calculates mean distance to points in another cluster.
func (cm *ClusteringMetrics) calculateMeanInterClusterDistance(X core.Tensor, labels []int, sampleIdx, cluster int) float64 {
	samplePoint := X.Row(sampleIdx)
	totalDist := 0.0
	count := 0

	nSamples := len(labels)
	for i := 0; i < nSamples; i++ {
		if labels[i] == cluster {
			otherPoint := X.Row(i)
			dist := cm.euclideanDistance(samplePoint, otherPoint)
			totalDist += dist
			count++
		}
	}

	if count == 0 {
		return math.Inf(1)
	}

	return totalDist / float64(count)
}

// euclideanDistance calculates the Euclidean distance between two points.
func (cm *ClusteringMetrics) euclideanDistance(a, b core.Tensor) float64 {
	_, aCols := a.Dims()
	_, bCols := b.Dims()

	if aCols != bCols {
		return math.Inf(1)
	}

	dist := 0.0
	for j := 0; j < aCols; j++ {
		diff := a.At(0, j) - b.At(0, j)
		dist += diff * diff
	}

	return math.Sqrt(dist)
}

// euclideanDistanceSquared calculates the squared Euclidean distance between two points.
func (cm *ClusteringMetrics) euclideanDistanceSquared(a, b core.Tensor) float64 {
	_, aCols := a.Dims()
	_, bCols := b.Dims()

	if aCols != bCols {
		return math.Inf(1)
	}

	dist := 0.0
	for j := 0; j < aCols; j++ {
		diff := a.At(0, j) - b.At(0, j)
		dist += diff * diff
	}

	return dist
}
