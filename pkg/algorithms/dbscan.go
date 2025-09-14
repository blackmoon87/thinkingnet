package algorithms

import (
	"math"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// DBSCAN implements the Density-Based Spatial Clustering of Applications with Noise algorithm.
type DBSCAN struct {
	// Configuration
	eps        float64 // Maximum distance between two samples for one to be considered as in the neighborhood of the other
	minSamples int     // Number of samples in a neighborhood for a point to be considered as a core point

	// State
	fitted      bool  // Whether the model has been fitted
	labels      []int // Cluster labels (-1 for noise, >= 0 for cluster ID)
	coreIndices []int // Indices of core samples
	nClusters   int   // Number of clusters found
	nNoise      int   // Number of noise points
}

// NewDBSCAN creates a new DBSCAN clusterer.
func NewDBSCAN(eps float64, minSamples int, options ...DBSCANOption) *DBSCAN {
	dbscan := &DBSCAN{
		eps:        eps,
		minSamples: minSamples,
		fitted:     false,
	}

	// Apply options
	for _, option := range options {
		option(dbscan)
	}

	return dbscan
}

// DBSCANOption represents a functional option for DBSCAN configuration.
type DBSCANOption func(*DBSCAN)

// WithEps sets the epsilon parameter.
func WithEps(eps float64) DBSCANOption {
	return func(db *DBSCAN) { db.eps = eps }
}

// WithMinSamples sets the minimum samples parameter.
func WithMinSamples(minSamples int) DBSCANOption {
	return func(db *DBSCAN) { db.minSamples = minSamples }
}

// Fit learns cluster parameters from data.
func (db *DBSCAN) Fit(X core.Tensor) error {
	if err := db.validateInput(X); err != nil {
		return err
	}

	nSamples, _ := X.Dims()
	db.labels = make([]int, nSamples)
	db.coreIndices = make([]int, 0)

	// Initialize all points as unvisited (-2)
	for i := range db.labels {
		db.labels[i] = -2
	}

	clusterID := 0

	// Process each point
	for i := 0; i < nSamples; i++ {
		if db.labels[i] != -2 {
			continue // Already processed
		}

		// Find neighbors
		neighbors := db.findNeighbors(X, i)

		if len(neighbors) < db.minSamples {
			// Mark as noise
			db.labels[i] = -1
		} else {
			// Start new cluster
			db.coreIndices = append(db.coreIndices, i)
			db.expandCluster(X, i, neighbors, clusterID)
			clusterID++
		}
	}

	db.nClusters = clusterID
	db.countNoise()
	db.fitted = true

	return nil
}

// Predict assigns cluster labels to data.
func (db *DBSCAN) Predict(X core.Tensor) ([]int, error) {
	if !db.fitted {
		return nil, core.NewError(core.ErrNotFitted, "DBSCAN must be fitted before prediction")
	}

	if err := db.validateInput(X); err != nil {
		return nil, err
	}

	nSamples, _ := X.Dims()
	labels := make([]int, nSamples)

	// For new data, assign to nearest core point's cluster or mark as noise
	for i := 0; i < nSamples; i++ {
		// This is a limitation of DBSCAN - it doesn't naturally predict on new data
		// without access to the original training data
		labels[i] = -1 // Mark as noise by default
	}

	return labels, nil
}

// FitPredict fits and predicts in one step.
func (db *DBSCAN) FitPredict(X core.Tensor) ([]int, error) {
	if err := db.Fit(X); err != nil {
		return nil, err
	}
	return db.labels, nil
}

// ClusterCenters returns approximate cluster centers (centroids of core points in each cluster).
func (db *DBSCAN) ClusterCenters() core.Tensor {
	if !db.fitted || db.nClusters == 0 {
		return nil
	}

	// Note: DBSCAN doesn't have explicit centroids like K-means
	// This method returns approximate centers based on core points
	// In a full implementation, you'd need to store the training data
	return nil
}

// Name returns the clusterer name.
func (db *DBSCAN) Name() string {
	return "DBSCAN"
}

// NClusters returns the number of clusters found.
func (db *DBSCAN) NClusters() int {
	return db.nClusters
}

// NNoise returns the number of noise points.
func (db *DBSCAN) NNoise() int {
	return db.nNoise
}

// CoreIndices returns the indices of core samples.
func (db *DBSCAN) CoreIndices() []int {
	if !db.fitted {
		return nil
	}
	result := make([]int, len(db.coreIndices))
	copy(result, db.coreIndices)
	return result
}

// validateInput validates the input data.
func (db *DBSCAN) validateInput(X core.Tensor) error {
	if X == nil {
		return core.NewError(core.ErrInvalidInput, "input data cannot be nil")
	}

	nSamples, nFeatures := X.Dims()
	if nSamples == 0 || nFeatures == 0 {
		return core.NewError(core.ErrInvalidInput, "input data cannot be empty")
	}

	if db.eps <= 0 {
		return core.NewError(core.ErrInvalidInput, "eps must be positive")
	}

	if db.minSamples <= 0 {
		return core.NewError(core.ErrInvalidInput, "minSamples must be positive")
	}

	if !X.IsFinite() {
		return core.NewError(core.ErrNumericalInstability, "input data contains NaN or infinite values")
	}

	return nil
}

// findNeighbors finds all points within eps distance of the given point.
func (db *DBSCAN) findNeighbors(X core.Tensor, pointIdx int) []int {
	nSamples, _ := X.Dims()
	neighbors := make([]int, 0)
	point := X.Row(pointIdx)

	for i := 0; i < nSamples; i++ {
		if i == pointIdx {
			neighbors = append(neighbors, i)
			continue
		}

		otherPoint := X.Row(i)
		if db.euclideanDistance(point, otherPoint) <= db.eps {
			neighbors = append(neighbors, i)
		}
	}

	return neighbors
}

// expandCluster expands a cluster from a core point.
func (db *DBSCAN) expandCluster(X core.Tensor, pointIdx int, neighbors []int, clusterID int) {
	db.labels[pointIdx] = clusterID

	i := 0
	for i < len(neighbors) {
		neighborIdx := neighbors[i]

		if db.labels[neighborIdx] == -1 {
			// Change noise to border point
			db.labels[neighborIdx] = clusterID
		} else if db.labels[neighborIdx] == -2 {
			// Unvisited point
			db.labels[neighborIdx] = clusterID

			// Find neighbors of this neighbor
			neighborNeighbors := db.findNeighbors(X, neighborIdx)

			if len(neighborNeighbors) >= db.minSamples {
				// This neighbor is also a core point
				db.coreIndices = append(db.coreIndices, neighborIdx)
				// Add its neighbors to the expansion list
				neighbors = db.mergeNeighbors(neighbors, neighborNeighbors)
			}
		}

		i++
	}
}

// mergeNeighbors merges two neighbor lists, avoiding duplicates.
func (db *DBSCAN) mergeNeighbors(existing, new []int) []int {
	existingSet := make(map[int]bool)
	for _, idx := range existing {
		existingSet[idx] = true
	}

	for _, idx := range new {
		if !existingSet[idx] {
			existing = append(existing, idx)
			existingSet[idx] = true
		}
	}

	return existing
}

// countNoise counts the number of noise points.
func (db *DBSCAN) countNoise() {
	count := 0
	for _, label := range db.labels {
		if label == -1 {
			count++
		}
	}
	db.nNoise = count
}

// euclideanDistance calculates the Euclidean distance between two points.
func (db *DBSCAN) euclideanDistance(a, b core.Tensor) float64 {
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
