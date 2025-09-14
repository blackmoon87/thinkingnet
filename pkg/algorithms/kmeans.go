package algorithms

import (
	"math"
	"math/rand"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// KMeans implements the K-means clustering algorithm.
type KMeans struct {
	// Configuration
	k          int     // Number of clusters
	maxIters   int     // Maximum number of iterations
	tolerance  float64 // Convergence tolerance
	initMethod string  // Initialization method: "random", "kmeans++"
	randomSeed int64   // Random seed for reproducibility

	// State
	fitted    bool        // Whether the model has been fitted
	centroids core.Tensor // Cluster centroids
	labels    []int       // Cluster labels for training data
	inertia   float64     // Sum of squared distances to centroids
	nIters    int         // Number of iterations performed
	rng       *rand.Rand  // Random number generator
}

// NewKMeans creates a new K-means clusterer.
func NewKMeans(k int, options ...KMeansOption) *KMeans {
	kmeans := &KMeans{
		k:          k,
		maxIters:   300,
		tolerance:  1e-4,
		initMethod: "kmeans++",
		randomSeed: time.Now().UnixNano(),
		fitted:     false,
	}

	// Apply options
	for _, option := range options {
		option(kmeans)
	}

	// Initialize random number generator
	kmeans.rng = rand.New(rand.NewSource(kmeans.randomSeed))

	return kmeans
}

// KMeansOption represents a functional option for K-means configuration.
type KMeansOption func(*KMeans)

// WithMaxIters sets the maximum number of iterations.
func WithMaxIters(maxIters int) KMeansOption {
	return func(km *KMeans) { km.maxIters = maxIters }
}

// WithTolerance sets the convergence tolerance.
func WithTolerance(tolerance float64) KMeansOption {
	return func(km *KMeans) { km.tolerance = tolerance }
}

// WithInitMethod sets the initialization method.
func WithInitMethod(method string) KMeansOption {
	return func(km *KMeans) { km.initMethod = method }
}

// WithRandomSeed sets the random seed.
func WithRandomSeed(seed int64) KMeansOption {
	return func(km *KMeans) { km.randomSeed = seed }
}

// EasyKMeans creates a K-means clustering model with sensible defaults.
// This is a simplified constructor for quick usage without needing to configure options.
func EasyKMeans(k int) *KMeans {
	return NewKMeans(k,
		WithMaxIters(300),
		WithTolerance(1e-4),
		WithInitMethod("kmeans++"),
		WithRandomSeed(42),
	)
}

// Fit learns cluster parameters from data.
func (km *KMeans) Fit(X core.Tensor) error {
	if err := km.validateInput(X); err != nil {
		return err
	}

	nSamples, _ := X.Dims()

	// Initialize centroids
	centroids, err := km.initializeCentroids(X)
	if err != nil {
		return err
	}

	km.centroids = centroids
	km.labels = make([]int, nSamples)
	prevInertia := math.Inf(1)

	// Main K-means loop
	for iter := 0; iter < km.maxIters; iter++ {
		// Assign points to nearest centroids
		km.assignClusters(X)

		// Update centroids
		newCentroids := km.updateCentroids(X)

		// Calculate inertia (within-cluster sum of squares)
		km.inertia = km.calculateInertia(X)

		// Check for convergence
		if math.Abs(prevInertia-km.inertia) < km.tolerance {
			km.nIters = iter + 1
			break
		}

		km.centroids = newCentroids
		prevInertia = km.inertia
		km.nIters = iter + 1
	}

	km.fitted = true
	return nil
}

// Predict assigns cluster labels to data.
func (km *KMeans) Predict(X core.Tensor) ([]int, error) {
	if !km.fitted {
		return nil, core.NewError(core.ErrNotFitted, "KMeans must be fitted before prediction")
	}

	if err := km.validateInput(X); err != nil {
		return nil, err
	}

	nSamples, _ := X.Dims()
	labels := make([]int, nSamples)

	for i := 0; i < nSamples; i++ {
		point := X.Row(i)
		labels[i] = km.findNearestCentroid(point)
	}

	return labels, nil
}

// FitPredict fits and predicts in one step.
func (km *KMeans) FitPredict(X core.Tensor) ([]int, error) {
	if err := km.Fit(X); err != nil {
		return nil, err
	}
	return km.labels, nil
}

// ClusterCenters returns the cluster centers.
func (km *KMeans) ClusterCenters() core.Tensor {
	if !km.fitted {
		return nil
	}
	return km.centroids.Copy()
}

// Name returns the clusterer name.
func (km *KMeans) Name() string {
	return "KMeans"
}

// Inertia returns the within-cluster sum of squares.
func (km *KMeans) Inertia() float64 {
	return km.inertia
}

// NIters returns the number of iterations performed.
func (km *KMeans) NIters() int {
	return km.nIters
}

// validateInput validates the input data.
func (km *KMeans) validateInput(X core.Tensor) error {
	if X == nil {
		return core.NewError(core.ErrInvalidInput, "input data cannot be nil")
	}

	nSamples, nFeatures := X.Dims()
	if nSamples == 0 || nFeatures == 0 {
		return core.NewError(core.ErrInvalidInput, "input data cannot be empty")
	}

	if km.k <= 0 {
		return core.NewError(core.ErrInvalidInput, "number of clusters must be positive")
	}

	if km.k > nSamples {
		return core.NewError(core.ErrInvalidInput, "number of clusters cannot exceed number of samples")
	}

	if !X.IsFinite() {
		return core.NewError(core.ErrNumericalInstability, "input data contains NaN or infinite values")
	}

	return nil
}

// initializeCentroids initializes cluster centroids using the specified method.
func (km *KMeans) initializeCentroids(X core.Tensor) (core.Tensor, error) {
	_, nFeatures := X.Dims()

	switch km.initMethod {
	case "random":
		return km.initializeRandom(X, nFeatures)
	case "kmeans++":
		return km.initializeKMeansPlusPlus(X)
	default:
		return nil, core.NewError(core.ErrConfigurationError, "unsupported initialization method: "+km.initMethod)
	}
}

// initializeRandom randomly selects k data points as initial centroids.
func (km *KMeans) initializeRandom(X core.Tensor, nFeatures int) (core.Tensor, error) {
	nSamples, _ := X.Dims()
	centroids := core.NewZerosTensor(km.k, nFeatures)

	// Randomly select k unique indices
	indices := km.rng.Perm(nSamples)[:km.k]

	for i, idx := range indices {
		for j := 0; j < nFeatures; j++ {
			centroids.Set(i, j, X.At(idx, j))
		}
	}

	return centroids, nil
}

// initializeKMeansPlusPlus implements K-means++ initialization for better convergence.
func (km *KMeans) initializeKMeansPlusPlus(X core.Tensor) (core.Tensor, error) {
	nSamples, nFeatures := X.Dims()
	centroids := core.NewZerosTensor(km.k, nFeatures)

	// Choose first centroid randomly
	firstIdx := km.rng.Intn(nSamples)
	for j := 0; j < nFeatures; j++ {
		centroids.Set(0, j, X.At(firstIdx, j))
	}

	// Choose remaining centroids
	for c := 1; c < km.k; c++ {
		distances := make([]float64, nSamples)
		totalDistance := 0.0

		// Calculate squared distances to nearest existing centroid
		for i := 0; i < nSamples; i++ {
			point := X.Row(i)
			minDist := math.Inf(1)

			for existingC := 0; existingC < c; existingC++ {
				centroid := centroids.Row(existingC)
				dist := km.euclideanDistanceSquared(point, centroid)
				if dist < minDist {
					minDist = dist
				}
			}

			distances[i] = minDist
			totalDistance += minDist
		}

		// Choose next centroid with probability proportional to squared distance
		target := km.rng.Float64() * totalDistance
		cumSum := 0.0

		for i := 0; i < nSamples; i++ {
			cumSum += distances[i]
			if cumSum >= target {
				for j := 0; j < nFeatures; j++ {
					centroids.Set(c, j, X.At(i, j))
				}
				break
			}
		}
	}

	return centroids, nil
}

// assignClusters assigns each data point to the nearest centroid.
func (km *KMeans) assignClusters(X core.Tensor) {
	nSamples, _ := X.Dims()

	for i := 0; i < nSamples; i++ {
		point := X.Row(i)
		km.labels[i] = km.findNearestCentroid(point)
	}
}

// findNearestCentroid finds the index of the nearest centroid to a point.
func (km *KMeans) findNearestCentroid(point core.Tensor) int {
	minDist := math.Inf(1)
	nearestIdx := 0

	for c := 0; c < km.k; c++ {
		centroid := km.centroids.Row(c)
		dist := km.euclideanDistanceSquared(point, centroid)
		if dist < minDist {
			minDist = dist
			nearestIdx = c
		}
	}

	return nearestIdx
}

// updateCentroids updates centroids based on current cluster assignments.
func (km *KMeans) updateCentroids(X core.Tensor) core.Tensor {
	nSamples, nFeatures := X.Dims()
	newCentroids := core.NewZerosTensor(km.k, nFeatures)
	clusterCounts := make([]int, km.k)

	// Sum points in each cluster
	for i := 0; i < nSamples; i++ {
		cluster := km.labels[i]
		clusterCounts[cluster]++

		for j := 0; j < nFeatures; j++ {
			current := newCentroids.At(cluster, j)
			newCentroids.Set(cluster, j, current+X.At(i, j))
		}
	}

	// Average to get new centroids
	for c := 0; c < km.k; c++ {
		if clusterCounts[c] > 0 {
			for j := 0; j < nFeatures; j++ {
				avg := newCentroids.At(c, j) / float64(clusterCounts[c])
				newCentroids.Set(c, j, avg)
			}
		} else {
			// Handle empty cluster by reinitializing randomly
			for j := 0; j < nFeatures; j++ {
				randomIdx := km.rng.Intn(nSamples)
				newCentroids.Set(c, j, X.At(randomIdx, j))
			}
		}
	}

	return newCentroids
}

// calculateInertia calculates the within-cluster sum of squares.
func (km *KMeans) calculateInertia(X core.Tensor) float64 {
	nSamples, _ := X.Dims()
	inertia := 0.0

	for i := 0; i < nSamples; i++ {
		point := X.Row(i)
		centroid := km.centroids.Row(km.labels[i])
		inertia += km.euclideanDistanceSquared(point, centroid)
	}

	return inertia
}

// euclideanDistanceSquared calculates the squared Euclidean distance between two points.
func (km *KMeans) euclideanDistanceSquared(a, b core.Tensor) float64 {
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
