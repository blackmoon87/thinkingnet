package algorithms

import (
	"math"
	"sort"

	"github.com/blackmoon87/thinkingnet/pkg/core"
	"gonum.org/v1/gonum/mat"
)

// PCA implements Principal Component Analysis for dimensionality reduction.
type PCA struct {
	// Configuration
	nComponents int   // Number of components to keep
	whiten      bool  // Whether to whiten the components
	randomSeed  int64 // Random seed for reproducibility

	// State
	fitted            bool        // Whether the model has been fitted
	components        core.Tensor // Principal components (eigenvectors)
	explainedVariance []float64   // Explained variance for each component
	explainedVarRatio []float64   // Explained variance ratio for each component
	singularValues    []float64   // Singular values from SVD
	mean              core.Tensor // Mean of the training data
	nSamples          int         // Number of training samples
	nFeatures         int         // Number of original features
	totalVariance     float64     // Total variance in the data
}

// NewPCA creates a new PCA instance.
func NewPCA(nComponents int, options ...PCAOption) *PCA {
	pca := &PCA{
		nComponents: nComponents,
		whiten:      false,
		randomSeed:  42,
		fitted:      false,
	}

	// Apply options
	for _, option := range options {
		option(pca)
	}

	return pca
}

// PCAOption represents a functional option for PCA configuration.
type PCAOption func(*PCA)

// WithWhiten sets whether to whiten the components.
func WithWhiten(whiten bool) PCAOption {
	return func(pca *PCA) { pca.whiten = whiten }
}

// WithPCARandomSeed sets the random seed.
func WithPCARandomSeed(seed int64) PCAOption {
	return func(pca *PCA) { pca.randomSeed = seed }
}

// Fit learns the principal components from the data.
func (pca *PCA) Fit(X core.Tensor) error {
	if err := pca.validateInput(X); err != nil {
		return err
	}

	nSamples, nFeatures := X.Dims()
	pca.nSamples = nSamples
	pca.nFeatures = nFeatures

	// Validate number of components
	if pca.nComponents <= 0 {
		pca.nComponents = min(nSamples, nFeatures)
	} else if pca.nComponents > min(nSamples, nFeatures) {
		return core.NewError(core.ErrInvalidInput,
			"n_components cannot be larger than min(n_samples, n_features)")
	}

	// Center the data by computing and subtracting the mean
	pca.mean = pca.computeMean(X)
	XCentered := pca.centerData(X)

	// Perform SVD on the centered data
	if err := pca.performSVD(XCentered); err != nil {
		return err
	}

	pca.fitted = true
	return nil
}

// Transform applies the dimensionality reduction to the data.
func (pca *PCA) Transform(X core.Tensor) (core.Tensor, error) {
	if !pca.fitted {
		return nil, core.NewError(core.ErrNotFitted, "PCA must be fitted before transformation")
	}

	if err := pca.validateTransformInput(X); err != nil {
		return nil, err
	}

	// Center the data using the training mean
	XCentered := pca.centerDataWithMean(X, pca.mean)

	// Project onto principal components
	transformed := XCentered.Mul(pca.components.T())

	// Apply whitening if requested
	if pca.whiten {
		transformed = pca.applyWhitening(transformed)
	}

	return transformed, nil
}

// FitTransform fits the PCA and transforms the data in one step.
func (pca *PCA) FitTransform(X core.Tensor) (core.Tensor, error) {
	if err := pca.Fit(X); err != nil {
		return nil, err
	}
	return pca.Transform(X)
}

// InverseTransform transforms data back to the original space.
func (pca *PCA) InverseTransform(X core.Tensor) (core.Tensor, error) {
	if !pca.fitted {
		return nil, core.NewError(core.ErrNotFitted, "PCA must be fitted before inverse transformation")
	}

	nSamples, nComponents := X.Dims()
	if nComponents != pca.nComponents {
		return nil, core.NewError(core.ErrDimensionMismatch,
			"input must have same number of components as fitted PCA")
	}

	// Reverse whitening if it was applied
	XUnwhitened := X
	if pca.whiten {
		XUnwhitened = pca.reverseWhitening(X)
	}

	// Project back to original space
	XReconstructed := XUnwhitened.Mul(pca.components)

	// Add back the mean
	for i := range nSamples {
		for j := range pca.nFeatures {
			current := XReconstructed.At(i, j)
			XReconstructed.Set(i, j, current+pca.mean.At(0, j))
		}
	}

	return XReconstructed, nil
}

// Components returns the principal components.
func (pca *PCA) Components() core.Tensor {
	if !pca.fitted {
		return nil
	}
	return pca.components.Copy()
}

// ExplainedVariance returns the explained variance for each component.
func (pca *PCA) ExplainedVariance() []float64 {
	if !pca.fitted {
		return nil
	}
	result := make([]float64, len(pca.explainedVariance))
	copy(result, pca.explainedVariance)
	return result
}

// ExplainedVarianceRatio returns the explained variance ratio for each component.
func (pca *PCA) ExplainedVarianceRatio() []float64 {
	if !pca.fitted {
		return nil
	}
	result := make([]float64, len(pca.explainedVarRatio))
	copy(result, pca.explainedVarRatio)
	return result
}

// SingularValues returns the singular values.
func (pca *PCA) SingularValues() []float64 {
	if !pca.fitted {
		return nil
	}
	result := make([]float64, len(pca.singularValues))
	copy(result, pca.singularValues)
	return result
}

// Mean returns the mean of the training data.
func (pca *PCA) Mean() core.Tensor {
	if !pca.fitted {
		return nil
	}
	return pca.mean.Copy()
}

// Name returns the algorithm name.
func (pca *PCA) Name() string {
	return "PCA"
}

// NComponents returns the number of components.
func (pca *PCA) NComponents() int {
	return pca.nComponents
}

// validateInput validates the input data for fitting.
func (pca *PCA) validateInput(X core.Tensor) error {
	if X == nil {
		return core.NewError(core.ErrInvalidInput, "input data cannot be nil")
	}

	nSamples, nFeatures := X.Dims()
	if nSamples <= 0 || nFeatures <= 0 {
		return core.NewError(core.ErrInvalidInput, "input data cannot be empty")
	}

	if nSamples < 2 {
		return core.NewError(core.ErrInvalidInput, "PCA requires at least 2 samples")
	}

	if !X.IsFinite() {
		return core.NewError(core.ErrNumericalInstability,
			"input data contains NaN or infinite values")
	}

	return nil
}

// validateTransformInput validates input for transformation.
func (pca *PCA) validateTransformInput(X core.Tensor) error {
	if X == nil {
		return core.NewError(core.ErrInvalidInput, "input data cannot be nil")
	}

	_, nFeatures := X.Dims()
	if nFeatures != pca.nFeatures {
		return core.NewError(core.ErrDimensionMismatch,
			"input must have same number of features as training data")
	}

	if !X.IsFinite() {
		return core.NewError(core.ErrNumericalInstability,
			"input data contains NaN or infinite values")
	}

	return nil
}

// computeMean computes the mean of each feature.
func (pca *PCA) computeMean(X core.Tensor) core.Tensor {
	nSamples, nFeatures := X.Dims()
	mean := core.NewZerosTensor(1, nFeatures)

	for j := range nFeatures {
		var sum float64
		for i := range nSamples {
			sum += X.At(i, j)
		}
		mean.Set(0, j, sum/float64(nSamples))
	}

	return mean
}

// centerData centers the data by subtracting the mean.
func (pca *PCA) centerData(X core.Tensor) core.Tensor {
	return pca.centerDataWithMean(X, pca.mean)
}

// centerDataWithMean centers data using a provided mean.
func (pca *PCA) centerDataWithMean(X, mean core.Tensor) core.Tensor {
	nSamples, nFeatures := X.Dims()
	centered := core.NewZerosTensor(nSamples, nFeatures)

	for i := range nSamples {
		for j := range nFeatures {
			centered.Set(i, j, X.At(i, j)-mean.At(0, j))
		}
	}

	return centered
}

// performSVD performs Singular Value Decomposition on the centered data.
func (pca *PCA) performSVD(XCentered core.Tensor) error {
	nSamples, nFeatures := XCentered.Dims()

	// For efficiency, we choose between X^T*X or X*X^T based on dimensions
	var covMatrix core.Tensor
	var useTranspose bool

	if nSamples > nFeatures {
		// More samples than features: compute X^T * X
		covMatrix = XCentered.T().Mul(XCentered).Scale(1.0 / float64(nSamples-1))
		useTranspose = false
	} else {
		// More features than samples: compute X * X^T
		covMatrix = XCentered.Mul(XCentered.T()).Scale(1.0 / float64(nSamples-1))
		useTranspose = true
	}

	// Perform eigendecomposition using gonum's SVD
	var svd mat.SVD
	if !svd.Factorize(covMatrix.RawMatrix(), mat.SVDThin) {
		return core.NewError(core.ErrNumericalInstability, "SVD factorization failed")
	}

	// Get singular values and vectors
	var u, vt mat.Dense
	svd.UTo(&u)
	svd.VTo(&vt)
	singularValues := svd.Values(nil)

	// Sort eigenvalues and eigenvectors in descending order
	pca.sortEigenComponents(singularValues, &u, &vt, useTranspose, XCentered)

	// Compute explained variance
	pca.computeExplainedVariance()

	return nil
}

// sortEigenComponents sorts eigenvalues and eigenvectors in descending order.
func (pca *PCA) sortEigenComponents(singularValues []float64, u, vt *mat.Dense, useTranspose bool, XCentered core.Tensor) {
	// Create indices for sorting
	indices := make([]int, len(singularValues))
	for i := range indices {
		indices[i] = i
	}

	// Sort indices by singular values in descending order
	sort.Slice(indices, func(i, j int) bool {
		return singularValues[indices[i]] > singularValues[indices[j]]
	})

	// Extract the top n_components
	nComponents := min(pca.nComponents, len(singularValues))
	pca.singularValues = make([]float64, nComponents)

	var components *mat.Dense
	if useTranspose {
		// When using X*X^T, we need to compute the eigenvectors of X^T*X
		// from the eigenvectors of X*X^T
		uRows, _ := u.Dims()
		components = mat.NewDense(nComponents, pca.nFeatures, nil)

		for i := range nComponents {
			idx := indices[i]
			pca.singularValues[i] = singularValues[idx]

			// Compute eigenvector: X^T * u_i / sigma_i
			uCol := mat.Col(nil, idx, u)
			if len(uCol) != uRows {
				continue
			}

			// Create tensor from column
			uTensor := core.NewTensorFromData(uRows, 1, uCol)

			// Compute X^T * u_i
			eigenvec := XCentered.T().Mul(uTensor)

			// Normalize by singular value
			if pca.singularValues[i] > 1e-10 {
				eigenvec = eigenvec.Scale(1.0 / pca.singularValues[i])
			}

			// Normalize the eigenvector
			norm := eigenvec.Norm()
			if norm > 1e-10 {
				eigenvec = eigenvec.Scale(1.0 / norm)
			}

			// Set the component
			for j := range pca.nFeatures {
				components.Set(i, j, eigenvec.At(j, 0))
			}
		}
	} else {
		// Direct case: eigenvectors are in vt (transposed)
		components = mat.NewDense(nComponents, pca.nFeatures, nil)

		for i := range nComponents {
			idx := indices[i]
			pca.singularValues[i] = singularValues[idx]

			// Copy the eigenvector (row from vt)
			for j := range pca.nFeatures {
				components.Set(i, j, vt.At(idx, j))
			}
		}
	}

	pca.components = core.NewTensor(components)
}

// computeExplainedVariance computes the explained variance and variance ratios.
func (pca *PCA) computeExplainedVariance() {
	nComponents := len(pca.singularValues)
	pca.explainedVariance = make([]float64, nComponents)

	// Convert singular values to explained variance
	// For SVD of covariance matrix: explained_variance = singular_values
	totalVar := 0.0
	for i := range nComponents {
		pca.explainedVariance[i] = pca.singularValues[i]
		totalVar += pca.explainedVariance[i]
	}

	pca.totalVariance = totalVar

	// Compute explained variance ratios
	pca.explainedVarRatio = make([]float64, nComponents)
	if totalVar > 0 {
		for i := range nComponents {
			pca.explainedVarRatio[i] = pca.explainedVariance[i] / totalVar
		}
	}
}

// applyWhitening applies whitening transformation to the data.
func (pca *PCA) applyWhitening(X core.Tensor) core.Tensor {
	nSamples, nComponents := X.Dims()
	whitened := core.NewZerosTensor(nSamples, nComponents)

	for i := range nSamples {
		for j := range nComponents {
			if pca.explainedVariance[j] > 1e-10 {
				val := X.At(i, j) / math.Sqrt(pca.explainedVariance[j])
				whitened.Set(i, j, val)
			} else {
				whitened.Set(i, j, 0.0)
			}
		}
	}

	return whitened
}

// reverseWhitening reverses the whitening transformation.
func (pca *PCA) reverseWhitening(X core.Tensor) core.Tensor {
	nSamples, nComponents := X.Dims()
	unwhitened := core.NewZerosTensor(nSamples, nComponents)

	for i := range nSamples {
		for j := range nComponents {
			val := X.At(i, j) * math.Sqrt(pca.explainedVariance[j])
			unwhitened.Set(i, j, val)
		}
	}

	return unwhitened
}

// min returns the minimum of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
