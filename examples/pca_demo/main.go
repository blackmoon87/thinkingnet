package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/algorithms"
	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func main() {
	fmt.Println("=== PCA Dimensionality Reduction Demo ===")

	// Set random seed for reproducibility
	rng := rand.New(rand.NewSource(42))

	// Create synthetic 3D data with correlation
	fmt.Println("\n1. Creating synthetic 3D dataset...")
	nSamples := 100
	data := make([][]float64, nSamples)

	for i := range nSamples {
		// Create correlated data: z = x + y + noise
		x := rng.NormFloat64() * 2
		y := rng.NormFloat64() * 1.5
		z := x + y + rng.NormFloat64()*0.1 // Strong correlation with small noise

		data[i] = []float64{x, y, z}
	}

	X := core.NewTensorFromSlice(data)
	fmt.Printf("Original data shape: %v\n", X.Shape())

	// Demonstrate basic PCA
	fmt.Println("\n2. Fitting PCA with 2 components...")
	pca := algorithms.NewPCA(2)

	err := pca.Fit(X)
	if err != nil {
		fmt.Printf("Error fitting PCA: %v\n", err)
		return
	}

	fmt.Printf("PCA fitted successfully!\n")
	fmt.Printf("Number of components: %d\n", pca.NComponents())

	// Show explained variance
	explainedVar := pca.ExplainedVariance()
	explainedVarRatio := pca.ExplainedVarianceRatio()

	fmt.Println("\n3. Explained Variance Analysis:")
	totalExplained := 0.0
	for i, ratio := range explainedVarRatio {
		fmt.Printf("Component %d: %.4f variance (%.2f%%)\n",
			i+1, explainedVar[i], ratio*100)
		totalExplained += ratio
	}
	fmt.Printf("Total explained variance: %.2f%%\n", totalExplained*100)

	// Transform the data
	fmt.Println("\n4. Transforming data to 2D...")
	transformed, err := pca.Transform(X)
	if err != nil {
		fmt.Printf("Error transforming data: %v\n", err)
		return
	}

	fmt.Printf("Transformed data shape: %v\n", transformed.Shape())

	// Show some sample transformations
	fmt.Println("\nSample transformations (first 5 points):")
	fmt.Println("Original (3D) -> Transformed (2D)")
	for i := range 5 {
		fmt.Printf("[%.2f, %.2f, %.2f] -> [%.2f, %.2f]\n",
			X.At(i, 0), X.At(i, 1), X.At(i, 2),
			transformed.At(i, 0), transformed.At(i, 1))
	}

	// Demonstrate inverse transform
	fmt.Println("\n5. Reconstructing original data...")
	reconstructed, err := pca.InverseTransform(transformed)
	if err != nil {
		fmt.Printf("Error reconstructing data: %v\n", err)
		return
	}

	fmt.Printf("Reconstructed data shape: %v\n", reconstructed.Shape())

	// Calculate reconstruction error
	fmt.Println("\nReconstruction quality (first 5 points):")
	fmt.Println("Original -> Reconstructed (Error)")
	totalError := 0.0
	for i := range 5 {
		error := 0.0
		for j := range 3 {
			diff := X.At(i, j) - reconstructed.At(i, j)
			error += diff * diff
		}
		error = error / 3 // Mean squared error per feature
		totalError += error

		fmt.Printf("[%.2f, %.2f, %.2f] -> [%.2f, %.2f, %.2f] (MSE: %.4f)\n",
			X.At(i, 0), X.At(i, 1), X.At(i, 2),
			reconstructed.At(i, 0), reconstructed.At(i, 1), reconstructed.At(i, 2),
			error)
	}
	avgError := totalError / 5
	fmt.Printf("Average reconstruction MSE: %.4f\n", avgError)

	// Demonstrate PCA with whitening
	fmt.Println("\n6. PCA with Whitening...")
	pcaWhiten := algorithms.NewPCA(2, algorithms.WithWhiten(true))

	transformedWhiten, err := pcaWhiten.FitTransform(X)
	if err != nil {
		fmt.Printf("Error with whitened PCA: %v\n", err)
		return
	}

	fmt.Println("Whitened transformation (first 5 points):")
	for i := range 5 {
		fmt.Printf("[%.2f, %.2f]\n", transformedWhiten.At(i, 0), transformedWhiten.At(i, 1))
	}

	// Compare variance of whitened vs non-whitened components
	fmt.Println("\n7. Comparing component variances:")

	// Calculate variance of each component in transformed data
	fmt.Println("Non-whitened component variances:")
	for comp := range 2 {
		var sum, sumSq float64
		for i := range nSamples {
			val := transformed.At(i, comp)
			sum += val
			sumSq += val * val
		}
		mean := sum / float64(nSamples)
		variance := (sumSq / float64(nSamples)) - (mean * mean)
		fmt.Printf("Component %d variance: %.4f\n", comp+1, variance)
	}

	fmt.Println("Whitened component variances:")
	for comp := range 2 {
		var sum, sumSq float64
		for i := range nSamples {
			val := transformedWhiten.At(i, comp)
			sum += val
			sumSq += val * val
		}
		mean := sum / float64(nSamples)
		variance := (sumSq / float64(nSamples)) - (mean * mean)
		fmt.Printf("Component %d variance: %.4f\n", comp+1, variance)
	}

	// Demonstrate automatic component selection
	fmt.Println("\n8. Automatic Component Selection...")
	pcaAuto := algorithms.NewPCA(0) // 0 means auto-select all components

	err = pcaAuto.Fit(X)
	if err != nil {
		fmt.Printf("Error with auto PCA: %v\n", err)
		return
	}

	fmt.Printf("Auto-selected components: %d\n", pcaAuto.NComponents())
	autoExplainedRatio := pcaAuto.ExplainedVarianceRatio()
	fmt.Println("All component explained variance ratios:")
	for i, ratio := range autoExplainedRatio {
		fmt.Printf("Component %d: %.2f%%\n", i+1, ratio*100)
	}

	// Performance demonstration
	fmt.Println("\n9. Performance Test...")
	start := time.Now()

	// Create larger dataset
	largeData := make([][]float64, 1000)
	for i := range 1000 {
		x := rng.NormFloat64()
		y := rng.NormFloat64()
		z := x + y + rng.NormFloat64()*0.1
		w := x - y + rng.NormFloat64()*0.1
		largeData[i] = []float64{x, y, z, w}
	}

	XLarge := core.NewTensorFromSlice(largeData)
	pcaLarge := algorithms.NewPCA(2)

	err = pcaLarge.Fit(XLarge)
	if err != nil {
		fmt.Printf("Error with large dataset: %v\n", err)
		return
	}

	_, err = pcaLarge.Transform(XLarge)
	if err != nil {
		fmt.Printf("Error transforming large dataset: %v\n", err)
		return
	}

	elapsed := time.Since(start)
	fmt.Printf("Processed 1000x4 dataset in %v\n", elapsed)

	fmt.Println("\n=== PCA Demo Complete ===")
}
