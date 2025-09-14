package main

import (
	"fmt"
	"log"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func main() {
	fmt.Println("ThinkingNet Tensor Demo")
	fmt.Println("======================")

	// Create tensors from slices
	fmt.Println("\n1. Creating tensors:")
	a := core.NewTensorFromSlice([][]float64{{1, 2, 3}, {4, 5, 6}})
	b := core.NewTensorFromSlice([][]float64{{2, 3, 4}, {5, 6, 7}})

	fmt.Printf("Tensor A:\n%s\n", a.String())
	fmt.Printf("Tensor B:\n%s\n", b.String())

	// Basic arithmetic operations
	fmt.Println("\n2. Arithmetic operations:")

	// Addition
	c := a.Add(b)
	fmt.Printf("A + B:\n%s\n", c.String())

	// Subtraction
	d := a.Sub(b)
	fmt.Printf("A - B:\n%s\n", d.String())

	// Element-wise multiplication
	e := a.MulElem(b)
	fmt.Printf("A ⊙ B (element-wise):\n%s\n", e.String())

	// Matrix multiplication (need compatible dimensions)
	fmt.Println("\n3. Matrix multiplication:")
	f := core.NewTensorFromSlice([][]float64{{1, 2}, {3, 4}, {5, 6}})
	g := a.Mul(f) // (2x3) × (3x2) = (2x2)
	fmt.Printf("A × F:\n%s\n", g.String())

	// Mathematical functions
	fmt.Println("\n4. Mathematical functions:")
	h := core.NewTensorFromSlice([][]float64{{1, 4, 9}, {16, 25, 36}})
	fmt.Printf("Original tensor H:\n%s\n", h.String())

	sqrt_h := h.Sqrt()
	fmt.Printf("√H:\n%s\n", sqrt_h.String())

	pow_h := h.Pow(0.5)
	fmt.Printf("H^0.5:\n%s\n", pow_h.String())

	// Statistics
	fmt.Println("\n5. Statistics:")
	fmt.Printf("Sum of A: %.2f\n", a.Sum())
	fmt.Printf("Mean of A: %.2f\n", a.Mean())
	fmt.Printf("Max of A: %.2f\n", a.Max())
	fmt.Printf("Min of A: %.2f\n", a.Min())
	fmt.Printf("Standard deviation of A: %.2f\n", a.Std())
	fmt.Printf("Frobenius norm of A: %.2f\n", a.Norm())

	// Shape operations
	fmt.Println("\n6. Shape operations:")
	fmt.Printf("A shape: %v\n", a.Shape())

	// Transpose
	at := a.T()
	fmt.Printf("A transpose:\n%s\n", at.String())

	// Reshape
	reshaped := a.Reshape(3, 2)
	fmt.Printf("A reshaped to (3,2):\n%s\n", reshaped.String())

	// Flatten
	flattened := a.Flatten()
	fmt.Printf("A flattened:\n%s\n", flattened.String())

	// Slicing operations
	fmt.Println("\n7. Slicing operations:")
	row1 := a.Row(1)
	fmt.Printf("Row 1 of A:\n%s\n", row1.String())

	col2 := a.Col(2)
	fmt.Printf("Column 2 of A:\n%s\n", col2.String())

	// Utility operations
	fmt.Println("\n8. Utility operations:")

	// Clamp values
	i := core.NewTensorFromSlice([][]float64{{-2, 0, 3}, {5, -1, 8}})
	clamped := i.Clamp(0, 5)
	fmt.Printf("Original: %s\n", i.String())
	fmt.Printf("Clamped [0,5]: %s\n", clamped.String())

	// Apply custom function
	doubled := a.Apply(func(i, j int, v float64) float64 {
		return v * 2
	})
	fmt.Printf("A doubled (custom function):\n%s\n", doubled.String())

	// Tensor properties
	fmt.Println("\n9. Tensor properties:")
	square := core.NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	vector := core.NewTensorFromSlice([][]float64{{1, 2, 3}})

	fmt.Printf("Square tensor is square: %t\n", square.IsSquare())
	fmt.Printf("Vector tensor is vector: %t\n", vector.IsVector())
	fmt.Printf("Square tensor is vector: %t\n", square.IsVector())

	// Memory management demonstration
	fmt.Println("\n10. Memory management:")
	fmt.Printf("Matrix pool stats before: %v\n", core.MatrixPoolStats())

	// Create and release tensors
	for i := 0; i < 5; i++ {
		temp := core.NewZerosTensor(10, 10)
		temp.Fill(float64(i))
		temp.Release() // Return to pool
	}

	fmt.Printf("Matrix pool stats after: %v\n", core.MatrixPoolStats())

	// Configuration
	fmt.Println("\n11. Configuration:")
	fmt.Printf("Current epsilon: %e\n", core.GetEpsilon())
	fmt.Printf("Pooling enabled: %t\n", core.IsPoolingEnabled())
	fmt.Printf("Debug mode: %t\n", core.IsDebugMode())

	// Error handling demonstration
	fmt.Println("\n12. Error handling:")
	defer func() {
		if r := recover(); r != nil {
			if err, ok := r.(*core.ThinkingNetError); ok {
				fmt.Printf("Caught ThinkingNet error: %s\n", err.Error())
			} else {
				log.Printf("Caught panic: %v", r)
			}
		}
	}()

	// This will cause a dimension mismatch error
	incompatible1 := core.NewTensorFromSlice([][]float64{{1, 2}})
	incompatible2 := core.NewTensorFromSlice([][]float64{{1}, {2}})
	_ = incompatible1.Add(incompatible2) // This will panic with our error

	fmt.Println("\nDemo completed successfully!")
}
