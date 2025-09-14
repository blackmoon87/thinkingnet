package main

import (
	"fmt"
	"math"

	"github.com/blackmoon87/thinkingnet/pkg/activations"
	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func main() {
	fmt.Println("ThinkingNet Activation Functions Demo")
	fmt.Println("====================================")

	// Test input values
	testInputs := []float64{-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0}

	// Create all activation functions
	activationFuncs := []core.Activation{
		activations.NewReLU(),
		activations.NewLeakyReLU(0.01),
		activations.NewELU(1.0),
		activations.NewSigmoid(),
		activations.NewTanh(),
		activations.NewLinear(),
		activations.NewSwish(),
		activations.NewGELU(),
	}

	fmt.Println("\n1. Forward Pass Comparison:")
	fmt.Printf("%-12s", "Input")
	for _, activation := range activationFuncs {
		fmt.Printf("%-12s", activation.Name())
	}
	fmt.Println()

	for _, input := range testInputs {
		fmt.Printf("%-12.2f", input)
		for _, activation := range activationFuncs {
			output := activation.Forward(input)
			fmt.Printf("%-12.4f", output)
		}
		fmt.Println()
	}

	fmt.Println("\n2. Backward Pass (Derivatives):")
	fmt.Printf("%-12s", "Input")
	for _, activation := range activationFuncs {
		fmt.Printf("%-12s", activation.Name())
	}
	fmt.Println()

	for _, input := range testInputs {
		fmt.Printf("%-12.2f", input)
		for _, activation := range activationFuncs {
			derivative := activation.Backward(input)
			fmt.Printf("%-12.4f", derivative)
		}
		fmt.Println()
	}

	// Demonstrate Softmax with tensor input
	fmt.Println("\n3. Softmax Demonstration:")
	softmax := activations.NewSoftmax()

	// Create test tensor
	logits := core.NewTensorFromSlice([][]float64{
		{1.0, 2.0, 3.0, 4.0},
		{0.0, 0.0, 0.0, 0.0},
		{-1.0, 0.0, 1.0, 2.0},
	})

	fmt.Printf("Input logits:\n%s\n", logits.String())

	softmaxOutput := softmax.ApplyTensorwise(logits)
	fmt.Printf("Softmax output:\n%s\n", softmaxOutput.String())

	// Verify that each row sums to 1
	fmt.Println("Row sums (should be ~1.0):")
	rows, cols := softmaxOutput.Dims()
	for i := 0; i < rows; i++ {
		sum := 0.0
		for j := 0; j < cols; j++ {
			sum += softmaxOutput.At(i, j)
		}
		fmt.Printf("Row %d: %.6f\n", i, sum)
	}

	// Demonstrate activation registry
	fmt.Println("\n4. Activation Registry:")
	availableActivations := activations.ListActivations()
	fmt.Printf("Available activations: %v\n", availableActivations)

	// Get activation by name
	reluFromRegistry, err := activations.GetActivation("relu")
	if err != nil {
		fmt.Printf("Error getting ReLU: %v\n", err)
	} else {
		fmt.Printf("ReLU from registry: %s\n", reluFromRegistry.Name())
		fmt.Printf("ReLU(2.5) = %.4f\n", reluFromRegistry.Forward(2.5))
	}

	// Register custom activation
	activations.RegisterActivation("custom_relu", func() core.Activation {
		return activations.NewLeakyReLU(0.1) // Custom LeakyReLU with alpha=0.1
	})

	customActivation, err := activations.GetActivation("custom_relu")
	if err != nil {
		fmt.Printf("Error getting custom activation: %v\n", err)
	} else {
		fmt.Printf("Custom activation: %s\n", customActivation.Name())
		fmt.Printf("Custom activation(-1.0) = %.4f\n", customActivation.Forward(-1.0))
	}

	// Demonstrate numerical stability
	fmt.Println("\n5. Numerical Stability Test:")
	extremeValues := []float64{-1000.0, -100.0, 100.0, 1000.0}

	stableActivations := []core.Activation{
		activations.NewSigmoid(),
		activations.NewTanh(),
		activations.NewReLU(),
	}

	for _, activation := range stableActivations {
		fmt.Printf("\n%s with extreme values:\n", activation.Name())
		for _, x := range extremeValues {
			forward := activation.Forward(x)
			backward := activation.Backward(x)
			fmt.Printf("  x=%-8.1f -> f(x)=%-12.6f, f'(x)=%-12.6f\n", x, forward, backward)
		}
	}

	// Demonstrate gradient checking
	fmt.Println("\n6. Gradient Checking Example:")
	testActivation := activations.NewSigmoid()
	testPoint := 1.0
	h := 1e-5

	// Analytical gradient
	analytical := testActivation.Backward(testPoint)

	// Numerical gradient
	numerical := (testActivation.Forward(testPoint+h) - testActivation.Forward(testPoint-h)) / (2 * h)

	fmt.Printf("Gradient check for %s at x=%.1f:\n", testActivation.Name(), testPoint)
	fmt.Printf("  Analytical: %.8f\n", analytical)
	fmt.Printf("  Numerical:  %.8f\n", numerical)
	fmt.Printf("  Difference: %.2e\n", math.Abs(analytical-numerical))

	// Performance comparison
	fmt.Println("\n7. Performance Characteristics:")
	fmt.Println("(Note: Actual benchmarks should be run with 'go test -bench=.')")

	performanceActivations := []core.Activation{
		activations.NewReLU(),
		activations.NewSigmoid(),
		activations.NewTanh(),
		activations.NewSwish(),
	}

	testValue := 1.5
	iterations := 1000000

	for _, activation := range performanceActivations {
		// Simple timing (not as accurate as proper benchmarks)
		result := 0.0
		for i := 0; i < iterations; i++ {
			result += activation.Forward(testValue)
		}
		fmt.Printf("%s: Computed %d forward passes (result: %.2f)\n",
			activation.Name(), iterations, result/float64(iterations))
	}

	// Activation function properties
	fmt.Println("\n8. Activation Function Properties:")

	properties := map[string]map[string]string{
		"relu": {
			"Range":         "[0, +∞)",
			"Derivative":    "0 or 1",
			"Zero-centered": "No",
			"Saturating":    "Left side only",
		},
		"sigmoid": {
			"Range":         "(0, 1)",
			"Derivative":    "(0, 0.25]",
			"Zero-centered": "No",
			"Saturating":    "Both sides",
		},
		"tanh": {
			"Range":         "(-1, 1)",
			"Derivative":    "(0, 1]",
			"Zero-centered": "Yes",
			"Saturating":    "Both sides",
		},
		"swish": {
			"Range":         "(-∞, +∞)",
			"Derivative":    "Varies",
			"Zero-centered": "Approximately",
			"Saturating":    "Left side only",
		},
	}

	for name, props := range properties {
		fmt.Printf("\n%s:\n", name)
		for prop, value := range props {
			fmt.Printf("  %-15s: %s\n", prop, value)
		}
	}

	fmt.Println("\nDemo completed successfully!")
}
