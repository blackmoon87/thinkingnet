package main

import (
	"fmt"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func main() {
	fmt.Println("=== ThinkingNet-Go Bilingual Error Messages Demo ===")
	fmt.Println()

	// Demonstrate different types of bilingual errors
	demonstrateErrors()
}

func demonstrateErrors() {
	fmt.Println("1. Demonstrating common error messages:")
	fmt.Println()

	// Test nil tensor error
	fmt.Println("• Nil Tensor Error:")
	err := core.NewCommonError(core.ErrInvalidInput, "nil_tensor")
	printError(err)

	// Test model not compiled error
	fmt.Println("• Model Not Compiled Error:")
	err = core.NewCommonError(core.ErrModelNotCompiled, "model_not_compiled")
	printError(err)

	// Test NaN values error
	fmt.Println("• NaN Values Error:")
	err = core.NewCommonError(core.ErrNumericalInstability, "nan_values")
	printError(err)

	fmt.Println("\n2. Demonstrating custom bilingual errors:")
	fmt.Println()

	// Test custom bilingual error
	fmt.Println("• Custom Bilingual Error:")
	err = core.NewBilingualError(
		core.ErrInvalidInput,
		"معدل التعلم يجب أن يكون بين 0.001 و 1.0",
		"learning rate must be between 0.001 and 1.0",
		"جرب قيمة مثل 0.01 | Try a value like 0.01",
	)
	printError(err)

	fmt.Println("\n3. Demonstrating validation function errors:")
	fmt.Println()

	// Test validation errors
	fmt.Println("• Range Validation Error:")
	rangeErr := core.ValidateRange(-0.5, 0.0, 1.0, "learning_rate")
	if rangeErr != nil {
		printError(rangeErr)
	}

	fmt.Println("• Positive Validation Error:")
	positiveErr := core.ValidatePositive(-1.0, "batch_size")
	if positiveErr != nil {
		printError(positiveErr)
	}

	fmt.Println("\n4. Demonstrating error with cause:")
	fmt.Println()

	originalErr := fmt.Errorf("division by zero")
	err = core.NewBilingualErrorWithCause(
		core.ErrNumericalInstability,
		"خطأ في العملية الحسابية",
		"numerical computation error",
		"تحقق من القيم المدخلة | Check input values",
		originalErr,
	)
	printError(err)
}

func printError(err error) {
	if tnErr, ok := err.(*core.ThinkingNetError); ok {
		fmt.Printf("  Full Error: %s\n", tnErr.Error())
		fmt.Printf("  English: %s\n", tnErr.ErrorEnglish())
		fmt.Printf("  Arabic: %s\n", tnErr.ErrorArabic())
		if tip := tnErr.GetTip(); tip != "" {
			fmt.Printf("  Tip: %s\n", tip)
		}
		fmt.Println()
	} else {
		fmt.Printf("  Error: %s\n\n", err.Error())
	}
}
