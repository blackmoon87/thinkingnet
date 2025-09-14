package main

import (
	"fmt"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func main() {
	fmt.Println("=== ThinkingNet Helper Functions Demo ===")

	// Demonstrate EasyTensor function
	fmt.Println("\n1. EasyTensor Demo:")
	fmt.Println("Creating a tensor from 2D slice...")

	data := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
		{10.0, 11.0, 12.0},
	}

	X := core.EasyTensor(data)
	rows, cols := X.Dims()
	fmt.Printf("Created tensor with dimensions: (%d, %d)\n", rows, cols)
	fmt.Printf("Tensor content:\n%s\n", X.String())

	// Demonstrate EasySplit function
	fmt.Println("\n2. EasySplit Demo:")
	fmt.Println("Creating labels and splitting data...")

	// Create labels (y)
	labels := [][]float64{
		{0.0},
		{1.0},
		{0.0},
		{1.0},
	}

	y := core.EasyTensor(labels)
	yRows, yCols := y.Dims()
	fmt.Printf("Labels tensor dimensions: (%d, %d)\n", yRows, yCols)

	// Split the data
	fmt.Println("Splitting data with 30% test size...")
	XTrain, XTest, yTrain, yTest := core.EasySplit(X, y, 0.3)

	trainRows, trainCols := XTrain.Dims()
	testRows, testCols := XTest.Dims()

	fmt.Printf("Training set: (%d, %d)\n", trainRows, trainCols)
	fmt.Printf("Test set: (%d, %d)\n", testRows, testCols)

	fmt.Printf("\nTraining data:\n%s\n", XTrain.String())
	fmt.Printf("Training labels:\n%s\n", yTrain.String())

	fmt.Printf("\nTest data:\n%s\n", XTest.String())
	fmt.Printf("Test labels:\n%s\n", yTest.String())

	// Demonstrate error handling
	fmt.Println("\n3. Error Handling Demo:")
	fmt.Println("Trying to create tensor with inconsistent row lengths...")

	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("Caught expected error: %v\n", r)
		}
	}()

	// This should panic with a helpful error message
	badData := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0, 5.0}, // Different length!
	}

	core.EasyTensor(badData)

	fmt.Println("\n=== Demo Complete ===")
}
