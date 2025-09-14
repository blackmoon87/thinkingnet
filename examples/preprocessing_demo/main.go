package main

import (
	"fmt"
	"log"

	"github.com/blackmoon87/thinkingnet/pkg/core"
	"github.com/blackmoon87/thinkingnet/pkg/preprocessing"
)

func main() {
	fmt.Println("ThinkingNet Preprocessing Demo")
	fmt.Println("==============================")

	// Demo data preprocessing utilities
	demoStandardScaler()
	fmt.Println()
	demoMinMaxScaler()
	fmt.Println()
	demoOneHotEncoder()
	fmt.Println()
	demoLabelEncoder()
}

func demoStandardScaler() {
	fmt.Println("StandardScaler Demo:")
	fmt.Println("-------------------")

	// Create sample data
	data := [][]float64{
		{1.0, 100.0, 0.1},
		{2.0, 200.0, 0.2},
		{3.0, 300.0, 0.3},
		{4.0, 400.0, 0.4},
		{5.0, 500.0, 0.5},
	}
	tensor := core.NewTensorFromSlice(data)

	fmt.Printf("Original data:\n%s\n", tensor.String())

	// Create and fit StandardScaler
	scaler := preprocessing.NewStandardScaler()
	err := scaler.Fit(tensor)
	if err != nil {
		log.Fatalf("Failed to fit StandardScaler: %v", err)
	}

	// Transform the data
	scaled, err := scaler.Transform(tensor)
	if err != nil {
		log.Fatalf("Failed to transform data: %v", err)
	}

	fmt.Printf("Standardized data (mean=0, std=1):\n%s\n", scaled.String())

	// Show learned parameters
	mean := scaler.GetMean()
	std := scaler.GetStd()
	fmt.Printf("Learned mean: %s\n", mean.String())
	fmt.Printf("Learned std: %s\n", std.String())

	// Inverse transform to recover original data
	recovered, err := scaler.InverseTransform(scaled)
	if err != nil {
		log.Fatalf("Failed to inverse transform: %v", err)
	}

	fmt.Printf("Recovered data:\n%s\n", recovered.String())
}

func demoMinMaxScaler() {
	fmt.Println("MinMaxScaler Demo:")
	fmt.Println("-----------------")

	// Create sample data
	data := [][]float64{
		{10.0, 1000.0},
		{20.0, 2000.0},
		{30.0, 3000.0},
		{40.0, 4000.0},
		{50.0, 5000.0},
	}
	tensor := core.NewTensorFromSlice(data)

	fmt.Printf("Original data:\n%s\n", tensor.String())

	// Create MinMaxScaler with custom range [-1, 1]
	scaler := preprocessing.NewMinMaxScaler(preprocessing.WithFeatureRange(-1.0, 1.0))

	// Fit and transform
	scaled, err := scaler.FitTransform(tensor)
	if err != nil {
		log.Fatalf("Failed to fit and transform: %v", err)
	}

	fmt.Printf("Min-Max scaled data (range [-1, 1]):\n%s\n", scaled.String())

	// Show learned parameters
	dataMin := scaler.GetDataMin()
	dataMax := scaler.GetDataMax()
	fmt.Printf("Learned min: %s\n", dataMin.String())
	fmt.Printf("Learned max: %s\n", dataMax.String())

	min, max := scaler.GetFeatureRange()
	fmt.Printf("Feature range: [%.1f, %.1f]\n", min, max)
}

func demoOneHotEncoder() {
	fmt.Println("OneHotEncoder Demo:")
	fmt.Println("------------------")

	// Sample categorical data
	categories := []string{"cat", "dog", "bird", "cat", "dog", "fish", "bird"}

	fmt.Printf("Original categories: %v\n", categories)

	// Create and fit OneHotEncoder
	encoder := preprocessing.NewOneHotEncoder()
	encoded, err := encoder.FitTransform(categories)
	if err != nil {
		log.Fatalf("Failed to fit and transform: %v", err)
	}

	fmt.Printf("One-hot encoded data:\n%s\n", encoded.String())

	// Show learned classes
	classes := encoder.Classes()
	fmt.Printf("Learned classes: %v\n", classes)
	fmt.Printf("Number of classes: %d\n", encoder.GetNumClasses())

	// Show mapping
	fmt.Println("Class mapping:")
	for i, class := range classes {
		fmt.Printf("  %s -> column %d\n", class, i)
	}
}

func demoLabelEncoder() {
	fmt.Println("LabelEncoder Demo:")
	fmt.Println("-----------------")

	// Sample categorical data
	categories := []string{"small", "medium", "large", "small", "large", "medium"}

	fmt.Printf("Original categories: %v\n", categories)

	// Create and fit LabelEncoder
	encoder := preprocessing.NewLabelEncoder()
	encoded, err := encoder.FitTransform(categories)
	if err != nil {
		log.Fatalf("Failed to fit and transform: %v", err)
	}

	fmt.Printf("Label encoded data:\n%s\n", encoded.String())

	// Show learned classes
	classes := encoder.Classes()
	fmt.Printf("Learned classes: %v\n", classes)

	// Show mapping
	fmt.Println("Class mapping:")
	for i, class := range classes {
		fmt.Printf("  %s -> %d\n", class, i)
	}

	// Inverse transform to recover original categories
	decoded, err := encoder.InverseTransform(encoded)
	if err != nil {
		log.Fatalf("Failed to inverse transform: %v", err)
	}

	fmt.Printf("Decoded categories: %v\n", decoded)
}
