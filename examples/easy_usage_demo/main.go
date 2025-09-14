package main

import (
	"fmt"
	"log"

	"github.com/blackmoon87/thinkingnet/pkg/activations"
	"github.com/blackmoon87/thinkingnet/pkg/core"
	"github.com/blackmoon87/thinkingnet/pkg/layers"
	"github.com/blackmoon87/thinkingnet/pkg/losses"
	"github.com/blackmoon87/thinkingnet/pkg/models"
	"github.com/blackmoon87/thinkingnet/pkg/optimizers"
)

func main() {
	fmt.Println("=== ThinkingNet-Go Easy Usage Demo ===")
	fmt.Println("مثال على الاستخدام السهل لمكتبة ThinkingNet-Go")
	fmt.Println()

	// إنشاء بيانات بسيطة للتدريب (XOR problem)
	// Create simple training data (XOR problem)
	X := core.NewTensorFromSlice([][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	})

	y := core.NewTensorFromSlice([][]float64{
		{0},
		{1},
		{1},
		{0},
	})

	fmt.Println("Training data created:")
	fmt.Printf("X shape: %v\n", X.Shape())
	fmt.Printf("y shape: %v\n", y.Shape())
	fmt.Println()

	// إنشاء نموذج بسيط
	// Create a simple model
	model := models.NewSequential()

	// إضافة طبقات
	// Add layers
	model.AddLayer(layers.NewDense(4, activations.NewReLU()))
	model.AddLayer(layers.NewDense(1, activations.NewSigmoid()))

	fmt.Println("Model layers added")

	// تجميع النموذج
	// Compile the model
	optimizer := optimizers.NewAdam(0.01)
	loss := losses.NewBinaryCrossEntropy()

	err := model.Compile(optimizer, loss)
	if err != nil {
		log.Fatalf("خطأ في تجميع النموذج / Model compilation error: %v", err)
	}

	fmt.Println("Model compiled successfully")
	fmt.Println("النموذج تم تجميعه بنجاح")
	fmt.Println()

	// تدريب النموذج باستخدام EasyTrain
	// Train the model using EasyTrain
	fmt.Println("Training model using EasyTrain()...")
	fmt.Println("تدريب النموذج باستخدام EasyTrain()...")

	history, err := model.EasyTrain(X, y)
	if err != nil {
		log.Fatalf("خطأ في التدريب / Training error: %v", err)
	}

	fmt.Printf("Training completed! Final loss: %.4f\n", history.Loss[len(history.Loss)-1])
	fmt.Printf("التدريب مكتمل! الخسارة النهائية: %.4f\n", history.Loss[len(history.Loss)-1])
	fmt.Println()

	// التنبؤ باستخدام EasyPredict
	// Make predictions using EasyPredict
	fmt.Println("Making predictions using EasyPredict()...")
	fmt.Println("التنبؤ باستخدام EasyPredict()...")

	predictions, err := model.EasyPredict(X)
	if err != nil {
		log.Fatalf("خطأ في التنبؤ / Prediction error: %v", err)
	}

	fmt.Println("Predictions:")
	fmt.Println("التنبؤات:")
	rows, _ := predictions.Dims()
	for i := 0; i < rows; i++ {
		input0 := X.At(i, 0)
		input1 := X.At(i, 1)
		expected := y.At(i, 0)
		predicted := predictions.At(i, 0)

		fmt.Printf("Input: [%.0f, %.0f] -> Expected: %.0f, Predicted: %.4f\n",
			input0, input1, expected, predicted)
	}

	fmt.Println()
	fmt.Println("=== Demo completed successfully! ===")
	fmt.Println("=== المثال اكتمل بنجاح! ===")
}
