package losses

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
	"gonum.org/v1/gonum/mat"
)

// Helper function to compare floats with tolerance
func floatEqual(a, b, tolerance float64) bool {
	return math.Abs(a-b) < tolerance
}

func TestBinaryCrossEntropy_Compute(t *testing.T) {
	bce := NewBinaryCrossEntropy()

	// Test case 1: Perfect predictions
	yTrue := core.NewTensor(mat.NewDense(3, 1, []float64{1.0, 0.0, 1.0}))
	yPred := core.NewTensor(mat.NewDense(3, 1, []float64{0.999, 0.001, 0.999}))

	loss := bce.Compute(yTrue, yPred)
	if loss > 0.01 {
		t.Errorf("Expected very low loss for perfect predictions, got %f", loss)
	}

	// Test case 2: Worst predictions
	yTrue = core.NewTensor(mat.NewDense(3, 1, []float64{1.0, 0.0, 1.0}))
	yPred = core.NewTensor(mat.NewDense(3, 1, []float64{0.001, 0.999, 0.001}))

	loss = bce.Compute(yTrue, yPred)
	if loss < 5.0 {
		t.Errorf("Expected high loss for worst predictions, got %f", loss)
	}

	// Test case 3: Random predictions
	yTrue = core.NewTensor(mat.NewDense(2, 1, []float64{1.0, 0.0}))
	yPred = core.NewTensor(mat.NewDense(2, 1, []float64{0.5, 0.5}))

	loss = bce.Compute(yTrue, yPred)
	expectedLoss := -math.Log(0.5) // Should be ln(2) ≈ 0.693
	if !floatEqual(loss, expectedLoss, 0.001) {
		t.Errorf("Expected loss %f, got %f", expectedLoss, loss)
	}
}

func TestBinaryCrossEntropy_Gradient(t *testing.T) {
	bce := NewBinaryCrossEntropy()

	yTrue := core.NewTensor(mat.NewDense(2, 1, []float64{1.0, 0.0}))
	yPred := core.NewTensor(mat.NewDense(2, 1, []float64{0.8, 0.3}))

	grad := bce.Gradient(yTrue, yPred)

	// Check gradient dimensions
	r, c := grad.Dims()
	if r != 2 || c != 1 {
		t.Errorf("Expected gradient dimensions (2,1), got (%d,%d)", r, c)
	}

	// For binary cross-entropy, gradient should be (p-t)/(p*(1-p)*N)
	// For first sample: (0.8-1.0)/(0.8*0.2*2) = -0.2/0.32 = -0.625
	expectedGrad1 := -0.2 / (0.8 * 0.2 * 2.0)
	if !floatEqual(grad.At(0, 0), expectedGrad1, 0.001) {
		t.Errorf("Expected gradient %f for first sample, got %f", expectedGrad1, grad.At(0, 0))
	}
}

func TestCategoricalCrossEntropy_Compute(t *testing.T) {
	cce := NewCategoricalCrossEntropy()

	// Test case 1: Perfect predictions (one-hot)
	yTrue := core.NewTensor(mat.NewDense(2, 3, []float64{1.0, 0.0, 0.0, 0.0, 1.0, 0.0}))
	yPred := core.NewTensor(mat.NewDense(2, 3, []float64{0.999, 0.0005, 0.0005, 0.0005, 0.999, 0.0005}))

	loss := cce.Compute(yTrue, yPred)
	if loss > 0.01 {
		t.Errorf("Expected very low loss for perfect predictions, got %f", loss)
	}

	// Test case 2: Uniform predictions
	yTrue = core.NewTensor(mat.NewDense(1, 3, []float64{1.0, 0.0, 0.0}))
	yPred = core.NewTensor(mat.NewDense(1, 3, []float64{0.333, 0.333, 0.334}))

	loss = cce.Compute(yTrue, yPred)
	expectedLoss := -math.Log(0.333) // Should be approximately 1.1
	if !floatEqual(loss, expectedLoss, 0.01) {
		t.Errorf("Expected loss around %f, got %f", expectedLoss, loss)
	}
}

func TestCategoricalCrossEntropy_Gradient(t *testing.T) {
	cce := NewCategoricalCrossEntropy()

	yTrue := core.NewTensor(mat.NewDense(2, 2, []float64{1.0, 0.0, 0.0, 1.0}))
	yPred := core.NewTensor(mat.NewDense(2, 2, []float64{0.8, 0.2, 0.3, 0.7}))

	grad := cce.Gradient(yTrue, yPred)

	// Check gradient dimensions
	r, c := grad.Dims()
	if r != 2 || c != 2 {
		t.Errorf("Expected gradient dimensions (2,2), got (%d,%d)", r, c)
	}

	// For categorical cross-entropy with softmax, gradient is (yPred - yTrue) / batch_size
	expectedGrad00 := (0.8 - 1.0) / 2.0 // -0.1
	expectedGrad01 := (0.2 - 0.0) / 2.0 // 0.1

	if !floatEqual(grad.At(0, 0), expectedGrad00, 0.001) {
		t.Errorf("Expected gradient %f at (0,0), got %f", expectedGrad00, grad.At(0, 0))
	}
	if !floatEqual(grad.At(0, 1), expectedGrad01, 0.001) {
		t.Errorf("Expected gradient %f at (0,1), got %f", expectedGrad01, grad.At(0, 1))
	}
}

func TestMeanSquaredError_Compute(t *testing.T) {
	mse := NewMeanSquaredError()

	// Test case 1: Perfect predictions
	yTrue := core.NewTensor(mat.NewDense(3, 1, []float64{1.0, 2.0, 3.0}))
	yPred := core.NewTensor(mat.NewDense(3, 1, []float64{1.0, 2.0, 3.0}))

	loss := mse.Compute(yTrue, yPred)
	if loss != 0.0 {
		t.Errorf("Expected zero loss for perfect predictions, got %f", loss)
	}

	// Test case 2: Known error
	yTrue = core.NewTensor(mat.NewDense(2, 1, []float64{1.0, 2.0}))
	yPred = core.NewTensor(mat.NewDense(2, 1, []float64{2.0, 1.0}))

	loss = mse.Compute(yTrue, yPred)
	// MSE = [(1-2)² + (2-1)²] / (2*2) = [1 + 1] / 4 = 0.5
	expectedLoss := 0.5
	if !floatEqual(loss, expectedLoss, 0.001) {
		t.Errorf("Expected loss %f, got %f", expectedLoss, loss)
	}
}

func TestMeanSquaredError_Gradient(t *testing.T) {
	mse := NewMeanSquaredError()

	yTrue := core.NewTensor(mat.NewDense(2, 1, []float64{1.0, 2.0}))
	yPred := core.NewTensor(mat.NewDense(2, 1, []float64{2.0, 1.0}))

	grad := mse.Gradient(yTrue, yPred)

	// For MSE, gradient is (yPred - yTrue) / batch_size
	// First sample: (2.0 - 1.0) / 2 = 0.5
	// Second sample: (1.0 - 2.0) / 2 = -0.5
	if !floatEqual(grad.At(0, 0), 0.5, 0.001) {
		t.Errorf("Expected gradient 0.5 for first sample, got %f", grad.At(0, 0))
	}
	if !floatEqual(grad.At(1, 0), -0.5, 0.001) {
		t.Errorf("Expected gradient -0.5 for second sample, got %f", grad.At(1, 0))
	}
}

func TestLossNames(t *testing.T) {
	tests := []struct {
		loss core.Loss
		name string
	}{
		{NewBinaryCrossEntropy(), "binary_crossentropy"},
		{NewCategoricalCrossEntropy(), "categorical_crossentropy"},
		{NewMeanSquaredError(), "mean_squared_error"},
	}

	for _, test := range tests {
		if test.loss.Name() != test.name {
			t.Errorf("Expected name %s, got %s", test.name, test.loss.Name())
		}
	}
}

// Test edge cases and error handling
func TestLossEdgeCases(t *testing.T) {
	bce := NewBinaryCrossEntropy()

	// Test dimension mismatch
	yTrue := core.NewTensor(mat.NewDense(1, 1, []float64{1.0}))
	yPred := core.NewTensor(mat.NewDense(2, 1, []float64{0.5, 0.3}))

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for dimension mismatch")
		}
	}()

	bce.Compute(yTrue, yPred)
}

// Test numerical stability
func TestNumericalStability(t *testing.T) {
	bce := NewBinaryCrossEntropy()

	// Test with extreme values that could cause log(0)
	yTrue := core.NewTensor(mat.NewDense(2, 1, []float64{1.0, 0.0}))
	yPred := core.NewTensor(mat.NewDense(2, 1, []float64{0.0, 1.0})) // Extreme predictions

	loss := bce.Compute(yTrue, yPred)
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		t.Errorf("Loss should be finite, got %f", loss)
	}

	grad := bce.Gradient(yTrue, yPred)
	for i := 0; i < 2; i++ {
		for j := 0; j < 1; j++ {
			val := grad.At(i, j)
			if math.IsNaN(val) || math.IsInf(val, 0) {
				t.Errorf("Gradient should be finite at (%d,%d), got %f", i, j, val)
			}
		}
	}
}

// Benchmark tests
func BenchmarkBinaryCrossEntropy_Compute(b *testing.B) {
	bce := NewBinaryCrossEntropy()
	yTrue := core.NewTensor(mat.NewDense(4, 1, []float64{1.0, 0.0, 1.0, 0.0}))
	yPred := core.NewTensor(mat.NewDense(4, 1, []float64{0.8, 0.2, 0.9, 0.1}))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bce.Compute(yTrue, yPred)
	}
}

func BenchmarkCategoricalCrossEntropy_Compute(b *testing.B) {
	cce := NewCategoricalCrossEntropy()
	yTrue := core.NewTensor(mat.NewDense(2, 3, []float64{1.0, 0.0, 0.0, 0.0, 1.0, 0.0}))
	yPred := core.NewTensor(mat.NewDense(2, 3, []float64{0.8, 0.1, 0.1, 0.2, 0.7, 0.1}))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cce.Compute(yTrue, yPred)
	}
}

func BenchmarkMeanSquaredError_Compute(b *testing.B) {
	mse := NewMeanSquaredError()
	yTrue := core.NewTensor(mat.NewDense(4, 1, []float64{1.0, 2.0, 3.0, 4.0}))
	yPred := core.NewTensor(mat.NewDense(4, 1, []float64{1.1, 2.1, 2.9, 3.8}))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mse.Compute(yTrue, yPred)
	}
}
