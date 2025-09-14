package losses

import (
"testing"

"github.com/blackmoon87/thinkingnet/pkg/core"
"gonum.org/v1/gonum/mat"
)

func TestBinaryCrossEntropy_Basic(t *testing.T) {
bce := NewBinaryCrossEntropy()

// Create simple test data
yTrueData := mat.NewDense(2, 1, []float64{1.0, 0.0})
yPredData := mat.NewDense(2, 1, []float64{0.9, 0.1})

yTrue := core.NewTensor(yTrueData)
yPred := core.NewTensor(yPredData)

// Test compute
loss := bce.Compute(yTrue, yPred)
if loss < 0 {
t.Errorf("Loss should be positive, got %f", loss)
}

// Test gradient
grad := bce.Gradient(yTrue, yPred)
r, c := grad.Dims()
if r != 2 || c != 1 {
t.Errorf("Expected gradient dimensions (2,1), got (%d,%d)", r, c)
}

// Test name
if bce.Name() != "binary_crossentropy" {
t.Errorf("Expected name 'binary_crossentropy', got '%s'", bce.Name())
}
}

func TestMeanSquaredError_Basic(t *testing.T) {
mse := NewMeanSquaredError()

// Create simple test data
yTrueData := mat.NewDense(2, 1, []float64{1.0, 2.0})
yPredData := mat.NewDense(2, 1, []float64{1.1, 1.9})

yTrue := core.NewTensor(yTrueData)
yPred := core.NewTensor(yPredData)

// Test compute
loss := mse.Compute(yTrue, yPred)
if loss < 0 {
t.Errorf("Loss should be positive, got %f", loss)
}

// Test gradient
grad := mse.Gradient(yTrue, yPred)
r, c := grad.Dims()
if r != 2 || c != 1 {
t.Errorf("Expected gradient dimensions (2,1), got (%d,%d)", r, c)
}

// Test name
if mse.Name() != "mean_squared_error" {
t.Errorf("Expected name 'mean_squared_error', got '%s'", mse.Name())
}
}

func TestCategoricalCrossEntropy_Basic(t *testing.T) {
cce := NewCategoricalCrossEntropy()

// Create simple test data (one-hot encoded)
yTrueData := mat.NewDense(2, 3, []float64{1.0, 0.0, 0.0, 0.0, 1.0, 0.0})
yPredData := mat.NewDense(2, 3, []float64{0.8, 0.1, 0.1, 0.2, 0.7, 0.1})

yTrue := core.NewTensor(yTrueData)
yPred := core.NewTensor(yPredData)

// Test compute
loss := cce.Compute(yTrue, yPred)
if loss < 0 {
t.Errorf("Loss should be positive, got %f", loss)
}

// Test gradient
grad := cce.Gradient(yTrue, yPred)
r, c := grad.Dims()
if r != 2 || c != 3 {
t.Errorf("Expected gradient dimensions (2,3), got (%d,%d)", r, c)
}

// Test name
if cce.Name() != "categorical_crossentropy" {
t.Errorf("Expected name 'categorical_crossentropy', got '%s'", cce.Name())
}
}
