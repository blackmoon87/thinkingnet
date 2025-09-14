package optimizers

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
	"gonum.org/v1/gonum/mat"
)

// mockTensor implements the core.Tensor interface for testing.
type mockTensor struct {
	data *mat.Dense
	name string
}

func newMockTensor(rows, cols int, data []float64) *mockTensor {
	return &mockTensor{
		data: mat.NewDense(rows, cols, data),
	}
}

func (t *mockTensor) Dims() (int, int)        { return t.data.Dims() }
func (t *mockTensor) At(i, j int) float64     { return t.data.At(i, j) }
func (t *mockTensor) Set(i, j int, v float64) { t.data.Set(i, j, v) }

func (t *mockTensor) Copy() core.Tensor {
	rows, cols := t.data.Dims()
	newData := mat.NewDense(rows, cols, nil)
	newData.Copy(t.data)
	return &mockTensor{data: newData, name: t.name}
}

func (t *mockTensor) Add(other core.Tensor) core.Tensor {
	result := t.Copy().(*mockTensor)
	result.data.Add(result.data, other.(*mockTensor).data)
	return result
}

func (t *mockTensor) Sub(other core.Tensor) core.Tensor {
	result := t.Copy().(*mockTensor)
	result.data.Sub(result.data, other.(*mockTensor).data)
	return result
}

func (t *mockTensor) Mul(other core.Tensor) core.Tensor {
	result := t.Copy().(*mockTensor)
	result.data.Mul(result.data, other.(*mockTensor).data)
	return result
}

func (t *mockTensor) MulElem(other core.Tensor) core.Tensor {
	result := t.Copy().(*mockTensor)
	result.data.MulElem(result.data, other.(*mockTensor).data)
	return result
}

func (t *mockTensor) Div(other core.Tensor) core.Tensor {
	result := t.Copy().(*mockTensor)
	result.data.DivElem(result.data, other.(*mockTensor).data)
	return result
}

func (t *mockTensor) Scale(scalar float64) core.Tensor {
	t.data.Scale(scalar, t.data)
	return t
}

func (t *mockTensor) Zero() {
	t.data.Zero()
}

func (t *mockTensor) Fill(value float64) {
	rows, cols := t.data.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			t.data.Set(i, j, value)
		}
	}
}

func (t *mockTensor) Apply(fn func(i, j int, v float64) float64) core.Tensor {
	result := t.Copy().(*mockTensor)
	result.data.Apply(fn, result.data)
	return result
}

func (t *mockTensor) Norm() float64 {
	return mat.Norm(t.data, 2)
}

func (t *mockTensor) Sum() float64 {
	return mat.Sum(t.data)
}

func (t *mockTensor) Name() string        { return t.name }
func (t *mockTensor) SetName(name string) { t.name = name }

// Implement remaining interface methods with minimal functionality for testing
func (t *mockTensor) Pow(power float64) core.Tensor            { return t.Copy() }
func (t *mockTensor) Sqrt() core.Tensor                        { return t.Copy() }
func (t *mockTensor) Exp() core.Tensor                         { return t.Copy() }
func (t *mockTensor) Log() core.Tensor                         { return t.Copy() }
func (t *mockTensor) Abs() core.Tensor                         { return t.Copy() }
func (t *mockTensor) Sign() core.Tensor                        { return t.Copy() }
func (t *mockTensor) Clamp(min, max float64) core.Tensor       { return t.Copy() }
func (t *mockTensor) T() core.Tensor                           { return t.Copy() }
func (t *mockTensor) Mean() float64                            { return 0 }
func (t *mockTensor) Std() float64                             { return 0 }
func (t *mockTensor) Max() float64                             { return 0 }
func (t *mockTensor) Min() float64                             { return 0 }
func (t *mockTensor) Reshape(newRows, newCols int) core.Tensor { return t.Copy() }
func (t *mockTensor) Flatten() core.Tensor                     { return t.Copy() }
func (t *mockTensor) Shape() []int                             { r, c := t.Dims(); return []int{r, c} }
func (t *mockTensor) Equal(other core.Tensor) bool             { return false }
func (t *mockTensor) Release()                                 {}
func (t *mockTensor) Row(i int) core.Tensor                    { return t.Copy() }
func (t *mockTensor) Col(j int) core.Tensor                    { return t.Copy() }
func (t *mockTensor) Slice(r0, r1, c0, c1 int) core.Tensor     { return t.Copy() }
func (t *mockTensor) SetRow(i int, data []float64)             {}
func (t *mockTensor) SetCol(j int, data []float64)             {}
func (t *mockTensor) IsEmpty() bool                            { return false }
func (t *mockTensor) IsSquare() bool                           { return false }
func (t *mockTensor) IsVector() bool                           { return false }
func (t *mockTensor) String() string                           { return "" }
func (t *mockTensor) RawMatrix() *mat.Dense                    { return t.data }

// Additional methods required by the Tensor interface
func (t *mockTensor) AddScalar(scalar float64) core.Tensor {
	result := t.Copy().(*mockTensor)
	rows, cols := result.data.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.data.Set(i, j, result.data.At(i, j)+scalar)
		}
	}
	return result
}

func (t *mockTensor) SubScalar(scalar float64) core.Tensor {
	result := t.Copy().(*mockTensor)
	rows, cols := result.data.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.data.Set(i, j, result.data.At(i, j)-scalar)
		}
	}
	return result
}

func (t *mockTensor) DivScalar(scalar float64) core.Tensor {
	result := t.Copy().(*mockTensor)
	rows, cols := result.data.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.data.Set(i, j, result.data.At(i, j)/scalar)
		}
	}
	return result
}

func (t *mockTensor) Dot(other core.Tensor) float64 {
	sum := 0.0
	rows, cols := t.data.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			sum += t.data.At(i, j) * other.At(i, j)
		}
	}
	return sum
}

func (t *mockTensor) Trace() float64 {
	rows, cols := t.data.Dims()
	minDim := rows
	if cols < minDim {
		minDim = cols
	}
	sum := 0.0
	for i := 0; i < minDim; i++ {
		sum += t.data.At(i, i)
	}
	return sum
}

func (t *mockTensor) Diagonal() core.Tensor {
	rows, cols := t.data.Dims()
	minDim := rows
	if cols < minDim {
		minDim = cols
	}
	result := newMockTensor(minDim, 1, nil)
	for i := 0; i < minDim; i++ {
		result.data.Set(i, 0, t.data.At(i, i))
	}
	return result
}

func (t *mockTensor) Validate() error {
	return nil
}

func (t *mockTensor) HasNaN() bool {
	rows, cols := t.data.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if math.IsNaN(t.data.At(i, j)) {
				return true
			}
		}
	}
	return false
}

func (t *mockTensor) HasInf() bool {
	rows, cols := t.data.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if math.IsInf(t.data.At(i, j), 0) {
				return true
			}
		}
	}
	return false
}

func (t *mockTensor) IsFinite() bool {
	return !t.HasNaN() && !t.HasInf()
}

// Test helper functions
func almostEqual(a, b, tolerance float64) bool {
	return math.Abs(a-b) < tolerance
}

func TestValidateParameters(t *testing.T) {
	// Test matching parameters and gradients
	params := []core.Tensor{
		newMockTensor(2, 3, []float64{1, 2, 3, 4, 5, 6}),
		newMockTensor(1, 2, []float64{7, 8}),
	}
	grads := []core.Tensor{
		newMockTensor(2, 3, []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
		newMockTensor(1, 2, []float64{0.7, 0.8}),
	}

	err := validateParameters(params, grads)
	if err != nil {
		t.Errorf("Expected no error for matching parameters, got: %v", err)
	}

	// Test mismatched count
	gradsShort := grads[:1]
	err = validateParameters(params, gradsShort)
	if err == nil {
		t.Error("Expected error for mismatched parameter count")
	}

	// Test mismatched dimensions
	gradsBadDim := []core.Tensor{
		newMockTensor(2, 3, []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
		newMockTensor(2, 2, []float64{0.7, 0.8, 0.9, 1.0}), // Wrong dimensions
	}
	err = validateParameters(params, gradsBadDim)
	if err == nil {
		t.Error("Expected error for mismatched dimensions")
	}
}

func TestClampGradients(t *testing.T) {
	grads := []core.Tensor{
		newMockTensor(2, 2, []float64{3, 4, 0, 0}), // Norm = 5
		newMockTensor(2, 2, []float64{0, 0, 3, 4}), // Norm = 5
	}

	// Total norm = sqrt(25 + 25) = sqrt(50) ≈ 7.07
	// With maxNorm = 5, clipCoeff = 5/7.07 ≈ 0.707

	clampGradients(grads, 5.0)

	// Check that gradients were scaled down
	expectedScale := 5.0 / math.Sqrt(50)
	expected1 := 3.0 * expectedScale
	expected2 := 4.0 * expectedScale

	tolerance := 1e-6
	if !almostEqual(grads[0].At(0, 0), expected1, tolerance) {
		t.Errorf("Expected %f, got %f", expected1, grads[0].At(0, 0))
	}
	if !almostEqual(grads[0].At(0, 1), expected2, tolerance) {
		t.Errorf("Expected %f, got %f", expected2, grads[0].At(0, 1))
	}
}

func TestBaseOptimizer(t *testing.T) {
	base := BaseOptimizer{
		name:         "test",
		learningRate: 0.01,
		stepCount:    0,
	}

	// Test getters
	if base.Name() != "test" {
		t.Errorf("Expected name 'test', got '%s'", base.Name())
	}
	if base.LearningRate() != 0.01 {
		t.Errorf("Expected learning rate 0.01, got %f", base.LearningRate())
	}

	// Test step counter
	base.Step()
	if base.stepCount != 1 {
		t.Errorf("Expected step count 1, got %d", base.stepCount)
	}

	// Test reset
	base.Reset()
	if base.stepCount != 0 {
		t.Errorf("Expected step count 0 after reset, got %d", base.stepCount)
	}

	// Test learning rate setter
	err := base.SetLearningRate(0.02)
	if err != nil {
		t.Fatalf("Failed to set learning rate: %v", err)
	}
	if base.LearningRate() != 0.02 {
		t.Errorf("Expected learning rate 0.02, got %f", base.LearningRate())
	}

	// Test invalid learning rate
	err = base.SetLearningRate(-0.01)
	if err == nil {
		t.Error("Expected error for negative learning rate")
	}
}
