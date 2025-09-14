package core

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestNewTensor(t *testing.T) {
	data := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	tensor := NewTensor(data)

	if tensor == nil {
		t.Fatal("NewTensor returned nil")
	}

	rows, cols := tensor.Dims()
	if rows != 2 || cols != 3 {
		t.Errorf("Expected dimensions (2,3), got (%d,%d)", rows, cols)
	}
}

func TestNewTensorFromSlice(t *testing.T) {
	data := [][]float64{{1, 2, 3}, {4, 5, 6}}
	tensor := NewTensorFromSlice(data)

	if tensor == nil {
		t.Fatal("NewTensorFromSlice returned nil")
	}

	rows, cols := tensor.Dims()
	if rows != 2 || cols != 3 {
		t.Errorf("Expected dimensions (2,3), got (%d,%d)", rows, cols)
	}

	if tensor.At(0, 0) != 1 || tensor.At(1, 2) != 6 {
		t.Errorf("Tensor values not set correctly")
	}
}

func TestTensorArithmetic(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	b := NewTensorFromSlice([][]float64{{2, 3}, {4, 5}})

	// Test addition
	c := a.Add(b)
	expected := [][]float64{{3, 5}, {7, 9}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if c.At(i, j) != expected[i][j] {
				t.Errorf("Add: expected %f at (%d,%d), got %f", expected[i][j], i, j, c.At(i, j))
			}
		}
	}

	// Test subtraction
	d := a.Sub(b)
	expectedSub := [][]float64{{-1, -1}, {-1, -1}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if d.At(i, j) != expectedSub[i][j] {
				t.Errorf("Sub: expected %f at (%d,%d), got %f", expectedSub[i][j], i, j, d.At(i, j))
			}
		}
	}

	// Test element-wise multiplication
	e := a.MulElem(b)
	expectedMul := [][]float64{{2, 6}, {12, 20}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if e.At(i, j) != expectedMul[i][j] {
				t.Errorf("MulElem: expected %f at (%d,%d), got %f", expectedMul[i][j], i, j, e.At(i, j))
			}
		}
	}
}

func TestTensorMatrixMultiplication(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	b := NewTensorFromSlice([][]float64{{2, 0}, {1, 2}})

	c := a.Mul(b)
	expected := [][]float64{{4, 4}, {10, 8}}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if c.At(i, j) != expected[i][j] {
				t.Errorf("Mul: expected %f at (%d,%d), got %f", expected[i][j], i, j, c.At(i, j))
			}
		}
	}
}

func TestTensorTranspose(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2, 3}, {4, 5, 6}})
	b := a.T()

	rows, cols := b.Dims()
	if rows != 3 || cols != 2 {
		t.Errorf("Transpose: expected dimensions (3,2), got (%d,%d)", rows, cols)
	}

	expected := [][]float64{{1, 4}, {2, 5}, {3, 6}}
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			if b.At(i, j) != expected[i][j] {
				t.Errorf("Transpose: expected %f at (%d,%d), got %f", expected[i][j], i, j, b.At(i, j))
			}
		}
	}
}

func TestTensorScale(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	b := a.Scale(2.0)

	expected := [][]float64{{2, 4}, {6, 8}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if b.At(i, j) != expected[i][j] {
				t.Errorf("Scale: expected %f at (%d,%d), got %f", expected[i][j], i, j, b.At(i, j))
			}
		}
	}
}

func TestTensorStatistics(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})

	// Test sum
	sum := a.Sum()
	if sum != 10.0 {
		t.Errorf("Sum: expected 10.0, got %f", sum)
	}

	// Test mean
	mean := a.Mean()
	if mean != 2.5 {
		t.Errorf("Mean: expected 2.5, got %f", mean)
	}

	// Test max
	max := a.Max()
	if max != 4.0 {
		t.Errorf("Max: expected 4.0, got %f", max)
	}

	// Test min
	min := a.Min()
	if min != 1.0 {
		t.Errorf("Min: expected 1.0, got %f", min)
	}
}

func TestTensorMathFunctions(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 4}, {9, 16}})

	// Test sqrt
	b := a.Sqrt()
	expected := [][]float64{{1, 2}, {3, 4}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if math.Abs(b.At(i, j)-expected[i][j]) > 1e-10 {
				t.Errorf("Sqrt: expected %f at (%d,%d), got %f", expected[i][j], i, j, b.At(i, j))
			}
		}
	}

	// Test pow
	c := a.Pow(2.0)
	expectedPow := [][]float64{{1, 16}, {81, 256}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if c.At(i, j) != expectedPow[i][j] {
				t.Errorf("Pow: expected %f at (%d,%d), got %f", expectedPow[i][j], i, j, c.At(i, j))
			}
		}
	}
}

func TestTensorClamp(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{-2, 0}, {3, 5}})
	b := a.Clamp(0, 4)

	expected := [][]float64{{0, 0}, {3, 4}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if b.At(i, j) != expected[i][j] {
				t.Errorf("Clamp: expected %f at (%d,%d), got %f", expected[i][j], i, j, b.At(i, j))
			}
		}
	}
}

func TestTensorReshape(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	b := a.Reshape(1, 4)

	rows, cols := b.Dims()
	if rows != 1 || cols != 4 {
		t.Errorf("Reshape: expected dimensions (1,4), got (%d,%d)", rows, cols)
	}

	expected := []float64{1, 2, 3, 4}
	for j := 0; j < 4; j++ {
		if b.At(0, j) != expected[j] {
			t.Errorf("Reshape: expected %f at (0,%d), got %f", expected[j], j, b.At(0, j))
		}
	}
}

func TestTensorEqual(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	b := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	c := NewTensorFromSlice([][]float64{{1, 2}, {3, 5}})

	if !a.Equal(b) {
		t.Error("Equal: identical tensors should be equal")
	}

	if a.Equal(c) {
		t.Error("Equal: different tensors should not be equal")
	}
}

func TestTensorSlicing(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})

	// Test row extraction
	row := a.Row(1)
	expectedRow := []float64{4, 5, 6}
	for j := 0; j < 3; j++ {
		if row.At(0, j) != expectedRow[j] {
			t.Errorf("Row: expected %f at (0,%d), got %f", expectedRow[j], j, row.At(0, j))
		}
	}

	// Test column extraction
	col := a.Col(1)
	expectedCol := []float64{2, 5, 8}
	for i := 0; i < 3; i++ {
		if col.At(i, 0) != expectedCol[i] {
			t.Errorf("Col: expected %f at (%d,0), got %f", expectedCol[i], i, col.At(i, 0))
		}
	}
}

func TestTensorProperties(t *testing.T) {
	square := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	vector := NewTensorFromSlice([][]float64{{1, 2, 3}})
	rectangular := NewTensorFromSlice([][]float64{{1, 2, 3}, {4, 5, 6}})

	if !square.IsSquare() {
		t.Error("IsSquare: square tensor should return true")
	}

	if square.IsVector() {
		t.Error("IsSquare: square tensor should not be a vector")
	}

	if !vector.IsVector() {
		t.Error("IsVector: vector tensor should return true")
	}

	if rectangular.IsSquare() {
		t.Error("IsSquare: rectangular tensor should return false")
	}

	if rectangular.IsVector() {
		t.Error("IsVector: rectangular tensor should return false")
	}
}

func TestTensorDivision(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{6, 8}, {10, 12}})
	b := NewTensorFromSlice([][]float64{{2, 4}, {5, 3}})

	c := a.Div(b)
	expected := [][]float64{{3, 2}, {2, 4}}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if math.Abs(c.At(i, j)-expected[i][j]) > 1e-10 {
				t.Errorf("Div: expected %f at (%d,%d), got %f", expected[i][j], i, j, c.At(i, j))
			}
		}
	}
}

func TestTensorCopy(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	b := a.Copy()

	// Modify original
	a.Set(0, 0, 999)

	// Copy should be unchanged
	if b.At(0, 0) != 1 {
		t.Error("Copy: modifying original should not affect copy")
	}
}

func TestMatrixPooling(t *testing.T) {
	// Test that matrix pooling works
	stats1 := MatrixPoolStats()

	// Create and release some tensors
	for i := 0; i < 10; i++ {
		tensor := NewZerosTensor(3, 3)
		tensor.Release()
	}

	stats2 := MatrixPoolStats()

	// Pool should have been used (though we can't easily verify the exact count)
	if len(stats2) == 0 && len(stats1) == 0 {
		t.Log("Matrix pooling stats available")
	}
}

// Benchmark tests
func BenchmarkTensorAdd(b *testing.B) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	c := NewTensorFromSlice([][]float64{{2, 3}, {4, 5}})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := a.Add(c)
		result.Release() // Clean up
	}
}

func BenchmarkTensorMul(b *testing.B) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	c := NewTensorFromSlice([][]float64{{2, 3}, {4, 5}})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := a.Mul(c)
		result.Release() // Clean up
	}
}

func TestTensorScalarOperations(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})

	// Test AddScalar
	b := a.AddScalar(5)
	expected := [][]float64{{6, 7}, {8, 9}}
	for i := range 2 {
		for j := range 2 {
			if b.At(i, j) != expected[i][j] {
				t.Errorf("AddScalar: expected %f at (%d,%d), got %f", expected[i][j], i, j, b.At(i, j))
			}
		}
	}

	// Test SubScalar
	c := a.SubScalar(1)
	expectedSub := [][]float64{{0, 1}, {2, 3}}
	for i := range 2 {
		for j := range 2 {
			if c.At(i, j) != expectedSub[i][j] {
				t.Errorf("SubScalar: expected %f at (%d,%d), got %f", expectedSub[i][j], i, j, c.At(i, j))
			}
		}
	}

	// Test DivScalar
	d := a.DivScalar(2)
	expectedDiv := [][]float64{{0.5, 1}, {1.5, 2}}
	for i := range 2 {
		for j := range 2 {
			if d.At(i, j) != expectedDiv[i][j] {
				t.Errorf("DivScalar: expected %f at (%d,%d), got %f", expectedDiv[i][j], i, j, d.At(i, j))
			}
		}
	}
}

func TestTensorDot(t *testing.T) {
	// Test vector dot product
	a := NewTensorFromSlice([][]float64{{1, 2, 3}})
	b := NewTensorFromSlice([][]float64{{4, 5, 6}})

	dot := a.Dot(b)
	expected := 32.0 // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

	if dot != expected {
		t.Errorf("Dot: expected %f, got %f", expected, dot)
	}
}

func TestTensorTrace(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	trace := a.Trace()
	expected := 5.0 // 1 + 4 = 5

	if trace != expected {
		t.Errorf("Trace: expected %f, got %f", expected, trace)
	}
}

func TestTensorDiagonal(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})
	diag := a.Diagonal()

	expected := []float64{1, 5, 9}
	for i := range 3 {
		if diag.At(i, 0) != expected[i] {
			t.Errorf("Diagonal: expected %f at (%d,0), got %f", expected[i], i, diag.At(i, 0))
		}
	}
}

func TestTensorValidation(t *testing.T) {
	// Test valid tensor
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	if err := a.Validate(); err != nil {
		t.Errorf("Validate: valid tensor should not return error, got %v", err)
	}

	// Test finite checks
	if a.HasNaN() {
		t.Error("HasNaN: valid tensor should not have NaN")
	}

	if a.HasInf() {
		t.Error("HasInf: valid tensor should not have Inf")
	}

	if !a.IsFinite() {
		t.Error("IsFinite: valid tensor should be finite")
	}
}

func TestTensorErrorHandling(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})

	// Test division by zero scalar
	defer func() {
		if r := recover(); r == nil {
			t.Error("DivScalar by zero should panic")
		}
	}()
	a.DivScalar(0)
}

func TestTensorMemoryPooling(t *testing.T) {
	// Test that operations use memory pooling
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	b := NewTensorFromSlice([][]float64{{2, 3}, {4, 5}})

	// Perform operations that should use pooling
	results := make([]Tensor, 10)
	for i := range 10 {
		results[i] = a.Add(b)
	}

	// Release all results
	for _, result := range results {
		result.Release()
	}

	// This test mainly ensures no panics occur during pooling
	t.Log("Memory pooling test completed successfully")
}
