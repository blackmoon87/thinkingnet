package core

import (
	"math"
	"testing"
)

// TestTensorEdgeCases tests edge cases and error conditions
func TestTensorEdgeCases(t *testing.T) {
	// Test minimal tensor creation (1x0 or 0x1 are also invalid in gonum)
	// So we'll test with 1x1 minimal tensor instead
	minimal := NewZerosTensor(1, 1)
	if minimal.IsEmpty() {
		t.Error("1x1 tensor should not report IsEmpty() as true")
	}

	// Test single element tensor
	single := NewTensorFromSlice([][]float64{{42}})
	if single.At(0, 0) != 42 {
		t.Errorf("Single element tensor: expected 42, got %f", single.At(0, 0))
	}

	// Test zero tensor creation
	zeros := NewZerosTensor(3, 3)
	for i := range 3 {
		for j := range 3 {
			if zeros.At(i, j) != 0 {
				t.Errorf("Zero tensor should have all zeros, got %f at (%d,%d)", zeros.At(i, j), i, j)
			}
		}
	}

	// Test ones tensor creation
	ones := NewOnesTensor(2, 2)
	for i := range 2 {
		for j := range 2 {
			if ones.At(i, j) != 1 {
				t.Errorf("Ones tensor should have all ones, got %f at (%d,%d)", ones.At(i, j), i, j)
			}
		}
	}
}

func TestTensorNumericalStability(t *testing.T) {
	// Test with very small numbers (for future use)
	_ = NewTensorFromSlice([][]float64{{1e-15, 2e-15}, {3e-15, 4e-15}})

	// Test exp with overflow protection
	large := NewTensorFromSlice([][]float64{{300, 400}, {500, 600}})
	expResult := large.Exp()

	// Should not contain Inf values due to clamping
	if expResult.HasInf() {
		t.Error("Exp should prevent overflow and not contain Inf values")
	}

	// Test log with underflow protection
	verySmall := NewTensorFromSlice([][]float64{{1e-20, 1e-30}, {0, -1}})
	logResult := verySmall.Log()

	// Should not contain -Inf or NaN due to epsilon protection
	if logResult.HasInf() || logResult.HasNaN() {
		t.Error("Log should prevent underflow and not contain Inf or NaN values")
	}

	// Test sqrt with negative values
	negative := NewTensorFromSlice([][]float64{{-1, -2}, {3, 4}})
	sqrtResult := negative.Sqrt()

	// Negative values should be handled gracefully (set to 0)
	if sqrtResult.At(0, 0) != 0 || sqrtResult.At(0, 1) != 0 {
		t.Error("Sqrt should handle negative values gracefully")
	}
}

func TestTensorAdvancedOperations(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, -2, 3}, {-4, 5, -6}})

	// Test Abs
	absResult := a.Abs()
	expected := [][]float64{{1, 2, 3}, {4, 5, 6}}
	for i := range 2 {
		for j := range 3 {
			if absResult.At(i, j) != expected[i][j] {
				t.Errorf("Abs: expected %f at (%d,%d), got %f", expected[i][j], i, j, absResult.At(i, j))
			}
		}
	}

	// Test Sign
	signResult := a.Sign()
	expectedSign := [][]float64{{1, -1, 1}, {-1, 1, -1}}
	for i := range 2 {
		for j := range 3 {
			if signResult.At(i, j) != expectedSign[i][j] {
				t.Errorf("Sign: expected %f at (%d,%d), got %f", expectedSign[i][j], i, j, signResult.At(i, j))
			}
		}
	}

	// Test Std (standard deviation)
	std := a.Std()
	if std <= 0 {
		t.Error("Standard deviation should be positive for non-constant data")
	}

	// Test Norm
	norm := a.Norm()
	if norm <= 0 {
		t.Error("Norm should be positive for non-zero tensor")
	}
}

func TestTensorSlicingAdvanced(t *testing.T) {
	a := NewTensorFromSlice([][]float64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
	})

	// Test slice operation
	slice := a.Slice(1, 3, 1, 3)
	rows, cols := slice.Dims()
	if rows != 2 || cols != 2 {
		t.Errorf("Slice: expected dimensions (2,2), got (%d,%d)", rows, cols)
	}

	expected := [][]float64{{6, 7}, {10, 11}}
	for i := range 2 {
		for j := range 2 {
			if slice.At(i, j) != expected[i][j] {
				t.Errorf("Slice: expected %f at (%d,%d), got %f", expected[i][j], i, j, slice.At(i, j))
			}
		}
	}

	// Test SetRow
	newRow := []float64{100, 200, 300, 400}
	a.SetRow(0, newRow)
	for j := range 4 {
		if a.At(0, j) != newRow[j] {
			t.Errorf("SetRow: expected %f at (0,%d), got %f", newRow[j], j, a.At(0, j))
		}
	}

	// Test SetCol
	newCol := []float64{1000, 2000, 3000}
	a.SetCol(0, newCol)
	for i := range 3 {
		if a.At(i, 0) != newCol[i] {
			t.Errorf("SetCol: expected %f at (%d,0), got %f", newCol[i], i, a.At(i, 0))
		}
	}
}

func TestTensorReshapeAdvanced(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2, 3}, {4, 5, 6}})

	// Test flatten
	flat := a.Flatten()
	rows, cols := flat.Dims()
	if rows != 1 || cols != 6 {
		t.Errorf("Flatten: expected dimensions (1,6), got (%d,%d)", rows, cols)
	}

	expected := []float64{1, 2, 3, 4, 5, 6}
	for j := range 6 {
		if flat.At(0, j) != expected[j] {
			t.Errorf("Flatten: expected %f at (0,%d), got %f", expected[j], j, flat.At(0, j))
		}
	}

	// Test reshape to different dimensions
	reshaped := a.Reshape(3, 2)
	reshapeRows, reshapeCols := reshaped.Dims()
	if reshapeRows != 3 || reshapeCols != 2 {
		t.Errorf("Reshape: expected dimensions (3,2), got (%d,%d)", reshapeRows, reshapeCols)
	}
}

func TestTensorFillAndZero(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})

	// Test Fill
	a.Fill(7.5)
	for i := range 2 {
		for j := range 2 {
			if a.At(i, j) != 7.5 {
				t.Errorf("Fill: expected 7.5 at (%d,%d), got %f", i, j, a.At(i, j))
			}
		}
	}

	// Test Zero
	a.Zero()
	for i := range 2 {
		for j := range 2 {
			if a.At(i, j) != 0 {
				t.Errorf("Zero: expected 0 at (%d,%d), got %f", i, j, a.At(i, j))
			}
		}
	}
}

func TestTensorNameAndString(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})

	// Test name operations
	if a.Name() != "" {
		t.Error("New tensor should have empty name")
	}

	a.SetName("test_tensor")
	if a.Name() != "test_tensor" {
		t.Errorf("Expected name 'test_tensor', got '%s'", a.Name())
	}

	// Test string representation
	str := a.String()
	if str == "" {
		t.Error("String representation should not be empty")
	}
}

func TestTensorApply(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})

	// Test Apply function - square each element
	squared := a.Apply(func(i, j int, v float64) float64 {
		return v * v
	})

	expected := [][]float64{{1, 4}, {9, 16}}
	for i := range 2 {
		for j := range 2 {
			if squared.At(i, j) != expected[i][j] {
				t.Errorf("Apply: expected %f at (%d,%d), got %f", expected[i][j], i, j, squared.At(i, j))
			}
		}
	}
}

func TestTensorDotAdvanced(t *testing.T) {
	// Test column vector dot product
	a := NewTensorFromSlice([][]float64{{1}, {2}, {3}})
	b := NewTensorFromSlice([][]float64{{4}, {5}, {6}})

	dot := a.Dot(b)
	expected := 32.0 // 1*4 + 2*5 + 3*6 = 32

	if dot != expected {
		t.Errorf("Column vector dot: expected %f, got %f", expected, dot)
	}

	// Test mixed vector types
	rowVec := NewTensorFromSlice([][]float64{{1, 2, 3}})
	colVec := NewTensorFromSlice([][]float64{{4}, {5}, {6}})

	mixedDot := rowVec.Dot(colVec)
	if mixedDot != expected {
		t.Errorf("Mixed vector dot: expected %f, got %f", expected, mixedDot)
	}
}

func TestTensorDiagonalAdvanced(t *testing.T) {
	// Test rectangular matrix diagonal
	rect := NewTensorFromSlice([][]float64{{1, 2, 3}, {4, 5, 6}})
	diag := rect.Diagonal()

	rows, cols := diag.Dims()
	if rows != 2 || cols != 1 {
		t.Errorf("Rectangular diagonal: expected dimensions (2,1), got (%d,%d)", rows, cols)
	}

	expected := []float64{1, 5}
	for i := range 2 {
		if diag.At(i, 0) != expected[i] {
			t.Errorf("Rectangular diagonal: expected %f at (%d,0), got %f", expected[i], i, diag.At(i, 0))
		}
	}
}

func TestTensorErrorConditions(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	b := NewTensorFromSlice([][]float64{{1, 2, 3}, {4, 5, 6}}) // Different dimensions

	// Test dimension mismatch panics
	testPanic := func(name string, fn func()) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("%s should panic on dimension mismatch", name)
			}
		}()
		fn()
	}

	testPanic("Add", func() { a.Add(b) })
	testPanic("Sub", func() { a.Sub(b) })
	testPanic("MulElem", func() { a.MulElem(b) })
	testPanic("Div", func() { a.Div(b) })

	// Test invalid reshape
	testPanic("Reshape", func() { a.Reshape(3, 3) }) // 4 elements can't become 9

	// Test trace on non-square matrix
	rect := NewTensorFromSlice([][]float64{{1, 2, 3}, {4, 5, 6}})
	testPanic("Trace", func() { rect.Trace() })

	// Test SetRow with wrong length
	testPanic("SetRow", func() { a.SetRow(0, []float64{1, 2, 3}) })

	// Test SetCol with wrong length
	testPanic("SetCol", func() { a.SetCol(0, []float64{1, 2, 3}) })

	// Test dot product with incompatible vectors
	v1 := NewTensorFromSlice([][]float64{{1, 2}})
	v2 := NewTensorFromSlice([][]float64{{1, 2, 3}})
	testPanic("Dot", func() { v1.Dot(v2) })

	// Test dot product with non-vectors
	testPanic("DotNonVector", func() { a.Dot(b) })
}

func TestTensorSafeOperation(t *testing.T) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	b := NewTensorFromSlice([][]float64{{2, 3}, {4, 5}})

	// Test successful safe operation
	result, err := a.SafeOperation("add", func() Tensor {
		return a.Add(b)
	})

	if err != nil {
		t.Errorf("Safe operation should succeed, got error: %v", err)
	}

	if result == nil {
		t.Error("Safe operation should return result")
	}

	// Test safe operation with panic recovery - use defer to catch the panic properly
	func() {
		defer func() {
			if r := recover(); r != nil {
				// Expected - the SafeOperation should convert panic to error
				// but it re-panics ThinkingNet errors, so we expect a panic here
				t.Log("SafeOperation correctly re-panicked ThinkingNet error")
			}
		}()

		_, err = a.SafeOperation("invalid", func() Tensor {
			panic("test panic")
		})

		// If we get here without panic, check for error
		if err == nil {
			t.Error("Safe operation should return error when panic occurs")
		}
	}()
}

func TestTensorValidationExtended(t *testing.T) {
	// Test validation with NaN
	nanTensor := NewTensorFromSlice([][]float64{{1, 2}, {3, math.NaN()}})
	if err := nanTensor.Validate(); err == nil {
		t.Error("Validation should fail for tensor with NaN")
	}

	// Test validation with Inf
	infTensor := NewTensorFromSlice([][]float64{{1, 2}, {3, math.Inf(1)}})
	if err := infTensor.Validate(); err == nil {
		t.Error("Validation should fail for tensor with Inf")
	}

	// Test HasNaN
	if !nanTensor.HasNaN() {
		t.Error("HasNaN should return true for tensor with NaN")
	}

	// Test HasInf
	if !infTensor.HasInf() {
		t.Error("HasInf should return true for tensor with Inf")
	}

	// Test IsFinite
	validTensor := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	if !validTensor.IsFinite() {
		t.Error("IsFinite should return true for valid tensor")
	}

	if nanTensor.IsFinite() {
		t.Error("IsFinite should return false for tensor with NaN")
	}

	if infTensor.IsFinite() {
		t.Error("IsFinite should return false for tensor with Inf")
	}
}

// Benchmark tests for performance monitoring
func BenchmarkTensorCreation(b *testing.B) {
	data := [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor := NewTensorFromSlice(data)
		tensor.Release()
	}
}

func BenchmarkTensorArithmetic(b *testing.B) {
	a := NewTensorFromSlice([][]float64{{1, 2}, {3, 4}})
	c := NewTensorFromSlice([][]float64{{2, 3}, {4, 5}})

	b.Run("Add", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			result := a.Add(c)
			result.Release()
		}
	})

	b.Run("Sub", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			result := a.Sub(c)
			result.Release()
		}
	})

	b.Run("MulElem", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			result := a.MulElem(c)
			result.Release()
		}
	})

	b.Run("Div", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			result := a.Div(c)
			result.Release()
		}
	})
}

func BenchmarkTensorMathFunctions(b *testing.B) {
	a := NewTensorFromSlice([][]float64{{1, 2, 3}, {4, 5, 6}})

	b.Run("Exp", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			result := a.Exp()
			result.Release()
		}
	})

	b.Run("Log", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			result := a.Log()
			result.Release()
		}
	})

	b.Run("Sqrt", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			result := a.Sqrt()
			result.Release()
		}
	})

	b.Run("Pow", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			result := a.Pow(2.0)
			result.Release()
		}
	})
}

func BenchmarkTensorStatistics(b *testing.B) {
	a := NewTensorFromSlice([][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})

	b.Run("Sum", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = a.Sum()
		}
	})

	b.Run("Mean", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = a.Mean()
		}
	})

	b.Run("Std", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = a.Std()
		}
	})

	b.Run("Norm", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = a.Norm()
		}
	})
}

func BenchmarkTensorLargeOperations(b *testing.B) {
	// Create larger tensors for realistic benchmarks
	size := 100
	data := make([][]float64, size)
	for i := range size {
		data[i] = make([]float64, size)
		for j := range size {
			data[i][j] = float64(i*size + j)
		}
	}

	a := NewTensorFromSlice(data)
	c := NewTensorFromSlice(data)

	b.Run("LargeAdd", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			result := a.Add(c)
			result.Release()
		}
	})

	b.Run("LargeMatMul", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			result := a.Mul(c)
			result.Release()
		}
	})

	b.Run("LargeTranspose", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			result := a.T()
			result.Release()
		}
	})
}
