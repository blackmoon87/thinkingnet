package core

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// DenseTensor implements the Tensor interface using gonum's Dense matrix.
type DenseTensor struct {
	data  *mat.Dense
	shape []int
	name  string
}

// NewTensor creates a new tensor from a gonum Dense matrix.
func NewTensor(data *mat.Dense) *DenseTensor {
	rows, cols := data.Dims()
	return &DenseTensor{
		data:  data,
		shape: []int{rows, cols},
		name:  "",
	}
}

// NewTensorFromSlice creates a new tensor from a 2D slice.
func NewTensorFromSlice(data [][]float64) *DenseTensor {
	if len(data) == 0 {
		return NewTensor(mat.NewDense(0, 0, nil))
	}

	rows := len(data)
	cols := len(data[0])

	// Flatten the 2D slice
	flat := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			flat[i*cols+j] = data[i][j]
		}
	}

	return NewTensor(mat.NewDense(rows, cols, flat))
}

// NewTensorFromData creates a new tensor with given dimensions and data.
func NewTensorFromData(rows, cols int, data []float64) *DenseTensor {
	return NewTensor(mat.NewDense(rows, cols, data))
}

// NewZerosTensor creates a tensor filled with zeros.
func NewZerosTensor(rows, cols int) *DenseTensor {
	return NewTensor(mat.NewDense(rows, cols, nil))
}

// NewOnesTensor creates a tensor filled with ones.
func NewOnesTensor(rows, cols int) *DenseTensor {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = 1.0
	}
	return NewTensor(mat.NewDense(rows, cols, data))
}

// Dims returns the dimensions of the tensor.
func (t *DenseTensor) Dims() (int, int) {
	return t.data.Dims()
}

// At returns the value at position (i, j).
func (t *DenseTensor) At(i, j int) float64 {
	return t.data.At(i, j)
}

// Set sets the value at position (i, j).
func (t *DenseTensor) Set(i, j int, v float64) {
	t.data.Set(i, j, v)
}

// Copy creates a deep copy of the tensor.
func (t *DenseTensor) Copy() Tensor {
	rows, cols := t.Dims()
	newData := GetMatrix(rows, cols)
	newData.Copy(t.data)

	result := NewTensor(newData)
	result.name = t.name
	return result
}

// Add performs element-wise addition.
func (t *DenseTensor) Add(other Tensor) Tensor {
	if err := ValidateDimensions(t, other, "add"); err != nil {
		panic(err) // In production, we might want to return error instead
	}

	rows, cols := t.Dims()
	config := GetParallelConfig()

	// Use parallel processing for large tensors
	if config.Enabled && rows*cols >= config.MinSize {
		result, err := ParallelElementWiseOperation(t, other, func(a, b float64) float64 {
			return a + b
		})
		if err == nil {
			return result
		}
		// Fallback to sequential if parallel fails
	}

	// Sequential implementation
	result := GetMatrix(rows, cols)
	result.Add(t.data, other.RawMatrix())

	return NewTensor(result)
}

// Mul performs matrix multiplication.
func (t *DenseTensor) Mul(other Tensor) Tensor {
	if err := ValidateDimensions(t, other, "mul"); err != nil {
		panic(err)
	}

	rows, _ := t.Dims()
	_, cols := other.Dims()
	config := GetParallelConfig()

	// Use parallel processing for large matrices
	if config.Enabled && rows >= config.MinSize {
		result, err := ParallelMatrixMultiply(t, other)
		if err == nil {
			return result
		}
		// Fallback to sequential if parallel fails
	}

	// Sequential implementation using optimized BLAS
	result := GetMatrix(rows, cols)
	result.Mul(t.data, other.RawMatrix())

	return NewTensor(result)
}

// MulElem performs element-wise multiplication.
func (t *DenseTensor) MulElem(other Tensor) Tensor {
	if err := ValidateDimensions(t, other, "mul_elem"); err != nil {
		panic(err)
	}

	rows, cols := t.Dims()
	config := GetParallelConfig()

	// Use parallel processing for large tensors
	if config.Enabled && rows*cols >= config.MinSize {
		result, err := ParallelElementWiseOperation(t, other, func(a, b float64) float64 {
			return a * b
		})
		if err == nil {
			return result
		}
		// Fallback to sequential if parallel fails
	}

	// Sequential implementation
	result := GetMatrix(rows, cols)
	result.MulElem(t.data, other.RawMatrix())

	return NewTensor(result)
}

// Sub performs element-wise subtraction.
func (t *DenseTensor) Sub(other Tensor) Tensor {
	if err := ValidateDimensions(t, other, "sub"); err != nil {
		panic(err)
	}

	rows, cols := t.Dims()
	config := GetParallelConfig()

	// Use parallel processing for large tensors
	if config.Enabled && rows*cols >= config.MinSize {
		result, err := ParallelElementWiseOperation(t, other, func(a, b float64) float64 {
			return a - b
		})
		if err == nil {
			return result
		}
		// Fallback to sequential if parallel fails
	}

	// Sequential implementation
	result := GetMatrix(rows, cols)
	result.Sub(t.data, other.RawMatrix())

	return NewTensor(result)
}

// Scale multiplies all elements by a scalar.
func (t *DenseTensor) Scale(scalar float64) Tensor {
	rows, cols := t.Dims()
	result := GetMatrix(rows, cols)
	result.Scale(scalar, t.data)

	return NewTensor(result)
}

// T returns the transpose.
func (t *DenseTensor) T() Tensor {
	cols, rows := t.Dims() // Note: transposed dimensions
	result := GetMatrix(rows, cols)
	result.Copy(t.data.T())

	return NewTensor(result)
}

// RawMatrix returns the underlying gonum matrix.
func (t *DenseTensor) RawMatrix() *mat.Dense {
	return t.data
}

// Shape returns the shape as a slice.
func (t *DenseTensor) Shape() []int {
	return []int{t.shape[0], t.shape[1]}
}

// Name returns the tensor name.
func (t *DenseTensor) Name() string {
	return t.name
}

// SetName sets the tensor name.
func (t *DenseTensor) SetName(name string) {
	t.name = name
}

// String returns a string representation of the tensor.
func (t *DenseTensor) String() string {
	if t.name != "" {
		return fmt.Sprintf("Tensor(%s): %v", t.name, mat.Formatted(t.data, mat.Prefix(""), mat.Squeeze()))
	}
	return fmt.Sprintf("Tensor: %v", mat.Formatted(t.data, mat.Prefix(""), mat.Squeeze()))
}

// Apply applies a function element-wise to the tensor.
func (t *DenseTensor) Apply(fn func(i, j int, v float64) float64) Tensor {
	rows, cols := t.Dims()
	result := GetMatrix(rows, cols)
	result.Apply(fn, t.data)

	return NewTensor(result)
}

// Sum returns the sum of all elements.
func (t *DenseTensor) Sum() float64 {
	return mat.Sum(t.data)
}

// Max returns the maximum element.
func (t *DenseTensor) Max() float64 {
	rows, cols := t.Dims()
	if rows == 0 || cols == 0 {
		return 0
	}

	max := t.At(0, 0)
	for i := range rows {
		for j := range cols {
			if val := t.At(i, j); val > max {
				max = val
			}
		}
	}
	return max
}

// Min returns the minimum element.
func (t *DenseTensor) Min() float64 {
	rows, cols := t.Dims()
	if rows == 0 || cols == 0 {
		return 0
	}

	min := t.At(0, 0)
	for i := range rows {
		for j := range cols {
			if val := t.At(i, j); val < min {
				min = val
			}
		}
	}
	return min
}

// Row returns a specific row as a new tensor.
func (t *DenseTensor) Row(i int) Tensor {
	_, cols := t.Dims()
	data := make([]float64, cols)
	for j := 0; j < cols; j++ {
		data[j] = t.At(i, j)
	}
	return NewTensorFromData(1, cols, data)
}

// Col returns a specific column as a new tensor.
func (t *DenseTensor) Col(j int) Tensor {
	rows, _ := t.Dims()
	data := make([]float64, rows)
	for i := 0; i < rows; i++ {
		data[i] = t.At(i, j)
	}
	return NewTensorFromData(rows, 1, data)
}

// Slice returns a slice of the tensor.
func (t *DenseTensor) Slice(r0, r1, c0, c1 int) Tensor {
	sliced := t.data.Slice(r0, r1, c0, c1).(*mat.Dense)
	return NewTensor(mat.DenseCopyOf(sliced))
}

// SetRow sets a specific row.
func (t *DenseTensor) SetRow(i int, data []float64) {
	_, cols := t.Dims()
	if len(data) != cols {
		panic(NewError(ErrDimensionMismatch,
			fmt.Sprintf("row data length %d doesn't match tensor columns %d", len(data), cols)))
	}

	for j := range cols {
		t.Set(i, j, data[j])
	}
}

// SetCol sets a specific column.
func (t *DenseTensor) SetCol(j int, data []float64) {
	rows, _ := t.Dims()
	if len(data) != rows {
		panic(NewError(ErrDimensionMismatch,
			fmt.Sprintf("column data length %d doesn't match tensor rows %d", len(data), rows)))
	}

	for i := range rows {
		t.Set(i, j, data[i])
	}
}

// IsEmpty returns true if the tensor has zero elements.
func (t *DenseTensor) IsEmpty() bool {
	rows, cols := t.Dims()
	return rows == 0 || cols == 0
}

// IsSquare returns true if the tensor is square.
func (t *DenseTensor) IsSquare() bool {
	rows, cols := t.Dims()
	return rows == cols
}

// IsVector returns true if the tensor is a vector (single row or column).
func (t *DenseTensor) IsVector() bool {
	rows, cols := t.Dims()
	return rows == 1 || cols == 1
}

// Norm returns the Frobenius norm of the tensor.
func (t *DenseTensor) Norm() float64 {
	return mat.Norm(t.data, 2)
}

// Div performs element-wise division.
func (t *DenseTensor) Div(other Tensor) Tensor {
	if err := ValidateDimensions(t, other, "div"); err != nil {
		panic(err)
	}

	rows, cols := t.Dims()
	result := GetMatrix(rows, cols)

	for i := range rows {
		for j := range cols {
			divisor := other.At(i, j)
			if math.Abs(divisor) < GetEpsilon() {
				if divisor >= 0 {
					result.Set(i, j, t.At(i, j)/GetEpsilon())
				} else {
					result.Set(i, j, t.At(i, j)/(-GetEpsilon()))
				}
			} else {
				result.Set(i, j, t.At(i, j)/divisor)
			}
		}
	}

	return NewTensor(result)
}

// Pow raises each element to the given power.
func (t *DenseTensor) Pow(power float64) Tensor {
	rows, cols := t.Dims()
	result := GetMatrix(rows, cols)

	result.Apply(func(i, j int, v float64) float64 {
		return math.Pow(t.At(i, j), power)
	}, result)

	return NewTensor(result)
}

// Sqrt computes the square root of each element.
func (t *DenseTensor) Sqrt() Tensor {
	rows, cols := t.Dims()
	result := GetMatrix(rows, cols)

	result.Apply(func(i, j int, v float64) float64 {
		val := t.At(i, j)
		if val < 0 {
			return 0 // Handle negative values gracefully
		}
		return math.Sqrt(val)
	}, result)

	return NewTensor(result)
}

// Exp computes the exponential of each element with overflow protection.
func (t *DenseTensor) Exp() Tensor {
	rows, cols := t.Dims()
	result := GetMatrix(rows, cols)

	result.Apply(func(i, j int, v float64) float64 {
		val := t.At(i, j)
		// Clamp to prevent overflow
		val = math.Max(-250, math.Min(250, val))
		return math.Exp(val)
	}, result)

	return NewTensor(result)
}

// Log computes the natural logarithm of each element with numerical stability.
func (t *DenseTensor) Log() Tensor {
	rows, cols := t.Dims()
	result := GetMatrix(rows, cols)

	result.Apply(func(i, j int, v float64) float64 {
		val := t.At(i, j)
		return math.Log(math.Max(val, GetEpsilon()))
	}, result)

	return NewTensor(result)
}

// Abs computes the absolute value of each element.
func (t *DenseTensor) Abs() Tensor {
	rows, cols := t.Dims()
	result := GetMatrix(rows, cols)

	result.Apply(func(i, j int, v float64) float64 {
		return math.Abs(t.At(i, j))
	}, result)

	return NewTensor(result)
}

// Sign returns the sign of each element (-1, 0, or 1).
func (t *DenseTensor) Sign() Tensor {
	rows, cols := t.Dims()
	result := GetMatrix(rows, cols)

	result.Apply(func(i, j int, v float64) float64 {
		val := t.At(i, j)
		if val > GetEpsilon() {
			return 1
		} else if val < -GetEpsilon() {
			return -1
		}
		return 0
	}, result)

	return NewTensor(result)
}

// Clamp constrains all elements to be within [min, max].
func (t *DenseTensor) Clamp(min, max float64) Tensor {
	rows, cols := t.Dims()
	result := GetMatrix(rows, cols)

	result.Apply(func(i, j int, v float64) float64 {
		val := t.At(i, j)
		if val < min {
			return min
		} else if val > max {
			return max
		}
		return val
	}, result)

	return NewTensor(result)
}

// Mean computes the mean of all elements.
func (t *DenseTensor) Mean() float64 {
	rows, cols := t.Dims()
	if rows == 0 || cols == 0 {
		return 0
	}
	return t.Sum() / float64(rows*cols)
}

// Std computes the standard deviation of all elements.
func (t *DenseTensor) Std() float64 {
	rows, cols := t.Dims()
	if rows == 0 || cols == 0 {
		return 0
	}

	mean := t.Mean()
	var sumSq float64

	for i := range rows {
		for j := range cols {
			diff := t.At(i, j) - mean
			sumSq += diff * diff
		}
	}

	return math.Sqrt(sumSq / float64(rows*cols))
}

// Reshape returns a new tensor with the specified dimensions.
func (t *DenseTensor) Reshape(newRows, newCols int) Tensor {
	rows, cols := t.Dims()
	if rows*cols != newRows*newCols {
		panic(NewError(ErrDimensionMismatch,
			fmt.Sprintf("cannot reshape tensor from (%d,%d) to (%d,%d): different number of elements",
				rows, cols, newRows, newCols)))
	}

	// Extract data in row-major order
	data := make([]float64, rows*cols)
	for i := range rows {
		for j := range cols {
			data[i*cols+j] = t.At(i, j)
		}
	}

	return NewTensorFromData(newRows, newCols, data)
}

// Flatten returns a new tensor with shape (1, rows*cols).
func (t *DenseTensor) Flatten() Tensor {
	rows, cols := t.Dims()
	return t.Reshape(1, rows*cols)
}

// Equal checks if two tensors are element-wise equal within epsilon tolerance.
func (t *DenseTensor) Equal(other Tensor) bool {
	if err := ValidateDimensions(t, other, "equal"); err != nil {
		return false
	}

	rows, cols := t.Dims()
	for i := range rows {
		for j := range cols {
			if math.Abs(t.At(i, j)-other.At(i, j)) > GetEpsilon() {
				return false
			}
		}
	}
	return true
}

// Fill sets all elements to the specified value.
func (t *DenseTensor) Fill(value float64) {
	rows, cols := t.Dims()
	for i := range rows {
		for j := range cols {
			t.Set(i, j, value)
		}
	}
}

// Zero sets all elements to zero.
func (t *DenseTensor) Zero() {
	t.data.Zero()
}

// Release returns the tensor's matrix to the pool for reuse.
func (t *DenseTensor) Release() {
	if t.data != nil {
		PutMatrix(t.data)
		t.data = nil
	}
}

// Dot performs dot product for vectors or matrix multiplication for matrices.
func (t *DenseTensor) Dot(other Tensor) float64 {
	// For vectors, compute dot product
	if t.IsVector() && other.IsVector() {
		tRows, tCols := t.Dims()
		oRows, oCols := other.Dims()

		// Flatten both vectors
		tSize := tRows * tCols
		oSize := oRows * oCols

		if tSize != oSize {
			panic(NewError(ErrDimensionMismatch,
				fmt.Sprintf("vectors must have same length for dot product: %d vs %d", tSize, oSize)))
		}

		var result float64

		// Handle row vector
		if tRows == 1 && oRows == 1 {
			for j := range tCols {
				result += t.At(0, j) * other.At(0, j)
			}
		} else if tCols == 1 && oCols == 1 {
			// Handle column vector
			for i := range tRows {
				result += t.At(i, 0) * other.At(i, 0)
			}
		} else {
			// Mixed vector types - flatten and compute
			tFlat := make([]float64, tSize)
			oFlat := make([]float64, oSize)

			idx := 0
			for i := range tRows {
				for j := range tCols {
					tFlat[idx] = t.At(i, j)
					idx++
				}
			}

			idx = 0
			for i := range oRows {
				for j := range oCols {
					oFlat[idx] = other.At(i, j)
					idx++
				}
			}

			for i := range tSize {
				result += tFlat[i] * oFlat[i]
			}
		}

		return result
	}

	panic(NewError(ErrUnsupportedOperation, "dot product only supported for vectors"))
}

// AddScalar adds a scalar value to all elements.
func (t *DenseTensor) AddScalar(scalar float64) Tensor {
	rows, cols := t.Dims()
	result := GetMatrix(rows, cols)

	for i := range rows {
		for j := range cols {
			result.Set(i, j, t.At(i, j)+scalar)
		}
	}

	return NewTensor(result)
}

// SubScalar subtracts a scalar value from all elements.
func (t *DenseTensor) SubScalar(scalar float64) Tensor {
	rows, cols := t.Dims()
	result := GetMatrix(rows, cols)

	for i := range rows {
		for j := range cols {
			result.Set(i, j, t.At(i, j)-scalar)
		}
	}

	return NewTensor(result)
}

// DivScalar divides all elements by a scalar value.
func (t *DenseTensor) DivScalar(scalar float64) Tensor {
	if math.Abs(scalar) < GetEpsilon() {
		panic(NewError(ErrInvalidInput, "cannot divide by zero or near-zero scalar"))
	}

	rows, cols := t.Dims()
	result := GetMatrix(rows, cols)

	for i := range rows {
		for j := range cols {
			result.Set(i, j, t.At(i, j)/scalar)
		}
	}

	return NewTensor(result)
}

// Trace computes the trace (sum of diagonal elements) for square matrices.
func (t *DenseTensor) Trace() float64 {
	if !t.IsSquare() {
		panic(NewError(ErrDimensionMismatch, "trace is only defined for square matrices"))
	}

	rows, _ := t.Dims()
	var trace float64
	for i := range rows {
		trace += t.At(i, i)
	}
	return trace
}

// Diagonal returns the diagonal elements as a vector.
func (t *DenseTensor) Diagonal() Tensor {
	rows, cols := t.Dims()
	minDim := rows
	if cols < minDim {
		minDim = cols
	}

	data := make([]float64, minDim)
	for i := range minDim {
		data[i] = t.At(i, i)
	}

	return NewTensorFromData(minDim, 1, data)
}

// Validate performs comprehensive validation of the tensor.
func (t *DenseTensor) Validate() error {
	if t.data == nil {
		return NewError(ErrInvalidInput, "tensor data is nil")
	}

	rows, cols := t.Dims()
	if rows < 0 || cols < 0 {
		return NewError(ErrDimensionMismatch, "tensor dimensions cannot be negative")
	}

	// Check for NaN or Inf values
	for i := range rows {
		for j := range cols {
			val := t.At(i, j)
			if math.IsNaN(val) {
				return NewError(ErrNumericalInstability,
					fmt.Sprintf("NaN value found at position (%d,%d)", i, j))
			}
			if math.IsInf(val, 0) {
				return NewError(ErrNumericalInstability,
					fmt.Sprintf("Inf value found at position (%d,%d)", i, j))
			}
		}
	}

	return nil
}

// HasNaN checks if the tensor contains any NaN values.
func (t *DenseTensor) HasNaN() bool {
	rows, cols := t.Dims()
	for i := range rows {
		for j := range cols {
			if math.IsNaN(t.At(i, j)) {
				return true
			}
		}
	}
	return false
}

// HasInf checks if the tensor contains any infinite values.
func (t *DenseTensor) HasInf() bool {
	rows, cols := t.Dims()
	for i := range rows {
		for j := range cols {
			if math.IsInf(t.At(i, j), 0) {
				return true
			}
		}
	}
	return false
}

// IsFinite checks if all values in the tensor are finite.
func (t *DenseTensor) IsFinite() bool {
	return !t.HasNaN() && !t.HasInf()
}

// SafeOperation performs an operation with error recovery.
func (t *DenseTensor) SafeOperation(operation string, fn func() Tensor) (Tensor, error) {
	defer func() {
		if r := recover(); r != nil {
			// Convert panic to error
			if err, ok := r.(error); ok {
				if tnErr, ok := err.(*ThinkingNetError); ok {
					// Re-throw ThinkingNet errors
					panic(tnErr)
				}
			}
			// Convert other panics to ThinkingNet errors
			panic(NewError(ErrNumericalInstability, fmt.Sprintf("operation %s failed: %v", operation, r)))
		}
	}()

	result := fn()

	// Validate result
	if err := ValidateTensorFinite(result, "operation_result"); err != nil {
		return nil, NewErrorWithCause(ErrNumericalInstability,
			fmt.Sprintf("operation %s produced non-finite values", operation), err)
	}

	return result, nil
}

// Helper Functions for Easy Usage

// EasyTensor creates a tensor from a 2D slice with simple error handling.
// This is a convenience function that provides better error messages for common use cases.
func EasyTensor(data [][]float64) Tensor {
	if len(data) == 0 {
		panic(NewError(ErrInvalidInput, "لا يمكن إنشاء tensor من بيانات فارغة / Cannot create tensor from empty data"))
	}

	if len(data[0]) == 0 {
		panic(NewError(ErrInvalidInput, "لا يمكن إنشاء tensor من صفوف فارغة / Cannot create tensor from empty rows"))
	}

	// Validate that all rows have the same length
	cols := len(data[0])
	for i, row := range data {
		if len(row) != cols {
			panic(NewError(ErrDimensionMismatch,
				fmt.Sprintf("جميع الصفوف يجب أن تكون بنفس الطول. الصف %d له %d عناصر بينما متوقع %d / All rows must have the same length. Row %d has %d elements, expected %d",
					i, len(row), cols, i, len(row), cols)))
		}
	}

	return NewTensorFromSlice(data)
}

// EasySplit splits data into training and testing sets with sensible defaults.
// This is a convenience wrapper around the preprocessing.TrainTestSplit function.
func EasySplit(X, y Tensor, testSize float64) (XTrain, XTest, yTrain, yTest Tensor) {
	// Import the preprocessing package function
	// We need to use the existing TrainTestSplit function from preprocessing
	// For now, we'll implement a simple version here to avoid circular imports

	if testSize <= 0 || testSize >= 1 {
		panic(NewError(ErrInvalidInput,
			fmt.Sprintf("حجم بيانات الاختبار يجب أن يكون بين 0 و 1، تم تمرير %f / Test size must be between 0 and 1, got %f", testSize, testSize)))
	}

	// Validate input tensors
	if err := ValidateTrainingData(X, y); err != nil {
		panic(NewErrorWithCause(ErrInvalidInput, "بيانات الإدخال غير صحيحة / Invalid input data", err))
	}

	nSamples, _ := X.Dims()
	if nSamples < 2 {
		panic(NewError(ErrInvalidInput, "يجب أن يكون هناك على الأقل عينتان لتقسيم البيانات / Need at least 2 samples for data splitting"))
	}

	// Calculate split indices
	testSamples := int(float64(nSamples) * testSize)
	if testSamples == 0 {
		testSamples = 1 // Ensure at least one test sample
	}
	if testSamples >= nSamples {
		testSamples = nSamples - 1 // Ensure at least one train sample
	}

	trainSamples := nSamples - testSamples

	// Create index arrays
	trainIndices := make([]int, trainSamples)
	testIndices := make([]int, testSamples)

	for i := 0; i < trainSamples; i++ {
		trainIndices[i] = i
	}
	for i := 0; i < testSamples; i++ {
		testIndices[i] = trainSamples + i
	}

	// Extract training data
	_, xCols := X.Dims()
	_, yCols := y.Dims()

	XTrain = NewZerosTensor(trainSamples, xCols)
	yTrain = NewZerosTensor(trainSamples, yCols)

	for i, idx := range trainIndices {
		for j := 0; j < xCols; j++ {
			XTrain.Set(i, j, X.At(idx, j))
		}
		for j := 0; j < yCols; j++ {
			yTrain.Set(i, j, y.At(idx, j))
		}
	}

	// Extract testing data
	XTest = NewZerosTensor(testSamples, xCols)
	yTest = NewZerosTensor(testSamples, yCols)

	for i, idx := range testIndices {
		for j := 0; j < xCols; j++ {
			XTest.Set(i, j, X.At(idx, j))
		}
		for j := 0; j < yCols; j++ {
			yTest.Set(i, j, y.At(idx, j))
		}
	}

	return XTrain, XTest, yTrain, yTest
}
