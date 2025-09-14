package core

import (
	"fmt"
	"math"
)

// ErrorType represents different types of errors in the library.
type ErrorType int

const (
	// ErrInvalidInput indicates invalid input parameters
	ErrInvalidInput ErrorType = iota

	// ErrDimensionMismatch indicates incompatible tensor dimensions
	ErrDimensionMismatch

	// ErrNotFitted indicates a component hasn't been fitted/trained
	ErrNotFitted

	// ErrNumericalInstability indicates numerical computation issues
	ErrNumericalInstability

	// ErrConfigurationError indicates invalid configuration
	ErrConfigurationError

	// ErrModelNotCompiled indicates model hasn't been compiled
	ErrModelNotCompiled

	// ErrFileIO indicates file input/output errors
	ErrFileIO

	// ErrUnsupportedOperation indicates unsupported operations
	ErrUnsupportedOperation

	// ErrConvergence indicates convergence failures
	ErrConvergence

	// ErrMemory indicates memory allocation issues
	ErrMemory
)

// String returns a string representation of the error type.
func (et ErrorType) String() string {
	switch et {
	case ErrInvalidInput:
		return "InvalidInput"
	case ErrDimensionMismatch:
		return "DimensionMismatch"
	case ErrNotFitted:
		return "NotFitted"
	case ErrNumericalInstability:
		return "NumericalInstability"
	case ErrConfigurationError:
		return "ConfigurationError"
	case ErrModelNotCompiled:
		return "ModelNotCompiled"
	case ErrFileIO:
		return "FileIO"
	case ErrUnsupportedOperation:
		return "UnsupportedOperation"
	case ErrConvergence:
		return "Convergence"
	case ErrMemory:
		return "Memory"
	default:
		return "Unknown"
	}
}

// BilingualMessage represents an error message in both Arabic and English.
type BilingualMessage struct {
	Arabic  string `json:"arabic"`
	English string `json:"english"`
	Tip     string `json:"tip,omitempty"` // Helpful tip for resolving the error
}

// ThinkingNetError represents a structured error in the ThinkingNet library.
type ThinkingNetError struct {
	Type    ErrorType        `json:"type"`
	Message BilingualMessage `json:"message"`
	Context map[string]any   `json:"context,omitempty"`
	Cause   error            `json:"-"` // Original error, not serialized
}

// Error implements the error interface.
func (e *ThinkingNetError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("[%s] %s | %s: %v", e.Type.String(), e.Message.English, e.Message.Arabic, e.Cause)
	}
	return fmt.Sprintf("[%s] %s | %s", e.Type.String(), e.Message.English, e.Message.Arabic)
}

// ErrorEnglish returns the English error message.
func (e *ThinkingNetError) ErrorEnglish() string {
	if e.Cause != nil {
		return fmt.Sprintf("[%s] %s: %v", e.Type.String(), e.Message.English, e.Cause)
	}
	return fmt.Sprintf("[%s] %s", e.Type.String(), e.Message.English)
}

// ErrorArabic returns the Arabic error message.
func (e *ThinkingNetError) ErrorArabic() string {
	if e.Cause != nil {
		return fmt.Sprintf("[%s] %s: %v", e.Type.String(), e.Message.Arabic, e.Cause)
	}
	return fmt.Sprintf("[%s] %s", e.Type.String(), e.Message.Arabic)
}

// GetTip returns the helpful tip for resolving the error.
func (e *ThinkingNetError) GetTip() string {
	return e.Message.Tip
}

// Unwrap returns the underlying error.
func (e *ThinkingNetError) Unwrap() error {
	return e.Cause
}

// WithContext adds context information to the error.
func (e *ThinkingNetError) WithContext(key string, value any) *ThinkingNetError {
	if e.Context == nil {
		e.Context = make(map[string]any)
	}
	e.Context[key] = value
	return e
}

// Common bilingual error messages
var CommonErrorMessages = map[ErrorType]map[string]BilingualMessage{
	ErrInvalidInput: {
		"nil_tensor": {
			Arabic:  "المصفوفة لا يمكن أن تكون فارغة",
			English: "tensor cannot be nil",
			Tip:     "تأكد من إنشاء المصفوفة بشكل صحيح | Make sure to create the tensor properly",
		},
		"zero_dimensions": {
			Arabic:  "المصفوفة لا يمكن أن تحتوي على أبعاد صفرية",
			English: "tensor cannot have zero dimensions",
			Tip:     "تحقق من أن البيانات تحتوي على قيم | Check that data contains values",
		},
		"empty_data": {
			Arabic:  "البيانات لا يمكن أن تكون فارغة",
			English: "data cannot be empty",
			Tip:     "تأكد من وجود بيانات للمعالجة | Ensure there is data to process",
		},
		"invalid_range": {
			Arabic:  "القيمة خارج النطاق المسموح",
			English: "value is out of allowed range",
			Tip:     "تحقق من القيم المدخلة | Check input values",
		},
	},
	ErrDimensionMismatch: {
		"incompatible_shapes": {
			Arabic:  "أشكال المصفوفات غير متوافقة",
			English: "tensor shapes are incompatible",
			Tip:     "تأكد من أن أبعاد المصفوفات متوافقة للعملية | Ensure tensor dimensions are compatible for the operation",
		},
		"matrix_multiplication": {
			Arabic:  "أبعاد غير متوافقة لضرب المصفوفات",
			English: "incompatible dimensions for matrix multiplication",
			Tip:     "عدد الأعمدة في المصفوفة الأولى يجب أن يساوي عدد الصفوف في الثانية | Number of columns in first matrix must equal rows in second",
		},
		"training_data_mismatch": {
			Arabic:  "عدد العينات في X و y غير متطابق",
			English: "number of samples in X and y don't match",
			Tip:     "تأكد من أن X و y لهما نفس عدد الصفوف | Ensure X and y have the same number of rows",
		},
	},
	ErrNotFitted: {
		"component_not_fitted": {
			Arabic:  "المكون لم يتم تدريبه بعد",
			English: "component has not been fitted yet",
			Tip:     "استخدم fit() أو train() أولاً | Use fit() or train() first",
		},
	},
	ErrModelNotCompiled: {
		"model_not_compiled": {
			Arabic:  "النموذج لم يتم تجميعه بعد",
			English: "model has not been compiled yet",
			Tip:     "استخدم Compile() مع optimizer و loss function | Use Compile() with optimizer and loss function",
		},
	},
	ErrNumericalInstability: {
		"nan_values": {
			Arabic:  "المصفوفة تحتوي على قيم NaN أو لا نهائية",
			English: "tensor contains NaN or infinite values",
			Tip:     "تحقق من البيانات المدخلة والمعاملات | Check input data and parameters",
		},
		"convergence_failed": {
			Arabic:  "فشل في التقارب",
			English: "convergence failed",
			Tip:     "جرب تقليل معدل التعلم أو زيادة عدد التكرارات | Try reducing learning rate or increasing iterations",
		},
	},
}

// NewError creates a new ThinkingNetError with English message only (for backward compatibility).
func NewError(errType ErrorType, message string) *ThinkingNetError {
	return &ThinkingNetError{
		Type: errType,
		Message: BilingualMessage{
			Arabic:  message, // Fallback to same message
			English: message,
			Tip:     "",
		},
		Context: make(map[string]any),
	}
}

// NewBilingualError creates a new ThinkingNetError with bilingual messages.
func NewBilingualError(errType ErrorType, arabic, english, tip string) *ThinkingNetError {
	return &ThinkingNetError{
		Type: errType,
		Message: BilingualMessage{
			Arabic:  arabic,
			English: english,
			Tip:     tip,
		},
		Context: make(map[string]any),
	}
}

// NewCommonError creates a new ThinkingNetError using predefined common messages.
func NewCommonError(errType ErrorType, messageKey string) *ThinkingNetError {
	if messages, exists := CommonErrorMessages[errType]; exists {
		if msg, exists := messages[messageKey]; exists {
			return &ThinkingNetError{
				Type:    errType,
				Message: msg,
				Context: make(map[string]any),
			}
		}
	}
	// Fallback to generic message
	return NewError(errType, fmt.Sprintf("error of type %s with key %s", errType.String(), messageKey))
}

// NewErrorWithCause creates a new ThinkingNetError with an underlying cause.
func NewErrorWithCause(errType ErrorType, message string, cause error) *ThinkingNetError {
	return &ThinkingNetError{
		Type: errType,
		Message: BilingualMessage{
			Arabic:  message, // Fallback to same message
			English: message,
			Tip:     "",
		},
		Context: make(map[string]any),
		Cause:   cause,
	}
}

// NewBilingualErrorWithCause creates a new ThinkingNetError with bilingual messages and underlying cause.
func NewBilingualErrorWithCause(errType ErrorType, arabic, english, tip string, cause error) *ThinkingNetError {
	return &ThinkingNetError{
		Type: errType,
		Message: BilingualMessage{
			Arabic:  arabic,
			English: english,
			Tip:     tip,
		},
		Context: make(map[string]any),
		Cause:   cause,
	}
}

// NewCommonErrorWithCause creates a new ThinkingNetError using predefined common messages with underlying cause.
func NewCommonErrorWithCause(errType ErrorType, messageKey string, cause error) *ThinkingNetError {
	if messages, exists := CommonErrorMessages[errType]; exists {
		if msg, exists := messages[messageKey]; exists {
			return &ThinkingNetError{
				Type:    errType,
				Message: msg,
				Context: make(map[string]any),
				Cause:   cause,
			}
		}
	}
	// Fallback to generic message
	return NewErrorWithCause(errType, fmt.Sprintf("error of type %s with key %s", errType.String(), messageKey), cause)
}

// Validation helper functions

// ValidateInput validates input tensor dimensions and properties.
func ValidateInput(X Tensor, expectedShape []int) error {
	if X == nil {
		return NewCommonError(ErrInvalidInput, "nil_tensor")
	}

	rows, cols := X.Dims()
	if len(expectedShape) >= 2 {
		if expectedShape[1] != -1 && cols != expectedShape[1] {
			return NewBilingualError(ErrDimensionMismatch,
				fmt.Sprintf("متوقع %d عمود، تم الحصول على %d", expectedShape[1], cols),
				fmt.Sprintf("expected %d columns, got %d", expectedShape[1], cols),
				"تحقق من شكل البيانات المدخلة | Check input data shape")
		}
	}

	if rows == 0 || cols == 0 {
		return NewCommonError(ErrInvalidInput, "zero_dimensions")
	}

	return nil
}

// ValidateTrainingData validates training data consistency.
func ValidateTrainingData(X, y Tensor) error {
	if X == nil || y == nil {
		return NewBilingualError(ErrInvalidInput,
			"مصفوفات بيانات التدريب لا يمكن أن تكون فارغة",
			"training data tensors cannot be nil",
			"تأكد من تحميل البيانات بشكل صحيح | Make sure to load data properly")
	}

	xRows, _ := X.Dims()
	yRows, _ := y.Dims()

	if xRows != yRows {
		return NewBilingualError(ErrDimensionMismatch,
			fmt.Sprintf("X تحتوي على %d عينة لكن y تحتوي على %d عينة", xRows, yRows),
			fmt.Sprintf("X has %d samples but y has %d samples", xRows, yRows),
			"تأكد من أن X و y لهما نفس عدد العينات | Ensure X and y have the same number of samples")
	}

	if xRows == 0 {
		return NewCommonError(ErrInvalidInput, "empty_data")
	}

	return nil
}

// ValidateDimensions validates that two tensors have compatible dimensions for an operation.
func ValidateDimensions(a, b Tensor, operation string) error {
	if a == nil || b == nil {
		return NewCommonError(ErrInvalidInput, "nil_tensor")
	}

	aRows, aCols := a.Dims()
	bRows, bCols := b.Dims()

	switch operation {
	case "add", "sub", "mul_elem", "div", "equal":
		if aRows != bRows || aCols != bCols {
			return NewBilingualError(ErrDimensionMismatch,
				fmt.Sprintf("المصفوفات يجب أن تحتوي على نفس الأبعاد للعملية %s: (%d,%d) مقابل (%d,%d)",
					operation, aRows, aCols, bRows, bCols),
				fmt.Sprintf("tensors must have same dimensions for %s: (%d,%d) vs (%d,%d)",
					operation, aRows, aCols, bRows, bCols),
				"تأكد من أن المصفوفات لها نفس الشكل | Ensure tensors have the same shape")
		}
	case "mul":
		if aCols != bRows {
			return NewBilingualError(ErrDimensionMismatch,
				fmt.Sprintf("أبعاد غير متوافقة لضرب المصفوفات: (%d,%d) x (%d,%d)",
					aRows, aCols, bRows, bCols),
				fmt.Sprintf("incompatible dimensions for matrix multiplication: (%d,%d) x (%d,%d)",
					aRows, aCols, bRows, bCols),
				"عدد الأعمدة في المصفوفة الأولى يجب أن يساوي عدد الصفوف في الثانية | Number of columns in first matrix must equal rows in second")
		}
	}

	return nil
}

// ValidateRange validates that a value is within a specified range.
func ValidateRange(value float64, min, max float64, name string) error {
	if value < min || value > max {
		return NewBilingualError(ErrInvalidInput,
			fmt.Sprintf("%s يجب أن تكون بين %f و %f، تم الحصول على %f", name, min, max, value),
			fmt.Sprintf("%s must be between %f and %f, got %f", name, min, max, value),
			"تحقق من القيم المدخلة | Check input values")
	}
	return nil
}

// ValidatePositive validates that a value is positive.
func ValidatePositive(value float64, name string) error {
	if value <= 0 {
		return NewBilingualError(ErrInvalidInput,
			fmt.Sprintf("%s يجب أن تكون موجبة، تم الحصول على %f", name, value),
			fmt.Sprintf("%s must be positive, got %f", name, value),
			"استخدم قيمة أكبر من الصفر | Use a value greater than zero")
	}
	return nil
}

// ValidateNonNegative validates that a value is non-negative.
func ValidateNonNegative(value float64, name string) error {
	if value < 0 {
		return NewError(ErrInvalidInput,
			fmt.Sprintf("%s must be non-negative, got %f", name, value))
	}
	return nil
}

// ValidateNotFitted checks if a component has been fitted.
func ValidateNotFitted(fitted bool, componentName string) error {
	if !fitted {
		return NewBilingualError(ErrNotFitted,
			fmt.Sprintf("%s يجب تدريبه قبل الاستخدام", componentName),
			fmt.Sprintf("%s must be fitted before use", componentName),
			"استخدم fit() أو train() أولاً | Use fit() or train() first")
	}
	return nil
}

// ValidateCompiled checks if a model has been compiled.
func ValidateCompiled(compiled bool) error {
	if !compiled {
		return NewCommonError(ErrModelNotCompiled, "model_not_compiled")
	}
	return nil
}

// ValidateTensorFinite validates that a tensor contains only finite values.
func ValidateTensorFinite(tensor Tensor, name string) error {
	if tensor == nil {
		return NewBilingualError(ErrInvalidInput,
			fmt.Sprintf("مصفوفة %s لا يمكن أن تكون فارغة", name),
			fmt.Sprintf("%s tensor cannot be nil", name),
			"تأكد من إنشاء المصفوفة بشكل صحيح | Make sure to create the tensor properly")
	}

	if !tensor.IsFinite() {
		return NewCommonError(ErrNumericalInstability, "nan_values")
	}

	return nil
}

// ValidateScalar validates that a scalar value is finite and within bounds.
func ValidateScalar(value float64, name string) error {
	if math.IsNaN(value) {
		return NewError(ErrInvalidInput, fmt.Sprintf("%s cannot be NaN", name))
	}

	if math.IsInf(value, 0) {
		return NewError(ErrInvalidInput, fmt.Sprintf("%s cannot be infinite", name))
	}

	return nil
}

// ValidateSquareMatrix validates that a tensor is a square matrix.
func ValidateSquareMatrix(tensor Tensor, name string) error {
	if tensor == nil {
		return NewError(ErrInvalidInput, fmt.Sprintf("%s tensor cannot be nil", name))
	}

	if !tensor.IsSquare() {
		rows, cols := tensor.Dims()
		return NewError(ErrDimensionMismatch,
			fmt.Sprintf("%s must be square matrix, got (%d,%d)", name, rows, cols))
	}

	return nil
}

// ValidateVector validates that a tensor is a vector.
func ValidateVector(tensor Tensor, name string) error {
	if tensor == nil {
		return NewError(ErrInvalidInput, fmt.Sprintf("%s tensor cannot be nil", name))
	}

	if !tensor.IsVector() {
		rows, cols := tensor.Dims()
		return NewError(ErrDimensionMismatch,
			fmt.Sprintf("%s must be a vector, got (%d,%d)", name, rows, cols))
	}

	return nil
}

// ValidateNonEmpty validates that a tensor is not empty.
func ValidateNonEmpty(tensor Tensor, name string) error {
	if tensor == nil {
		return NewError(ErrInvalidInput, fmt.Sprintf("%s tensor cannot be nil", name))
	}

	if tensor.IsEmpty() {
		return NewError(ErrInvalidInput, fmt.Sprintf("%s tensor cannot be empty", name))
	}

	return nil
}

// IsThinkingNetError checks if an error is a ThinkingNetError.
func IsThinkingNetError(err error) bool {
	_, ok := err.(*ThinkingNetError)
	return ok
}

// GetErrorType returns the error type if it's a ThinkingNetError.
func GetErrorType(err error) (ErrorType, bool) {
	if tnErr, ok := err.(*ThinkingNetError); ok {
		return tnErr.Type, true
	}
	return ErrInvalidInput, false
}

// ValidateIntRange validates that an integer value is within a specified range.
func ValidateIntRange(value, min, max int, name string) error {
	if value < min || value > max {
		return NewError(ErrInvalidInput,
			fmt.Sprintf("%s must be between %d and %d, got %d", name, min, max, value))
	}
	return nil
}

// ValidatePositiveInt validates that an integer value is positive.
func ValidatePositiveInt(value int, name string) error {
	if value <= 0 {
		return NewError(ErrInvalidInput,
			fmt.Sprintf("%s must be positive, got %d", name, value))
	}
	return nil
}

// ValidateNonNegativeInt validates that an integer value is non-negative.
func ValidateNonNegativeInt(value int, name string) error {
	if value < 0 {
		return NewError(ErrInvalidInput,
			fmt.Sprintf("%s must be non-negative, got %d", name, value))
	}
	return nil
}

// ValidateSliceNotEmpty validates that a slice is not empty.
func ValidateSliceNotEmpty(slice interface{}, name string) error {
	switch s := slice.(type) {
	case []string:
		if len(s) == 0 {
			return NewError(ErrInvalidInput, fmt.Sprintf("%s slice cannot be empty", name))
		}
	case []int:
		if len(s) == 0 {
			return NewError(ErrInvalidInput, fmt.Sprintf("%s slice cannot be empty", name))
		}
	case []float64:
		if len(s) == 0 {
			return NewError(ErrInvalidInput, fmt.Sprintf("%s slice cannot be empty", name))
		}
	case []Tensor:
		if len(s) == 0 {
			return NewError(ErrInvalidInput, fmt.Sprintf("%s slice cannot be empty", name))
		}
	default:
		return NewError(ErrInvalidInput, fmt.Sprintf("unsupported slice type for %s", name))
	}
	return nil
}

// ValidateStringInSet validates that a string value is in a set of allowed values.
func ValidateStringInSet(value string, allowedValues []string, name string) error {
	for _, allowed := range allowedValues {
		if value == allowed {
			return nil
		}
	}
	return NewError(ErrInvalidInput,
		fmt.Sprintf("%s must be one of %v, got '%s'", name, allowedValues, value))
}

// ValidateCompatibleShapes validates that two tensors have compatible shapes for broadcasting.
func ValidateCompatibleShapes(a, b Tensor, operation string) error {
	if a == nil || b == nil {
		return NewError(ErrInvalidInput, "tensors cannot be nil")
	}

	aRows, aCols := a.Dims()
	bRows, bCols := b.Dims()

	// Check if shapes are compatible for broadcasting
	compatible := false

	switch operation {
	case "broadcast_add", "broadcast_sub", "broadcast_mul", "broadcast_div":
		// Broadcasting rules: dimensions are compatible if they are equal or one of them is 1
		compatible = (aRows == bRows || aRows == 1 || bRows == 1) &&
			(aCols == bCols || aCols == 1 || bCols == 1)
	case "matmul":
		// Matrix multiplication: inner dimensions must match
		compatible = aCols == bRows
	default:
		return ValidateDimensions(a, b, operation)
	}

	if !compatible {
		return NewError(ErrDimensionMismatch,
			fmt.Sprintf("incompatible shapes for %s: (%d,%d) and (%d,%d)",
				operation, aRows, aCols, bRows, bCols))
	}

	return nil
}

// ValidateModelState validates the state of a model before operations.
func ValidateModelState(compiled, fitted bool, operation string) error {
	switch operation {
	case "fit", "train":
		if !compiled {
			return NewError(ErrModelNotCompiled, "model must be compiled before training")
		}
	case "predict", "evaluate":
		if !compiled {
			return NewError(ErrModelNotCompiled, "model must be compiled before prediction")
		}
		if !fitted {
			return NewError(ErrNotFitted, "model must be fitted before prediction")
		}
	case "compile":
		// No prerequisites for compilation
	default:
		return NewError(ErrInvalidInput, fmt.Sprintf("unknown operation: %s", operation))
	}
	return nil
}

// ValidateOptimizationConfig validates optimizer configuration parameters.
func ValidateOptimizationConfig(config map[string]interface{}) error {
	if lr, exists := config["learning_rate"]; exists {
		if lrFloat, ok := lr.(float64); ok {
			if err := ValidatePositive(lrFloat, "learning_rate"); err != nil {
				return err
			}
		} else {
			return NewError(ErrInvalidInput, "learning_rate must be a float64")
		}
	}

	if epochs, exists := config["epochs"]; exists {
		if epochsInt, ok := epochs.(int); ok {
			if err := ValidatePositiveInt(epochsInt, "epochs"); err != nil {
				return err
			}
		} else {
			return NewError(ErrInvalidInput, "epochs must be an int")
		}
	}

	if batchSize, exists := config["batch_size"]; exists {
		if batchSizeInt, ok := batchSize.(int); ok {
			if err := ValidatePositiveInt(batchSizeInt, "batch_size"); err != nil {
				return err
			}
		} else {
			return NewError(ErrInvalidInput, "batch_size must be an int")
		}
	}

	if beta1, exists := config["beta1"]; exists {
		if beta1Float, ok := beta1.(float64); ok {
			if err := ValidateRange(beta1Float, 0.0, 1.0, "beta1"); err != nil {
				return err
			}
		} else {
			return NewError(ErrInvalidInput, "beta1 must be a float64")
		}
	}

	if beta2, exists := config["beta2"]; exists {
		if beta2Float, ok := beta2.(float64); ok {
			if err := ValidateRange(beta2Float, 0.0, 1.0, "beta2"); err != nil {
				return err
			}
		} else {
			return NewError(ErrInvalidInput, "beta2 must be a float64")
		}
	}

	return nil
}

// RecoverFromPanic recovers from panics and converts them to ThinkingNet errors.
// This function should be called from within a defer function.
func RecoverFromPanic(operation string) error {
	r := recover()
	if r != nil {
		if err, ok := r.(error); ok {
			if tnErr, ok := err.(*ThinkingNetError); ok {
				return tnErr
			}
			return NewErrorWithCause(ErrNumericalInstability,
				fmt.Sprintf("operation %s panicked", operation), err)
		}
		return NewError(ErrNumericalInstability,
			fmt.Sprintf("operation %s panicked: %v", operation, r))
	}
	return nil
}

// SafeExecute executes a function with panic recovery.
func SafeExecute(operation string, fn func() error) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if recoverErr, ok := r.(error); ok {
				if tnErr, ok := recoverErr.(*ThinkingNetError); ok {
					err = tnErr
				} else {
					err = NewErrorWithCause(ErrNumericalInstability,
						fmt.Sprintf("operation %s panicked", operation), recoverErr)
				}
			} else {
				err = NewError(ErrNumericalInstability,
					fmt.Sprintf("operation %s panicked: %v", operation, r))
			}
		}
	}()

	return fn()
}

// ValidateMemoryUsage validates that tensor operations won't exceed memory limits.
func ValidateMemoryUsage(tensors []Tensor, operation string) error {
	const maxElements = 1e8 // 100M elements as a reasonable limit

	var totalElements int64
	for i, tensor := range tensors {
		if tensor == nil {
			return NewError(ErrInvalidInput, fmt.Sprintf("tensor %d is nil", i))
		}

		rows, cols := tensor.Dims()
		elements := int64(rows) * int64(cols)
		totalElements += elements

		if totalElements > maxElements {
			return NewError(ErrMemory,
				fmt.Sprintf("operation %s would use too much memory: %d elements", operation, totalElements)).
				WithContext("max_elements", maxElements).
				WithContext("requested_elements", totalElements)
		}
	}

	return nil
}
