package core

import (
	"fmt"
	"strings"
	"testing"
)

func TestBilingualErrorMessages(t *testing.T) {
	tests := []struct {
		name            string
		errorFunc       func() error
		expectedArabic  string
		expectedEnglish string
		expectedTip     string
	}{
		{
			name: "nil tensor error",
			errorFunc: func() error {
				return NewCommonError(ErrInvalidInput, "nil_tensor")
			},
			expectedArabic:  "المصفوفة لا يمكن أن تكون فارغة",
			expectedEnglish: "tensor cannot be nil",
			expectedTip:     "تأكد من إنشاء المصفوفة بشكل صحيح | Make sure to create the tensor properly",
		},
		{
			name: "zero dimensions error",
			errorFunc: func() error {
				return NewCommonError(ErrInvalidInput, "zero_dimensions")
			},
			expectedArabic:  "المصفوفة لا يمكن أن تحتوي على أبعاد صفرية",
			expectedEnglish: "tensor cannot have zero dimensions",
			expectedTip:     "تحقق من أن البيانات تحتوي على قيم | Check that data contains values",
		},
		{
			name: "model not compiled error",
			errorFunc: func() error {
				return NewCommonError(ErrModelNotCompiled, "model_not_compiled")
			},
			expectedArabic:  "النموذج لم يتم تجميعه بعد",
			expectedEnglish: "model has not been compiled yet",
			expectedTip:     "استخدم Compile() مع optimizer و loss function | Use Compile() with optimizer and loss function",
		},
		{
			name: "component not fitted error",
			errorFunc: func() error {
				return NewCommonError(ErrNotFitted, "component_not_fitted")
			},
			expectedArabic:  "المكون لم يتم تدريبه بعد",
			expectedEnglish: "component has not been fitted yet",
			expectedTip:     "استخدم fit() أو train() أولاً | Use fit() or train() first",
		},
		{
			name: "NaN values error",
			errorFunc: func() error {
				return NewCommonError(ErrNumericalInstability, "nan_values")
			},
			expectedArabic:  "المصفوفة تحتوي على قيم NaN أو لا نهائية",
			expectedEnglish: "tensor contains NaN or infinite values",
			expectedTip:     "تحقق من البيانات المدخلة والمعاملات | Check input data and parameters",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.errorFunc()
			if err == nil {
				t.Fatal("expected error, got nil")
			}

			tnErr, ok := err.(*ThinkingNetError)
			if !ok {
				t.Fatalf("expected ThinkingNetError, got %T", err)
			}

			// Test Arabic message
			if tnErr.Message.Arabic != tt.expectedArabic {
				t.Errorf("Arabic message mismatch:\nexpected: %s\ngot: %s", tt.expectedArabic, tnErr.Message.Arabic)
			}

			// Test English message
			if tnErr.Message.English != tt.expectedEnglish {
				t.Errorf("English message mismatch:\nexpected: %s\ngot: %s", tt.expectedEnglish, tnErr.Message.English)
			}

			// Test tip
			if tnErr.Message.Tip != tt.expectedTip {
				t.Errorf("Tip mismatch:\nexpected: %s\ngot: %s", tt.expectedTip, tnErr.Message.Tip)
			}

			// Test Error() method includes both languages
			errorStr := tnErr.Error()
			if !strings.Contains(errorStr, tt.expectedArabic) {
				t.Errorf("Error() should contain Arabic message: %s", errorStr)
			}
			if !strings.Contains(errorStr, tt.expectedEnglish) {
				t.Errorf("Error() should contain English message: %s", errorStr)
			}

			// Test ErrorArabic() method
			arabicStr := tnErr.ErrorArabic()
			if !strings.Contains(arabicStr, tt.expectedArabic) {
				t.Errorf("ErrorArabic() should contain Arabic message: %s", arabicStr)
			}

			// Test ErrorEnglish() method
			englishStr := tnErr.ErrorEnglish()
			if !strings.Contains(englishStr, tt.expectedEnglish) {
				t.Errorf("ErrorEnglish() should contain English message: %s", englishStr)
			}

			// Test GetTip() method
			tip := tnErr.GetTip()
			if tip != tt.expectedTip {
				t.Errorf("GetTip() mismatch:\nexpected: %s\ngot: %s", tt.expectedTip, tip)
			}
		})
	}
}

func TestNewBilingualError(t *testing.T) {
	arabic := "رسالة خطأ باللغة العربية"
	english := "Error message in English"
	tip := "نصيحة مفيدة | Helpful tip"

	err := NewBilingualError(ErrInvalidInput, arabic, english, tip)

	if err.Type != ErrInvalidInput {
		t.Errorf("expected error type %v, got %v", ErrInvalidInput, err.Type)
	}

	if err.Message.Arabic != arabic {
		t.Errorf("expected Arabic message %s, got %s", arabic, err.Message.Arabic)
	}

	if err.Message.English != english {
		t.Errorf("expected English message %s, got %s", english, err.Message.English)
	}

	if err.Message.Tip != tip {
		t.Errorf("expected tip %s, got %s", tip, err.Message.Tip)
	}
}

func TestValidationFunctionsWithBilingualErrors(t *testing.T) {
	// Test ValidateRange with bilingual error
	err := ValidateRange(-1.0, 0.0, 1.0, "learning_rate")
	if err == nil {
		t.Fatal("expected error for invalid range")
	}

	tnErr, ok := err.(*ThinkingNetError)
	if !ok {
		t.Fatalf("expected ThinkingNetError, got %T", err)
	}

	// Check that both Arabic and English messages are present
	if !strings.Contains(tnErr.Message.Arabic, "يجب أن تكون بين") {
		t.Error("Arabic message should contain range validation text")
	}
	if !strings.Contains(tnErr.Message.English, "must be between") {
		t.Error("English message should contain range validation text")
	}

	// Test ValidatePositive with bilingual error
	err = ValidatePositive(-0.5, "learning_rate")
	if err == nil {
		t.Fatal("expected error for negative value")
	}

	tnErr, ok = err.(*ThinkingNetError)
	if !ok {
		t.Fatalf("expected ThinkingNetError, got %T", err)
	}

	if !strings.Contains(tnErr.Message.Arabic, "يجب أن تكون موجبة") {
		t.Error("Arabic message should contain positive validation text")
	}
	if !strings.Contains(tnErr.Message.English, "must be positive") {
		t.Error("English message should contain positive validation text")
	}
}

func TestErrorWithCause(t *testing.T) {
	originalErr := NewError(ErrInvalidInput, "original error")

	err := NewBilingualErrorWithCause(
		ErrNumericalInstability,
		"خطأ عددي مع سبب",
		"numerical error with cause",
		"نصيحة للحل | Solution tip",
		originalErr,
	)

	if err.Cause != originalErr {
		t.Error("cause should be preserved")
	}

	errorStr := err.Error()
	if !strings.Contains(errorStr, "numerical error with cause") {
		t.Error("error string should contain main message")
	}
	if !strings.Contains(errorStr, "original error") {
		t.Error("error string should contain cause")
	}
}

func TestCommonErrorWithCause(t *testing.T) {
	originalErr := NewError(ErrInvalidInput, "original error")

	err := NewCommonErrorWithCause(ErrInvalidInput, "nil_tensor", originalErr)

	if err.Cause != originalErr {
		t.Error("cause should be preserved")
	}

	if err.Message.Arabic != "المصفوفة لا يمكن أن تكون فارغة" {
		t.Error("should use common Arabic message")
	}
	if err.Message.English != "tensor cannot be nil" {
		t.Error("should use common English message")
	}
}

func TestBackwardCompatibility(t *testing.T) {
	// Test that old NewError still works
	err := NewError(ErrInvalidInput, "test message")

	if err.Message.Arabic != "test message" {
		t.Error("backward compatibility: Arabic should fallback to message")
	}
	if err.Message.English != "test message" {
		t.Error("backward compatibility: English should fallback to message")
	}
	if err.Message.Tip != "" {
		t.Error("backward compatibility: Tip should be empty")
	}
}

func TestErrorTypeHelpers(t *testing.T) {
	err := NewCommonError(ErrModelNotCompiled, "model_not_compiled")

	// Test IsThinkingNetError
	if !IsThinkingNetError(err) {
		t.Error("IsThinkingNetError should return true for ThinkingNetError")
	}

	// Test GetErrorType
	errType, ok := GetErrorType(err)
	if !ok {
		t.Error("GetErrorType should return true for ThinkingNetError")
	}
	if errType != ErrModelNotCompiled {
		t.Errorf("expected error type %v, got %v", ErrModelNotCompiled, errType)
	}

	// Test with non-ThinkingNetError
	regularErr := fmt.Errorf("regular error")
	if IsThinkingNetError(regularErr) {
		t.Error("IsThinkingNetError should return false for regular error")
	}

	_, ok = GetErrorType(regularErr)
	if ok {
		t.Error("GetErrorType should return false for regular error")
	}
}
