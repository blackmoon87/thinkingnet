package utils

import (
	"math"
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

func TestClamp(t *testing.T) {
	tests := []struct {
		x, min, max, expected float64
	}{
		{5.0, 0.0, 10.0, 5.0},   // within range
		{-1.0, 0.0, 10.0, 0.0},  // below min
		{15.0, 0.0, 10.0, 10.0}, // above max
		{0.0, 0.0, 10.0, 0.0},   // at min
		{10.0, 0.0, 10.0, 10.0}, // at max
	}

	for _, test := range tests {
		result := Clamp(test.x, test.min, test.max)
		if result != test.expected {
			t.Errorf("Clamp(%f, %f, %f) = %f, expected %f",
				test.x, test.min, test.max, result, test.expected)
		}
	}
}

func TestClampToConfig(t *testing.T) {
	x := 1e10 // Very large number
	result := ClampToConfig(x)

	// Should be clamped to config max
	if result > core.GetMaxFloat() {
		t.Errorf("ClampToConfig should clamp to max config value")
	}
}

func TestSafeLog(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{1.0, 0.0},
		{math.E, 1.0},
		{0.0, math.Log(core.GetEpsilon())},  // Should use epsilon
		{-1.0, math.Log(core.GetEpsilon())}, // Should use epsilon for negative
	}

	for _, test := range tests {
		result := SafeLog(test.input)
		if math.Abs(result-test.expected) > 1e-10 {
			t.Errorf("SafeLog(%f) = %f, expected %f", test.input, result, test.expected)
		}
	}
}

func TestSafeExp(t *testing.T) {
	// Test normal values
	result := SafeExp(1.0)
	expected := math.E
	if math.Abs(result-expected) > 1e-10 {
		t.Errorf("SafeExp(1.0) = %f, expected %f", result, expected)
	}

	// Test overflow protection
	result = SafeExp(1000.0) // Should be clamped
	if math.IsInf(result, 0) {
		t.Error("SafeExp should prevent overflow")
	}

	// Test underflow protection
	result = SafeExp(-1000.0) // Should be clamped
	if result == 0.0 {
		t.Error("SafeExp should prevent complete underflow")
	}
}

func TestSafeDivide(t *testing.T) {
	tests := []struct {
		a, b     float64
		expected float64
	}{
		{10.0, 2.0, 5.0},
		{10.0, 0.0, 10.0 / core.GetEpsilon()}, // Positive zero
	}

	for _, test := range tests {
		result := SafeDivide(test.a, test.b)
		if math.Abs(result-test.expected) > 1e-6 {
			t.Errorf("SafeDivide(%f, %f) = %f, expected %f",
				test.a, test.b, result, test.expected)
		}
	}
}

func TestSigmoid(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{0.0, 0.5},
		{1000.0, 1.0},  // Should be clamped and approach 1
		{-1000.0, 0.0}, // Should be clamped and approach 0
	}

	for _, test := range tests {
		result := Sigmoid(test.input)
		if math.Abs(result-test.expected) > 1e-6 {
			t.Errorf("Sigmoid(%f) = %f, expected %f", test.input, result, test.expected)
		}
	}
}

func TestSoftmax(t *testing.T) {
	// Test empty slice
	empty := Softmax([]float64{})
	if len(empty) != 0 {
		t.Error("Softmax of empty slice should return empty slice")
	}

	// Test normal case
	input := []float64{1.0, 2.0, 3.0}
	result := Softmax(input)

	// Check that probabilities sum to 1
	sum := Sum(result)
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("Softmax probabilities should sum to 1, got %f", sum)
	}

	// Check that all values are positive
	for i, val := range result {
		if val <= 0 {
			t.Errorf("Softmax[%d] = %f should be positive", i, val)
		}
	}

	// Test numerical stability with large values
	large := []float64{1000.0, 1001.0, 1002.0}
	resultLarge := Softmax(large)
	sumLarge := Sum(resultLarge)
	if math.Abs(sumLarge-1.0) > 1e-10 {
		t.Errorf("Softmax with large values should still sum to 1, got %f", sumLarge)
	}
}

func TestLogSumExp(t *testing.T) {
	// Test empty slice
	empty := LogSumExp([]float64{})
	if !math.IsInf(empty, -1) {
		t.Error("LogSumExp of empty slice should return -Inf")
	}

	// Test normal case
	input := []float64{1.0, 2.0, 3.0}
	result := LogSumExp(input)

	// Verify against direct computation (for small values)
	expected := math.Log(math.Exp(1.0) + math.Exp(2.0) + math.Exp(3.0))
	if math.Abs(result-expected) > 1e-10 {
		t.Errorf("LogSumExp([1,2,3]) = %f, expected %f", result, expected)
	}

	// Test numerical stability
	large := []float64{1000.0, 1001.0, 1002.0}
	resultLarge := LogSumExp(large)
	if math.IsInf(resultLarge, 0) || math.IsNaN(resultLarge) {
		t.Error("LogSumExp should handle large values without overflow")
	}
}

func TestStatistics(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}

	// Test Mean
	mean := Mean(data)
	expected := 3.0
	if mean != expected {
		t.Errorf("Mean = %f, expected %f", mean, expected)
	}

	// Test empty slice
	emptyMean := Mean([]float64{})
	if emptyMean != 0 {
		t.Errorf("Mean of empty slice should be 0, got %f", emptyMean)
	}

	// Test Variance
	variance := Variance(data)
	expectedVar := 2.5 // Sample variance of [1,2,3,4,5]
	if math.Abs(variance-expectedVar) > 1e-10 {
		t.Errorf("Variance = %f, expected %f", variance, expectedVar)
	}

	// Test variance with single element
	singleVar := Variance([]float64{5.0})
	if singleVar != 0 {
		t.Errorf("Variance of single element should be 0, got %f", singleVar)
	}

	// Test StandardDeviation
	std := StandardDeviation(data)
	expectedStd := math.Sqrt(expectedVar)
	if math.Abs(std-expectedStd) > 1e-10 {
		t.Errorf("StandardDeviation = %f, expected %f", std, expectedStd)
	}
}

func TestMinMax(t *testing.T) {
	data := []float64{3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0}

	// Test Min
	min := Min(data)
	if min != 1.0 {
		t.Errorf("Min = %f, expected 1.0", min)
	}

	// Test Max
	max := Max(data)
	if max != 9.0 {
		t.Errorf("Max = %f, expected 9.0", max)
	}

	// Test empty slices
	emptyMin := Min([]float64{})
	if emptyMin != 0 {
		t.Errorf("Min of empty slice should be 0, got %f", emptyMin)
	}

	emptyMax := Max([]float64{})
	if emptyMax != 0 {
		t.Errorf("Max of empty slice should be 0, got %f", emptyMax)
	}
}

func TestArgMinMax(t *testing.T) {
	data := []float64{3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0}

	// Test ArgMin
	argMin := ArgMin(data)
	if argMin != 1 { // First occurrence of minimum
		t.Errorf("ArgMin = %d, expected 1", argMin)
	}

	// Test ArgMax
	argMax := ArgMax(data)
	if argMax != 5 {
		t.Errorf("ArgMax = %d, expected 5", argMax)
	}

	// Test empty slices
	emptyArgMin := ArgMin([]float64{})
	if emptyArgMin != -1 {
		t.Errorf("ArgMin of empty slice should be -1, got %d", emptyArgMin)
	}

	emptyArgMax := ArgMax([]float64{})
	if emptyArgMax != -1 {
		t.Errorf("ArgMax of empty slice should be -1, got %d", emptyArgMax)
	}
}

func TestSum(t *testing.T) {
	data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	sum := Sum(data)
	expected := 15.0

	if sum != expected {
		t.Errorf("Sum = %f, expected %f", sum, expected)
	}

	// Test empty slice
	emptySum := Sum([]float64{})
	if emptySum != 0 {
		t.Errorf("Sum of empty slice should be 0, got %f", emptySum)
	}
}

func TestDot(t *testing.T) {
	a := []float64{1.0, 2.0, 3.0}
	b := []float64{4.0, 5.0, 6.0}

	dot := Dot(a, b)
	expected := 32.0 // 1*4 + 2*5 + 3*6 = 32

	if dot != expected {
		t.Errorf("Dot = %f, expected %f", dot, expected)
	}

	// Test dimension mismatch panic
	defer func() {
		if r := recover(); r == nil {
			t.Error("Dot should panic on dimension mismatch")
		}
	}()
	Dot(a, []float64{1.0, 2.0}) // Different lengths
}

func TestNorm(t *testing.T) {
	data := []float64{3.0, 4.0} // 3-4-5 triangle
	norm := Norm(data)
	expected := 5.0

	if norm != expected {
		t.Errorf("Norm = %f, expected %f", norm, expected)
	}
}

func TestNormalize(t *testing.T) {
	data := []float64{3.0, 4.0}
	normalized := Normalize(data)

	// Check that norm is 1
	norm := Norm(normalized)
	if math.Abs(norm-1.0) > 1e-10 {
		t.Errorf("Normalized vector should have norm 1, got %f", norm)
	}

	// Test zero vector
	zero := []float64{0.0, 0.0}
	normalizedZero := Normalize(zero)
	if len(normalizedZero) != 2 || normalizedZero[0] != 0 || normalizedZero[1] != 0 {
		t.Error("Normalizing zero vector should return zero vector")
	}
}

func TestRandomGeneration(t *testing.T) {
	// Test SetSeed (just ensure no panic)
	SetSeed(42)
	SetSeedFromTime()

	// Test RandNormal
	normal := RandNormal(100, 0.0, 1.0)
	if len(normal) != 100 {
		t.Errorf("RandNormal should return 100 values, got %d", len(normal))
	}

	// Test RandUniform
	uniform := RandUniform(100, 0.0, 1.0)
	if len(uniform) != 100 {
		t.Errorf("RandUniform should return 100 values, got %d", len(uniform))
	}

	// Check that uniform values are in range
	for i, val := range uniform {
		if val < 0.0 || val > 1.0 {
			t.Errorf("RandUniform[%d] = %f should be in [0,1]", i, val)
		}
	}

	// Test RandInt
	for i := 0; i < 10; i++ {
		randInt := RandInt(0, 10)
		if randInt < 0 || randInt >= 10 {
			t.Errorf("RandInt should be in [0,10), got %d", randInt)
		}
	}
}

func TestShuffle(t *testing.T) {
	original := []int{1, 2, 3, 4, 5}
	toShuffle := make([]int, len(original))
	copy(toShuffle, original)

	Shuffle(toShuffle)

	// Check that all elements are still present
	if len(toShuffle) != len(original) {
		t.Error("Shuffle should preserve length")
	}

	// Check that it's a permutation (all elements present)
	counts := make(map[int]int)
	for _, val := range toShuffle {
		counts[val]++
	}

	for _, val := range original {
		if counts[val] != 1 {
			t.Errorf("Shuffle should preserve all elements, missing or duplicate: %d", val)
		}
	}
}

func TestChoice(t *testing.T) {
	slice := []int{1, 2, 3, 4, 5}

	// Test normal choice
	chosen := Choice(slice, 3)
	if len(chosen) != 3 {
		t.Errorf("Choice should return 3 elements, got %d", len(chosen))
	}

	// Test choice larger than slice
	chosenLarge := Choice(slice, 10)
	if len(chosenLarge) != len(slice) {
		t.Errorf("Choice should return at most slice length, got %d", len(chosenLarge))
	}

	// Check that chosen elements are from original slice
	originalSet := make(map[int]bool)
	for _, val := range slice {
		originalSet[val] = true
	}

	for _, val := range chosen {
		if !originalSet[val] {
			t.Errorf("Chosen element %d not in original slice", val)
		}
	}
}

func TestLinspace(t *testing.T) {
	// Test normal case
	result := Linspace(0.0, 10.0, 11)
	if len(result) != 11 {
		t.Errorf("Linspace should return 11 values, got %d", len(result))
	}

	if result[0] != 0.0 || result[10] != 10.0 {
		t.Errorf("Linspace endpoints incorrect: got [%f, %f], expected [0, 10]",
			result[0], result[10])
	}

	// Test single point
	single := Linspace(5.0, 10.0, 1)
	if len(single) != 1 || single[0] != 5.0 {
		t.Errorf("Linspace with n=1 should return [start], got %v", single)
	}

	// Test zero points
	zero := Linspace(0.0, 10.0, 0)
	if len(zero) != 0 {
		t.Errorf("Linspace with n=0 should return empty slice, got %v", zero)
	}

	// Test negative points
	negative := Linspace(0.0, 10.0, -1)
	if len(negative) != 0 {
		t.Errorf("Linspace with negative n should return empty slice, got %v", negative)
	}
}

func TestArange(t *testing.T) {
	// Test positive step
	result := Arange(0.0, 5.0, 1.0)
	expected := []float64{0.0, 1.0, 2.0, 3.0, 4.0}

	if len(result) != len(expected) {
		t.Errorf("Arange length mismatch: got %d, expected %d", len(result), len(expected))
	}

	for i, val := range result {
		if math.Abs(val-expected[i]) > 1e-10 {
			t.Errorf("Arange[%d] = %f, expected %f", i, val, expected[i])
		}
	}

	// Test negative step
	resultNeg := Arange(5.0, 0.0, -1.0)
	expectedNeg := []float64{5.0, 4.0, 3.0, 2.0, 1.0}

	if len(resultNeg) != len(expectedNeg) {
		t.Errorf("Arange negative step length mismatch: got %d, expected %d",
			len(resultNeg), len(expectedNeg))
	}

	// Test zero step panic
	defer func() {
		if r := recover(); r == nil {
			t.Error("Arange should panic on zero step")
		}
	}()
	Arange(0.0, 5.0, 0.0)
}

func TestFiniteChecks(t *testing.T) {
	// Test IsNaN
	if !IsNaN(math.NaN()) {
		t.Error("IsNaN should return true for NaN")
	}

	if IsNaN(1.0) {
		t.Error("IsNaN should return false for normal number")
	}

	// Test IsInf
	if !IsInf(math.Inf(1)) {
		t.Error("IsInf should return true for +Inf")
	}

	if !IsInf(math.Inf(-1)) {
		t.Error("IsInf should return true for -Inf")
	}

	if IsInf(1.0) {
		t.Error("IsInf should return false for normal number")
	}

	// Test IsFinite
	if !IsFinite(1.0) {
		t.Error("IsFinite should return true for normal number")
	}

	if IsFinite(math.NaN()) {
		t.Error("IsFinite should return false for NaN")
	}

	if IsFinite(math.Inf(1)) {
		t.Error("IsFinite should return false for Inf")
	}
}

func TestRound(t *testing.T) {
	tests := []struct {
		input    float64
		places   int
		expected float64
	}{
		{3.14159, 2, 3.14},
		{3.14159, 4, 3.1416},
		{123.456, 0, 123.0},
		{123.456, -1, 120.0}, // Negative places should work
	}

	for _, test := range tests {
		result := Round(test.input, test.places)
		if math.Abs(result-test.expected) > 1e-10 {
			t.Errorf("Round(%f, %d) = %f, expected %f",
				test.input, test.places, result, test.expected)
		}
	}
}

// Benchmark tests
func BenchmarkSoftmax(b *testing.B) {
	data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Softmax(data)
	}
}

func BenchmarkLogSumExp(b *testing.B) {
	data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		LogSumExp(data)
	}
}

func BenchmarkStatistics(b *testing.B) {
	data := make([]float64, 1000)
	for i := range data {
		data[i] = float64(i)
	}

	b.Run("Mean", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			Mean(data)
		}
	})

	b.Run("Variance", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			Variance(data)
		}
	})

	b.Run("StandardDeviation", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			StandardDeviation(data)
		}
	})
}

func BenchmarkDot(b *testing.B) {
	a := make([]float64, 1000)
	c := make([]float64, 1000)
	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Dot(a, c)
	}
}
