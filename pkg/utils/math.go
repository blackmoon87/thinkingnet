// Package utils provides common utility functions for the ThinkingNet library.
package utils

import (
	"math"
	"math/rand"
	"time"

	"github.com/blackmoon87/thinkingnet/pkg/core"
)

// Clamp constrains a value between min and max.
func Clamp(x, min, max float64) float64 {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}

// ClampToConfig constrains a value using global config limits.
func ClampToConfig(x float64) float64 {
	return Clamp(x, core.GetMinFloat(), core.GetMaxFloat())
}

// SafeLog computes log with numerical stability.
func SafeLog(x float64) float64 {
	return math.Log(math.Max(x, core.GetEpsilon()))
}

// SafeExp computes exp with overflow protection.
func SafeExp(x float64) float64 {
	x = Clamp(x, -250, 250) // Prevent overflow/underflow
	return math.Exp(x)
}

// SafeDivide performs division with zero protection.
func SafeDivide(a, b float64) float64 {
	if math.Abs(b) < core.GetEpsilon() {
		if b >= 0 {
			return a / core.GetEpsilon()
		}
		return a / (-core.GetEpsilon())
	}
	return a / b
}

// Sigmoid computes the sigmoid function with numerical stability.
func Sigmoid(x float64) float64 {
	x = Clamp(x, -250, 250)
	return 1.0 / (1.0 + math.Exp(-x))
}

// Softmax computes softmax over a slice with numerical stability.
func Softmax(x []float64) []float64 {
	if len(x) == 0 {
		return x
	}

	// Find max for numerical stability
	max := x[0]
	for _, val := range x[1:] {
		if val > max {
			max = val
		}
	}

	// Compute exp and sum
	result := make([]float64, len(x))
	var sum float64
	for i, val := range x {
		exp := math.Exp(val - max)
		result[i] = exp
		sum += exp
	}

	// Normalize
	for i := range result {
		result[i] /= sum
	}

	return result
}

// LogSumExp computes log(sum(exp(x))) with numerical stability.
func LogSumExp(x []float64) float64 {
	if len(x) == 0 {
		return math.Inf(-1)
	}

	// Find max
	max := x[0]
	for _, val := range x[1:] {
		if val > max {
			max = val
		}
	}

	// Compute sum of exp(x - max)
	var sum float64
	for _, val := range x {
		sum += math.Exp(val - max)
	}

	return max + math.Log(sum)
}

// Mean computes the mean of a slice.
func Mean(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}

	var sum float64
	for _, val := range x {
		sum += val
	}
	return sum / float64(len(x))
}

// Variance computes the variance of a slice.
func Variance(x []float64) float64 {
	if len(x) <= 1 {
		return 0
	}

	mean := Mean(x)
	var sum float64
	for _, val := range x {
		diff := val - mean
		sum += diff * diff
	}
	return sum / float64(len(x)-1)
}

// StandardDeviation computes the standard deviation of a slice.
func StandardDeviation(x []float64) float64 {
	return math.Sqrt(Variance(x))
}

// Min returns the minimum value in a slice.
func Min(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}

	min := x[0]
	for _, val := range x[1:] {
		if val < min {
			min = val
		}
	}
	return min
}

// Max returns the maximum value in a slice.
func Max(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}

	max := x[0]
	for _, val := range x[1:] {
		if val > max {
			max = val
		}
	}
	return max
}

// ArgMax returns the index of the maximum value.
func ArgMax(x []float64) int {
	if len(x) == 0 {
		return -1
	}

	maxIdx := 0
	maxVal := x[0]
	for i, val := range x[1:] {
		if val > maxVal {
			maxVal = val
			maxIdx = i + 1
		}
	}
	return maxIdx
}

// ArgMin returns the index of the minimum value.
func ArgMin(x []float64) int {
	if len(x) == 0 {
		return -1
	}

	minIdx := 0
	minVal := x[0]
	for i, val := range x[1:] {
		if val < minVal {
			minVal = val
			minIdx = i + 1
		}
	}
	return minIdx
}

// Sum computes the sum of a slice.
func Sum(x []float64) float64 {
	var sum float64
	for _, val := range x {
		sum += val
	}
	return sum
}

// Dot computes the dot product of two slices.
func Dot(a, b []float64) float64 {
	if len(a) != len(b) {
		panic(core.NewError(core.ErrDimensionMismatch, "slices must have same length for dot product"))
	}

	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// Norm computes the L2 norm of a slice.
func Norm(x []float64) float64 {
	return math.Sqrt(Dot(x, x))
}

// Normalize normalizes a slice to unit length.
func Normalize(x []float64) []float64 {
	norm := Norm(x)
	if norm < core.GetEpsilon() {
		return x
	}

	result := make([]float64, len(x))
	for i, val := range x {
		result[i] = val / norm
	}
	return result
}

// Random number generation utilities

// SetSeed sets the random seed.
func SetSeed(seed int64) {
	rand.Seed(seed)
}

// SetSeedFromTime sets the random seed from current time.
func SetSeedFromTime() {
	rand.Seed(time.Now().UnixNano())
}

// RandNormal generates random numbers from normal distribution.
func RandNormal(n int, mean, std float64) []float64 {
	result := make([]float64, n)
	for i := range result {
		result[i] = rand.NormFloat64()*std + mean
	}
	return result
}

// RandUniform generates random numbers from uniform distribution.
func RandUniform(n int, min, max float64) []float64 {
	result := make([]float64, n)
	for i := range result {
		result[i] = rand.Float64()*(max-min) + min
	}
	return result
}

// RandInt generates random integers in range [min, max).
func RandInt(min, max int) int {
	return rand.Intn(max-min) + min
}

// Shuffle randomly shuffles a slice of integers.
func Shuffle(slice []int) {
	rand.Shuffle(len(slice), func(i, j int) {
		slice[i], slice[j] = slice[j], slice[i]
	})
}

// Choice randomly selects n elements from a slice without replacement.
func Choice(slice []int, n int) []int {
	if n > len(slice) {
		n = len(slice)
	}

	// Create a copy to avoid modifying original
	temp := make([]int, len(slice))
	copy(temp, slice)

	Shuffle(temp)
	return temp[:n]
}

// Linspace generates n evenly spaced numbers between start and stop.
func Linspace(start, stop float64, n int) []float64 {
	if n <= 0 {
		return []float64{}
	}
	if n == 1 {
		return []float64{start}
	}

	result := make([]float64, n)
	step := (stop - start) / float64(n-1)

	for i := 0; i < n; i++ {
		result[i] = start + float64(i)*step
	}

	return result
}

// Arange generates numbers in a range with a step.
func Arange(start, stop, step float64) []float64 {
	if step == 0 {
		panic(core.NewError(core.ErrInvalidInput, "step cannot be zero"))
	}

	var result []float64
	if step > 0 {
		for x := start; x < stop; x += step {
			result = append(result, x)
		}
	} else {
		for x := start; x > stop; x += step {
			result = append(result, x)
		}
	}

	return result
}

// IsNaN checks if a value is NaN.
func IsNaN(x float64) bool {
	return math.IsNaN(x)
}

// IsInf checks if a value is infinite.
func IsInf(x float64) bool {
	return math.IsInf(x, 0)
}

// IsFinite checks if a value is finite (not NaN or Inf).
func IsFinite(x float64) bool {
	return !IsNaN(x) && !IsInf(x)
}

// Round rounds a value to n decimal places.
func Round(x float64, n int) float64 {
	pow := math.Pow(10, float64(n))
	return math.Round(x*pow) / pow
}
