package models

import (
	"testing"

	"github.com/blackmoon87/thinkingnet/pkg/core"
	"gonum.org/v1/gonum/mat"
)

// MockTensor implements core.Tensor for testing
type MockTensor struct {
	rows, cols int
	data       [][]float64
	name       string
}

func NewMockTensor(rows, cols int) *MockTensor {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return &MockTensor{rows: rows, cols: cols, data: data}
}

func (t *MockTensor) Dims() (int, int) { return t.rows, t.cols }
func (t *MockTensor) At(i, j int) float64 {
	if i >= 0 && i < t.rows && j >= 0 && j < t.cols {
		return t.data[i][j]
	}
	return 0.0
}
func (t *MockTensor) Set(i, j int, v float64) {
	if i >= 0 && i < t.rows && j >= 0 && j < t.cols {
		t.data[i][j] = v
	}
}
func (t *MockTensor) Copy() core.Tensor {
	copy := NewMockTensor(t.rows, t.cols)
	for i := 0; i < t.rows; i++ {
		for j := 0; j < t.cols; j++ {
			copy.data[i][j] = t.data[i][j]
		}
	}
	return copy
}

func (t *MockTensor) Add(other core.Tensor) core.Tensor     { return t }
func (t *MockTensor) Sub(other core.Tensor) core.Tensor     { return t }
func (t *MockTensor) Mul(other core.Tensor) core.Tensor     { return t }
func (t *MockTensor) MulElem(other core.Tensor) core.Tensor { return t }
func (t *MockTensor) Div(other core.Tensor) core.Tensor     { return t }
func (t *MockTensor) Scale(scalar float64) core.Tensor      { return t }
func (t *MockTensor) Pow(power float64) core.Tensor         { return t }
func (t *MockTensor) Sqrt() core.Tensor                     { return t }
func (t *MockTensor) Exp() core.Tensor                      { return t }
func (t *MockTensor) Log() core.Tensor                      { return t }
func (t *MockTensor) Abs() core.Tensor                      { return t }
func (t *MockTensor) Sign() core.Tensor                     { return t }
func (t *MockTensor) Clamp(min, max float64) core.Tensor    { return t }
func (t *MockTensor) T() core.Tensor                        { return t }
func (t *MockTensor) Sum() float64                          { return 0 }
func (t *MockTensor) Mean() float64                         { return 0 }
func (t *MockTensor) Std() float64                          { return 0 }
func (t *MockTensor) Max() float64                          { return 0 }
func (t *MockTensor) Min() float64                          { return 0 }
func (t *MockTensor) Norm() float64                         { return 0 }
func (t *MockTensor) Reshape(newRows, newCols int) core.Tensor {
	return NewMockTensor(newRows, newCols)
}
func (t *MockTensor) Flatten() core.Tensor                                   { return t }
func (t *MockTensor) Shape() []int                                           { return []int{t.rows, t.cols} }
func (t *MockTensor) Apply(fn func(i, j int, v float64) float64) core.Tensor { return t }
func (t *MockTensor) Equal(other core.Tensor) bool                           { return true }
func (t *MockTensor) Fill(value float64) {
	for i := 0; i < t.rows; i++ {
		for j := 0; j < t.cols; j++ {
			t.data[i][j] = value
		}
	}
}
func (t *MockTensor) Zero() {
	for i := 0; i < t.rows; i++ {
		for j := 0; j < t.cols; j++ {
			t.data[i][j] = 0
		}
	}
}
func (t *MockTensor) Release()              {}
func (t *MockTensor) Row(i int) core.Tensor { return NewMockTensor(1, t.cols) }
func (t *MockTensor) Col(j int) core.Tensor { return NewMockTensor(t.rows, 1) }
func (t *MockTensor) Slice(r0, r1, c0, c1 int) core.Tensor {
	return NewMockTensor(r1-r0, c1-c0)
}
func (t *MockTensor) SetRow(i int, data []float64)         {}
func (t *MockTensor) SetCol(j int, data []float64)         {}
func (t *MockTensor) IsEmpty() bool                        { return t.rows == 0 || t.cols == 0 }
func (t *MockTensor) IsSquare() bool                       { return t.rows == t.cols }
func (t *MockTensor) IsVector() bool                       { return t.rows == 1 || t.cols == 1 }
func (t *MockTensor) Name() string                         { return t.name }
func (t *MockTensor) SetName(name string)                  { t.name = name }
func (t *MockTensor) String() string                       { return "MockTensor" }
func (t *MockTensor) RawMatrix() *mat.Dense                { return nil }
func (t *MockTensor) Dot(other core.Tensor) float64        { return 0 }
func (t *MockTensor) AddScalar(scalar float64) core.Tensor { return t }
func (t *MockTensor) SubScalar(scalar float64) core.Tensor { return t }
func (t *MockTensor) DivScalar(scalar float64) core.Tensor { return t }
func (t *MockTensor) Trace() float64                       { return 0 }
func (t *MockTensor) Diagonal() core.Tensor                { return t }
func (t *MockTensor) Validate() error                      { return nil }
func (t *MockTensor) HasNaN() bool                         { return false }
func (t *MockTensor) HasInf() bool                         { return false }
func (t *MockTensor) IsFinite() bool                       { return true }

// MockLayer implements core.Layer for testing
type MockLayer struct {
	name        string
	trainable   bool
	paramCount  int
	outputShape []int
	params      []core.Tensor
	grads       []core.Tensor
}

func NewMockLayer(name string, trainable bool, paramCount int, outputShape []int) *MockLayer {
	return &MockLayer{
		name:        name,
		trainable:   trainable,
		paramCount:  paramCount,
		outputShape: outputShape,
		params:      make([]core.Tensor, 0),
		grads:       make([]core.Tensor, 0),
	}
}

func (l *MockLayer) Forward(input core.Tensor) (core.Tensor, error) {
	return NewMockTensor(l.outputShape[0], l.outputShape[1]), nil
}
func (l *MockLayer) Backward(gradient core.Tensor) (core.Tensor, error) {
	return gradient, nil
}
func (l *MockLayer) Parameters() []core.Tensor                   { return l.params }
func (l *MockLayer) Gradients() []core.Tensor                    { return l.grads }
func (l *MockLayer) IsTrainable() bool                           { return l.trainable }
func (l *MockLayer) Name() string                                { return l.name }
func (l *MockLayer) SetName(name string)                         { l.name = name }
func (l *MockLayer) OutputShape(inputShape []int) ([]int, error) { return l.outputShape, nil }
func (l *MockLayer) ParameterCount() int                         { return l.paramCount }

// MockOptimizer implements core.Optimizer for testing
type MockOptimizer struct {
	lr   float64
	name string
}

func NewMockOptimizer(lr float64) *MockOptimizer {
	return &MockOptimizer{lr: lr, name: "mock"}
}

func (o *MockOptimizer) Update(params []core.Tensor, grads []core.Tensor) {}
func (o *MockOptimizer) Step()                                            {}
func (o *MockOptimizer) Reset()                                           {}
func (o *MockOptimizer) Config() core.OptimizerConfig {
	return core.OptimizerConfig{Name: o.name, LearningRate: o.lr}
}
func (o *MockOptimizer) Name() string               { return o.name }
func (o *MockOptimizer) LearningRate() float64      { return o.lr }
func (o *MockOptimizer) SetLearningRate(lr float64) { o.lr = lr }

// MockLoss implements core.Loss for testing
type MockLoss struct {
	name string
}

func NewMockLoss() *MockLoss {
	return &MockLoss{name: "mock_loss"}
}

func (l *MockLoss) Compute(yTrue, yPred core.Tensor) float64 { return 0.5 }
func (l *MockLoss) Gradient(yTrue, yPred core.Tensor) core.Tensor {
	rows, cols := yPred.Dims()
	return NewMockTensor(rows, cols)
}
func (l *MockLoss) Name() string { return l.name }

func TestNewSequential(t *testing.T) {
	model := NewSequential()

	if model == nil {
		t.Fatal("NewSequential() returned nil")
	}

	if len(model.layers) != 0 {
		t.Errorf("Expected 0 layers, got %d", len(model.layers))
	}

	if model.compiled {
		t.Error("Model should not be compiled initially")
	}
}

func TestSequentialAddLayer(t *testing.T) {
	model := NewSequential()
	layer := NewMockLayer("test_layer", true, 10, []int{1, 5})

	err := model.AddLayer(layer)
	if err != nil {
		t.Fatalf("AddLayer() failed: %v", err)
	}

	if len(model.layers) != 1 {
		t.Errorf("Expected 1 layer, got %d", len(model.layers))
	}

	if model.layers[0] != layer {
		t.Error("Layer not added correctly")
	}
}

func TestSequentialAddLayerNil(t *testing.T) {
	model := NewSequential()

	err := model.AddLayer(nil)
	if err == nil {
		t.Error("AddLayer(nil) should return an error")
	}
}

func TestSequentialCompile(t *testing.T) {
	model := NewSequential()
	layer := NewMockLayer("test", true, 10, []int{1, 5})
	model.AddLayer(layer)

	optimizer := NewMockOptimizer(0.001)
	loss := NewMockLoss()

	err := model.Compile(optimizer, loss)
	if err != nil {
		t.Fatalf("Compile() failed: %v", err)
	}

	if !model.compiled {
		t.Error("Model should be compiled after Compile()")
	}

	if model.optimizer != optimizer {
		t.Error("Optimizer not set correctly")
	}

	if model.loss != loss {
		t.Error("Loss function not set correctly")
	}
}

func TestSequentialCompileNilOptimizer(t *testing.T) {
	model := NewSequential()
	loss := NewMockLoss()

	err := model.Compile(nil, loss)
	if err == nil {
		t.Error("Compile() with nil optimizer should return an error")
	}
}

func TestSequentialCompileNilLoss(t *testing.T) {
	model := NewSequential()
	optimizer := NewMockOptimizer(0.001)

	err := model.Compile(optimizer, nil)
	if err == nil {
		t.Error("Compile() with nil loss should return an error")
	}
}

func TestSequentialForward(t *testing.T) {
	model := NewSequential()
	layer1 := NewMockLayer("layer1", true, 10, []int{1, 5})
	layer2 := NewMockLayer("layer2", true, 5, []int{1, 3})

	model.AddLayer(layer1)
	model.AddLayer(layer2)

	input := NewMockTensor(1, 10)
	output, err := model.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	if output == nil {
		t.Fatal("Forward() returned nil")
	}

	rows, cols := output.Dims()
	if rows != 1 || cols != 3 {
		t.Errorf("Expected output shape (1, 3), got (%d, %d)", rows, cols)
	}
}

func TestSequentialPredict(t *testing.T) {
	model := NewSequential()
	optimizer := NewMockOptimizer(0.001)
	loss := NewMockLoss()
	layer := NewMockLayer("layer", true, 10, []int{1, 5})

	model.AddLayer(layer)
	model.Compile(optimizer, loss)

	input := NewMockTensor(1, 10)
	output, err := model.Predict(input)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	if output == nil {
		t.Fatal("Predict() returned nil")
	}
}

func TestSequentialPredictNotCompiled(t *testing.T) {
	model := NewSequential()
	layer := NewMockLayer("layer", true, 10, []int{1, 5})
	model.AddLayer(layer)

	input := NewMockTensor(1, 10)

	_, err := model.Predict(input)
	if err == nil {
		t.Error("Predict() on uncompiled model should return error")
	}
}

func TestSequentialFit(t *testing.T) {
	model := NewSequential()
	optimizer := NewMockOptimizer(0.001)
	loss := NewMockLoss()
	layer := NewMockLayer("layer", true, 10, []int{10, 1})

	model.AddLayer(layer)
	model.Compile(optimizer, loss)

	X := NewMockTensor(10, 5)
	y := NewMockTensor(10, 1)

	config := core.TrainingConfig{
		Epochs:    2,
		BatchSize: 5,
		Metrics:   []string{"accuracy"},
		Verbose:   0,
	}

	history, err := model.Fit(X, y, config)
	if err != nil {
		t.Fatalf("Fit() failed: %v", err)
	}

	if history == nil {
		t.Fatal("Fit() returned nil history")
	}

	if len(history.Loss) != 2 {
		t.Errorf("Expected 2 loss values, got %d", len(history.Loss))
	}
}

func TestSequentialFitNotCompiled(t *testing.T) {
	model := NewSequential()
	layer := NewMockLayer("layer", true, 10, []int{10, 1})
	model.AddLayer(layer)

	X := NewMockTensor(10, 5)
	y := NewMockTensor(10, 1)

	config := core.TrainingConfig{
		Epochs:    1,
		BatchSize: 5,
	}

	_, err := model.Fit(X, y, config)
	if err == nil {
		t.Error("Fit() on uncompiled model should return an error")
	}
}

func TestSequentialFitNilData(t *testing.T) {
	model := NewSequential()
	optimizer := NewMockOptimizer(0.001)
	loss := NewMockLoss()
	model.Compile(optimizer, loss)

	config := core.TrainingConfig{
		Epochs:    1,
		BatchSize: 5,
	}

	_, err := model.Fit(nil, nil, config)
	if err == nil {
		t.Error("Fit() with nil data should return an error")
	}
}

func TestSequentialEvaluate(t *testing.T) {
	model := NewSequential()
	optimizer := NewMockOptimizer(0.001)
	loss := NewMockLoss()
	layer := NewMockLayer("layer", true, 10, []int{10, 1})

	model.AddLayer(layer)
	model.Compile(optimizer, loss)

	X := NewMockTensor(10, 5)
	y := NewMockTensor(10, 1)

	metrics, err := model.Evaluate(X, y)
	if err != nil {
		t.Fatalf("Evaluate() failed: %v", err)
	}

	if metrics == nil {
		t.Fatal("Evaluate() returned nil metrics")
	}
}

func TestSequentialSummary(t *testing.T) {
	model := NewSequential()
	layer1 := NewMockLayer("layer1", true, 10, []int{1, 5})
	layer2 := NewMockLayer("layer2", true, 5, []int{1, 3})

	model.AddLayer(layer1)
	model.AddLayer(layer2)

	summary := model.Summary()
	if summary == "" {
		t.Error("Summary() returned empty string")
	}

	// Check that summary contains expected information
	if !contains(summary, "Model: Sequential") {
		t.Error("Summary should contain model type")
	}

	if !contains(summary, "layer1") {
		t.Error("Summary should contain layer names")
	}
}

func TestSequentialIsCompiled(t *testing.T) {
	model := NewSequential()

	if model.IsCompiled() {
		t.Error("Model should not be compiled initially")
	}

	layer := NewMockLayer("test", true, 10, []int{1, 5})
	model.AddLayer(layer)

	optimizer := NewMockOptimizer(0.001)
	loss := NewMockLoss()
	model.Compile(optimizer, loss)

	if !model.IsCompiled() {
		t.Error("Model should be compiled after Compile()")
	}
}

func TestSequentialEasyTrain(t *testing.T) {
	model := NewSequential()
	optimizer := NewMockOptimizer(0.001)
	loss := NewMockLoss()
	layer := NewMockLayer("layer", true, 10, []int{10, 1})

	model.AddLayer(layer)
	model.Compile(optimizer, loss)

	X := NewMockTensor(10, 5)
	y := NewMockTensor(10, 1)

	history, err := model.EasyTrain(X, y)
	if err != nil {
		t.Fatalf("EasyTrain() failed: %v", err)
	}

	if history == nil {
		t.Fatal("EasyTrain() returned nil history")
	}

	// Should have 10 epochs by default
	if len(history.Loss) != 10 {
		t.Errorf("Expected 10 loss values, got %d", len(history.Loss))
	}
}

func TestSequentialEasyTrainNotCompiled(t *testing.T) {
	model := NewSequential()
	layer := NewMockLayer("layer", true, 10, []int{10, 1})
	model.AddLayer(layer)

	X := NewMockTensor(10, 5)
	y := NewMockTensor(10, 1)

	_, err := model.EasyTrain(X, y)
	if err == nil {
		t.Error("EasyTrain() on uncompiled model should return an error")
	}

	// Check that error message contains Arabic text
	if !contains(err.Error(), "يجب تجميع النموذج أولاً") {
		t.Error("Error message should contain Arabic text")
	}
}

func TestSequentialEasyTrainNilData(t *testing.T) {
	model := NewSequential()
	optimizer := NewMockOptimizer(0.001)
	loss := NewMockLoss()
	layer := NewMockLayer("layer", true, 10, []int{10, 1})

	model.AddLayer(layer)
	model.Compile(optimizer, loss)

	// Test with nil X
	_, err := model.EasyTrain(nil, NewMockTensor(10, 1))
	if err == nil {
		t.Error("EasyTrain() with nil X should return an error")
	}
	if !contains(err.Error(), "بيانات الإدخال لا يمكن أن تكون فارغة") {
		t.Error("Error message should contain Arabic text for nil input")
	}

	// Test with nil y
	_, err = model.EasyTrain(NewMockTensor(10, 5), nil)
	if err == nil {
		t.Error("EasyTrain() with nil y should return an error")
	}
	if !contains(err.Error(), "بيانات الهدف لا يمكن أن تكون فارغة") {
		t.Error("Error message should contain Arabic text for nil target")
	}
}

func TestSequentialEasyPredict(t *testing.T) {
	model := NewSequential()
	optimizer := NewMockOptimizer(0.001)
	loss := NewMockLoss()
	layer := NewMockLayer("layer", true, 10, []int{1, 5})

	model.AddLayer(layer)
	model.Compile(optimizer, loss)

	input := NewMockTensor(1, 10)
	output, err := model.EasyPredict(input)
	if err != nil {
		t.Fatalf("EasyPredict() failed: %v", err)
	}

	if output == nil {
		t.Fatal("EasyPredict() returned nil")
	}
}

func TestSequentialEasyPredictNotCompiled(t *testing.T) {
	model := NewSequential()
	layer := NewMockLayer("layer", true, 10, []int{1, 5})
	model.AddLayer(layer)

	input := NewMockTensor(1, 10)

	_, err := model.EasyPredict(input)
	if err == nil {
		t.Error("EasyPredict() on uncompiled model should return an error")
	}

	// Check that error message contains Arabic text
	if !contains(err.Error(), "يجب تجميع النموذج أولاً") {
		t.Error("Error message should contain Arabic text")
	}
}

func TestSequentialEasyPredictNilInput(t *testing.T) {
	model := NewSequential()
	optimizer := NewMockOptimizer(0.001)
	loss := NewMockLoss()
	layer := NewMockLayer("layer", true, 10, []int{1, 5})

	model.AddLayer(layer)
	model.Compile(optimizer, loss)

	_, err := model.EasyPredict(nil)
	if err == nil {
		t.Error("EasyPredict() with nil input should return an error")
	}

	// Check that error message contains Arabic text
	if !contains(err.Error(), "بيانات الإدخال لا يمكن أن تكون فارغة") {
		t.Error("Error message should contain Arabic text for nil input")
	}
}

func TestSequentialEasyPredictNoLayers(t *testing.T) {
	model := NewSequential()
	optimizer := NewMockOptimizer(0.001)
	loss := NewMockLoss()

	// Try to compile model with no layers - this should fail
	err := model.Compile(optimizer, loss)
	if err == nil {
		t.Error("Compile() on model with no layers should return an error")
		return
	}

	// Since compile failed, the model is not compiled
	// EasyPredict should fail with "not compiled" error
	input := NewMockTensor(1, 10)

	_, err = model.EasyPredict(input)
	if err == nil {
		t.Error("EasyPredict() on uncompiled model should return an error")
	}

	// Check that error message contains Arabic text for not compiled
	if !contains(err.Error(), "يجب تجميع النموذج أولاً") {
		t.Error("Error message should contain Arabic text for not compiled")
	}
}

// Helper function to check if string contains substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr ||
		(len(s) > len(substr) && (s[:len(substr)] == substr ||
			s[len(s)-len(substr):] == substr ||
			containsAt(s, substr))))
}

func containsAt(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
