package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unsafe"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

const (
	Epsilon     = 1e-8
	DefaultSeed = 42
)

// ============ Core Types ============

type Config struct {
	LearningRate    float64
	Epochs          int
	BatchSize       int
	ValidationSplit float64
	EarlyStopping   bool
	Patience        int
	Verbose         int
	Seed            int64
	UseGPU          bool
	NumWorkers      int
	SaveCheckpoints bool
	CheckpointDir   string
}

func NewConfig() *Config {
	return &Config{
		LearningRate:    0.001,
		Epochs:          100,
		BatchSize:       32,
		ValidationSplit: 0.2,
		EarlyStopping:   false,
		Patience:        10,
		Verbose:         1,
		Seed:            DefaultSeed,
		UseGPU:          false,
		NumWorkers:      1,
		SaveCheckpoints: false,
		CheckpointDir:   "./checkpoints",
	}
}

type History struct {
	Loss        []float64
	Accuracy    []float64
	ValLoss     []float64
	ValAccuracy []float64
	Epochs      []int
}

func NewHistory() *History {
	return &History{
		Loss:        make([]float64, 0),
		Accuracy:    make([]float64, 0),
		ValLoss:     make([]float64, 0),
		ValAccuracy: make([]float64, 0),
		Epochs:      make([]int, 0),
	}
}

// ============ Automatic Differentiation (Autograd) ============

type Tensor struct {
	Data         *mat.Dense
	Grad         *mat.Dense
	RequiresGrad bool
	GradFn       GradFunction
	IsLeaf       bool
	Name         string
}

type GradFunction interface {
	Apply(gradOutput *Tensor) []*Tensor
	NextFunctions() []GradFunction
}

func NewTensor(data *mat.Dense, requiresGrad bool) *Tensor {
	rows, cols := data.Dims()
	return &Tensor{
		Data:         mat.DenseCopyOf(data),
		Grad:         mat.NewDense(rows, cols, nil),
		RequiresGrad: requiresGrad,
		IsLeaf:       true,
	}
}

func (t *Tensor) Backward() {
	if !t.RequiresGrad {
		return
	}

	// Initialize gradient for backward pass
	if t.Grad == nil {
		rows, cols := t.Data.Dims()
		t.Grad = mat.NewDense(rows, cols, nil)
		// Set to ones for scalar loss
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				t.Grad.Set(i, j, 1.0)
			}
		}
	}

	// Perform backward pass through computation graph
	if t.GradFn != nil {
		_ = t.GradFn.Apply(t)
		// Propagate gradients to input tensors
		// This would continue the chain rule in a full implementation
	}
}

func (t *Tensor) ZeroGrad() {
	if t.Grad != nil {
		t.Grad.Zero()
	}
}

// Matrix multiplication with autograd
type MatMulFunction struct {
	savedTensors []*Tensor
}

func (mmf *MatMulFunction) Apply(gradOutput *Tensor) []*Tensor {
	a, b := mmf.savedTensors[0], mmf.savedTensors[1]

	// Compute gradients
	gradA := &Tensor{Data: mat.NewDense(a.Data.RawMatrix().Rows, a.Data.RawMatrix().Cols, nil)}
	gradB := &Tensor{Data: mat.NewDense(b.Data.RawMatrix().Rows, b.Data.RawMatrix().Cols, nil)}

	gradA.Data.Mul(gradOutput.Data, b.Data.T())
	gradB.Data.Mul(a.Data.T(), gradOutput.Data)

	return []*Tensor{gradA, gradB}
}

func (mmf *MatMulFunction) NextFunctions() []GradFunction {
	return nil
}

func MatMul(a, b *Tensor) *Tensor {
	result := mat.NewDense(a.Data.RawMatrix().Rows, b.Data.RawMatrix().Cols, nil)
	result.Mul(a.Data, b.Data)

	output := &Tensor{
		Data:         result,
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
		IsLeaf:       false,
	}

	if output.RequiresGrad {
		output.GradFn = &MatMulFunction{savedTensors: []*Tensor{a, b}}
	}

	return output
}

// ============ GPU Acceleration Interface ============

type Device interface {
	ToDevice(data *mat.Dense) GPUMatrix
	ToHost(data GPUMatrix) *mat.Dense
	MatMul(a, b GPUMatrix) GPUMatrix
	Add(a, b GPUMatrix) GPUMatrix
	Activation(data GPUMatrix, activation string) GPUMatrix
	GetType() string
}

type GPUMatrix interface {
	Dims() (int, int)
	At(i, j int) float64
	Set(i, j int, v float64)
	Copy() GPUMatrix
}

// CPU implementation (fallback)
type CPUDevice struct{}
type CPUMatrix struct{ *mat.Dense }

func (c CPUMatrix) Copy() GPUMatrix { return CPUMatrix{mat.DenseCopyOf(c.Dense)} }

func (d *CPUDevice) GetType() string { return "cpu" }

func (d *CPUDevice) ToDevice(data *mat.Dense) GPUMatrix {
	return CPUMatrix{mat.DenseCopyOf(data)}
}

func (d *CPUDevice) ToHost(data GPUMatrix) *mat.Dense {
	return mat.DenseCopyOf(data.(CPUMatrix).Dense)
}

func (d *CPUDevice) MatMul(a, b GPUMatrix) GPUMatrix {
	r, _ := a.Dims()
	_, c := b.Dims()
	result := mat.NewDense(r, c, nil)
	result.Mul(a.(CPUMatrix).Dense, b.(CPUMatrix).Dense)
	return CPUMatrix{result}
}

func (d *CPUDevice) Add(a, b GPUMatrix) GPUMatrix {
	r, c := a.Dims()
	result := mat.NewDense(r, c, nil)
	result.Add(a.(CPUMatrix).Dense, b.(CPUMatrix).Dense)
	return CPUMatrix{result}
}

func (d *CPUDevice) Activation(data GPUMatrix, activation string) GPUMatrix {
	rows, cols := data.Dims()
	result := mat.NewDense(rows, cols, nil)

	switch activation {
	case "relu":
		result.Apply(func(i, j int, v float64) float64 {
			return math.Max(0, data.At(i, j))
		}, result)
	case "sigmoid":
		result.Apply(func(i, j int, v float64) float64 {
			return 1.0 / (1.0 + math.Exp(-data.At(i, j)))
		}, result)
	case "tanh":
		result.Apply(func(i, j int, v float64) float64 {
			return math.Tanh(data.At(i, j))
		}, result)
	}

	return CPUMatrix{result}
}

// GPU device interface (would implement CUDA operations)
type CUDADevice struct {
	deviceID int
}

func NewCUDADevice(deviceID int) *CUDADevice {
	return &CUDADevice{deviceID: deviceID}
}

func (d *CUDADevice) GetType() string {
	return fmt.Sprintf("cuda:%d", d.deviceID)
}

// Note: In a real implementation, these would interface with CUDA
func (d *CUDADevice) ToDevice(data *mat.Dense) GPUMatrix {
	// Simulate GPU transfer - in reality this would copy to GPU memory
	fmt.Printf("Transferring data to GPU device %d\n", d.deviceID)
	return CPUMatrix{mat.DenseCopyOf(data)} // Fallback to CPU for demo
}

func (d *CUDADevice) ToHost(data GPUMatrix) *mat.Dense {
	return mat.DenseCopyOf(data.(CPUMatrix).Dense)
}

func (d *CUDADevice) MatMul(a, b GPUMatrix) GPUMatrix {
	// In reality, this would use cuBLAS
	fmt.Println("Performing GPU matrix multiplication")
	return (&CPUDevice{}).MatMul(a, b)
}

func (d *CUDADevice) Add(a, b GPUMatrix) GPUMatrix {
	return (&CPUDevice{}).Add(a, b)
}

func (d *CUDADevice) Activation(data GPUMatrix, activation string) GPUMatrix {
	fmt.Printf("Performing GPU %s activation\n", activation)
	return (&CPUDevice{}).Activation(data, activation)
}

// ============ Enhanced Activation Functions ============

type Activation interface {
	Forward(x float64) float64
	Backward(x float64) float64
	GetName() string
}

type ReLU struct{}

func (r *ReLU) GetName() string           { return "relu" }
func (r *ReLU) Forward(x float64) float64 { return math.Max(0, x) }
func (r *ReLU) Backward(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

type LeakyReLU struct{ Alpha float64 }

func NewLeakyReLU(alpha float64) *LeakyReLU { return &LeakyReLU{Alpha: alpha} }
func (l *LeakyReLU) GetName() string        { return "leaky_relu" }
func (l *LeakyReLU) Forward(x float64) float64 {
	if x > 0 {
		return x
	}
	return l.Alpha * x
}
func (l *LeakyReLU) Backward(x float64) float64 {
	if x > 0 {
		return 1
	}
	return l.Alpha
}

type ELU struct{ Alpha float64 }

func NewELU(alpha float64) *ELU { return &ELU{Alpha: alpha} }
func (e *ELU) GetName() string  { return "elu" }
func (e *ELU) Forward(x float64) float64 {
	if x > 0 {
		return x
	}
	return e.Alpha * (math.Exp(x) - 1)
}
func (e *ELU) Backward(x float64) float64 {
	if x > 0 {
		return 1
	}
	return e.Alpha * math.Exp(x)
}

type Swish struct{}

func (s *Swish) GetName() string { return "swish" }
func (s *Swish) Forward(x float64) float64 {
	sigmoid := 1.0 / (1.0 + math.Exp(-x))
	return x * sigmoid
}
func (s *Swish) Backward(x float64) float64 {
	sigmoid := 1.0 / (1.0 + math.Exp(-x))
	return sigmoid + x*sigmoid*(1-sigmoid)
}

type GELU struct{}

func (g *GELU) GetName() string { return "gelu" }
func (g *GELU) Forward(x float64) float64 {
	return 0.5 * x * (1.0 + math.Tanh(math.Sqrt(2.0/math.Pi)*(x+0.044715*math.Pow(x, 3))))
}
func (g *GELU) Backward(x float64) float64 {
	// Approximate derivative
	tanh_input := math.Sqrt(2.0/math.Pi) * (x + 0.044715*math.Pow(x, 3))
	tanh_val := math.Tanh(tanh_input)
	sech2 := 1 - tanh_val*tanh_val
	return 0.5*(1+tanh_val) + 0.5*x*sech2*math.Sqrt(2.0/math.Pi)*(1+3*0.044715*x*x)
}

// Original activations
type Sigmoid struct{}

func (s *Sigmoid) GetName() string { return "sigmoid" }
func (s *Sigmoid) Forward(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-clamp(x, -250, 250)))
}
func (s *Sigmoid) Backward(x float64) float64 {
	sig := s.Forward(x)
	return sig * (1 - sig)
}

type Tanh struct{}

func (t *Tanh) GetName() string           { return "tanh" }
func (t *Tanh) Forward(x float64) float64 { return math.Tanh(x) }
func (t *Tanh) Backward(x float64) float64 {
	tanh := math.Tanh(x)
	return 1 - tanh*tanh
}

type Linear struct{}

func (l *Linear) GetName() string            { return "linear" }
func (l *Linear) Forward(x float64) float64  { return x }
func (l *Linear) Backward(x float64) float64 { return 1 }

type Softmax struct{}

func (s *Softmax) GetName() string            { return "softmax" }
func (s *Softmax) Forward(x float64) float64  { return x }
func (s *Softmax) Backward(x float64) float64 { return 1 }

// ============ Loss Functions ============

type LossFunction interface {
	Compute(yTrue, yPred *mat.Dense) float64
	ComputeGrad(yTrue, yPred *mat.Dense) *mat.Dense
	GetName() string
}

type BinaryCrossentropy struct{}

func (bce *BinaryCrossentropy) GetName() string { return "binary_crossentropy" }

func (bce *BinaryCrossentropy) Compute(yTrue, yPred *mat.Dense) float64 {
	rows, _ := yPred.Dims()
	var total float64
	for i := 0; i < rows; i++ {
		p := clamp(yPred.At(i, 0), Epsilon, 1-Epsilon)
		t := yTrue.At(i, 0)
		total += t*math.Log(p) + (1-t)*math.Log(1-p)
	}
	return -total / float64(rows)
}

func (bce *BinaryCrossentropy) ComputeGrad(yTrue, yPred *mat.Dense) *mat.Dense {
	rows, cols := yPred.Dims()
	grad := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			p := clamp(yPred.At(i, j), Epsilon, 1-Epsilon)
			t := yTrue.At(i, j)
			grad.Set(i, j, (p-t)/(p*(1-p)*float64(rows)))
		}
	}
	return grad
}

type CategoricalCrossentropy struct{}

func (cce *CategoricalCrossentropy) GetName() string { return "categorical_crossentropy" }

func (cce *CategoricalCrossentropy) Compute(yTrue, yPred *mat.Dense) float64 {
	rows, cols := yPred.Dims()
	var total float64
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			p := clamp(yPred.At(i, j), Epsilon, 1-Epsilon)
			t := yTrue.At(i, j)
			if t > 0 {
				total += t * math.Log(p)
			}
		}
	}
	return -total / float64(rows)
}

func (cce *CategoricalCrossentropy) ComputeGrad(yTrue, yPred *mat.Dense) *mat.Dense {
	rows, cols := yPred.Dims()
	grad := mat.NewDense(rows, cols, nil)
	grad.Sub(yPred, yTrue)
	grad.Scale(1.0/float64(rows), grad)
	return grad
}

type FocalLoss struct {
	Alpha float64
	Gamma float64
}

func NewFocalLoss(alpha, gamma float64) *FocalLoss {
	return &FocalLoss{Alpha: alpha, Gamma: gamma}
}

func (fl *FocalLoss) GetName() string { return "focal_loss" }

func (fl *FocalLoss) Compute(yTrue, yPred *mat.Dense) float64 {
	rows, cols := yPred.Dims()
	var total float64

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			p := clamp(yPred.At(i, j), Epsilon, 1-Epsilon)
			t := yTrue.At(i, j)

			if t > 0 {
				focal_weight := fl.Alpha * math.Pow(1-p, fl.Gamma)
				total += focal_weight * math.Log(p)
			}
		}
	}
	return -total / float64(rows)
}

func (fl *FocalLoss) ComputeGrad(yTrue, yPred *mat.Dense) *mat.Dense {
	rows, cols := yPred.Dims()
	grad := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			p := clamp(yPred.At(i, j), Epsilon, 1-Epsilon)
			t := yTrue.At(i, j)

			alpha_t := fl.Alpha
			if t == 0 {
				alpha_t = 1 - fl.Alpha
			}

			pt := p
			if t == 0 {
				pt = 1 - p
			}

			focal_weight := alpha_t * math.Pow(1-pt, fl.Gamma)
			grad_val := focal_weight * (fl.Gamma*pt*math.Log(pt) + pt - 1) / float64(rows)

			if t == 0 {
				grad_val *= -1
			}
			grad.Set(i, j, grad_val)
		}
	}
	return grad
}

type MeanSquaredError struct{}

func (mse *MeanSquaredError) GetName() string { return "mean_squared_error" }

func (mse *MeanSquaredError) Compute(yTrue, yPred *mat.Dense) float64 {
	rows, cols := yPred.Dims()
	var total float64
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			diff := yTrue.At(i, j) - yPred.At(i, j)
			total += diff * diff
		}
	}
	return total / (2.0 * float64(rows))
}

func (mse *MeanSquaredError) ComputeGrad(yTrue, yPred *mat.Dense) *mat.Dense {
	rows, cols := yPred.Dims()
	grad := mat.NewDense(rows, cols, nil)
	grad.Sub(yPred, yTrue)
	grad.Scale(1.0/float64(rows), grad)
	return grad
}

// ============ Enhanced Layer Interface ============

type Layer interface {
	Forward(inputs ...*mat.Dense) *mat.Dense
	Backward(grad *mat.Dense) []*mat.Dense
	IsTrainable() bool
	GetWeights() []*mat.Dense
	GetGradients() []*mat.Dense
	GetName() string
	SetName(name string)
	GetOutputShape(inputShape []int) []int
	GetParams() int
	SetDevice(device Device)
	Train(training bool)
}

type BaseLayer struct {
	name     string
	device   Device
	training bool
}

func (l *BaseLayer) GetName() string         { return l.name }
func (l *BaseLayer) SetName(name string)     { l.name = name }
func (l *BaseLayer) GetParams() int          { return 0 }
func (l *BaseLayer) SetDevice(device Device) { l.device = device }
func (l *BaseLayer) Train(training bool)     { l.training = training }

// ============ Dense Layer (Enhanced) ============

type Dense struct {
	BaseLayer
	inputDim, outputDim      int
	weights, biases          *mat.Dense
	gradWeights, gradBiases  *mat.Dense
	lastInput, preActivation *mat.Dense
	activation               Activation
	weightInit               string
	useBias                  bool
}

func NewDense(units int, activation Activation, options ...func(*Dense)) *Dense {
	layer := &Dense{
		outputDim:  units,
		activation: activation,
		weightInit: "glorot_uniform",
		useBias:    true,
	}
	layer.device = &CPUDevice{}
	layer.training = true

	for _, opt := range options {
		opt(layer)
	}

	if layer.activation == nil {
		layer.activation = &Linear{}
	}

	return layer
}

func WithInputDim(dim int) func(*Dense) {
	return func(d *Dense) { d.inputDim = dim }
}

func WithWeightInit(method string) func(*Dense) {
	return func(d *Dense) { d.weightInit = method }
}

func WithBias(useBias bool) func(*Dense) {
	return func(d *Dense) { d.useBias = useBias }
}

func (d *Dense) Build(inputDim int) {
	d.inputDim = inputDim

	var std float64
	switch d.weightInit {
	case "glorot_uniform", "xavier_uniform":
		std = math.Sqrt(6.0 / float64(d.inputDim+d.outputDim))
	case "he_uniform":
		std = math.Sqrt(6.0 / float64(d.inputDim))
	case "glorot_normal", "xavier_normal":
		std = math.Sqrt(2.0 / float64(d.inputDim+d.outputDim))
	case "he_normal":
		std = math.Sqrt(2.0 / float64(d.inputDim))
	default:
		std = 0.01
	}

	d.weights = mat.NewDense(d.inputDim, d.outputDim, randomNormal(d.inputDim*d.outputDim, std))
	if d.useBias {
		d.biases = mat.NewDense(1, d.outputDim, zeros(d.outputDim))
	}
}

func (d *Dense) GetOutputShape(inputShape []int) []int {
	return []int{inputShape[0], d.outputDim}
}

func (d *Dense) GetParams() int {
	if d.weights == nil {
		return 0
	}
	params := d.inputDim * d.outputDim
	if d.useBias {
		params += d.outputDim
	}
	return params
}

func (d *Dense) Forward(inputs ...*mat.Dense) *mat.Dense {
	x := inputs[0]
	d.lastInput = mat.DenseCopyOf(x)

	if d.weights == nil {
		_, inputDim := x.Dims()
		d.Build(inputDim)
	}

	rows, _ := x.Dims()
	d.preActivation = mat.NewDense(rows, d.outputDim, nil)

	// Use device for matrix multiplication if available
	if d.device != nil && d.device.GetType() != "cpu" {
		gpuX := d.device.ToDevice(x)
		gpuW := d.device.ToDevice(d.weights)
		result := d.device.MatMul(gpuX, gpuW)
		d.preActivation = d.device.ToHost(result)
	} else {
		d.preActivation.Mul(x, d.weights)
	}

	// Add bias if enabled
	if d.useBias && d.biases != nil {
		d.preActivation.Apply(func(i, j int, v float64) float64 {
			return v + d.biases.At(0, j)
		}, d.preActivation)
	}

	// Apply activation
	output := mat.NewDense(rows, d.outputDim, nil)
	if _, isSoftmax := d.activation.(*Softmax); isSoftmax {
		for i := 0; i < rows; i++ {
			row := mat.Row(nil, i, d.preActivation)
			softmaxRow := applySoftmax(row)
			for j := 0; j < d.outputDim; j++ {
				output.Set(i, j, softmaxRow[j])
			}
		}
	} else {
		output.Apply(func(i, j int, v float64) float64 {
			return d.activation.Forward(d.preActivation.At(i, j))
		}, d.preActivation)
	}

	return output
}

func (d *Dense) Backward(grad *mat.Dense) []*mat.Dense {
	rows, cols := grad.Dims()

	// Compute activation gradient
	activationGrad := mat.NewDense(rows, cols, nil)
	if _, isSoftmax := d.activation.(*Softmax); isSoftmax {
		activationGrad.Copy(grad)
	} else {
		activationGrad.Apply(func(i, j int, v float64) float64 {
			return d.activation.Backward(d.preActivation.At(i, j))
		}, d.preActivation)
		activationGrad.MulElem(activationGrad, grad)
	}

	// Compute weight gradients
	d.gradWeights = mat.NewDense(d.inputDim, d.outputDim, nil)
	d.gradWeights.Mul(d.lastInput.T(), activationGrad)

	// Compute bias gradients
	if d.useBias {
		d.gradBiases = mat.NewDense(1, d.outputDim, nil)
		for j := 0; j < d.outputDim; j++ {
			d.gradBiases.Set(0, j, floats.Sum(mat.Col(nil, j, activationGrad)))
		}
	}

	// Compute input gradients
	inputGrad := mat.NewDense(rows, d.inputDim, nil)
	inputGrad.Mul(activationGrad, d.weights.T())

	return []*mat.Dense{inputGrad}
}

func (d *Dense) IsTrainable() bool { return true }

func (d *Dense) GetWeights() []*mat.Dense {
	if d.useBias && d.biases != nil {
		return []*mat.Dense{d.weights, d.biases}
	}
	return []*mat.Dense{d.weights}
}

func (d *Dense) GetGradients() []*mat.Dense {
	if d.useBias && d.gradBiases != nil {
		return []*mat.Dense{d.gradWeights, d.gradBiases}
	}
	return []*mat.Dense{d.gradWeights}
}

// ============ Convolutional Layer ============

type Conv2D struct {
	BaseLayer
	filters     int
	kernelSize  []int
	stride      []int
	padding     string
	activation  Activation
	useBias     bool
	weights     *mat.Dense
	biases      *mat.Dense
	gradWeights *mat.Dense
	gradBiases  *mat.Dense
}

func NewConv2D(filters int, kernelSize []int, options ...func(*Conv2D)) *Conv2D {
	layer := &Conv2D{
		filters:    filters,
		kernelSize: kernelSize,
		stride:     []int{1, 1},
		padding:    "valid",
		useBias:    true,
	}
	layer.device = &CPUDevice{}
	layer.training = true

	for _, opt := range options {
		opt(layer)
	}

	return layer
}

func WithStride(stride []int) func(*Conv2D) {
	return func(c *Conv2D) { c.stride = stride }
}

func WithPadding(padding string) func(*Conv2D) {
	return func(c *Conv2D) { c.padding = padding }
}

func WithConvActivation(activation Activation) func(*Conv2D) {
	return func(c *Conv2D) { c.activation = activation }
}

func (c *Conv2D) GetOutputShape(inputShape []int) []int {
	// Simplified output shape calculation
	// In a full implementation, this would account for padding and stride
	height := (inputShape[1]-c.kernelSize[0])/c.stride[0] + 1
	width := (inputShape[2]-c.kernelSize[1])/c.stride[1] + 1
	return []int{inputShape[0], height, width, c.filters}
}

func (c *Conv2D) GetParams() int {
	if c.weights == nil {
		return 0
	}
	params := c.kernelSize[0] * c.kernelSize[1] * c.filters
	if c.useBias {
		params += c.filters
	}
	return params
}

// Simplified convolution implementation (for demonstration)
func (c *Conv2D) Forward(inputs ...*mat.Dense) *mat.Dense {
	x := inputs[0]

	// In a real implementation, this would perform 2D convolution
	// For now, we'll return a reshaped version to demonstrate the interface
	rows, _ := x.Dims()
	output := mat.NewDense(rows, c.filters, nil)

	// Placeholder convolution operation
	for i := 0; i < rows; i++ {
		for j := 0; j < c.filters; j++ {
			output.Set(i, j, rand.NormFloat64()*0.1)
		}
	}

	if c.activation != nil {
		output.Apply(func(i, j int, v float64) float64 {
			return c.activation.Forward(v)
		}, output)
	}

	return output
}

func (c *Conv2D) Backward(grad *mat.Dense) []*mat.Dense {
	// Simplified backward pass - in reality would compute proper convolution gradients
	rows, _ := grad.Dims()
	inputGrad := mat.NewDense(rows, 784, nil) // Assuming flattened 28x28 input
	return []*mat.Dense{inputGrad}
}

func (c *Conv2D) IsTrainable() bool { return true }
func (c *Conv2D) GetWeights() []*mat.Dense {
	if c.useBias && c.biases != nil {
		return []*mat.Dense{c.weights, c.biases}
	}
	return []*mat.Dense{c.weights}
}
func (c *Conv2D) GetGradients() []*mat.Dense {
	if c.useBias && c.gradBiases != nil {
		return []*mat.Dense{c.gradWeights, c.gradBiases}
	}
	return []*mat.Dense{c.gradWeights}
}

// ============ Max Pooling Layer ============

type MaxPooling2D struct {
	BaseLayer
	poolSize []int
	stride   []int
	padding  string
}

func NewMaxPooling2D(poolSize []int, options ...func(*MaxPooling2D)) *MaxPooling2D {
	layer := &MaxPooling2D{
		poolSize: poolSize,
		stride:   poolSize, // Default stride equals pool size
		padding:  "valid",
	}
	layer.device = &CPUDevice{}
	layer.training = true

	for _, opt := range options {
		opt(layer)
	}

	return layer
}

func WithPoolStride(stride []int) func(*MaxPooling2D) {
	return func(m *MaxPooling2D) { m.stride = stride }
}

func (m *MaxPooling2D) GetOutputShape(inputShape []int) []int {
	height := (inputShape[1]-m.poolSize[0])/m.stride[0] + 1
	width := (inputShape[2]-m.poolSize[1])/m.stride[1] + 1
	return []int{inputShape[0], height, width, inputShape[3]}
}

func (m *MaxPooling2D) Forward(inputs ...*mat.Dense) *mat.Dense {
	x := inputs[0]
	rows, cols := x.Dims()

	// Simplified pooling - in reality would perform 2D max pooling
	outputCols := cols / 2 // Assuming 2x2 pooling
	output := mat.NewDense(rows, outputCols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < outputCols; j++ {
			// Take max of 2 adjacent elements as simplified pooling
			val1 := x.At(i, j*2)
			val2 := 0.0
			if j*2+1 < cols {
				val2 = x.At(i, j*2+1)
			}
			output.Set(i, j, math.Max(val1, val2))
		}
	}

	return output
}

func (m *MaxPooling2D) Backward(grad *mat.Dense) []*mat.Dense {
	rows, cols := grad.Dims()
	inputGrad := mat.NewDense(rows, cols*2, nil) // Reverse pooling

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			// Distribute gradient (simplified)
			val := grad.At(i, j)
			inputGrad.Set(i, j*2, val)
			if j*2+1 < cols*2 {
				inputGrad.Set(i, j*2+1, 0) // Only max value gets gradient
			}
		}
	}

	return []*mat.Dense{inputGrad}
}

func (m *MaxPooling2D) IsTrainable() bool          { return false }
func (m *MaxPooling2D) GetWeights() []*mat.Dense   { return nil }
func (m *MaxPooling2D) GetGradients() []*mat.Dense { return nil }

// ============ Embedding Layer ============

type Embedding struct {
	BaseLayer
	vocabSize    int
	embeddingDim int
	weights      *mat.Dense
	gradWeights  *mat.Dense
}

func NewEmbedding(vocabSize, embeddingDim int, options ...func(*Embedding)) *Embedding {
	layer := &Embedding{
		vocabSize:    vocabSize,
		embeddingDim: embeddingDim,
	}
	layer.device = &CPUDevice{}
	layer.training = true

	// Initialize embedding weights
	std := math.Sqrt(2.0 / float64(embeddingDim))
	layer.weights = mat.NewDense(vocabSize, embeddingDim, randomNormal(vocabSize*embeddingDim, std))

	for _, opt := range options {
		opt(layer)
	}

	return layer
}

func (e *Embedding) GetOutputShape(inputShape []int) []int {
	return []int{inputShape[0], inputShape[1], e.embeddingDim}
}

func (e *Embedding) GetParams() int {
	return e.vocabSize * e.embeddingDim
}

func (e *Embedding) Forward(inputs ...*mat.Dense) *mat.Dense {
	indices := inputs[0]
	batchSize, seqLen := indices.Dims()

	output := mat.NewDense(batchSize*seqLen, e.embeddingDim, nil)

	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			idx := int(indices.At(i, j))
			if idx >= 0 && idx < e.vocabSize {
				for k := 0; k < e.embeddingDim; k++ {
					output.Set(i*seqLen+j, k, e.weights.At(idx, k))
				}
			}
		}
	}

	return output
}

func (e *Embedding) Backward(grad *mat.Dense) []*mat.Dense {
	// In practice, this would accumulate gradients for each embedding vector
	rows, cols := grad.Dims()
	inputGrad := mat.NewDense(rows, cols, nil) // Pass-through for indices

	// Update embedding gradients (simplified)
	e.gradWeights = mat.NewDense(e.vocabSize, e.embeddingDim, nil)

	return []*mat.Dense{inputGrad}
}

func (e *Embedding) IsTrainable() bool          { return true }
func (e *Embedding) GetWeights() []*mat.Dense   { return []*mat.Dense{e.weights} }
func (e *Embedding) GetGradients() []*mat.Dense { return []*mat.Dense{e.gradWeights} }

// ============ LSTM Layer ============

type LSTM struct {
	BaseLayer
	units       int
	returnSeq   bool
	weights     []*mat.Dense // Wi, Wf, Wo, Wc, Ui, Uf, Uo, Uc
	biases      []*mat.Dense // bi, bf, bo, bc
	gradWeights []*mat.Dense
	gradBiases  []*mat.Dense
}

func NewLSTM(units int, options ...func(*LSTM)) *LSTM {
	layer := &LSTM{
		units:     units,
		returnSeq: false,
	}
	layer.device = &CPUDevice{}
	layer.training = true

	for _, opt := range options {
		opt(layer)
	}

	return layer
}

func WithReturnSequences(returnSeq bool) func(*LSTM) {
	return func(l *LSTM) { l.returnSeq = returnSeq }
}

func (l *LSTM) Build(inputDim int) {
	// Initialize LSTM weights (simplified)
	std := math.Sqrt(2.0 / float64(inputDim+l.units))

	// Weight matrices for input, forget, output, and candidate gates
	l.weights = make([]*mat.Dense, 8)
	l.biases = make([]*mat.Dense, 4)

	for i := 0; i < 4; i++ {
		l.weights[i] = mat.NewDense(inputDim, l.units, randomNormal(inputDim*l.units, std)) // W matrices
		l.weights[i+4] = mat.NewDense(l.units, l.units, randomNormal(l.units*l.units, std)) // U matrices
		l.biases[i] = mat.NewDense(1, l.units, zeros(l.units))
	}
}

func (l *LSTM) GetOutputShape(inputShape []int) []int {
	if l.returnSeq {
		return []int{inputShape[0], inputShape[1], l.units}
	}
	return []int{inputShape[0], l.units}
}

func (l *LSTM) GetParams() int {
	if len(l.weights) == 0 {
		return 0
	}
	// 4 gates × (input_dim + units) × units + 4 × units (biases)
	inputDim := l.weights[0].RawMatrix().Rows
	return 4*(inputDim+l.units)*l.units + 4*l.units
}

func (l *LSTM) Forward(inputs ...*mat.Dense) *mat.Dense {
	x := inputs[0]
	batchSize, seqLen, inputDim := x.RawMatrix().Rows, 1, x.RawMatrix().Cols

	if len(l.weights) == 0 {
		l.Build(inputDim)
	}

	// Simplified LSTM forward pass
	// In practice, this would implement the full LSTM equations
	h := mat.NewDense(batchSize, l.units, nil)
	c := mat.NewDense(batchSize, l.units, nil)

	// Process each time step (simplified for demo)
	for t := 0; t < seqLen; t++ {
		// Extract input at time t
		xt := x // Simplified - should extract time slice

		// Compute gates (simplified)
		i_gate := mat.NewDense(batchSize, l.units, nil)
		i_gate.Mul(xt, l.weights[0]) // Input gate

		f_gate := mat.NewDense(batchSize, l.units, nil)
		f_gate.Mul(xt, l.weights[1]) // Forget gate

		o_gate := mat.NewDense(batchSize, l.units, nil)
		o_gate.Mul(xt, l.weights[2]) // Output gate

		c_candidate := mat.NewDense(batchSize, l.units, nil)
		c_candidate.Mul(xt, l.weights[3]) // Candidate values

		// Apply activations (simplified)
		i_gate.Apply(func(i, j int, v float64) float64 {
			return 1.0 / (1.0 + math.Exp(-v)) // Sigmoid
		}, i_gate)

		f_gate.Apply(func(i, j int, v float64) float64 {
			return 1.0 / (1.0 + math.Exp(-v)) // Sigmoid
		}, f_gate)

		o_gate.Apply(func(i, j int, v float64) float64 {
			return 1.0 / (1.0 + math.Exp(-v)) // Sigmoid
		}, o_gate)

		c_candidate.Apply(func(i, j int, v float64) float64 {
			return math.Tanh(v)
		}, c_candidate)

		// Update cell state and hidden state
		c.MulElem(c, f_gate)
		temp := mat.NewDense(batchSize, l.units, nil)
		temp.MulElem(i_gate, c_candidate)
		c.Add(c, temp)

		h.Apply(func(i, j int, v float64) float64 {
			return o_gate.At(i, j) * math.Tanh(c.At(i, j))
		}, h)
	}

	if l.returnSeq {
		// Should return sequence of hidden states
		return h
	}

	return h // Return final hidden state
}

func (l *LSTM) Backward(grad *mat.Dense) []*mat.Dense {
	// Simplified LSTM backward pass
	rows, cols := grad.Dims()
	inputGrad := mat.NewDense(rows, cols, nil)

	// Initialize gradient matrices
	l.gradWeights = make([]*mat.Dense, len(l.weights))
	for i := range l.gradWeights {
		r, c := l.weights[i].Dims()
		l.gradWeights[i] = mat.NewDense(r, c, nil)
	}

	l.gradBiases = make([]*mat.Dense, len(l.biases))
	for i := range l.gradBiases {
		r, c := l.biases[i].Dims()
		l.gradBiases[i] = mat.NewDense(r, c, nil)
	}

	return []*mat.Dense{inputGrad}
}

func (l *LSTM) IsTrainable() bool { return true }
func (l *LSTM) GetWeights() []*mat.Dense {
	weights := make([]*mat.Dense, len(l.weights)+len(l.biases))
	copy(weights[:len(l.weights)], l.weights)
	copy(weights[len(l.weights):], l.biases)
	return weights
}
func (l *LSTM) GetGradients() []*mat.Dense {
	grads := make([]*mat.Dense, len(l.gradWeights)+len(l.gradBiases))
	copy(grads[:len(l.gradWeights)], l.gradWeights)
	copy(grads[len(l.gradWeights):], l.gradBiases)
	return grads
}

// ============ Batch Normalization ============

type BatchNorm struct {
	BaseLayer
	epsilon     float64
	momentum    float64
	gamma       *mat.Dense
	beta        *mat.Dense
	runningMean *mat.Dense
	runningVar  *mat.Dense
	gradGamma   *mat.Dense
	gradBeta    *mat.Dense
}

func NewBatchNorm(options ...func(*BatchNorm)) *BatchNorm {
	layer := &BatchNorm{
		epsilon:  1e-5,
		momentum: 0.9,
	}
	layer.device = &CPUDevice{}
	layer.training = true

	for _, opt := range options {
		opt(layer)
	}

	return layer
}

func WithBNMomentum(momentum float64) func(*BatchNorm) {
	return func(bn *BatchNorm) { bn.momentum = momentum }
}

func (bn *BatchNorm) Build(numFeatures int) {
	bn.gamma = mat.NewDense(1, numFeatures, nil)
	bn.beta = mat.NewDense(1, numFeatures, nil)
	bn.runningMean = mat.NewDense(1, numFeatures, nil)
	bn.runningVar = mat.NewDense(1, numFeatures, nil)

	// Initialize gamma to 1, beta to 0
	for i := 0; i < numFeatures; i++ {
		bn.gamma.Set(0, i, 1.0)
		bn.beta.Set(0, i, 0.0)
		bn.runningVar.Set(0, i, 1.0)
	}
}

func (bn *BatchNorm) GetOutputShape(inputShape []int) []int {
	return inputShape
}

func (bn *BatchNorm) GetParams() int {
	if bn.gamma == nil {
		return 0
	}
	_, cols := bn.gamma.Dims()
	return 2 * cols // gamma + beta
}

func (bn *BatchNorm) Forward(inputs ...*mat.Dense) *mat.Dense {
	x := inputs[0]
	rows, cols := x.Dims()

	if bn.gamma == nil {
		bn.Build(cols)
	}

	output := mat.NewDense(rows, cols, nil)

	if bn.training {
		// Compute batch statistics
		mean := mat.NewDense(1, cols, nil)
		variance := mat.NewDense(1, cols, nil)

		// Compute mean
		for j := 0; j < cols; j++ {
			col := mat.Col(nil, j, x)
			mean.Set(0, j, floats.Sum(col)/float64(rows))
		}

		// Compute variance
		for j := 0; j < cols; j++ {
			var var_sum float64
			for i := 0; i < rows; i++ {
				diff := x.At(i, j) - mean.At(0, j)
				var_sum += diff * diff
			}
			variance.Set(0, j, var_sum/float64(rows))
		}

		// Normalize
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				normalized := (x.At(i, j) - mean.At(0, j)) / math.Sqrt(variance.At(0, j)+bn.epsilon)
				scaled := normalized*bn.gamma.At(0, j) + bn.beta.At(0, j)
				output.Set(i, j, scaled)
			}
		}

		// Update running statistics
		for j := 0; j < cols; j++ {
			oldMean := bn.runningMean.At(0, j)
			oldVar := bn.runningVar.At(0, j)
			bn.runningMean.Set(0, j, bn.momentum*oldMean+(1-bn.momentum)*mean.At(0, j))
			bn.runningVar.Set(0, j, bn.momentum*oldVar+(1-bn.momentum)*variance.At(0, j))
		}
	} else {
		// Use running statistics for inference
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				normalized := (x.At(i, j) - bn.runningMean.At(0, j)) / math.Sqrt(bn.runningVar.At(0, j)+bn.epsilon)
				scaled := normalized*bn.gamma.At(0, j) + bn.beta.At(0, j)
				output.Set(i, j, scaled)
			}
		}
	}

	return output
}

func (bn *BatchNorm) Backward(grad *mat.Dense) []*mat.Dense {
	rows, cols := grad.Dims()

	// Initialize gradients
	bn.gradGamma = mat.NewDense(1, cols, nil)
	bn.gradBeta = mat.NewDense(1, cols, nil)
	inputGrad := mat.NewDense(rows, cols, nil)

	// Simplified gradient computation
	// In practice, this would implement the full batch norm backward pass
	inputGrad.Copy(grad)

	return []*mat.Dense{inputGrad}
}

func (bn *BatchNorm) IsTrainable() bool          { return true }
func (bn *BatchNorm) GetWeights() []*mat.Dense   { return []*mat.Dense{bn.gamma, bn.beta} }
func (bn *BatchNorm) GetGradients() []*mat.Dense { return []*mat.Dense{bn.gradGamma, bn.gradBeta} }

// ============ Enhanced Dropout Layer ============

type Dropout struct {
	BaseLayer
	rate     float64
	mask     *mat.Dense
	seedLock sync.Mutex
}

func NewDropout(rate float64) *Dropout {
	return &Dropout{
		rate: rate,
	}
}

func (d *Dropout) GetOutputShape(inputShape []int) []int {
	return inputShape
}

func (d *Dropout) Forward(inputs ...*mat.Dense) *mat.Dense {
	x := inputs[0]
	rows, cols := x.Dims()

	if !d.training || d.rate == 0 {
		return mat.DenseCopyOf(x)
	}

	// Thread-safe random number generation
	d.seedLock.Lock()
	defer d.seedLock.Unlock()

	// Generate dropout mask
	maskData := make([]float64, rows*cols)
	scale := 1.0 / (1.0 - d.rate)
	for i := range maskData {
		if rand.Float64() < d.rate {
			maskData[i] = 0
		} else {
			maskData[i] = scale
		}
	}
	d.mask = mat.NewDense(rows, cols, maskData)

	output := mat.NewDense(rows, cols, nil)
	output.MulElem(x, d.mask)
	return output
}

func (d *Dropout) Backward(grad *mat.Dense) []*mat.Dense {
	if !d.training || d.rate == 0 {
		return []*mat.Dense{grad}
	}

	rows, cols := grad.Dims()
	gradInput := mat.NewDense(rows, cols, nil)
	gradInput.MulElem(grad, d.mask)
	return []*mat.Dense{gradInput}
}

func (d *Dropout) IsTrainable() bool          { return false }
func (d *Dropout) GetWeights() []*mat.Dense   { return nil }
func (d *Dropout) GetGradients() []*mat.Dense { return nil }

// ============ Enhanced Optimizers ============

type Optimizer interface {
	Update(param *mat.Dense, grad *mat.Dense)
	Step()
	GetName() string
	GetLR() float64
	SetLR(lr float64)
	GetState() map[string]interface{}
	LoadState(state map[string]interface{})
}

type Adam struct {
	lr, beta1, beta2, epsilon float64
	m, v                      map[uintptr]*mat.Dense
	t                         int
	weightDecay               float64
	amsgrad                   bool
	vMax                      map[uintptr]*mat.Dense
}

func NewAdam(lr float64, options ...func(*Adam)) *Adam {
	adam := &Adam{
		lr:      lr,
		beta1:   0.9,
		beta2:   0.999,
		epsilon: 1e-8,
		m:       make(map[uintptr]*mat.Dense),
		v:       make(map[uintptr]*mat.Dense),
		vMax:    make(map[uintptr]*mat.Dense),
	}

	for _, opt := range options {
		opt(adam)
	}

	return adam
}

func WithBetas(beta1, beta2 float64) func(*Adam) {
	return func(a *Adam) {
		a.beta1 = beta1
		a.beta2 = beta2
	}
}

func WithWeightDecay(wd float64) func(*Adam) {
	return func(a *Adam) { a.weightDecay = wd }
}

func WithAMSGrad(amsgrad bool) func(*Adam) {
	return func(a *Adam) { a.amsgrad = amsgrad }
}

func (a *Adam) GetName() string  { return "adam" }
func (a *Adam) GetLR() float64   { return a.lr }
func (a *Adam) SetLR(lr float64) { a.lr = lr }
func (a *Adam) Step()            { a.t++ }

func (a *Adam) Update(param *mat.Dense, grad *mat.Dense) {
	rows, cols := param.Dims()
	addr := uintptr(unsafe.Pointer(&param.RawMatrix().Data[0]))

	if _, exists := a.m[addr]; !exists {
		a.m[addr] = mat.NewDense(rows, cols, nil)
		a.v[addr] = mat.NewDense(rows, cols, nil)
		if a.amsgrad {
			a.vMax[addr] = mat.NewDense(rows, cols, nil)
		}
	}

	m, v := a.m[addr], a.v[addr]

	// Apply weight decay if specified
	if a.weightDecay > 0 {
		decayGrad := mat.NewDense(rows, cols, nil)
		decayGrad.Scale(a.weightDecay, param)
		grad.Add(grad, decayGrad)
	}

	// Update biased first moment estimate
	m.Scale(a.beta1, m)
	gradScaled := mat.NewDense(rows, cols, nil)
	gradScaled.Scale(1-a.beta1, grad)
	m.Add(m, gradScaled)

	// Update biased second moment estimate
	v.Scale(a.beta2, v)
	gradSq := mat.NewDense(rows, cols, nil)
	gradSq.MulElem(grad, grad)
	gradSq.Scale(1-a.beta2, gradSq)
	v.Add(v, gradSq)

	// Bias correction
	tFloat := float64(a.t + 1)
	mHat := mat.NewDense(rows, cols, nil)
	mHat.Scale(1/(1-math.Pow(a.beta1, tFloat)), m)

	var vHat *mat.Dense
	if a.amsgrad {
		vMax := a.vMax[addr]
		vMax.Apply(func(i, j int, oldVal float64) float64 {
			return math.Max(oldVal, v.At(i, j))
		}, vMax)
		vHat = mat.NewDense(rows, cols, nil)
		vHat.Scale(1/(1-math.Pow(a.beta2, tFloat)), vMax)
	} else {
		vHat = mat.NewDense(rows, cols, nil)
		vHat.Scale(1/(1-math.Pow(a.beta2, tFloat)), v)
	}

	// Parameter update
	update := mat.NewDense(rows, cols, nil)
	update.Apply(func(i, j int, val float64) float64 {
		return a.lr * mHat.At(i, j) / (math.Sqrt(vHat.At(i, j)) + a.epsilon)
	}, update)

	param.Sub(param, update)
}

func (a *Adam) GetState() map[string]interface{} {
	return map[string]interface{}{
		"lr":           a.lr,
		"beta1":        a.beta1,
		"beta2":        a.beta2,
		"epsilon":      a.epsilon,
		"t":            a.t,
		"weight_decay": a.weightDecay,
		"amsgrad":      a.amsgrad,
	}
}

func (a *Adam) LoadState(state map[string]interface{}) {
	if lr, ok := state["lr"].(float64); ok {
		a.lr = lr
	}
	if beta1, ok := state["beta1"].(float64); ok {
		a.beta1 = beta1
	}
	if beta2, ok := state["beta2"].(float64); ok {
		a.beta2 = beta2
	}
	if eps, ok := state["epsilon"].(float64); ok {
		a.epsilon = eps
	}
	if t, ok := state["t"].(int); ok {
		a.t = t
	}
	if wd, ok := state["weight_decay"].(float64); ok {
		a.weightDecay = wd
	}
	if ams, ok := state["amsgrad"].(bool); ok {
		a.amsgrad = ams
	}
}

type AdamW struct {
	*Adam
}

func NewAdamW(lr, weightDecay float64, options ...func(*Adam)) *AdamW {
	adam := NewAdam(lr, options...)
	adam.weightDecay = weightDecay
	return &AdamW{Adam: adam}
}

func (aw *AdamW) GetName() string { return "adamw" }

// Enhanced SGD with more features
type SGD struct {
	lr, momentum, dampening, weightDecay float64
	nesterov                             bool
	velocities                           map[uintptr]*mat.Dense
}

func NewSGD(lr float64, options ...func(*SGD)) *SGD {
	sgd := &SGD{
		lr:          lr,
		momentum:    0,
		dampening:   0,
		weightDecay: 0,
		nesterov:    false,
		velocities:  make(map[uintptr]*mat.Dense),
	}

	for _, opt := range options {
		opt(sgd)
	}

	return sgd
}

func WithMomentum(momentum float64) func(*SGD) {
	return func(s *SGD) { s.momentum = momentum }
}

func WithDampening(dampening float64) func(*SGD) {
	return func(s *SGD) { s.dampening = dampening }
}

func WithSGDWeightDecay(wd float64) func(*SGD) {
	return func(s *SGD) { s.weightDecay = wd }
}

func WithNesterov(nesterov bool) func(*SGD) {
	return func(s *SGD) { s.nesterov = nesterov }
}

func (s *SGD) GetName() string  { return "sgd" }
func (s *SGD) GetLR() float64   { return s.lr }
func (s *SGD) SetLR(lr float64) { s.lr = lr }
func (s *SGD) Step()            {}

func (s *SGD) Update(param *mat.Dense, grad *mat.Dense) {
	rows, cols := param.Dims()
	addr := uintptr(unsafe.Pointer(&param.RawMatrix().Data[0]))

	// Apply weight decay
	if s.weightDecay > 0 {
		decayGrad := mat.NewDense(rows, cols, nil)
		decayGrad.Scale(s.weightDecay, param)
		grad.Add(grad, decayGrad)
	}

	if s.momentum > 0 {
		if _, exists := s.velocities[addr]; !exists {
			s.velocities[addr] = mat.NewDense(rows, cols, nil)
		}

		velocity := s.velocities[addr]
		velocity.Scale(s.momentum, velocity)
		gradScaled := mat.NewDense(rows, cols, nil)
		gradScaled.Scale(1-s.dampening, grad)
		velocity.Add(velocity, gradScaled)

		if s.nesterov {
			update := mat.NewDense(rows, cols, nil)
			update.Scale(s.momentum, velocity)
			update.Add(update, grad)
			update.Scale(s.lr, update)
			param.Sub(param, update)
		} else {
			update := mat.NewDense(rows, cols, nil)
			update.Scale(s.lr, velocity)
			param.Sub(param, update)
		}
	} else {
		update := mat.NewDense(rows, cols, nil)
		update.Scale(s.lr, grad)
		param.Sub(param, update)
	}
}

func (s *SGD) GetState() map[string]interface{} {
	return map[string]interface{}{
		"lr":           s.lr,
		"momentum":     s.momentum,
		"dampening":    s.dampening,
		"weight_decay": s.weightDecay,
		"nesterov":     s.nesterov,
	}
}

func (s *SGD) LoadState(state map[string]interface{}) {
	if lr, ok := state["lr"].(float64); ok {
		s.lr = lr
	}
	if mom, ok := state["momentum"].(float64); ok {
		s.momentum = mom
	}
	if damp, ok := state["dampening"].(float64); ok {
		s.dampening = damp
	}
	if wd, ok := state["weight_decay"].(float64); ok {
		s.weightDecay = wd
	}
	if nest, ok := state["nesterov"].(bool); ok {
		s.nesterov = nest
	}
}

// ============ Learning Rate Schedulers ============

type LRScheduler interface {
	Step() float64
	GetLR() float64
	SetOptimizer(optimizer Optimizer)
}

type StepLR struct {
	optimizer Optimizer
	stepSize  int
	gamma     float64
	lastEpoch int
	baseLR    float64
}

func NewStepLR(optimizer Optimizer, stepSize int, gamma float64) *StepLR {
	return &StepLR{
		optimizer: optimizer,
		stepSize:  stepSize,
		gamma:     gamma,
		lastEpoch: 0,
		baseLR:    optimizer.GetLR(),
	}
}

func (s *StepLR) SetOptimizer(optimizer Optimizer) {
	s.optimizer = optimizer
	s.baseLR = optimizer.GetLR()
}

func (s *StepLR) Step() float64 {
	s.lastEpoch++
	newLR := s.baseLR * math.Pow(s.gamma, float64(s.lastEpoch/s.stepSize))
	s.optimizer.SetLR(newLR)
	return newLR
}

func (s *StepLR) GetLR() float64 {
	return s.optimizer.GetLR()
}

type CosineAnnealingLR struct {
	optimizer Optimizer
	tMax      int
	etaMin    float64
	lastEpoch int
	baseLR    float64
}

func NewCosineAnnealingLR(optimizer Optimizer, tMax int, etaMin float64) *CosineAnnealingLR {
	return &CosineAnnealingLR{
		optimizer: optimizer,
		tMax:      tMax,
		etaMin:    etaMin,
		lastEpoch: 0,
		baseLR:    optimizer.GetLR(),
	}
}

func (c *CosineAnnealingLR) SetOptimizer(optimizer Optimizer) {
	c.optimizer = optimizer
	c.baseLR = optimizer.GetLR()
}

func (c *CosineAnnealingLR) Step() float64 {
	c.lastEpoch++
	newLR := c.etaMin + (c.baseLR-c.etaMin)*(1+math.Cos(math.Pi*float64(c.lastEpoch)/float64(c.tMax)))/2
	c.optimizer.SetLR(newLR)
	return newLR
}

func (c *CosineAnnealingLR) GetLR() float64 {
	return c.optimizer.GetLR()
}

// ============ Data Loading and Preprocessing ============

type Dataset interface {
	Len() int
	GetItem(index int) (*mat.Dense, *mat.Dense)
	GetBatch(indices []int) (*mat.Dense, *mat.Dense)
}

type TensorDataset struct {
	features *mat.Dense
	labels   *mat.Dense
}

func NewTensorDataset(features, labels *mat.Dense) *TensorDataset {
	return &TensorDataset{
		features: features,
		labels:   labels,
	}
}

func (td *TensorDataset) Len() int {
	rows, _ := td.features.Dims()
	return rows
}

func (td *TensorDataset) GetItem(index int) (*mat.Dense, *mat.Dense) {
	_, featCols := td.features.Dims()
	_, labelCols := td.labels.Dims()

	feature := mat.NewDense(1, featCols, nil)
	label := mat.NewDense(1, labelCols, nil)

	for j := 0; j < featCols; j++ {
		feature.Set(0, j, td.features.At(index, j))
	}

	for j := 0; j < labelCols; j++ {
		label.Set(0, j, td.labels.At(index, j))
	}

	return feature, label
}

func (td *TensorDataset) GetBatch(indices []int) (*mat.Dense, *mat.Dense) {
	_, featCols := td.features.Dims()
	_, labelCols := td.labels.Dims()

	features := mat.NewDense(len(indices), featCols, nil)
	labels := mat.NewDense(len(indices), labelCols, nil)

	for i, idx := range indices {
		for j := 0; j < featCols; j++ {
			features.Set(i, j, td.features.At(idx, j))
		}
		for j := 0; j < labelCols; j++ {
			labels.Set(i, j, td.labels.At(idx, j))
		}
	}

	return features, labels
}

type DataLoader struct {
	dataset    Dataset
	batchSize  int
	shuffle    bool
	numWorkers int
	indices    []int
	currentIdx int
}

func NewDataLoader(dataset Dataset, batchSize int, options ...func(*DataLoader)) *DataLoader {
	dl := &DataLoader{
		dataset:    dataset,
		batchSize:  batchSize,
		shuffle:    true,
		numWorkers: 1,
		indices:    make([]int, dataset.Len()),
	}

	for i := range dl.indices {
		dl.indices[i] = i
	}

	for _, opt := range options {
		opt(dl)
	}

	if dl.shuffle {
		rand.Shuffle(len(dl.indices), func(i, j int) {
			dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
		})
	}

	return dl
}

func WithShuffle(shuffle bool) func(*DataLoader) {
	return func(dl *DataLoader) { dl.shuffle = shuffle }
}

func WithNumWorkers(numWorkers int) func(*DataLoader) {
	return func(dl *DataLoader) { dl.numWorkers = numWorkers }
}

func (dl *DataLoader) Reset() {
	dl.currentIdx = 0
	if dl.shuffle {
		rand.Shuffle(len(dl.indices), func(i, j int) {
			dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
		})
	}
}

func (dl *DataLoader) HasNext() bool {
	return dl.currentIdx < len(dl.indices)
}

func (dl *DataLoader) NextBatch() (*mat.Dense, *mat.Dense) {
	if !dl.HasNext() {
		return nil, nil
	}

	endIdx := dl.currentIdx + dl.batchSize
	if endIdx > len(dl.indices) {
		endIdx = len(dl.indices)
	}

	batchIndices := dl.indices[dl.currentIdx:endIdx]
	dl.currentIdx = endIdx

	return dl.dataset.GetBatch(batchIndices)
}

// ============ Data Transformations ============

type Transform interface {
	Apply(data *mat.Dense) *mat.Dense
}

type Normalize struct {
	mean []float64
	std  []float64
}

func NewNormalize(mean, std []float64) *Normalize {
	return &Normalize{mean: mean, std: std}
}

func (n *Normalize) Apply(data *mat.Dense) *mat.Dense {
	rows, cols := data.Dims()
	result := mat.NewDense(rows, cols, nil)

	result.Apply(func(i, j int, v float64) float64 {
		if j < len(n.mean) && j < len(n.std) {
			return (v - n.mean[j]) / n.std[j]
		}
		return v
	}, data)

	return result
}

type RandomNoise struct {
	std float64
}

func NewRandomNoise(std float64) *RandomNoise {
	return &RandomNoise{std: std}
}

func (rn *RandomNoise) Apply(data *mat.Dense) *mat.Dense {
	rows, cols := data.Dims()
	result := mat.NewDense(rows, cols, nil)

	result.Apply(func(i, j int, v float64) float64 {
		return v + rand.NormFloat64()*rn.std
	}, data)

	return result
}

// ============ Model Flexibility - Multi-Input/Output ============

type ModuleDict map[string]Layer

type FunctionalModel struct {
	modules     ModuleDict
	forward     func(ModuleDict, map[string]*mat.Dense) map[string]*mat.Dense
	inputNames  []string
	outputNames []string
	optimizer   Optimizer
	loss        LossFunction
	compiled    bool
	history     *History
	device      Device
}

func NewFunctionalModel() *FunctionalModel {
	return &FunctionalModel{
		modules: make(ModuleDict),
		history: NewHistory(),
		device:  &CPUDevice{},
	}
}

func (fm *FunctionalModel) AddModule(name string, layer Layer) {
	fm.modules[name] = layer
	layer.SetDevice(fm.device)
}

func (fm *FunctionalModel) SetForward(forwardFn func(ModuleDict, map[string]*mat.Dense) map[string]*mat.Dense) {
	fm.forward = forwardFn
}

func (fm *FunctionalModel) SetInputNames(names []string) {
	fm.inputNames = names
}

func (fm *FunctionalModel) SetOutputNames(names []string) {
	fm.outputNames = names
}

func (fm *FunctionalModel) SetDevice(device Device) {
	fm.device = device
	for _, layer := range fm.modules {
		layer.SetDevice(device)
	}
}

func (fm *FunctionalModel) Compile(optimizer Optimizer, loss LossFunction) {
	fm.optimizer = optimizer
	fm.loss = loss
	fm.compiled = true
}

func (fm *FunctionalModel) Forward(inputs map[string]*mat.Dense) map[string]*mat.Dense {
	if fm.forward == nil {
		panic("Forward function not defined")
	}
	return fm.forward(fm.modules, inputs)
}

func (fm *FunctionalModel) Train(training bool) {
	for _, layer := range fm.modules {
		layer.Train(training)
	}
}

// Example usage for ResNet-style skip connections
func CreateResNetBlock() *FunctionalModel {
	model := NewFunctionalModel()

	model.AddModule("conv1", NewConv2D(64, []int{3, 3}, WithConvActivation(&ReLU{})))
	model.AddModule("bn1", NewBatchNorm())
	model.AddModule("conv2", NewConv2D(64, []int{3, 3}))
	model.AddModule("bn2", NewBatchNorm())

	model.SetForward(func(modules ModuleDict, inputs map[string]*mat.Dense) map[string]*mat.Dense {
		x := inputs["input"]

		// First conv block
		out := modules["conv1"].Forward(x)
		out = modules["bn1"].Forward(out)

		// Second conv block
		out = modules["conv2"].Forward(out)
		out = modules["bn2"].Forward(out)

		// Skip connection (element-wise addition)
		// In practice, this would handle dimension matching
		rows, cols := x.Dims()
		outRows, outCols := out.Dims()

		if rows == outRows && cols == outCols {
			result := mat.NewDense(rows, cols, nil)
			result.Add(x, out)
			out = result
		}

		// Apply ReLU activation
		out.Apply(func(i, j int, v float64) float64 {
			return math.Max(0, v)
		}, out)

		return map[string]*mat.Dense{"output": out}
	})

	return model
}

// ============ Enhanced Sequential Model ============

type Model struct {
	layers    []Layer
	optimizer Optimizer
	loss      LossFunction
	compiled  bool
	history   *History
	device    Device
	scheduler LRScheduler
	clipGrad  float64
}

func NewModel() *Model {
	return &Model{
		layers:  make([]Layer, 0),
		history: NewHistory(),
		device:  &CPUDevice{},
	}
}

func (m *Model) Add(layer Layer) {
	m.layers = append(m.layers, layer)
	layer.SetDevice(m.device)
}

func (m *Model) SetDevice(device Device) {
	m.device = device
	for _, layer := range m.layers {
		layer.SetDevice(device)
	}
}

func (m *Model) Compile(optimizer Optimizer, loss LossFunction, options ...func(*Model)) {
	m.optimizer = optimizer
	m.loss = loss
	m.compiled = true

	for _, opt := range options {
		opt(m)
	}
}

func WithLRScheduler(scheduler LRScheduler) func(*Model) {
	return func(m *Model) {
		m.scheduler = scheduler
		scheduler.SetOptimizer(m.optimizer)
	}
}

func WithGradClipping(clipValue float64) func(*Model) {
	return func(m *Model) { m.clipGrad = clipValue }
}

func (m *Model) Summary(inputShape []int) {
	fmt.Printf("Model: Sequential (%s)\n", m.device.GetType())
	fmt.Println("_________________________________________________________________")
	fmt.Printf("%-25s %-20s %-15s\n", "Layer (type)", "Output Shape", "Param #")
	fmt.Println("=================================================================")

	totalParams := 0
	trainableParams := 0
	currentShape := inputShape

	for i, layer := range m.layers {
		// Build layer if it's a Dense layer that hasn't been built
		if dense, ok := layer.(*Dense); ok && dense.weights == nil {
			if len(currentShape) > 1 {
				dense.Build(currentShape[1])
			}
		}

		layerName := fmt.Sprintf("%s_%d", layer.GetName(), i)
		layerType := fmt.Sprintf("(%s)", strings.Split(fmt.Sprintf("%T", layer), ".")[1])
		fullLayerName := fmt.Sprintf("%s %s", layerName, layerType)

		params := layer.GetParams()
		totalParams += params
		if layer.IsTrainable() {
			trainableParams += params
		}

		outputShape := layer.GetOutputShape(currentShape)
		shapeStr := fmt.Sprintf("(%s)", formatShape(outputShape))

		fmt.Printf("%-25s %-20s %-15d\n", fullLayerName, shapeStr, params)
		currentShape = outputShape
	}

	fmt.Println("=================================================================")
	fmt.Printf("Total params: %d\n", totalParams)
	fmt.Printf("Trainable params: %d\n", trainableParams)
	fmt.Printf("Non-trainable params: %d\n", totalParams-trainableParams)
	fmt.Println("_________________________________________________________________")
}

func (m *Model) FitGenerator(trainLoader *DataLoader, config *Config, validationLoader *DataLoader, callbacks ...Callback) *History {
	if !m.compiled {
		panic("Model must be compiled before fitting")
	}

	rand.Seed(config.Seed)

	// Create checkpoint directory if needed
	if config.SaveCheckpoints {
		os.MkdirAll(config.CheckpointDir, 0755)
	}

	// Initialize callbacks
	for _, cb := range callbacks {
		cb.OnTrainBegin()
	}

	bestLoss := math.Inf(1)
	patience := 0

	for epoch := 0; epoch < config.Epochs; epoch++ {
		start := time.Now()

		for _, cb := range callbacks {
			cb.OnEpochBegin(epoch)
		}

		// Training phase
		m.setTrainingMode(true)
		trainLoader.Reset()

		epochLoss := 0.0
		epochAccuracy := 0.0
		numBatches := 0

		for trainLoader.HasNext() {
			xBatch, yBatch := trainLoader.NextBatch()
			if xBatch == nil {
				break
			}

			// Forward pass
			yPred := m.predict(xBatch)
			loss := m.loss.Compute(yBatch, yPred)
			accuracy := m.computeAccuracy(yBatch, yPred)

			// Backward pass
			grad := m.loss.ComputeGrad(yBatch, yPred)
			m.backward(grad)

			// Gradient clipping
			if m.clipGrad > 0 {
				m.clipGradients(m.clipGrad)
			}

			m.updateWeights()

			epochLoss += loss
			epochAccuracy += accuracy
			numBatches++
		}

		// Average metrics over batches
		if numBatches > 0 {
			epochLoss /= float64(numBatches)
			epochAccuracy /= float64(numBatches)
		}

		m.history.Loss = append(m.history.Loss, epochLoss)
		m.history.Accuracy = append(m.history.Accuracy, epochAccuracy)
		m.history.Epochs = append(m.history.Epochs, epoch+1)

		// Validation phase
		var valLoss, valAccuracy float64
		if validationLoader != nil {
			m.setTrainingMode(false)
			validationLoader.Reset()

			valLoss, valAccuracy = m.evaluateGenerator(validationLoader)
			m.history.ValLoss = append(m.history.ValLoss, valLoss)
			m.history.ValAccuracy = append(m.history.ValAccuracy, valAccuracy)
		}

		// Learning rate scheduling
		if m.scheduler != nil {
			m.scheduler.Step()
		}

		elapsed := time.Since(start)

		// Logging
		if config.Verbose > 0 && (epoch+1)%config.Verbose == 0 {
			logMsg := fmt.Sprintf("Epoch %d/%d [%.2fs] - loss: %.4f - accuracy: %.4f",
				epoch+1, config.Epochs, elapsed.Seconds(), epochLoss, epochAccuracy)

			if validationLoader != nil {
				logMsg += fmt.Sprintf(" - val_loss: %.4f - val_accuracy: %.4f", valLoss, valAccuracy)
			}

			if m.scheduler != nil {
				logMsg += fmt.Sprintf(" - lr: %.6f", m.optimizer.GetLR())
			}

			fmt.Println(logMsg)
		}

		// Early stopping
		currentLoss := epochLoss
		if validationLoader != nil {
			currentLoss = valLoss
		}

		if config.EarlyStopping {
			if currentLoss < bestLoss {
				bestLoss = currentLoss
				patience = 0

				// Save best model checkpoint
				if config.SaveCheckpoints {
					m.SaveCheckpoint(filepath.Join(config.CheckpointDir, "best_model.json"))
				}
			} else {
				patience++
				if patience >= config.Patience {
					fmt.Printf("Early stopping at epoch %d\n", epoch+1)
					break
				}
			}
		}

		// Save periodic checkpoints
		if config.SaveCheckpoints && (epoch+1)%10 == 0 {
			filename := fmt.Sprintf("checkpoint_epoch_%d.json", epoch+1)
			m.SaveCheckpoint(filepath.Join(config.CheckpointDir, filename))
		}

		for _, cb := range callbacks {
			cb.OnEpochEnd(epoch, config.Epochs, epochLoss)
		}
	}

	for _, cb := range callbacks {
		cb.OnTrainEnd()
	}

	return m.history
}

func (m *Model) clipGradients(maxNorm float64) {
	var totalNorm float64

	// Calculate total gradient norm
	for _, layer := range m.layers {
		if layer.IsTrainable() {
			grads := layer.GetGradients()
			for _, grad := range grads {
				if grad != nil {
					rows, cols := grad.Dims()
					for i := 0; i < rows; i++ {
						for j := 0; j < cols; j++ {
							val := grad.At(i, j)
							totalNorm += val * val
						}
					}
				}
			}
		}
	}

	totalNorm = math.Sqrt(totalNorm)

	// Clip gradients if necessary
	if totalNorm > maxNorm {
		clipCoeff := maxNorm / totalNorm
		for _, layer := range m.layers {
			if layer.IsTrainable() {
				grads := layer.GetGradients()
				for _, grad := range grads {
					if grad != nil {
						grad.Scale(clipCoeff, grad)
					}
				}
			}
		}
	}
}

func (m *Model) setTrainingMode(training bool) {
	for _, layer := range m.layers {
		layer.Train(training)
	}
}

func (m *Model) computeAccuracy(yTrue, yPred *mat.Dense) float64 {
	rows, cols := yTrue.Dims()
	correct := 0

	if cols == 1 {
		// Binary classification
		for i := 0; i < rows; i++ {
			pred := 0.0
			if yPred.At(i, 0) >= 0.5 {
				pred = 1.0
			}
			if pred == yTrue.At(i, 0) {
				correct++
			}
		}
	} else {
		// Multi-class classification
		for i := 0; i < rows; i++ {
			// Find predicted class (argmax)
			predClass := 0
			maxPred := yPred.At(i, 0)
			for j := 1; j < cols; j++ {
				if yPred.At(i, j) > maxPred {
					maxPred = yPred.At(i, j)
					predClass = j
				}
			}

			// Find true class (argmax)
			trueClass := 0
			maxTrue := yTrue.At(i, 0)
			for j := 1; j < cols; j++ {
				if yTrue.At(i, j) > maxTrue {
					maxTrue = yTrue.At(i, j)
					trueClass = j
				}
			}

			if predClass == trueClass {
				correct++
			}
		}
	}

	return float64(correct) / float64(rows)
}

// ============ Enhanced Callbacks ============

type Callback interface {
	OnTrainBegin()
	OnTrainEnd()
	OnEpochBegin(epoch int)
	OnEpochEnd(epoch, maxEpochs int, loss float64)
}

type BaseCallback struct{}

func (cb *BaseCallback) OnTrainBegin()                                 {}
func (cb *BaseCallback) OnTrainEnd()                                   {}
func (cb *BaseCallback) OnEpochBegin(epoch int)                        {}
func (cb *BaseCallback) OnEpochEnd(epoch, maxEpochs int, loss float64) {}

type VerboseCallback struct {
	BaseCallback
	verbose int
}

func NewVerboseCallback(verbose int) *VerboseCallback {
	return &VerboseCallback{verbose: verbose}
}

func (vc *VerboseCallback) OnTrainBegin() {
	if vc.verbose > 0 {
		fmt.Println("Training started...")
	}
}

func (vc *VerboseCallback) OnTrainEnd() {
	if vc.verbose > 0 {
		fmt.Println("Training completed!")
	}
}

func (vc *VerboseCallback) OnEpochEnd(epoch, maxEpochs int, loss float64) {
	if vc.verbose > 0 && (epoch+1)%vc.verbose == 0 {
		fmt.Printf("Epoch %d/%d - Loss: %.4f\n", epoch+1, maxEpochs, loss)
	}
}

type EarlyStoppingCallback struct {
	BaseCallback
	patience   int
	minDelta   float64
	bestLoss   float64
	waitCount  int
	shouldStop bool
}

func NewEarlyStoppingCallback(patience int, minDelta float64) *EarlyStoppingCallback {
	return &EarlyStoppingCallback{
		patience:   patience,
		minDelta:   minDelta,
		bestLoss:   math.Inf(1),
		waitCount:  0,
		shouldStop: false,
	}
}

func (esc *EarlyStoppingCallback) OnEpochEnd(epoch, maxEpochs int, loss float64) {
	if loss < esc.bestLoss-esc.minDelta {
		esc.bestLoss = loss
		esc.waitCount = 0
	} else {
		esc.waitCount++
		if esc.waitCount >= esc.patience {
			esc.shouldStop = true
			fmt.Printf("Early stopping triggered at epoch %d\n", epoch+1)
		}
	}
}

func (esc *EarlyStoppingCallback) ShouldStop() bool {
	return esc.shouldStop
}

type ModelCheckpointCallback struct {
	BaseCallback
	filepath   string
	monitor    string
	saveBest   bool
	bestMetric float64
}

func NewModelCheckpointCallback(filepath string, monitor string, saveBest bool) *ModelCheckpointCallback {
	return &ModelCheckpointCallback{
		filepath:   filepath,
		monitor:    monitor,
		saveBest:   saveBest,
		bestMetric: math.Inf(1),
	}
}

func (mcc *ModelCheckpointCallback) OnEpochEnd(epoch, maxEpochs int, loss float64) {
	if mcc.saveBest && loss < mcc.bestMetric {
		mcc.bestMetric = loss
		fmt.Printf("Saving model checkpoint at epoch %d\n", epoch+1)
		// In practice, would save the model here
	}
}

type TensorBoardCallback struct {
	BaseCallback
	logDir  string
	metrics []string
}

func NewTensorBoardCallback(logDir string) *TensorBoardCallback {
	return &TensorBoardCallback{
		logDir:  logDir,
		metrics: []string{"loss", "accuracy"},
	}
}

func (tbc *TensorBoardCallback) OnTrainBegin() {
	os.MkdirAll(tbc.logDir, 0755)
	fmt.Printf("TensorBoard logging to: %s\n", tbc.logDir)
}

func (tbc *TensorBoardCallback) OnEpochEnd(epoch, maxEpochs int, loss float64) {
	// In practice, would log metrics to TensorBoard format
	logFile := filepath.Join(tbc.logDir, fmt.Sprintf("epoch_%d.log", epoch+1))
	logData := fmt.Sprintf("epoch: %d, loss: %.4f\n", epoch+1, loss)
	ioutil.WriteFile(logFile, []byte(logData), 0644)
}

// ============ Data Processing (Enhanced) ============

type OneHotEncoder struct {
	classes    [][]string
	numClasses []int
	fitted     bool
}

func NewOneHotEncoder() *OneHotEncoder {
	return &OneHotEncoder{
		classes:    make([][]string, 0),
		numClasses: make([]int, 0),
		fitted:     false,
	}
}

func (ohe *OneHotEncoder) Fit(X [][]string) *OneHotEncoder {
	if len(X) == 0 {
		return ohe
	}

	numFeatures := len(X[0])
	ohe.classes = make([][]string, numFeatures)
	ohe.numClasses = make([]int, numFeatures)

	for j := 0; j < numFeatures; j++ {
		uniqueSet := make(map[string]bool)
		for i := 0; i < len(X); i++ {
			uniqueSet[X[i][j]] = true
		}

		classes := make([]string, 0, len(uniqueSet))
		for class := range uniqueSet {
			classes = append(classes, class)
		}
		sort.Strings(classes)

		ohe.classes[j] = classes
		ohe.numClasses[j] = len(classes)
	}

	ohe.fitted = true
	return ohe
}

func (ohe *OneHotEncoder) Transform(X [][]string) *mat.Dense {
	if !ohe.fitted {
		panic("OneHotEncoder must be fitted before transform")
	}

	if len(X) == 0 {
		return mat.NewDense(0, 0, nil)
	}

	totalCols := 0
	for _, numClass := range ohe.numClasses {
		totalCols += numClass
	}

	data := make([]float64, len(X)*totalCols)

	for i := 0; i < len(X); i++ {
		colOffset := 0
		for j := 0; j < len(X[0]); j++ {
			for k, class := range ohe.classes[j] {
				if X[i][j] == class {
					data[i*totalCols+colOffset+k] = 1.0
					break
				}
			}
			colOffset += ohe.numClasses[j]
		}
	}

	return mat.NewDense(len(X), totalCols, data)
}

func (ohe *OneHotEncoder) FitTransform(X [][]string) *mat.Dense {
	return ohe.Fit(X).Transform(X)
}

type LabelEncoder struct {
	classes []string
	fitted  bool
}

func NewLabelEncoder() *LabelEncoder {
	return &LabelEncoder{
		classes: make([]string, 0),
		fitted:  false,
	}
}

func (le *LabelEncoder) Fit(y []string) *LabelEncoder {
	uniqueSet := make(map[string]bool)
	for _, label := range y {
		uniqueSet[label] = true
	}

	le.classes = make([]string, 0, len(uniqueSet))
	for class := range uniqueSet {
		le.classes = append(le.classes, class)
	}
	sort.Strings(le.classes)

	le.fitted = true
	return le
}

func (le *LabelEncoder) Transform(y []string) []int {
	if !le.fitted {
		panic("LabelEncoder must be fitted before transform")
	}

	classMap := make(map[string]int)
	for i, class := range le.classes {
		classMap[class] = i
	}

	encoded := make([]int, len(y))
	for i, label := range y {
		if idx, exists := classMap[label]; exists {
			encoded[i] = idx
		} else {
			panic(fmt.Sprintf("Unknown label: %s", label))
		}
	}

	return encoded
}

func (le *LabelEncoder) FitTransform(y []string) []int {
	return le.Fit(y).Transform(y)
}

func (le *LabelEncoder) GetClasses() []string {
	return le.classes
}

func (le *LabelEncoder) GetNumClasses() int {
	return len(le.classes)
}

// ============ Data Utilities ============

func trainTestSplit(X, y *mat.Dense, testSize float64, randomState int64) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
	rand.Seed(randomState)

	nSamples, _ := X.Dims()
	indices := make([]int, nSamples)
	for i := 0; i < nSamples; i++ {
		indices[i] = i
	}

	rand.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	testSamples := int(float64(nSamples) * testSize)
	trainIndices := indices[testSamples:]
	testIndices := indices[:testSamples]

	XTrain := extractRows(X, trainIndices)
	XTest := extractRows(X, testIndices)
	yTrain := extractRows(y, trainIndices)
	yTest := extractRows(y, testIndices)

	return XTrain, XTest, yTrain, yTest
}

func extractRows(m *mat.Dense, indices []int) *mat.Dense {
	_, cols := m.Dims()
	data := make([]float64, len(indices)*cols)

	for i, idx := range indices {
		for j := 0; j < cols; j++ {
			data[i*cols+j] = m.At(idx, j)
		}
	}

	return mat.NewDense(len(indices), cols, data)
}

func extractRowRange(m *mat.Dense, start, end int) *mat.Dense {
	_, cols := m.Dims()
	batchSize := end - start
	data := make([]float64, batchSize*cols)

	for i := 0; i < batchSize; i++ {
		for j := 0; j < cols; j++ {
			data[i*cols+j] = m.At(start+i, j)
		}
	}

	return mat.NewDense(batchSize, cols, data)
}

type StandardScaler struct {
	mean   []float64
	std    []float64
	fitted bool
}

func NewStandardScaler() *StandardScaler {
	return &StandardScaler{fitted: false}
}

func (ss *StandardScaler) Fit(X *mat.Dense) *StandardScaler {
	rows, cols := X.Dims()
	ss.mean = make([]float64, cols)
	ss.std = make([]float64, cols)

	for j := 0; j < cols; j++ {
		col := mat.Col(nil, j, X)
		ss.mean[j] = floats.Sum(col) / float64(rows)
	}

	for j := 0; j < cols; j++ {
		var variance float64
		for i := 0; i < rows; i++ {
			diff := X.At(i, j) - ss.mean[j]
			variance += diff * diff
		}
		variance /= float64(rows)
		ss.std[j] = math.Sqrt(variance)

		if ss.std[j] == 0 {
			ss.std[j] = 1.0
		}
	}

	ss.fitted = true
	return ss
}

func (ss *StandardScaler) Transform(X *mat.Dense) *mat.Dense {
	if !ss.fitted {
		panic("StandardScaler must be fitted before transform")
	}

	rows, cols := X.Dims()
	scaled := mat.NewDense(rows, cols, nil)

	scaled.Apply(func(i, j int, v float64) float64 {
		return (v - ss.mean[j]) / ss.std[j]
	}, X)

	return scaled
}

func (ss *StandardScaler) FitTransform(X *mat.Dense) *mat.Dense {
	return ss.Fit(X).Transform(X)
}

// ============ Dataset Generators ============

func GenerateXORData(nSamples int, noise float64) (*mat.Dense, *mat.Dense) {
	data := make([]float64, nSamples*2)
	for i := 0; i < nSamples*2; i += 2 {
		data[i] = rand.Float64()*2 - 1
		data[i+1] = rand.Float64()*2 - 1
	}
	X := mat.NewDense(nSamples, 2, data)

	yData := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		x1 := X.At(i, 0) + noise*rand.NormFloat64()
		x2 := X.At(i, 1) + noise*rand.NormFloat64()

		if (x1 > 0 && x2 <= 0) || (x1 <= 0 && x2 > 0) {
			yData[i] = 1.0
		} else {
			yData[i] = 0.0
		}
	}
	y := mat.NewDense(nSamples, 1, yData)

	return X, y
}

func GenerateCirclesData(nSamples int, noise float64, factor float64) (*mat.Dense, *mat.Dense) {
	nSamplesOut := nSamples / 2
	nSamplesIn := nSamples - nSamplesOut

	data := make([]float64, nSamples*2)
	labels := make([]float64, nSamples)

	for i := 0; i < nSamplesOut; i++ {
		angle := rand.Float64() * 2 * math.Pi
		radius := 1.0 + noise*rand.NormFloat64()

		data[i*2] = radius * math.Cos(angle)
		data[i*2+1] = radius * math.Sin(angle)
		labels[i] = 0.0
	}

	for i := 0; i < nSamplesIn; i++ {
		idx := nSamplesOut + i
		angle := rand.Float64() * 2 * math.Pi
		radius := factor + noise*rand.NormFloat64()

		data[idx*2] = radius * math.Cos(angle)
		data[idx*2+1] = radius * math.Sin(angle)
		labels[idx] = 1.0
	}

	X := mat.NewDense(nSamples, 2, data)
	y := mat.NewDense(nSamples, 1, labels)

	return X, y
}

func GenerateBlobsData(nSamples, nFeatures, nCenters int, clusterStd float64) (*mat.Dense, *mat.Dense) {
	samplesPerCenter := nSamples / nCenters

	data := make([]float64, nSamples*nFeatures)
	labels := make([]float64, nSamples)

	centers := make([][]float64, nCenters)
	for i := 0; i < nCenters; i++ {
		centers[i] = make([]float64, nFeatures)
		for j := 0; j < nFeatures; j++ {
			centers[i][j] = (rand.NormFloat64() + float64(i)*3) * 5
		}
	}

	sampleIdx := 0
	for centerIdx := 0; centerIdx < nCenters; centerIdx++ {
		for s := 0; s < samplesPerCenter && sampleIdx < nSamples; s++ {
			for j := 0; j < nFeatures; j++ {
				data[sampleIdx*nFeatures+j] = centers[centerIdx][j] + rand.NormFloat64()*clusterStd
			}
			labels[sampleIdx] = float64(centerIdx)
			sampleIdx++
		}
	}

	X := mat.NewDense(nSamples, nFeatures, data)
	y := mat.NewDense(nSamples, 1, labels)

	return X, y
}

func GenerateSpiralData(nSamples, nClasses int, noise float64) (*mat.Dense, *mat.Dense) {
	data := make([]float64, nSamples*2)
	labels := make([]float64, nSamples)
	samplesPerClass := nSamples / nClasses

	for j := 0; j < nClasses; j++ {
		for i := 0; i < samplesPerClass; i++ {
			radius := float64(i) / float64(samplesPerClass)
			angle := float64(i)*4/float64(samplesPerClass) + float64(j)*4 + noise*rand.NormFloat64()

			idx := j*samplesPerClass + i
			data[idx*2] = radius * math.Sin(angle)
			data[idx*2+1] = radius * math.Cos(angle)
			labels[idx] = float64(j)
		}
	}

	X := mat.NewDense(nSamples, 2, data)
	y := mat.NewDense(nSamples, 1, labels)

	return X, y
}

// ============ Utility Functions ============

func ToOneHot(labels []int, numClasses int) *mat.Dense {
	oneHot := mat.NewDense(len(labels), numClasses, nil)

	for i, label := range labels {
		if label >= 0 && label < numClasses {
			oneHot.Set(i, label, 1.0)
		}
	}

	return oneHot
}

func applySoftmax(x []float64) []float64 {
	maxVal := x[0]
	for _, val := range x[1:] {
		if val > maxVal {
			maxVal = val
		}
	}

	exp := make([]float64, len(x))
	sum := 0.0
	for i, val := range x {
		exp[i] = math.Exp(val - maxVal)
		sum += exp[i]
	}

	for i := range exp {
		exp[i] /= sum
	}

	return exp
}

func clamp(x, min, max float64) float64 {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}

func zeros(n int) []float64 {
	return make([]float64, n)
}

func randomNormal(n int, std float64) []float64 {
	data := make([]float64, n)
	for i := range data {
		data[i] = rand.NormFloat64() * std
	}
	return data
}

func formatShape(shape []int) string {
	if len(shape) == 0 {
		return "None"
	}
	result := ""
	for i, dim := range shape {
		if i > 0 {
			result += ", "
		}
		if i == 0 {
			result += "None"
		} else {
			result += fmt.Sprintf("%d", dim)
		}
	}
	return result
}

func (m *Model) evaluateGenerator(dataLoader *DataLoader) (float64, float64) {
	totalLoss := 0.0
	totalAccuracy := 0.0
	numBatches := 0

	for dataLoader.HasNext() {
		xBatch, yBatch := dataLoader.NextBatch()
		if xBatch == nil {
			break
		}

		yPred := m.predict(xBatch)
		loss := m.loss.Compute(yBatch, yPred)
		accuracy := m.computeAccuracy(yBatch, yPred)

		totalLoss += loss
		totalAccuracy += accuracy
		numBatches++
	}

	if numBatches > 0 {
		return totalLoss / float64(numBatches), totalAccuracy / float64(numBatches)
	}

	return 0, 0
}

// ============ Model Persistence ============

type ModelCheckpoint struct {
	Optimizer  map[string]interface{} `json:"optimizer"`
	History    *History               `json:"history"`
	Epoch      int                    `json:"epoch"`
	BestLoss   float64                `json:"best_loss"`
	ModelState map[string]interface{} `json:"model_state"`
}

func (m *Model) SaveCheckpoint(filename string) error {
	checkpoint := &ModelCheckpoint{
		History:  m.history,
		BestLoss: math.Inf(1),
	}

	if m.optimizer != nil {
		checkpoint.Optimizer = m.optimizer.GetState()
	}

	// In a full implementation, would save layer weights
	checkpoint.ModelState = make(map[string]interface{})

	data, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		return err
	}

	return ioutil.WriteFile(filename, data, 0644)
}

func (m *Model) LoadCheckpoint(filename string) error {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}

	var checkpoint ModelCheckpoint
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return err
	}

	m.history = checkpoint.History

	if m.optimizer != nil && checkpoint.Optimizer != nil {
		m.optimizer.LoadState(checkpoint.Optimizer)
	}

	return nil
}

// ============ Pretrained Models ============

type PretrainedModel interface {
	LoadWeights(url string) error
	GetFeatureExtractor() *Model
	GetClassifier() *Model
	FineTune(numClasses int, freezeFeatures bool) *Model
}

type ResNet50 struct {
	*Model
	featureExtractor *Model
	classifier       *Model
}

func NewResNet50(pretrained bool) *ResNet50 {
	model := NewModel()

	// Simplified ResNet50 architecture
	// Feature extractor
	featureExtractor := NewModel()
	featureExtractor.Add(NewConv2D(64, []int{7, 7}, WithStride([]int{2, 2}), WithConvActivation(&ReLU{})))
	featureExtractor.Add(NewMaxPooling2D([]int{3, 3}, WithPoolStride([]int{2, 2})))
	featureExtractor.Add(NewConv2D(256, []int{3, 3}, WithConvActivation(&ReLU{})))
	featureExtractor.Add(NewConv2D(512, []int{3, 3}, WithConvActivation(&ReLU{})))
	featureExtractor.Add(NewConv2D(1024, []int{3, 3}, WithConvActivation(&ReLU{})))
	featureExtractor.Add(NewConv2D(2048, []int{3, 3}, WithConvActivation(&ReLU{})))

	// Global average pooling would go here
	featureExtractor.Add(NewDense(2048, &ReLU{}))

	// Classifier
	classifier := NewModel()
	classifier.Add(NewDropout(0.5))
	classifier.Add(NewDense(1000, &Softmax{}))

	// Combine into full model
	for _, layer := range featureExtractor.layers {
		model.Add(layer)
	}
	for _, layer := range classifier.layers {
		model.Add(layer)
	}

	resnet := &ResNet50{
		Model:            model,
		featureExtractor: featureExtractor,
		classifier:       classifier,
	}

	if pretrained {
		// In practice, this would download and load pretrained weights
		fmt.Println("Loading pretrained ImageNet weights...")
	}

	return resnet
}

func (r *ResNet50) LoadWeights(url string) error {
	// In practice, this would download and load weights from URL
	fmt.Printf("Loading weights from %s\n", url)
	return nil
}

func (r *ResNet50) GetFeatureExtractor() *Model {
	return r.featureExtractor
}

func (r *ResNet50) GetClassifier() *Model {
	return r.classifier
}

func (r *ResNet50) FineTune(numClasses int, freezeFeatures bool) *Model {
	model := NewModel()

	// Copy feature extractor layers
	for _, layer := range r.featureExtractor.layers {
		model.Add(layer)
		if freezeFeatures {
			// In practice, would set layer.requires_grad = False
		}
	}

	// Add new classifier
	model.Add(NewDropout(0.5))
	model.Add(NewDense(numClasses, &Softmax{}))

	return model
}

// Keep all original utility functions and data generators
func (m *Model) Fit(xTrain, yTrain *mat.Dense, config *Config, callbacks ...Callback) *History {
	trainDataset := NewTensorDataset(xTrain, yTrain)
	trainLoader := NewDataLoader(trainDataset, config.BatchSize, WithShuffle(true))

	var validationLoader *DataLoader
	if config.ValidationSplit > 0 {
		xTrainSplit, xVal, yTrainSplit, yVal := trainTestSplit(xTrain, yTrain, config.ValidationSplit, config.Seed)
		trainDataset = NewTensorDataset(xTrainSplit, yTrainSplit)
		trainLoader = NewDataLoader(trainDataset, config.BatchSize, WithShuffle(true))

		validationDataset := NewTensorDataset(xVal, yVal)
		validationLoader = NewDataLoader(validationDataset, config.BatchSize, WithShuffle(false))
	}

	return m.FitGenerator(trainLoader, config, validationLoader, callbacks...)
}

func (m *Model) Predict(x *mat.Dense) *mat.Dense {
	m.setTrainingMode(false)
	return m.predict(x)
}

func (m *Model) Evaluate(xTest, yTest *mat.Dense, verbose int) (float64, float64) {
	m.setTrainingMode(false)
	yPred := m.predict(xTest)
	loss := m.loss.Compute(yTest, yPred)
	accuracy := m.computeAccuracy(yTest, yPred)

	if verbose > 0 {
		fmt.Printf("Test loss: %.4f - Test accuracy: %.4f\n", loss, accuracy)
	}

	return loss, accuracy
}

// Private methods
func (m *Model) predict(x *mat.Dense) *mat.Dense {
	output := x
	for _, layer := range m.layers {
		output = layer.Forward(output)
	}
	return output
}

func (m *Model) backward(grad *mat.Dense) {
	gradients := []*mat.Dense{grad}
	for i := len(m.layers) - 1; i >= 0; i-- {
		gradients = m.layers[i].Backward(gradients[0])
	}
}

func (m *Model) updateWeights() {
	m.optimizer.Step()
	for _, layer := range m.layers {
		if layer.IsTrainable() {
			weights := layer.GetWeights()
			grads := layer.GetGradients()
			for i := range weights {
				if i < len(grads) && grads[i] != nil {
					m.optimizer.Update(weights[i], grads[i])
				}
			}
		}
	}
}

// ============ Transformer Components ============

type MultiHeadAttention struct {
	BaseLayer
	dModel                         int
	numHeads                       int
	dK                             int
	wQ, wK, wV, wO                 *mat.Dense
	gradWQ, gradWK, gradWV, gradWO *mat.Dense
	lastQ, lastK, lastV            *mat.Dense
	dropout                        *Dropout
}

func NewMultiHeadAttention(dModel, numHeads int, options ...func(*MultiHeadAttention)) *MultiHeadAttention {
	if dModel%numHeads != 0 {
		panic("dModel must be divisible by numHeads")
	}

	layer := &MultiHeadAttention{
		dModel:   dModel,
		numHeads: numHeads,
		dK:       dModel / numHeads,
		dropout:  NewDropout(0.1),
	}
	layer.device = &CPUDevice{}
	layer.training = true

	// Initialize weight matrices
	std := math.Sqrt(2.0 / float64(dModel))
	layer.wQ = mat.NewDense(dModel, dModel, randomNormal(dModel*dModel, std))
	layer.wK = mat.NewDense(dModel, dModel, randomNormal(dModel*dModel, std))
	layer.wV = mat.NewDense(dModel, dModel, randomNormal(dModel*dModel, std))
	layer.wO = mat.NewDense(dModel, dModel, randomNormal(dModel*dModel, std))

	for _, opt := range options {
		opt(layer)
	}

	return layer
}

func WithAttentionDropout(rate float64) func(*MultiHeadAttention) {
	return func(mha *MultiHeadAttention) {
		mha.dropout = NewDropout(rate)
	}
}

func (mha *MultiHeadAttention) GetOutputShape(inputShape []int) []int {
	return inputShape
}

func (mha *MultiHeadAttention) GetParams() int {
	return 4 * mha.dModel * mha.dModel // Q, K, V, O matrices
}

func (mha *MultiHeadAttention) scaledDotProductAttention(q, k, v *mat.Dense) *mat.Dense {
	seqLen, _ := q.Dims()

	// Compute attention scores: Q * K^T
	scores := mat.NewDense(seqLen, seqLen, nil)
	scores.Mul(q, k.T())

	// Scale by sqrt(dK)
	scale := 1.0 / math.Sqrt(float64(mha.dK))
	scores.Scale(scale, scores)

	// Apply softmax
	for i := 0; i < seqLen; i++ {
		row := mat.Row(nil, i, scores)
		softmaxRow := applySoftmax(row)
		for j := 0; j < seqLen; j++ {
			scores.Set(i, j, softmaxRow[j])
		}
	}

	// Apply dropout during training
	if mha.training {
		scores = mha.dropout.Forward(scores)
	}

	// Compute attention output: Attention * V
	output := mat.NewDense(seqLen, mha.dK, nil)
	output.Mul(scores, v)

	return output
}

func (mha *MultiHeadAttention) Forward(inputs ...*mat.Dense) *mat.Dense {
	x := inputs[0]
	seqLen, dModel := x.Dims()

	// Linear projections
	mha.lastQ = mat.NewDense(seqLen, dModel, nil)
	mha.lastK = mat.NewDense(seqLen, dModel, nil)
	mha.lastV = mat.NewDense(seqLen, dModel, nil)

	mha.lastQ.Mul(x, mha.wQ)
	mha.lastK.Mul(x, mha.wK)
	mha.lastV.Mul(x, mha.wV)

	// Reshape for multi-head attention (simplified)
	outputs := make([]*mat.Dense, mha.numHeads)
	for h := 0; h < mha.numHeads; h++ {
		// Extract head slice (simplified - in practice would need proper tensor ops)
		startCol := h * mha.dK
		endCol := (h + 1) * mha.dK

		qHead := extractCols(mha.lastQ, startCol, endCol)
		kHead := extractCols(mha.lastK, startCol, endCol)
		vHead := extractCols(mha.lastV, startCol, endCol)

		outputs[h] = mha.scaledDotProductAttention(qHead, kHead, vHead)
	}

	// Concatenate heads (simplified)
	concatenated := mat.NewDense(seqLen, dModel, nil)
	for h := 0; h < mha.numHeads; h++ {
		startCol := h * mha.dK
		for i := 0; i < seqLen; i++ {
			for j := 0; j < mha.dK; j++ {
				concatenated.Set(i, startCol+j, outputs[h].At(i, j))
			}
		}
	}

	// Final linear projection
	output := mat.NewDense(seqLen, dModel, nil)
	output.Mul(concatenated, mha.wO)

	return output
}

func (mha *MultiHeadAttention) Backward(grad *mat.Dense) []*mat.Dense {
	seqLen, dModel := grad.Dims()

	// Initialize gradients
	mha.gradWQ = mat.NewDense(dModel, dModel, nil)
	mha.gradWK = mat.NewDense(dModel, dModel, nil)
	mha.gradWV = mat.NewDense(dModel, dModel, nil)
	mha.gradWO = mat.NewDense(dModel, dModel, nil)

	// Simplified backward pass
	inputGrad := mat.NewDense(seqLen, dModel, nil)
	inputGrad.Copy(grad)

	return []*mat.Dense{inputGrad}
}

func (mha *MultiHeadAttention) IsTrainable() bool { return true }
func (mha *MultiHeadAttention) GetWeights() []*mat.Dense {
	return []*mat.Dense{mha.wQ, mha.wK, mha.wV, mha.wO}
}
func (mha *MultiHeadAttention) GetGradients() []*mat.Dense {
	return []*mat.Dense{mha.gradWQ, mha.gradWK, mha.gradWV, mha.gradWO}
}

// ============ GRU Layer ============

type GRU struct {
	BaseLayer
	units       int
	returnSeq   bool
	weights     []*mat.Dense // Wr, Wu, Wh, Ur, Uu, Uh
	biases      []*mat.Dense // br, bu, bh
	gradWeights []*mat.Dense
	gradBiases  []*mat.Dense
}

func NewGRU(units int, options ...func(*GRU)) *GRU {
	layer := &GRU{
		units:     units,
		returnSeq: false,
	}
	layer.device = &CPUDevice{}
	layer.training = true

	for _, opt := range options {
		opt(layer)
	}

	return layer
}

func WithGRUReturnSequences(returnSeq bool) func(*GRU) {
	return func(g *GRU) { g.returnSeq = returnSeq }
}

func (g *GRU) Build(inputDim int) {
	std := math.Sqrt(2.0 / float64(inputDim+g.units))

	// Weight matrices for reset, update, and new gates
	g.weights = make([]*mat.Dense, 6)
	g.biases = make([]*mat.Dense, 3)

	// Input weights
	g.weights[0] = mat.NewDense(inputDim, g.units, randomNormal(inputDim*g.units, std)) // Wr
	g.weights[1] = mat.NewDense(inputDim, g.units, randomNormal(inputDim*g.units, std)) // Wu
	g.weights[2] = mat.NewDense(inputDim, g.units, randomNormal(inputDim*g.units, std)) // Wh

	// Hidden weights
	g.weights[3] = mat.NewDense(g.units, g.units, randomNormal(g.units*g.units, std)) // Ur
	g.weights[4] = mat.NewDense(g.units, g.units, randomNormal(g.units*g.units, std)) // Uu
	g.weights[5] = mat.NewDense(g.units, g.units, randomNormal(g.units*g.units, std)) // Uh

	// Biases
	for i := 0; i < 3; i++ {
		g.biases[i] = mat.NewDense(1, g.units, zeros(g.units))
	}
}

func (g *GRU) GetOutputShape(inputShape []int) []int {
	if g.returnSeq {
		return []int{inputShape[0], inputShape[1], g.units}
	}
	return []int{inputShape[0], g.units}
}

func (g *GRU) GetParams() int {
	if len(g.weights) == 0 {
		return 0
	}
	inputDim := g.weights[0].RawMatrix().Rows
	return 3*(inputDim+g.units)*g.units + 3*g.units
}

func (g *GRU) Forward(inputs ...*mat.Dense) *mat.Dense {
	x := inputs[0]
	batchSize, seqLen, inputDim := x.RawMatrix().Rows, 1, x.RawMatrix().Cols

	if len(g.weights) == 0 {
		g.Build(inputDim)
	}

	// Initialize hidden state
	h := mat.NewDense(batchSize, g.units, nil)

	var outputs []*mat.Dense
	if g.returnSeq {
		outputs = make([]*mat.Dense, seqLen)
	}

	// Process each time step
	for t := 0; t < seqLen; t++ {
		xt := x // Simplified - should extract time slice

		// Reset gate: rt = sigmoid(Wr @ xt + Ur @ h_{t-1} + br)
		reset := mat.NewDense(batchSize, g.units, nil)
		reset.Mul(xt, g.weights[0])
		temp := mat.NewDense(batchSize, g.units, nil)
		temp.Mul(h, g.weights[3])
		reset.Add(reset, temp)
		reset.Apply(func(i, j int, v float64) float64 {
			return 1.0 / (1.0 + math.Exp(-(v + g.biases[0].At(0, j))))
		}, reset)

		// Update gate: ut = sigmoid(Wu @ xt + Uu @ h_{t-1} + bu)
		update := mat.NewDense(batchSize, g.units, nil)
		update.Mul(xt, g.weights[1])
		temp.Mul(h, g.weights[4])
		update.Add(update, temp)
		update.Apply(func(i, j int, v float64) float64 {
			return 1.0 / (1.0 + math.Exp(-(v + g.biases[1].At(0, j))))
		}, update)

		// New gate: nt = tanh(Wh @ xt + Uh @ (rt * h_{t-1}) + bh)
		resetH := mat.NewDense(batchSize, g.units, nil)
		resetH.MulElem(reset, h)

		newGate := mat.NewDense(batchSize, g.units, nil)
		newGate.Mul(xt, g.weights[2])
		temp.Mul(resetH, g.weights[5])
		newGate.Add(newGate, temp)
		newGate.Apply(func(i, j int, v float64) float64 {
			return math.Tanh(v + g.biases[2].At(0, j))
		}, newGate)

		// Hidden state: h_t = (1 - ut) * nt + ut * h_{t-1}
		oneMinusUpdate := mat.NewDense(batchSize, g.units, nil)
		oneMinusUpdate.Apply(func(i, j int, v float64) float64 {
			return 1.0 - update.At(i, j)
		}, oneMinusUpdate)

		term1 := mat.NewDense(batchSize, g.units, nil)
		term1.MulElem(oneMinusUpdate, newGate)

		term2 := mat.NewDense(batchSize, g.units, nil)
		term2.MulElem(update, h)

		h.Add(term1, term2)

		if g.returnSeq {
			outputs[t] = mat.DenseCopyOf(h)
		}
	}

	if g.returnSeq {
		// Concatenate all outputs (simplified)
		return outputs[len(outputs)-1]
	}

	return h
}

func (g *GRU) Backward(grad *mat.Dense) []*mat.Dense {
	rows, cols := grad.Dims()
	inputGrad := mat.NewDense(rows, cols, nil)

	// Initialize gradient matrices
	g.gradWeights = make([]*mat.Dense, len(g.weights))
	for i := range g.gradWeights {
		r, c := g.weights[i].Dims()
		g.gradWeights[i] = mat.NewDense(r, c, nil)
	}

	g.gradBiases = make([]*mat.Dense, len(g.biases))
	for i := range g.gradBiases {
		r, c := g.biases[i].Dims()
		g.gradBiases[i] = mat.NewDense(r, c, nil)
	}

	return []*mat.Dense{inputGrad}
}

func (g *GRU) IsTrainable() bool { return true }
func (g *GRU) GetWeights() []*mat.Dense {
	weights := make([]*mat.Dense, len(g.weights)+len(g.biases))
	copy(weights[:len(g.weights)], g.weights)
	copy(weights[len(g.weights):], g.biases)
	return weights
}
func (g *GRU) GetGradients() []*mat.Dense {
	grads := make([]*mat.Dense, len(g.gradWeights)+len(g.gradBiases))
	copy(grads[:len(g.gradWeights)], g.gradWeights)
	copy(grads[len(g.gradWeights):], g.gradBiases)
	return grads
}

// ============ Transformer Block ============

type TransformerBlock struct {
	BaseLayer
	dModel      int
	attention   *MultiHeadAttention
	feedForward *Model
	layerNorm1  *LayerNorm
	layerNorm2  *LayerNorm
	dropout     *Dropout
}

func NewTransformerBlock(dModel, numHeads int, dFF int, options ...func(*TransformerBlock)) *TransformerBlock {
	layer := &TransformerBlock{
		dModel:     dModel,
		attention:  NewMultiHeadAttention(dModel, numHeads),
		layerNorm1: NewLayerNorm(dModel),
		layerNorm2: NewLayerNorm(dModel),
		dropout:    NewDropout(0.1),
	}

	// Feed-forward network
	layer.feedForward = NewModel()
	layer.feedForward.Add(NewDense(dFF, &ReLU{}))
	layer.feedForward.Add(NewDropout(0.1))
	layer.feedForward.Add(NewDense(dModel, &Linear{}))

	layer.device = &CPUDevice{}
	layer.training = true

	for _, opt := range options {
		opt(layer)
	}

	return layer
}

func WithTransformerDropout(rate float64) func(*TransformerBlock) {
	return func(tb *TransformerBlock) {
		tb.dropout = NewDropout(rate)
	}
}

func (tb *TransformerBlock) GetOutputShape(inputShape []int) []int {
	return inputShape
}

func (tb *TransformerBlock) GetParams() int {
	params := tb.attention.GetParams()
	params += tb.layerNorm1.GetParams()
	params += tb.layerNorm2.GetParams()
	for _, layer := range tb.feedForward.layers {
		params += layer.GetParams()
	}
	return params
}

func (tb *TransformerBlock) Forward(inputs ...*mat.Dense) *mat.Dense {
	x := inputs[0]

	// Multi-head self-attention with residual connection
	attnOutput := tb.attention.Forward(x)
	if tb.training {
		attnOutput = tb.dropout.Forward(attnOutput)
	}

	// Add & Norm 1
	residual1 := mat.NewDense(x.RawMatrix().Rows, x.RawMatrix().Cols, nil)
	residual1.Add(x, attnOutput)
	normed1 := tb.layerNorm1.Forward(residual1)

	// Feed-forward network
	ffOutput := normed1
	for _, layer := range tb.feedForward.layers {
		ffOutput = layer.Forward(ffOutput)
	}

	// Add & Norm 2
	residual2 := mat.NewDense(normed1.RawMatrix().Rows, normed1.RawMatrix().Cols, nil)
	residual2.Add(normed1, ffOutput)
	output := tb.layerNorm2.Forward(residual2)

	return output
}

func (tb *TransformerBlock) Backward(grad *mat.Dense) []*mat.Dense {
	// Simplified backward pass
	return []*mat.Dense{grad}
}

func (tb *TransformerBlock) IsTrainable() bool { return true }
func (tb *TransformerBlock) GetWeights() []*mat.Dense {
	weights := tb.attention.GetWeights()
	weights = append(weights, tb.layerNorm1.GetWeights()...)
	weights = append(weights, tb.layerNorm2.GetWeights()...)
	for _, layer := range tb.feedForward.layers {
		weights = append(weights, layer.GetWeights()...)
	}
	return weights
}
func (tb *TransformerBlock) GetGradients() []*mat.Dense {
	grads := tb.attention.GetGradients()
	grads = append(grads, tb.layerNorm1.GetGradients()...)
	grads = append(grads, tb.layerNorm2.GetGradients()...)
	for _, layer := range tb.feedForward.layers {
		grads = append(grads, layer.GetGradients()...)
	}
	return grads
}

// ============ Layer Normalization ============

type LayerNorm struct {
	BaseLayer
	numFeatures int
	epsilon     float64
	gamma       *mat.Dense
	beta        *mat.Dense
	gradGamma   *mat.Dense
	gradBeta    *mat.Dense
}

func NewLayerNorm(numFeatures int, options ...func(*LayerNorm)) *LayerNorm {
	layer := &LayerNorm{
		numFeatures: numFeatures,
		epsilon:     1e-5,
	}
	layer.device = &CPUDevice{}
	layer.training = true

	// Initialize parameters
	layer.gamma = mat.NewDense(1, numFeatures, nil)
	layer.beta = mat.NewDense(1, numFeatures, nil)

	for i := 0; i < numFeatures; i++ {
		layer.gamma.Set(0, i, 1.0)
		layer.beta.Set(0, i, 0.0)
	}

	for _, opt := range options {
		opt(layer)
	}

	return layer
}

func WithLNEpsilon(epsilon float64) func(*LayerNorm) {
	return func(ln *LayerNorm) { ln.epsilon = epsilon }
}

func (ln *LayerNorm) GetOutputShape(inputShape []int) []int {
	return inputShape
}

func (ln *LayerNorm) GetParams() int {
	return 2 * ln.numFeatures // gamma + beta
}

func (ln *LayerNorm) Forward(inputs ...*mat.Dense) *mat.Dense {
	x := inputs[0]
	rows, cols := x.Dims()
	output := mat.NewDense(rows, cols, nil)

	// Compute layer normalization for each sample
	for i := 0; i < rows; i++ {
		row := mat.Row(nil, i, x)

		// Compute mean and variance
		mean := floats.Sum(row) / float64(len(row))
		var variance float64
		for _, val := range row {
			diff := val - mean
			variance += diff * diff
		}
		variance /= float64(len(row))

		// Normalize and scale
		for j := 0; j < cols; j++ {
			normalized := (x.At(i, j) - mean) / math.Sqrt(variance+ln.epsilon)
			scaled := normalized*ln.gamma.At(0, j) + ln.beta.At(0, j)
			output.Set(i, j, scaled)
		}
	}

	return output
}

func (ln *LayerNorm) Backward(grad *mat.Dense) []*mat.Dense {
	rows, cols := grad.Dims()

	// Initialize gradients
	ln.gradGamma = mat.NewDense(1, cols, nil)
	ln.gradBeta = mat.NewDense(1, cols, nil)
	inputGrad := mat.NewDense(rows, cols, nil)

	// Simplified gradient computation
	inputGrad.Copy(grad)

	return []*mat.Dense{inputGrad}
}

func (ln *LayerNorm) IsTrainable() bool          { return true }
func (ln *LayerNorm) GetWeights() []*mat.Dense   { return []*mat.Dense{ln.gamma, ln.beta} }
func (ln *LayerNorm) GetGradients() []*mat.Dense { return []*mat.Dense{ln.gradGamma, ln.gradBeta} }

func CreateBERT(vocabSize, dModel, numHeads, numLayers, maxSeqLen int) *Model {
	model := NewModel()

	// Token embedding
	model.Add(NewEmbedding(vocabSize, dModel))

	// Add positional encoding (simplified)
	model.Add(NewDropout(0.1))

	// Transformer encoder blocks
	for i := 0; i < numLayers; i++ {
		model.Add(NewTransformerBlock(dModel, numHeads, dModel*4))
	}

	// Final layer norm
	model.Add(NewLayerNorm(dModel))

	return model
}

func CreateUNet(inputChannels, outputChannels int) *FunctionalModel {
	model := NewFunctionalModel()

	// Encoder path
	model.AddModule("conv1", NewConv2D(64, []int{3, 3}, WithConvActivation(&ReLU{})))
	model.AddModule("conv2", NewConv2D(64, []int{3, 3}, WithConvActivation(&ReLU{})))
	model.AddModule("pool1", NewMaxPooling2D([]int{2, 2}))

	model.AddModule("conv3", NewConv2D(128, []int{3, 3}, WithConvActivation(&ReLU{})))
	model.AddModule("conv4", NewConv2D(128, []int{3, 3}, WithConvActivation(&ReLU{})))
	model.AddModule("pool2", NewMaxPooling2D([]int{2, 2}))

	// Bottleneck
	model.AddModule("conv5", NewConv2D(256, []int{3, 3}, WithConvActivation(&ReLU{})))
	model.AddModule("conv6", NewConv2D(256, []int{3, 3}, WithConvActivation(&ReLU{})))

	// Decoder path (simplified - would need upsampling layers)
	model.AddModule("conv7", NewConv2D(128, []int{3, 3}, WithConvActivation(&ReLU{})))
	model.AddModule("conv8", NewConv2D(64, []int{3, 3}, WithConvActivation(&ReLU{})))
	model.AddModule("conv9", NewConv2D(outputChannels, []int{1, 1}, WithConvActivation(&Sigmoid{})))

	model.SetForward(func(modules ModuleDict, inputs map[string]*mat.Dense) map[string]*mat.Dense {
		x := inputs["input"]

		// Encoder
		conv1 := modules["conv1"].Forward(x)
		conv2 := modules["conv2"].Forward(conv1)
		pool1 := modules["pool1"].Forward(conv2)

		conv3 := modules["conv3"].Forward(pool1)
		conv4 := modules["conv4"].Forward(conv3)
		pool2 := modules["pool2"].Forward(conv4)

		// Bottleneck
		conv5 := modules["conv5"].Forward(pool2)
		conv6 := modules["conv6"].Forward(conv5)

		// Decoder (simplified)
		conv7 := modules["conv7"].Forward(conv6)
		conv8 := modules["conv8"].Forward(conv7)
		output := modules["conv9"].Forward(conv8)

		return map[string]*mat.Dense{"output": output}
	})

	return model
}

// ============ Advanced Data Augmentation ============

type Compose struct {
	transforms []Transform
}

func NewCompose(transforms ...Transform) *Compose {
	return &Compose{transforms: transforms}
}

func (c *Compose) Apply(data *mat.Dense) *mat.Dense {
	result := data
	for _, transform := range c.transforms {
		result = transform.Apply(result)
	}
	return result
}

type RandomRotation struct {
	degrees float64
}

func NewRandomRotation(degrees float64) *RandomRotation {
	return &RandomRotation{degrees: degrees}
}

func (rr *RandomRotation) Apply(data *mat.Dense) *mat.Dense {
	// Simplified rotation - in practice would apply proper image rotation
	angle := (rand.Float64()*2 - 1) * rr.degrees * math.Pi / 180
	_ = angle // Use angle for rotation transformation

	// For now, return original data with small noise as placeholder
	rows, cols := data.Dims()
	result := mat.NewDense(rows, cols, nil)
	result.Copy(data)

	noise := 0.01 * math.Abs(angle)
	result.Apply(func(i, j int, v float64) float64 {
		return v + rand.NormFloat64()*noise
	}, result)

	return result
}

type RandomHorizontalFlip struct {
	probability float64
}

func NewRandomHorizontalFlip(p float64) *RandomHorizontalFlip {
	return &RandomHorizontalFlip{probability: p}
}

func (rhf *RandomHorizontalFlip) Apply(data *mat.Dense) *mat.Dense {
	if rand.Float64() > rhf.probability {
		return mat.DenseCopyOf(data)
	}

	// Simplified flip - in practice would flip image horizontally
	rows, cols := data.Dims()
	result := mat.NewDense(rows, cols, nil)

	// Reverse column order as simple flip simulation
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result.Set(i, j, data.At(i, cols-1-j))
		}
	}

	return result
}

// ============ Model Zoo (Pretrained Models) ============

type ModelZoo struct {
	models map[string]PretrainedModel
}

func NewModelZoo() *ModelZoo {
	zoo := &ModelZoo{
		models: make(map[string]PretrainedModel),
	}

	// Register available models
	zoo.models["resnet50"] = NewResNet50(false)
	zoo.models["vit_base"] = NewViTBase(false)

	return zoo
}

func (mz *ModelZoo) LoadModel(name string, pretrained bool) (PretrainedModel, error) {
	if model, exists := mz.models[name]; exists {
		if pretrained {
			err := model.LoadWeights(fmt.Sprintf("https://models.zoo/%s/weights", name))
			if err != nil {
				return nil, err
			}
		}
		return model, nil
	}
	return nil, fmt.Errorf("model %s not found in zoo", name)
}

func (mz *ModelZoo) ListModels() []string {
	models := make([]string, 0, len(mz.models))
	for name := range mz.models {
		models = append(models, name)
	}
	return models
}

// Vision Transformer Implementation
type ViTBase struct {
	*Model
	patchSize  int
	imageSize  int
	numPatches int
	embedDim   int
	numHeads   int
	numLayers  int
	numClasses int
}

func NewViTBase(pretrained bool) *ViTBase {
	imageSize := 224
	patchSize := 16
	embedDim := 768
	numHeads := 12
	numLayers := 12
	numClasses := 1000

	model := CreateVisionTransformer(imageSize, patchSize, embedDim, numHeads, numLayers, numClasses)

	vit := &ViTBase{
		Model:      model,
		patchSize:  patchSize,
		imageSize:  imageSize,
		numPatches: (imageSize / patchSize) * (imageSize / patchSize),
		embedDim:   embedDim,
		numHeads:   numHeads,
		numLayers:  numLayers,
		numClasses: numClasses,
	}

	if pretrained {
		fmt.Println("Loading pretrained ViT-Base weights...")
	}

	return vit
}

func (vit *ViTBase) LoadWeights(url string) error {
	fmt.Printf("Loading ViT weights from %s\n", url)
	return nil
}

func (vit *ViTBase) GetFeatureExtractor() *Model {
	extractor := NewModel()
	// Copy all layers except the last classification layer
	for i := 0; i < len(vit.layers)-1; i++ {
		extractor.Add(vit.layers[i])
	}
	return extractor
}

func (vit *ViTBase) GetClassifier() *Model {
	classifier := NewModel()
	// Add only the last layer
	classifier.Add(vit.layers[len(vit.layers)-1])
	return classifier
}

func (vit *ViTBase) FineTune(numClasses int, freezeFeatures bool) *Model {
	model := NewModel()

	// Copy feature extractor layers
	for i := 0; i < len(vit.layers)-1; i++ {
		model.Add(vit.layers[i])
		if freezeFeatures {
			// In practice, would freeze these layers
		}
	}

	// Add new classification head
	model.Add(NewDense(numClasses, &Softmax{}))

	return model
}

// ============ Advanced Metrics and Evaluation ============

type Metrics struct {
	accuracy   float64
	precision  []float64
	recall     []float64
	f1Score    []float64
	confMatrix [][]int
	numClasses int
}

func NewMetrics(numClasses int) *Metrics {
	return &Metrics{
		precision:  make([]float64, numClasses),
		recall:     make([]float64, numClasses),
		f1Score:    make([]float64, numClasses),
		confMatrix: make([][]int, numClasses),
		numClasses: numClasses,
	}
}

func (m *Metrics) Update(yTrue, yPred *mat.Dense) {
	rows, cols := yTrue.Dims()

	// Initialize confusion matrix
	for i := range m.confMatrix {
		m.confMatrix[i] = make([]int, m.numClasses)
	}

	correct := 0

	for i := 0; i < rows; i++ {
		var trueClass, predClass int

		if cols == 1 {
			// Binary classification
			trueClass = int(yTrue.At(i, 0))
			predClass = 0
			if yPred.At(i, 0) >= 0.5 {
				predClass = 1
			}
		} else {
			// Multi-class classification - find argmax
			maxTrue, maxPred := yTrue.At(i, 0), yPred.At(i, 0)
			trueClass, predClass = 0, 0
			for j := 1; j < cols; j++ {
				if yTrue.At(i, j) > maxTrue {
					maxTrue = yTrue.At(i, j)
					trueClass = j
				}
				if yPred.At(i, j) > maxPred {
					maxPred = yPred.At(i, j)
					predClass = j
				}
			}
		}

		if trueClass == predClass {
			correct++
		}

		// Update confusion matrix
		if trueClass < m.numClasses && predClass < m.numClasses {
			m.confMatrix[trueClass][predClass]++
		}
	}

	m.accuracy = float64(correct) / float64(rows)
	m.computeClassificationMetrics()
}

func (m *Metrics) computeClassificationMetrics() {
	for i := 0; i < m.numClasses; i++ {
		tp := m.confMatrix[i][i]

		// Calculate precision and recall
		var fp, fn int
		for j := 0; j < m.numClasses; j++ {
			if j != i {
				fp += m.confMatrix[j][i] // False positives
				fn += m.confMatrix[i][j] // False negatives
			}
		}

		if tp+fp > 0 {
			m.precision[i] = float64(tp) / float64(tp+fp)
		}

		if tp+fn > 0 {
			m.recall[i] = float64(tp) / float64(tp+fn)
		}

		if m.precision[i]+m.recall[i] > 0 {
			m.f1Score[i] = 2 * m.precision[i] * m.recall[i] / (m.precision[i] + m.recall[i])
		}
	}
}

func (m *Metrics) GetAccuracy() float64 { return m.accuracy }

func (m *Metrics) GetPrecision() []float64 { return m.precision }

func (m *Metrics) GetRecall() []float64 { return m.recall }

func (m *Metrics) GetF1Score() []float64 { return m.f1Score }

func (m *Metrics) GetMacroF1() float64 {
	sum := 0.0
	for _, f1 := range m.f1Score {
		sum += f1
	}
	return sum / float64(len(m.f1Score))
}

func (m *Metrics) PrintReport() {
	fmt.Println("\nClassification Report:")
	fmt.Println(strings.Repeat("=", 50))
	fmt.Printf("%-10s %-10s %-10s %-10s\n", "Class", "Precision", "Recall", "F1-Score")
	fmt.Println(strings.Repeat("-", 50))

	for i := 0; i < m.numClasses; i++ {
		fmt.Printf("%-10d %-10.3f %-10.3f %-10.3f\n",
			i, m.precision[i], m.recall[i], m.f1Score[i])
	}

	fmt.Println(strings.Repeat("-", 50))
	fmt.Printf("%-10s %-10.3f\n", "Accuracy", m.accuracy)
	fmt.Printf("%-10s %-10.3f\n", "Macro F1", m.GetMacroF1())
	fmt.Println(strings.Repeat("=", 50))
}

// ============ Hyperparameter Optimization ============

type HyperparameterConfig struct {
	LearningRate []float64
	BatchSize    []int
	Epochs       []int
	Dropout      []float64
	Architecture map[string][]interface{}
}

type TrialResult struct {
	Config  map[string]interface{}
	Score   float64
	History *History
}

type HyperparameterTuner struct {
	searchSpace HyperparameterConfig
	objective   string // "accuracy", "loss", "f1_score"
	direction   string // "maximize", "minimize"
	trials      []TrialResult
	bestTrial   *TrialResult
}

func NewHyperparameterTuner(searchSpace HyperparameterConfig, objective, direction string) *HyperparameterTuner {
	return &HyperparameterTuner{
		searchSpace: searchSpace,
		objective:   objective,
		direction:   direction,
		trials:      make([]TrialResult, 0),
	}
}

func (ht *HyperparameterTuner) RandomSearch(nTrials int, buildModelFn func(map[string]interface{}) *Model,
	trainData, valData *TensorDataset) *TrialResult {

	fmt.Printf("Starting hyperparameter search with %d trials...\n", nTrials)

	for trial := 0; trial < nTrials; trial++ {
		// Sample random configuration
		config := ht.sampleConfig()

		fmt.Printf("\nTrial %d/%d - Config: %v\n", trial+1, nTrials, config)

		// Build and train model
		model := buildModelFn(config)

		// Create training configuration
		trainConfig := &Config{
			LearningRate: config["learning_rate"].(float64),
			BatchSize:    config["batch_size"].(int),
			Epochs:       config["epochs"].(int),
			Verbose:      0, // Silent training
		}

		trainLoader := NewDataLoader(trainData, trainConfig.BatchSize, WithShuffle(true))
		valLoader := NewDataLoader(valData, trainConfig.BatchSize, WithShuffle(false))

		history := model.FitGenerator(trainLoader, trainConfig, valLoader)

		// Evaluate trial
		score := ht.evaluateTrial(history)

		result := TrialResult{
			Config:  config,
			Score:   score,
			History: history,
		}

		ht.trials = append(ht.trials, result)

		// Update best trial
		if ht.bestTrial == nil || ht.isBetter(score, ht.bestTrial.Score) {
			ht.bestTrial = &result
			fmt.Printf("New best trial! Score: %.4f\n", score)
		}

		fmt.Printf("Trial %d score: %.4f\n", trial+1, score)
	}

	fmt.Printf("\nBest configuration found:\n")
	fmt.Printf("Score: %.4f\n", ht.bestTrial.Score)
	fmt.Printf("Config: %v\n", ht.bestTrial.Config)

	return ht.bestTrial
}

func (ht *HyperparameterTuner) sampleConfig() map[string]interface{} {
	config := make(map[string]interface{})

	// Sample learning rate
	if len(ht.searchSpace.LearningRate) > 0 {
		config["learning_rate"] = ht.searchSpace.LearningRate[rand.Intn(len(ht.searchSpace.LearningRate))]
	}

	// Sample batch size
	if len(ht.searchSpace.BatchSize) > 0 {
		config["batch_size"] = ht.searchSpace.BatchSize[rand.Intn(len(ht.searchSpace.BatchSize))]
	}

	// Sample epochs
	if len(ht.searchSpace.Epochs) > 0 {
		config["epochs"] = ht.searchSpace.Epochs[rand.Intn(len(ht.searchSpace.Epochs))]
	}

	// Sample dropout
	if len(ht.searchSpace.Dropout) > 0 {
		config["dropout"] = ht.searchSpace.Dropout[rand.Intn(len(ht.searchSpace.Dropout))]
	}

	return config
}

func (ht *HyperparameterTuner) evaluateTrial(history *History) float64 {
	if len(history.ValAccuracy) == 0 {
		return 0.0
	}

	switch ht.objective {
	case "accuracy":
		return floats.Max(history.ValAccuracy)
	case "loss":
		return floats.Min(history.ValLoss)
	case "final_accuracy":
		return history.ValAccuracy[len(history.ValAccuracy)-1]
	default:
		return floats.Max(history.ValAccuracy)
	}
}

func (ht *HyperparameterTuner) isBetter(score1, score2 float64) bool {
	if ht.direction == "maximize" {
		return score1 > score2
	}
	return score1 < score2
}

// ============ Advanced Training Techniques ============

// Gradient Accumulation
type GradientAccumulator struct {
	accumulatedGrads  map[uintptr][]*mat.Dense
	steps             int
	accumulationSteps int
}

func NewGradientAccumulator(accumulationSteps int) *GradientAccumulator {
	return &GradientAccumulator{
		accumulatedGrads:  make(map[uintptr][]*mat.Dense),
		accumulationSteps: accumulationSteps,
	}
}

func (ga *GradientAccumulator) AccumulateGradients(model *Model) {
	ga.steps++

	for i, layer := range model.layers {
		if !layer.IsTrainable() {
			continue
		}

		layerAddr := uintptr(unsafe.Pointer(&model.layers[i]))
		grads := layer.GetGradients()

		if ga.accumulatedGrads[layerAddr] == nil {
			ga.accumulatedGrads[layerAddr] = make([]*mat.Dense, len(grads))
			for j, grad := range grads {
				if grad != nil {
					rows, cols := grad.Dims()
					ga.accumulatedGrads[layerAddr][j] = mat.NewDense(rows, cols, nil)
				}
			}
		}

		// Accumulate gradients
		for j, grad := range grads {
			if grad != nil && ga.accumulatedGrads[layerAddr][j] != nil {
				ga.accumulatedGrads[layerAddr][j].Add(ga.accumulatedGrads[layerAddr][j], grad)
			}
		}
	}
}

func (ga *GradientAccumulator) ShouldUpdate() bool {
	return ga.steps%ga.accumulationSteps == 0
}

func (ga *GradientAccumulator) GetAverageGradients(model *Model) {
	if !ga.ShouldUpdate() {
		return
	}

	scale := 1.0 / float64(ga.accumulationSteps)

	for i, layer := range model.layers {
		if !layer.IsTrainable() {
			continue
		}

		layerAddr := uintptr(unsafe.Pointer(&model.layers[i]))
		if accumulated := ga.accumulatedGrads[layerAddr]; accumulated != nil {
			grads := layer.GetGradients()
			for j, accGrad := range accumulated {
				if accGrad != nil && j < len(grads) {
					accGrad.Scale(scale, accGrad)
					grads[j].Copy(accGrad)
				}
			}
		}
	}
}

func (ga *GradientAccumulator) Reset() {
	ga.steps = 0
	for _, grads := range ga.accumulatedGrads {
		for _, grad := range grads {
			if grad != nil {
				grad.Zero()
			}
		}
	}
}

// Mixed Precision Training (Simulation)
type MixedPrecisionTrainer struct {
	model     *Model
	optimizer Optimizer
	scaler    *GradScaler
	enabled   bool
}

type GradScaler struct {
	scale       float64
	growthRate  float64
	backoffRate float64
	interval    int
	counter     int
}

func NewGradScaler() *GradScaler {
	return &GradScaler{
		scale:       65536.0, // 2^16
		growthRate:  2.0,
		backoffRate: 0.5,
		interval:    2000,
	}
}

func (gs *GradScaler) Scale(loss float64) float64 {
	return loss * gs.scale
}

func (gs *GradScaler) Step(optimizer Optimizer, model *Model) {
	// Check for gradient overflow (simplified)
	hasInfGrad := false

	for _, layer := range model.layers {
		if !layer.IsTrainable() {
			continue
		}

		grads := layer.GetGradients()
		for _, grad := range grads {
			if grad != nil {
				rows, cols := grad.Dims()
				for i := 0; i < rows; i++ {
					for j := 0; j < cols; j++ {
						val := grad.At(i, j)
						if math.IsInf(val, 0) || math.IsNaN(val) {
							hasInfGrad = true
							break
						}
					}
					if hasInfGrad {
						break
					}
				}
			}
			if hasInfGrad {
				break
			}
		}
		if hasInfGrad {
			break
		}
	}

	if hasInfGrad {
		// Skip update and reduce scale
		gs.scale *= gs.backoffRate
		gs.counter = 0
	} else {
		// Unscale gradients and update
		for _, layer := range model.layers {
			if !layer.IsTrainable() {
				continue
			}

			weights := layer.GetWeights()
			grads := layer.GetGradients()

			for i, grad := range grads {
				if grad != nil && i < len(weights) {
					// Unscale gradients
					grad.Scale(1.0/gs.scale, grad)
					optimizer.Update(weights[i], grad)
				}
			}
		}

		optimizer.Step()

		// Try to increase scale
		gs.counter++
		if gs.counter >= gs.interval {
			gs.scale *= gs.growthRate
			gs.counter = 0
		}
	}
}

func NewMixedPrecisionTrainer(model *Model, optimizer Optimizer) *MixedPrecisionTrainer {
	return &MixedPrecisionTrainer{
		model:     model,
		optimizer: optimizer,
		scaler:    NewGradScaler(),
		enabled:   true,
	}
}

// ============ Utility Functions ============

func extractCols(m *mat.Dense, startCol, endCol int) *mat.Dense {
	rows, _ := m.Dims()
	cols := endCol - startCol
	data := make([]float64, rows*cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data[i*cols+j] = m.At(i, startCol+j)
		}
	}

	return mat.NewDense(rows, cols, data)
}

func getTotalParams(model *Model) int {
	total := 0
	for _, layer := range model.layers {
		total += layer.GetParams()
	}
	return total
}

func (m *Model) Build(inputShape []int) {
	currentShape := inputShape
	for _, layer := range m.layers {
		// This logic is primarily for layers that need their input dimension
		// to be explicitly set before they can report their parameters, like Dense.
		if dense, ok := layer.(*Dense); ok && dense.weights == nil {
			if len(currentShape) > 1 {
				// Assumes flattened input for dense layer if coming from multi-dim layer
				dense.Build(currentShape[len(currentShape)-1])
			}
		}
		// Add similar build logic for other layer types like LSTM, Conv2D if they
		// cannot infer their weights' shape from just the `units` parameter.
		currentShape = layer.GetOutputShape(currentShape)
	}
}

func CreateVisionTransformer(imageSize, patchSize, dModel, numHeads, numLayers, numClasses int) *Model {
	model := NewModel()

	// 1. Patch Embedding Layer
	// This is a simplification. A real implementation would first break the image
	// into patches and then flatten them. Here we assume the input is already a
	// flattened image and use a single Dense layer for linear projection.
	patchDim := patchSize * patchSize * 3 // e.g., 16x16x3 for RGB
	model.Add(NewDense(dModel, &Linear{}, WithInputDim(patchDim)))

	// 2. Positional Encoding (simulated with Dropout)
	// A real implementation would add a learnable or fixed positional embedding vector.
	model.Add(NewDropout(0.1))

	// 3. Transformer Encoder Blocks
	for i := 0; i < numLayers; i++ {
		model.Add(NewTransformerBlock(dModel, numHeads, dModel*4))
	}

	// 4. Classification Head
	// In a real ViT, you'd typically take the output corresponding to the [CLS] token.
	// Here we simplify by just passing the whole sequence to the LayerNorm.
	model.Add(NewLayerNorm(dModel))
	model.Add(NewDense(numClasses, &Softmax{}))

	return model
}

func DemonstrateAdvancedFeatures() {
	fmt.Println("🔬 ThinkingNet Advanced Features Demonstration")
	fmt.Println(strings.Repeat("=", 60))

	// === 1. Transformer for Text Classification ===
	fmt.Println("\n1️⃣  Building BERT-like Transformer for Text Classification")

	vocabSize := 10000
	maxSeqLen := 128
	dModel := 256
	numHeads := 8
	numLayers := 6
	numClasses := 5

	bert := CreateBERT(vocabSize, dModel, numHeads, numLayers, maxSeqLen)
	bert.Add(NewDense(numClasses, &Softmax{})) // Classification head
	bert.Build([]int{-1, maxSeqLen})           // Build the model to calculate params
	fmt.Printf("✅ BERT model created with %d parameters\n", getTotalParams(bert))

	// === 2. Vision Transformer ===
	fmt.Println("\n2️⃣  Creating Vision Transformer (ViT)")

	vit := NewViTBase(false)
	fmt.Printf("✅ ViT-Base created: %dx%d patches, %d embed dim, %d layers\n",
		vit.patchSize, vit.patchSize, vit.embedDim, vit.numLayers)

	// === 3. Hyperparameter Optimization Demo ===
	fmt.Println("\n3️⃣  Hyperparameter Optimization Demo")

	searchSpace := HyperparameterConfig{
		LearningRate: []float64{0.001, 0.005, 0.01},
		BatchSize:    []int{32, 64},
		Epochs:       []int{10, 20}, // Increased for meaningful training
		Dropout:      []float64{0.2, 0.4},
	}

	tuner := NewHyperparameterTuner(searchSpace, "accuracy", "maximize")

	// Create sample data for tuning
	X, y := GenerateBlobsData(500, 10, 3, 1.5) // Slightly more complex data
	yLabels := make([]int, y.RawMatrix().Rows)
	for i := 0; i < y.RawMatrix().Rows; i++ {
		yLabels[i] = int(y.At(i, 0))
	}
	yOneHot := ToOneHot(yLabels, 3)

	XTrain, XVal, yTrain, yVal := trainTestSplit(X, yOneHot, 0.2, 42)
	trainDataset := NewTensorDataset(XTrain, yTrain)
	valDataset := NewTensorDataset(XVal, yVal)

	// Model builder function
	buildModel := func(config map[string]interface{}) *Model {
		model := NewModel()
		dropout := 0.3
		if d, ok := config["dropout"].(float64); ok {
			dropout = d
		}

		model.Add(NewDense(64, &ReLU{}, WithInputDim(10)))
		model.Add(NewDropout(dropout))
		model.Add(NewDense(32, &ReLU{}))
		model.Add(NewDropout(dropout))
		model.Add(NewDense(3, &Softmax{}))

		lr := 0.001
		if l, ok := config["learning_rate"].(float64); ok {
			lr = l
		}

		optimizer := NewAdam(lr)
		model.Compile(optimizer, &CategoricalCrossentropy{})

		return model
	}

	// Run hyperparameter search (limited trials for demo)
	bestTrial := tuner.RandomSearch(2, buildModel, trainDataset, valDataset)
	fmt.Printf("✅ Best hyperparameters found: Score=%.4f\n", bestTrial.Score)

	// === 4. Advanced Data Augmentation ===
	fmt.Println("\n4️⃣  Data Augmentation Pipeline")

	augmentation := NewCompose(
		NewRandomNoise(0.05),
		NewRandomRotation(15.0),
		NewRandomHorizontalFlip(0.5),
	)

	// Apply augmentation
	originalData, _ := GenerateCirclesData(100, 0.1, 0.3)
	_ = augmentation.Apply(originalData)

	fmt.Printf("✅ Data augmentation applied to %d samples\n", originalData.RawMatrix().Rows)

	// === 5. Model Zoo Usage ===
	fmt.Println("\n5️⃣  Model Zoo Demonstration")

	zoo := NewModelZoo()
	availableModels := zoo.ListModels()
	fmt.Printf("📚 Available models: %v\n", availableModels)

	// Load pretrained model
	resnet, err := zoo.LoadModel("resnet50", true)
	if err == nil {
		fmt.Println("✅ ResNet50 loaded successfully")

		// Fine-tune for custom task
		customModel := resnet.FineTune(10, true)
		fmt.Printf("✅ Fine-tuned model created for 10 classes\n")
		_ = customModel
	}

	// === 6. Advanced Evaluation Metrics ===
	fmt.Println("\n6️⃣  Advanced Evaluation Metrics")

	// Create sample predictions for metrics demo
	nSamples := 200
	numClassesMetrics := 3

	yTrueData := make([]float64, nSamples*numClassesMetrics)
	yPredData := make([]float64, nSamples*numClassesMetrics)

	// Generate more realistic sample predictions
	for i := 0; i < nSamples; i++ {
		trueClass := rand.Intn(numClassesMetrics)
		yTrueData[i*numClassesMetrics+trueClass] = 1.0

		// Make prediction likely to be correct but not always
		predClass := trueClass
		if rand.Float64() < 0.2 { // 20% chance of being wrong
			predClass = rand.Intn(numClassesMetrics)
		}

		probs := make([]float64, numClassesMetrics)
		for j := 0; j < numClassesMetrics; j++ {
			if j == predClass {
				probs[j] = rand.Float64()*0.5 + 0.5 // High probability for predicted class
			} else {
				probs[j] = rand.Float64() * 0.2
			}
		}
		softmaxProbs := applySoftmax(probs)
		for j, prob := range softmaxProbs {
			yPredData[i*numClassesMetrics+j] = prob
		}
	}
	yTrue := mat.NewDense(nSamples, numClassesMetrics, yTrueData)
	yPred := mat.NewDense(nSamples, numClassesMetrics, yPredData)

	metrics := NewMetrics(numClassesMetrics)
	metrics.Update(yTrue, yPred)
	metrics.PrintReport()

	// === 7. Mixed Precision Training Demo ===
	fmt.Println("\n7️⃣  Mixed Precision Training")

	mpModel := NewModel()
	mpModel.Add(NewDense(128, &ReLU{}))
	mpModel.Add(NewDense(64, &ReLU{}))
	mpModel.Add(NewDense(3, &Softmax{}))
	mpOptimizer := NewAdam(0.001)
	mpModel.Compile(mpOptimizer, &CategoricalCrossentropy{})
	mpTrainer := NewMixedPrecisionTrainer(mpModel, mpOptimizer)
	fmt.Printf("✅ Mixed precision trainer initialized (scale: %.0f)\n", mpTrainer.scaler.scale)

	// === 8. Gradient Accumulation ===
	fmt.Println("\n8️⃣  Gradient Accumulation")

	accumulator := NewGradientAccumulator(4) // Accumulate over 4 steps
	fmt.Println("✅ Gradient accumulator initialized (4 steps)")

	// Simulate training with gradient accumulation
	for step := 1; step <= 8; step++ {
		accumulator.AccumulateGradients(mpModel)

		if accumulator.ShouldUpdate() {
			accumulator.GetAverageGradients(mpModel)
			fmt.Printf("   📊 Gradient update at step %d\n", step)
			accumulator.Reset()
		}
	}

	// === 9. Complex Architecture (U-Net) ===
	fmt.Println("\n9️⃣  Complex Architecture: U-Net for Segmentation")

	unet := CreateUNet(3, 1) // 3 input channels, 1 output channel
	fmt.Printf("✅ U-Net created with skip connections\n")

	// Test forward pass
	demoInput := map[string]*mat.Dense{
		"input": mat.NewDense(1, 256, randomNormal(256, 0.1)), // Flattened image
	}
	unetOutput := unet.Forward(demoInput)
	rows, cols := unetOutput["output"].Dims()
	fmt.Printf("   🔍 U-Net output shape: [%d, %d]\n", rows, cols)

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("🎉 All advanced features demonstrated successfully!")
	fmt.Println("✨ ThinkingNet is now a full-featured deep learning framework")
	fmt.Println(strings.Repeat("=", 60))
}

func mainxxxxxxx() {
	fmt.Println("🚀 ThinkingNet Advanced - Modern Deep Learning Framework")
	fmt.Println(strings.Repeat("=", 60))

	// Setup GPU device if available
	var device Device = &CPUDevice{}
	if false { // In practice, check for CUDA availability
		fmt.Println("🎯 Using CUDA device for demonstration")
		device = NewCUDADevice(0) // Would use GPU if available
	} else {
		fmt.Println("🎯 Using CPU device for demonstration")
	}

	// === Advanced Spiral Classification with Modern Features ===
	rand.Seed(42)

	// Generate more complex spiral dataset
	nClasses := 4
	X, y := GenerateSpiralData(2000, nClasses, 0.15)

	yLabels := make([]int, y.RawMatrix().Rows)
	for i := 0; i < y.RawMatrix().Rows; i++ {
		yLabels[i] = int(y.At(i, 0))
	}
	yOneHot := ToOneHot(yLabels, nClasses)

	// Enhanced data preprocessing
	scaler := NewStandardScaler()
	X = scaler.FitTransform(X)

	// Apply data augmentation
	noise := NewRandomNoise(0.02)
	X = noise.Apply(X)

	// Split data
	XTrain, XTest, yTrain, yTest := trainTestSplit(X, yOneHot, 0.15, 42)

	// === Build Advanced Model Architecture ===
	model := NewModel()
	model.SetDevice(device)

	// Feature extraction layers with modern activations
	model.Add(NewDense(128, NewLeakyReLU(0.01), WithInputDim(2)))
	model.Add(NewBatchNorm())
	model.Add(NewDropout(0.3))
	model.Add(NewDense(64, NewLeakyReLU(0.01)))
	model.Add(NewBatchNorm())
	model.Add(NewDropout(0.3))
	model.Add(NewDense(nClasses, &Softmax{})) // Final classification layer

	// Build the model to correctly calculate parameters before training
	model.Build([]int{-1, 2})
	fmt.Printf("✅ Model created with %d parameters\n", getTotalParams(model))

	// Compile model with advanced optimizer
	optimizer := NewAdam(0.001)
	loss := &CategoricalCrossentropy{}
	model.Compile(optimizer, loss)

	// === Train Model with Advanced Features ===
	fmt.Println("\n🔄 Training model with advanced features...")
	trainConfig := &Config{
		LearningRate: 0.001,
		BatchSize:    64,
		Epochs:       100,
		Verbose:      10,
	}
	trainDataset := NewTensorDataset(XTrain, yTrain)
	trainLoader := NewDataLoader(trainDataset, trainConfig.BatchSize, WithShuffle(true))

	validationDataset := NewTensorDataset(XTest, yTest)
	valLoader := NewDataLoader(validationDataset, trainConfig.BatchSize, WithShuffle(false))

	model.FitGenerator(trainLoader, trainConfig, valLoader)
	fmt.Println("✅ Training completed")

	// === Evaluate Model with Advanced Metrics ===
	fmt.Println("\n📊 Evaluating model with advanced metrics...")
	yPred := model.Predict(XTest)
	metrics := NewMetrics(nClasses)
	metrics.Update(yTest, yPred)
	metrics.PrintReport()
	fmt.Printf("   🔍 Final accuracy: %.4f\n", metrics.GetAccuracy())
	fmt.Printf("   🔍 Macro F1 Score: %.4f\n", metrics.GetMacroF1())
	fmt.Println("✅ Evaluation completed")

	// === Demonstrate Advanced Features ===
	DemonstrateAdvancedFeatures()
	fmt.Println("✨ Advanced features demonstration completed")
	fmt.Println("🚀 ThinkingNet Advanced is ready for modern deep learning tasks!")
	fmt.Println(strings.Repeat("=", 60))
}

func addRadialFeature(X *mat.Dense) *mat.Dense {
	rows, _ := X.Dims()
	// بيانات جديدة بعمود إضافي للميزة الجديدة
	newData := make([]float64, rows*3)

	for i := 0; i < rows; i++ {
		x := X.At(i, 0)
		y := X.At(i, 1)
		// حساب الميزة الجديدة والمفيدة
		radius := math.Sqrt(x*x + y*y)

		newData[i*3+0] = x
		newData[i*3+1] = y
		newData[i*3+2] = radius // الميزة الجديدة التي تساعد على التعلم
	}

	return mat.NewDense(rows, 3, newData)
}

func main() {
	fmt.Println("🚀 اختبار النموذج على بيانات دائرية معقدة مع إضافة مزايا (هندسة المزايا)")
	fmt.Println(strings.Repeat("=", 70))

	// الخطوة 1: توليد بيانات معقدة (نمط دوائر فوضوي)
	// هذه البيانات لا يمكن فصلها باستخدام خط مستقيم
	X, y_vec := GenerateCirclesData(1000, 0.1, 0.5)
	fmt.Println("✅ تم توليد 1000 نقطة بيانات بنمط دوائر متداخلة.")

	// الخطوة 2: هندسة المزايا - إضافة ميزة مفيدة
	// نضيف ميزة "نصف القطر" التي ستجعل المشكلة سهلة للغاية للنموذج
	fmt.Println("... إضافة ميزة نصف القطر لمساعدة النموذج على التعلم.")
	X_featured := addRadialFeature(X)

	// الخطوة 3: تجهيز وتقسيم البيانات
	// تحويل y إلى صيغة one-hot وتقسيم البيانات إلى مجموعات تدريب واختبار
	y_col_float := mat.Col(nil, 0, y_vec) // استخراج العمود كـ []float64

	// *** الإصلاح هنا: تحويل []float64 إلى []int ***
	y_col_int := make([]int, len(y_col_float))
	for i, v := range y_col_float {
		y_col_int[i] = int(v)
	}
	// ***********************************************

	yOneHot := ToOneHot(y_col_int, 2) // الآن نمرر النوع الصحيح
	XTrain, XTest, yTrain, yTest := trainTestSplit(X_featured, yOneHot, 0.2, 42)

	// الخطوة 4: بناء نموذج بسيط
	// مع الميزة الجديدة، لا نحتاج إلى نموذج معقد
	model := NewModel()
	// مدخلات النموذج الآن بحجم 3 (x, y, radius) بدلاً من 2
	model.Add(NewDense(10, &ReLU{}, WithInputDim(3)))
	model.Add(NewDense(10, &ReLU{}))
	model.Add(NewDense(2, &Softmax{})) // مخرجات لفئتين

	model.Build([]int{-1, 3})
	fmt.Printf("✅ تم بناء النموذج بـ %d مُعامِلاً (parameters)\n", getTotalParams(model))

	// الخطوة 5: تجميع وتدريب النموذج
	optimizer := NewAdam(0.005)
	loss := &CategoricalCrossentropy{}
	model.Compile(optimizer, loss)

	fmt.Println("\n🔄 بدء تدريب النموذج على البيانات المُحسَّنة...")
	config := &Config{
		Epochs:    50,
		BatchSize: 32,
		Verbose:   10,
	}
	model.Fit(XTrain, yTrain, config)
	fmt.Println("✅ اكتمل التدريب.")

	// الخطوة 6: تقييم أداء النموذج
	// نتوقع دقة عالية جدًا بفضل الميزة الجديدة
	fmt.Println("\n📊 تقييم أداء النموذج على بيانات الاختبار...")
	_, testAccuracy := model.Evaluate(XTest, yTest, 1)
	fmt.Printf("\n🎯 دقة النموذج النهائية على بيانات الاختبار: %.2f%%\n", testAccuracy*100)
	fmt.Println(strings.Repeat("=", 70))
}
