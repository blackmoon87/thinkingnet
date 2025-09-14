// Package core defines the fundamental interfaces and types for the ThinkingNet AI library.
package core

import (
	"gonum.org/v1/gonum/mat"
)

// Tensor represents a multi-dimensional array with mathematical operations.
type Tensor interface {
	// Basic operations
	Dims() (int, int)
	At(i, j int) float64
	Set(i, j int, v float64)
	Copy() Tensor

	// Arithmetic operations
	Add(other Tensor) Tensor
	Sub(other Tensor) Tensor
	Mul(other Tensor) Tensor
	MulElem(other Tensor) Tensor
	Div(other Tensor) Tensor
	Scale(scalar float64) Tensor

	// Mathematical functions
	Pow(power float64) Tensor
	Sqrt() Tensor
	Exp() Tensor
	Log() Tensor
	Abs() Tensor
	Sign() Tensor
	Clamp(min, max float64) Tensor

	// Linear algebra
	T() Tensor

	// Statistics
	Sum() float64
	Mean() float64
	Std() float64
	Max() float64
	Min() float64
	Norm() float64

	// Shape operations
	Reshape(newRows, newCols int) Tensor
	Flatten() Tensor
	Shape() []int

	// Utility operations
	Apply(fn func(i, j int, v float64) float64) Tensor
	Equal(other Tensor) bool
	Fill(value float64)
	Zero()
	Release()

	// Slicing and indexing
	Row(i int) Tensor
	Col(j int) Tensor
	Slice(r0, r1, c0, c1 int) Tensor
	SetRow(i int, data []float64)
	SetCol(j int, data []float64)

	// Properties
	IsEmpty() bool
	IsSquare() bool
	IsVector() bool

	// Metadata
	Name() string
	SetName(name string)
	String() string

	// Additional operations
	Dot(other Tensor) float64
	AddScalar(scalar float64) Tensor
	SubScalar(scalar float64) Tensor
	DivScalar(scalar float64) Tensor
	Trace() float64
	Diagonal() Tensor

	// Validation and checks
	Validate() error
	HasNaN() bool
	HasInf() bool
	IsFinite() bool

	// Low-level access
	RawMatrix() *mat.Dense
}

// Layer represents a neural network layer.
type Layer interface {
	// Forward performs the forward pass
	Forward(input Tensor) (Tensor, error)

	// Backward performs the backward pass and returns input gradients
	Backward(gradient Tensor) (Tensor, error)

	// Parameters returns all trainable parameters
	Parameters() []Tensor

	// Gradients returns gradients for all parameters
	Gradients() []Tensor

	// IsTrainable returns true if the layer has trainable parameters
	IsTrainable() bool

	// Name returns the layer name
	Name() string

	// SetName sets the layer name
	SetName(name string)

	// OutputShape returns the output shape given input shape
	OutputShape(inputShape []int) ([]int, error)

	// ParameterCount returns the number of trainable parameters
	ParameterCount() int
}

// Loss represents a loss function.
type Loss interface {
	// Compute calculates the loss value
	Compute(yTrue, yPred Tensor) float64

	// Gradient computes the gradient of the loss
	Gradient(yTrue, yPred Tensor) Tensor

	// Name returns the loss function name
	Name() string
}

// Optimizer represents an optimization algorithm.
type Optimizer interface {
	// Update updates parameters using gradients
	Update(params []Tensor, grads []Tensor)

	// Step increments the optimizer step counter
	Step()

	// Reset resets the optimizer state
	Reset()

	// Config returns the optimizer configuration
	Config() OptimizerConfig

	// Name returns the optimizer name
	Name() string

	// LearningRate returns the current learning rate
	LearningRate() float64

	// SetLearningRate sets the learning rate
	SetLearningRate(lr float64)
}

// Model represents a complete machine learning model.
type Model interface {
	// Forward performs forward pass through the model
	Forward(input Tensor) (Tensor, error)

	// Backward performs backward pass
	Backward(loss Loss, yTrue, yPred Tensor) error

	// Fit trains the model
	Fit(X, y Tensor, config TrainingConfig) (*History, error)

	// Predict makes predictions
	Predict(X Tensor) (Tensor, error)

	// Evaluate evaluates the model performance
	Evaluate(X, y Tensor) (*Metrics, error)

	// Summary returns a string representation of the model
	Summary() string

	// Save saves the model to a file
	Save(path string) error

	// Load loads the model from a file
	Load(path string) error

	// Compile compiles the model with optimizer and loss
	Compile(optimizer Optimizer, loss Loss) error

	// AddLayer adds a layer to the model
	AddLayer(layer Layer) error

	// Layers returns all layers in the model
	Layers() []Layer
}

// Activation represents an activation function.
type Activation interface {
	// Forward applies the activation function
	Forward(x float64) float64

	// Backward computes the derivative
	Backward(x float64) float64

	// Name returns the activation function name
	Name() string
}

// Preprocessor represents a data preprocessing component.
type Preprocessor interface {
	// Fit learns parameters from the data
	Fit(data Tensor) error

	// Transform applies the transformation
	Transform(data Tensor) (Tensor, error)

	// FitTransform fits and transforms in one step
	FitTransform(data Tensor) (Tensor, error)

	// InverseTransform reverses the transformation
	InverseTransform(data Tensor) (Tensor, error)

	// IsFitted returns true if the preprocessor has been fitted
	IsFitted() bool

	// Name returns the preprocessor name
	Name() string
}

// Encoder represents a categorical data encoder.
type Encoder interface {
	// Fit learns the encoding from categorical data
	Fit(data []string) error

	// Transform encodes categorical data to numerical
	Transform(data []string) (Tensor, error)

	// FitTransform fits and transforms in one step
	FitTransform(data []string) (Tensor, error)

	// Classes returns the learned classes
	Classes() []string

	// IsFitted returns true if the encoder has been fitted
	IsFitted() bool

	// Name returns the encoder name
	Name() string
}

// Clusterer represents a clustering algorithm.
type Clusterer interface {
	// Fit learns cluster parameters from data
	Fit(X Tensor) error

	// Predict assigns cluster labels to data
	Predict(X Tensor) ([]int, error)

	// FitPredict fits and predicts in one step
	FitPredict(X Tensor) ([]int, error)

	// ClusterCenters returns the cluster centers
	ClusterCenters() Tensor

	// Name returns the clusterer name
	Name() string
}

// Classifier represents a classification algorithm.
type Classifier interface {
	// Fit trains the classifier
	Fit(X, y Tensor) error

	// Predict makes class predictions
	Predict(X Tensor) (Tensor, error)

	// PredictProba predicts class probabilities
	PredictProba(X Tensor) (Tensor, error)

	// Score returns the accuracy score
	Score(X, y Tensor) (float64, error)

	// Name returns the classifier name
	Name() string
}

// Regressor represents a regression algorithm.
type Regressor interface {
	// Fit trains the regressor
	Fit(X, y Tensor) error

	// Predict makes predictions
	Predict(X Tensor) (Tensor, error)

	// Score returns the R² score
	Score(X, y Tensor) (float64, error)

	// Name returns the regressor name
	Name() string
}
