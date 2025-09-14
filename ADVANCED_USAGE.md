# ThinkingNet-Go Advanced Usage Guide

This guide covers advanced usage patterns and configurations for experienced developers who need fine-grained control over the library's behavior.

## Table of Contents

1. [Advanced Model Configuration](#advanced-model-configuration)
2. [Custom Training Configurations](#custom-training-configurations)
3. [Advanced Algorithms Configuration](#advanced-algorithms-configuration)
4. [Custom Preprocessing Pipelines](#custom-preprocessing-pipelines)
5. [Performance Optimization](#performance-optimization)
6. [Custom Callbacks and Monitoring](#custom-callbacks-and-monitoring)
7. [Low-Level Tensor Operations](#low-level-tensor-operations)
8. [Advanced Error Handling](#advanced-error-handling)

## Advanced Model Configuration

### Sequential Model with Custom Options

```go
package main

import (
    "thinkingnet/pkg/core"
    "thinkingnet/pkg/models"
    "thinkingnet/pkg/layers"
    "thinkingnet/pkg/activations"
    "thinkingnet/pkg/optimizers"
    "thinkingnet/pkg/losses"
)

func advancedModelSetup() {
    // Create model with custom configuration
    model := models.NewSequential(
        core.WithModelName("AdvancedClassifier"),
        core.WithModelSeed(12345),
        core.WithModelValidation(true),
    )

    // Add layers with specific configurations
    model.AddLayer(layers.NewDense(128, activations.NewReLU()))
    model.AddLayer(layers.NewDropout(0.3)) // 30% dropout
    model.AddLayer(layers.NewDense(64, activations.NewReLU()))
    model.AddLayer(layers.NewDropout(0.2))
    model.AddLayer(layers.NewDense(10, activations.NewSoftmax()))

    // Advanced optimizer configuration
    optimizer := optimizers.NewAdam(
        optimizers.WithAdamLearningRate(0.001),
        optimizers.WithAdamBeta1(0.9),
        optimizers.WithAdamBeta2(0.999),
        optimizers.WithAdamEpsilon(1e-8),
        optimizers.WithAdamWeightDecay(0.01),
        optimizers.WithAdamAMSGrad(true),
    )

    // Advanced loss configuration
    loss := losses.NewCategoricalCrossEntropy(
        losses.WithLabelSmoothing(0.1),
        losses.WithFromLogits(false),
    )

    model.Compile(optimizer, loss)
}
```

## Custom Training Configurations

### Comprehensive Training Setup

```go
func advancedTraining(model *models.Sequential, X, y core.Tensor) {
    // Advanced training configuration
    config := core.TrainingConfig{
        Epochs:          100,
        BatchSize:       64,
        ValidationSplit: 0.2,
        Shuffle:         true,
        Verbose:         10, // Log every 10 epochs
        Seed:            42,
        
        // Advanced metrics
        Metrics: []string{
            "accuracy", 
            "precision", 
            "recall", 
            "f1_score",
            "auc",
        },
        
        // Early stopping configuration
        EarlyStopping: core.EarlyStoppingConfig{
            Enabled:         true,
            Monitor:         "val_loss",
            Patience:        15,
            MinDelta:        0.001,
            Mode:           "min",
            RestoreBestWeights: true,
        },
        
        // Learning rate scheduling
        LearningRateSchedule: core.LRScheduleConfig{
            Enabled:     true,
            Type:        "exponential_decay",
            InitialLR:   0.001,
            DecayRate:   0.95,
            DecaySteps:  10,
        },
        
        // Data augmentation
        DataAugmentation: core.AugmentationConfig{
            Enabled:    true,
            Rotation:   15.0,
            Translation: 0.1,
            Scaling:    0.1,
            Noise:      0.01,
        },
    }

    history, err := model.Fit(X, y, config)
    if err != nil {
        panic(err)
    }
    
    // Analyze training history
    analyzeTrainingHistory(history)
}

func analyzeTrainingHistory(history *core.History) {
    fmt.Printf("Training completed in %v\n", history.Duration)
    fmt.Printf("Best epoch: %d\n", history.BestEpoch)
    fmt.Printf("Best score: %.4f\n", history.BestScore)
    
    // Plot learning curves (if visualization library available)
    // plotLearningCurves(history)
}
```

## Advanced Algorithms Configuration

### Linear Regression with Full Configuration

```go
func advancedLinearRegression() {
    // Create linear regression with all options
    lr := algorithms.NewLinearRegression(
        algorithms.WithLinearLearningRate(0.01),
        algorithms.WithLinearMaxIterations(2000),
        algorithms.WithLinearTolerance(1e-8),
        algorithms.WithLinearRegularization("elastic", 0.1),
        algorithms.WithLinearElasticNet(0.1, 0.7), // 70% L1, 30% L2
        algorithms.WithLinearFitIntercept(true),
        algorithms.WithLinearRandomSeed(42),
        algorithms.WithLinearSolver("gradient_descent"),
    )

    // Advanced data preparation
    X, y := prepareAdvancedData()
    
    // Fit with validation
    err := lr.Fit(X, y)
    if err != nil {
        panic(err)
    }

    // Get detailed model information
    weights := lr.Weights()
    fmt.Printf("Model converged: %v\n", lr.Converged())
    fmt.Printf("Iterations: %d\n", lr.Iterations())
    fmt.Printf("Final weights: %v\n", weights)
    
    // Advanced evaluation
    predictions, _ := lr.Predict(X)
    metrics := algorithms.CalculateRegressionMetrics(y, predictions)
    
    fmt.Printf("MSE: %.4f\n", metrics.MSE)
    fmt.Printf("RMSE: %.4f\n", metrics.RMSE)
    fmt.Printf("MAE: %.4f\n", metrics.MAE)
    fmt.Printf("R² Score: %.4f\n", metrics.R2Score)
}
```

### Logistic Regression with Advanced Options

```go
func advancedLogisticRegression() {
    clf := algorithms.NewLogisticRegression(
        algorithms.WithLogisticLearningRate(0.01),
        algorithms.WithLogisticMaxIterations(1000),
        algorithms.WithLogisticTolerance(1e-6),
        algorithms.WithLogisticRegularization("l2", 0.01),
        algorithms.WithLogisticSolver("lbfgs"),
        algorithms.WithLogisticMultiClass("ovr"), // One-vs-Rest
        algorithms.WithLogisticClassWeight("balanced"),
        algorithms.WithLogisticRandomSeed(42),
    )

    X, y := prepareClassificationData()
    
    err := clf.Fit(X, y)
    if err != nil {
        panic(err)
    }

    // Get prediction probabilities
    probabilities, _ := clf.PredictProba(X)
    predictions, _ := clf.Predict(X)
    
    // Advanced classification metrics
    metrics := algorithms.CalculateClassificationMetrics(y, predictions, probabilities)
    
    fmt.Printf("Accuracy: %.4f\n", metrics.Accuracy)
    fmt.Printf("Precision: %.4f\n", metrics.Precision)
    fmt.Printf("Recall: %.4f\n", metrics.Recall)
    fmt.Printf("F1-Score: %.4f\n", metrics.F1Score)
    fmt.Printf("AUC-ROC: %.4f\n", metrics.AUCROC)
}
```

### K-Means with Advanced Configuration

```go
func advancedKMeans() {
    kmeans := algorithms.NewKMeans(
        algorithms.WithKMeansNClusters(5),
        algorithms.WithKMeansInitMethod("k-means++"),
        algorithms.WithKMeansMaxIterations(300),
        algorithms.WithKMeansTolerance(1e-4),
        algorithms.WithKMeansNInit(10), // Run 10 times with different initializations
        algorithms.WithKMeansRandomSeed(42),
        algorithms.WithKMeansAlgorithm("lloyd"), // or "elkan"
    )

    X := prepareClusteringData()
    
    labels, err := kmeans.FitPredict(X)
    if err != nil {
        panic(err)
    }

    // Get cluster information
    centers := kmeans.ClusterCenters()
    inertia := kmeans.Inertia()
    nIters := kmeans.NIterations()
    
    fmt.Printf("Clustering completed in %d iterations\n", nIters)
    fmt.Printf("Final inertia: %.4f\n", inertia)
    fmt.Printf("Cluster centers shape: %v\n", centers.Shape())
    
    // Advanced clustering evaluation
    silhouetteScore := algorithms.CalculateSilhouetteScore(X, labels)
    calinskiScore := algorithms.CalculateCalinskiHarabaszScore(X, labels)
    
    fmt.Printf("Silhouette Score: %.4f\n", silhouetteScore)
    fmt.Printf("Calinski-Harabasz Score: %.4f\n", calinskiScore)
}
```

## Custom Preprocessing Pipelines

### Advanced Data Preprocessing

```go
func advancedPreprocessing(X core.Tensor) core.Tensor {
    // Create preprocessing pipeline
    pipeline := preprocessing.NewPipeline()
    
    // Add multiple preprocessing steps
    pipeline.AddStep(preprocessing.NewStandardScaler(
        preprocessing.WithScalerCopy(true),
        preprocessing.WithScalerWithMean(true),
        preprocessing.WithScalerWithStd(true),
    ))
    
    pipeline.AddStep(preprocessing.NewPCA(
        preprocessing.WithPCANComponents(50),
        preprocessing.WithPCAWhiten(true),
        preprocessing.WithPCASolver("auto"),
        preprocessing.WithPCARandomSeed(42),
    ))
    
    pipeline.AddStep(preprocessing.NewPolynomialFeatures(
        preprocessing.WithPolyDegree(2),
        preprocessing.WithPolyIncludeBias(false),
        preprocessing.WithPolyInteractionOnly(false),
    ))
    
    // Fit and transform
    XTransformed, err := pipeline.FitTransform(X)
    if err != nil {
        panic(err)
    }
    
    return XTransformed
}

// Custom feature selection
func advancedFeatureSelection(X, y core.Tensor) core.Tensor {
    selector := preprocessing.NewSelectKBest(
        preprocessing.WithSelectorScoreFunc("f_classif"),
        preprocessing.WithSelectorK(20),
    )
    
    XSelected, err := selector.FitTransform(X, y)
    if err != nil {
        panic(err)
    }
    
    // Get selected feature indices
    selectedFeatures := selector.GetSupport()
    fmt.Printf("Selected features: %v\n", selectedFeatures)
    
    return XSelected
}
```

## Performance Optimization

### Batch Processing and Parallelization

```go
func optimizedPrediction(model *models.Sequential, inputs []core.Tensor) {
    // Configure batch processor
    batchProcessor := core.GetBatchProcessor()
    batchProcessor.SetBatchSize(128)
    batchProcessor.SetNumWorkers(8)
    batchProcessor.SetBufferSize(1000)
    
    // Process in parallel batches
    results, err := model.PredictBatch(inputs)
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Processed %d inputs in parallel\n", len(results))
}

// Memory optimization
func optimizedMemoryUsage() {
    // Configure memory pool
    pool := core.GetMemoryPool()
    pool.SetMaxSize(1024 * 1024 * 1024) // 1GB
    pool.SetCleanupInterval(time.Minute * 5)
    pool.EnableStatistics(true)
    
    // Use pooled tensors
    tensor := pool.GetTensor(1000, 784)
    defer pool.ReleaseTensor(tensor)
    
    // Check memory statistics
    stats := pool.GetStatistics()
    fmt.Printf("Pool usage: %d/%d tensors\n", stats.InUse, stats.Total)
    fmt.Printf("Memory usage: %d bytes\n", stats.MemoryUsage)
}
```

## Custom Callbacks and Monitoring

### Advanced Callbacks

```go
func setupAdvancedCallbacks(model *models.Sequential) {
    // Learning rate scheduler callback
    lrScheduler := callbacks.NewLearningRateScheduler(
        callbacks.WithSchedulerFunction(func(epoch int, lr float64) float64 {
            if epoch < 10 {
                return lr
            } else if epoch < 20 {
                return lr * 0.5
            } else {
                return lr * 0.1
            }
        }),
    )
    
    // Model checkpoint callback
    checkpoint := callbacks.NewModelCheckpoint(
        callbacks.WithCheckpointFilepath("model_epoch_{epoch:02d}_loss_{val_loss:.4f}.h5"),
        callbacks.WithCheckpointMonitor("val_loss"),
        callbacks.WithCheckpointSaveBestOnly(true),
        callbacks.WithCheckpointMode("min"),
        callbacks.WithCheckpointVerbose(1),
    )
    
    // Custom metrics callback
    metricsLogger := callbacks.NewCustomCallback(
        callbacks.WithOnEpochEnd(func(epoch int, logs map[string]float64) {
            // Custom logging logic
            fmt.Printf("Epoch %d - Custom metrics processing\n", epoch)
            
            // Log to external system (e.g., TensorBoard, Weights & Biases)
            logToExternalSystem(epoch, logs)
        }),
    )
    
    // Add callbacks to model
    model.AddCallback(lrScheduler)
    model.AddCallback(checkpoint)
    model.AddCallback(metricsLogger)
}

func logToExternalSystem(epoch int, logs map[string]float64) {
    // Implementation for external logging
    // e.g., send to monitoring service
}
```

## Low-Level Tensor Operations

### Advanced Tensor Manipulation

```go
func advancedTensorOperations() {
    // Create tensors with specific properties
    tensor1 := core.NewTensor(100, 50)
    tensor1.SetName("InputFeatures")
    
    // Advanced mathematical operations
    tensor2 := tensor1.Pow(2.0).Add(tensor1.Sqrt()).Scale(0.5)
    
    // Statistical operations
    mean := tensor1.Mean()
    std := tensor1.Std()
    norm := tensor1.Norm()
    
    fmt.Printf("Tensor stats - Mean: %.4f, Std: %.4f, Norm: %.4f\n", mean, std, norm)
    
    // Advanced slicing and indexing
    subset := tensor1.Slice(10, 50, 0, 25) // Rows 10-50, Cols 0-25
    row := tensor1.Row(5)
    col := tensor1.Col(10)
    
    // Custom operations with Apply
    processed := tensor1.Apply(func(i, j int, v float64) float64 {
        // Custom transformation logic
        if i%2 == 0 {
            return v * 2.0
        }
        return v * 0.5
    })
    
    // Validation and checks
    if tensor1.HasNaN() {
        fmt.Println("Warning: Tensor contains NaN values")
    }
    
    if !tensor1.IsFinite() {
        fmt.Println("Warning: Tensor contains infinite values")
    }
    
    // Low-level matrix access
    rawMatrix := tensor1.RawMatrix()
    // Use gonum operations directly if needed
    _ = rawMatrix
}
```

## Advanced Error Handling

### Comprehensive Error Management

```go
func advancedErrorHandling() {
    // Enable detailed error context
    core.SetErrorVerbosity(core.ErrorVerbosityHigh)
    core.EnableErrorStackTrace(true)
    
    // Custom error handler
    core.SetErrorHandler(func(err *core.Error) {
        // Custom error processing
        fmt.Printf("Error Code: %s\n", err.Code())
        fmt.Printf("Error Message: %s\n", err.Message())
        fmt.Printf("Error Context: %v\n", err.Context())
        
        // Log to external system
        logErrorToSystem(err)
        
        // Send alerts if critical
        if err.IsCritical() {
            sendAlert(err)
        }
    })
    
    // Graceful error recovery
    model := models.NewSequential()
    
    err := core.WithErrorRecovery(func() error {
        // Operations that might fail
        return model.Compile(nil, nil) // This will fail
    }, func(err error) error {
        // Recovery logic
        fmt.Printf("Recovering from error: %v\n", err)
        
        // Provide default configuration
        optimizer := optimizers.NewAdam()
        loss := losses.NewMeanSquaredError()
        return model.Compile(optimizer, loss)
    })
    
    if err != nil {
        fmt.Printf("Final error after recovery: %v\n", err)
    }
}

func logErrorToSystem(err *core.Error) {
    // Implementation for error logging
}

func sendAlert(err *core.Error) {
    // Implementation for critical error alerts
}
```

## Advanced Configuration Examples

### Complete Advanced Setup

```go
func completeAdvancedExample() {
    // 1. Advanced data preparation
    X, y := loadAndPreprocessData()
    
    // 2. Create advanced model
    model := createAdvancedModel()
    
    // 3. Setup callbacks and monitoring
    setupAdvancedCallbacks(model)
    
    // 4. Configure advanced training
    config := createAdvancedTrainingConfig()
    
    // 5. Train with full monitoring
    history, err := model.Fit(X, y, config)
    if err != nil {
        handleTrainingError(err)
        return
    }
    
    // 6. Advanced evaluation
    performAdvancedEvaluation(model, history)
    
    // 7. Model optimization and deployment
    optimizeAndDeploy(model)
}

func loadAndPreprocessData() (core.Tensor, core.Tensor) {
    // Load data from multiple sources
    X := loadFromMultipleSources()
    y := loadTargets()
    
    // Advanced preprocessing pipeline
    X = advancedPreprocessing(X)
    y = preprocessTargets(y)
    
    return X, y
}

func createAdvancedModel() *models.Sequential {
    model := models.NewSequential(
        core.WithModelName("ProductionModel"),
        core.WithModelSeed(42),
        core.WithModelValidation(true),
    )
    
    // Add sophisticated architecture
    model.AddLayer(layers.NewDense(256, activations.NewReLU()))
    model.AddLayer(layers.NewBatchNormalization())
    model.AddLayer(layers.NewDropout(0.3))
    
    model.AddLayer(layers.NewDense(128, activations.NewReLU()))
    model.AddLayer(layers.NewBatchNormalization())
    model.AddLayer(layers.NewDropout(0.2))
    
    model.AddLayer(layers.NewDense(64, activations.NewReLU()))
    model.AddLayer(layers.NewDropout(0.1))
    
    model.AddLayer(layers.NewDense(10, activations.NewSoftmax()))
    
    // Advanced optimizer
    optimizer := optimizers.NewAdam(
        optimizers.WithAdamLearningRate(0.001),
        optimizers.WithAdamWeightDecay(0.01),
        optimizers.WithAdamAMSGrad(true),
    )
    
    loss := losses.NewCategoricalCrossEntropy(
        losses.WithLabelSmoothing(0.1),
    )
    
    model.Compile(optimizer, loss)
    return model
}
```

## Performance Benchmarking

### Advanced Performance Analysis

```go
func performanceBenchmarking() {
    // Setup benchmarking
    benchmark := core.NewBenchmark()
    benchmark.SetWarmupRuns(5)
    benchmark.SetBenchmarkRuns(100)
    benchmark.EnableMemoryProfiling(true)
    benchmark.EnableCPUProfiling(true)
    
    // Benchmark different configurations
    results := benchmark.RunComparison(map[string]func(){
        "EasyMode": func() {
            model := models.NewSequential()
            // ... easy mode setup
            model.EasyTrain(X, y)
        },
        "AdvancedMode": func() {
            model := createAdvancedModel()
            config := createAdvancedTrainingConfig()
            model.Fit(X, y, config)
        },
        "OptimizedMode": func() {
            model := createOptimizedModel()
            config := createOptimizedTrainingConfig()
            model.Fit(X, y, config)
        },
    })
    
    // Analyze results
    for name, result := range results {
        fmt.Printf("%s - Time: %v, Memory: %d bytes\n", 
            name, result.AverageTime, result.PeakMemory)
    }
}
```

This advanced usage guide demonstrates the full power and flexibility of ThinkingNet-Go beyond the simplified `Easy*` functions. The library provides comprehensive control over every aspect of machine learning workflows while maintaining the option for simplified usage when needed.

The advanced mode offers:
- Fine-grained control over all parameters
- Custom preprocessing pipelines
- Advanced optimization techniques
- Comprehensive monitoring and callbacks
- Performance optimization features
- Detailed error handling and recovery
- Low-level tensor operations
- Production-ready configurations