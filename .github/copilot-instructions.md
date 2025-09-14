# Copilot Instructions for thinkingnet-go

## Project Overview
- This is a Go-based machine learning/neural network framework, with custom implementations for training, loss functions, activations, and data handling.
- Main entry points: `main.go` and `thinkingnet.go`. Most core logic is in `thinkingnet.go`.
- Uses [gonum](https://gonum.org/) for matrix operations and numerical routines.
- JSON files like `intents.json` and `intents_large.json` are used for intent classification tasks (NLP/chatbot).

## Architecture & Key Patterns
- **Config Structs**: Training and model configs are defined in `Config`/`TrainingConfig` structs. Use `NewConfig()` or `NewDefaultConfig()` for defaults.
- **Loss Functions**: Implemented as interfaces (e.g., `LossFunction`), with concrete types for binary/categorical cross-entropy, MSE, etc. See `thinkingnet.go` and `main.go` for examples.
- **Activation Functions**: Defined as interfaces and concrete types (e.g., `Sigmoid`, `ReLU`, `Tanh`).
- **Data Flow**: Data is loaded from JSON, processed into matrices, and passed through model layers for training/evaluation.
- **Multiple Entry Points**: There are several main-like files (e.g., `main.go`, `thinkingnet.go`, `maincopy.go`). Each may run different experiments or variants.

## Developer Workflows
- **Build/Run**: Use `go run thinkingnet.go` or `go run main.go` to execute. Some files are for experiments/copies; prefer `thinkingnet.go` for main logic.
- **Dependencies**: Managed via Go modules. Only `gonum` is required (see `go.mod`).
- **Testing**: No standard Go test files found. Testing is typically done by running main files and inspecting output.
- **Debugging**: Print statements (`fmt.Println`) are used for debugging. No external logging framework.

## Conventions & Tips
- **Numerical Stability**: Constants like `Epsilon` are used to avoid log(0) and division by zero in loss/activation functions.
- **Randomness**: Seeded via config (`Seed`/`RandomSeed`).
- **Batching**: Training uses batch size from config.
- **No REST API or Web UI**: This is a pure Go/CLI project.
- **File Naming**: Many files are copies/experiments (e.g., `thinkingnet copy.go`). Focus on `thinkingnet.go` for canonical logic.

## Integration Points
- **External**: Only `gonum` is used for math. No other external APIs or services.
- **Data**: Expects input data in JSON format for intent classification.

## Examples
- To run the main model: `go run thinkingnet.go`
- To experiment with a different config: modify the `Config` struct or pass different data files.

---

For questions about architecture, always check `thinkingnet.go` first. If adding new model types or loss functions, follow the interface patterns already present.
