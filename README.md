# MiniTorch

A lightweight deep learning framework built from scratch, inspired by PyTorch and Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Overview

Training a neural network means repeating one loop:

1. **Forward:** compute a prediction and a loss.
2. **Backward:** compute the gradient of the loss with respect to every parameter.
3. **Update:** move each parameter against its gradient.

The hard part is step 2. MiniTorch implements it with **automatic differentiation**: every value remembers which values it was computed from, and the chain rule is applied backward through that graph.

## Goal

Understand deep learning frameworks by building every core piece ourselves, ending with a network that classifies handwritten digits (MNIST).

- Core algorithms are implemented from scratch.
- NumPy is used only where needed.
- The architecture is modular.
- Every component has unit tests.

## Project structure

```text
MiniTorch
├── Core          Value (scalar), Tensor, math operations
├── Autograd      Computational graph, topological sort, backpropagation
├── Neural Network  Module, layers, activation functions, loss functions
├── Optimizer     SGD, Adam
├── Data          Dataset, DataLoader
├── Tests
└── Examples
```

## Roadmap

1. Scalar `Value` and mathematical operations
2. Automatic differentiation and backpropagation
3. Neural networks and activation functions
4. Loss functions and gradient descent
5. Tensor operations and tensor-level autograd
6. Data loading and mini-batch training
7. MNIST handwritten digit classification

## Running the tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Status

Work in progress. `Value` supports addition and multiplication with backpropagation. More operations are next.
