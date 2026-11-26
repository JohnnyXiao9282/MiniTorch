# MiniTorch

A lightweight deep learning framework built from scratch, inspired by PyTorch and Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Project structure

```text
MiniTorch
├── Core          Value (scalar), Tensor, math operations
├── Autograd      Computational graph, topological sort, backpropagation
├── Neural Network Module, layers, activation functions, loss functions
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

## Requirements

- Core algorithms implemented from scratch
- NumPy only where necessary
- Modular architecture
- Unit tests for each component

## Getting started

```bash
pip install pytest
python -m pytest tests/ -v
```

## Status

Work in progress. The `Value` class scaffold is in place; operations are being implemented step by step.
