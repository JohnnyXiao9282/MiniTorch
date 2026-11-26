"""Scalar Value with reverse-mode automatic differentiation.

Your task: implement the TODO methods. Each operation should build a
computational graph by recording its children (`_prev`) and a local
backward function (`_backward`) that accumulates gradients into them.
"""


class Value:
  """A scalar that tracks its data, gradient, and the graph that produced it."""

  def __init__(self, data, _children=(), _op="", label=""):
    self.data = float(data)
    self.grad = 0.0
    # TODO: store the children as a set/tuple, the op name, and a label.
    self._prev = set(_children)
    self._op = _op
    self.label = label
    # TODO: default _backward to a no-op function (lambda: None).
    self._backward = lambda: None

  def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"

  # ---- Arithmetic operators (forward pass + local backward) ----

  def __add__(self, other):
    # TODO: wrap `other` in Value if needed, compute out = self + other,
    # define out._backward that does: self.grad += 1 * out.grad, etc.
    # Hint: use a closure. Think about why `+=` (not `=`) is required.
    raise NotImplementedError

  def __mul__(self, other):
    # TODO: product rule. d(a*b)/da = b, d(a*b)/db = a.
    raise NotImplementedError

  def __pow__(self, exponent):
    # TODO: only int/float exponents. d(x^n)/dx = n * x^(n-1).
    raise NotImplementedError

  def __neg__(self):
    # TODO: return self * -1
    raise NotImplementedError

  def __sub__(self, other):
    # TODO: return self + (-other)
    raise NotImplementedError

  def __truediv__(self, other):
    # TODO: return self * other**-1
    raise NotImplementedError

  # Reflected operators so that `2 + v` and `2 * v` work.
  def __radd__(self, other):
    return self + other

  def __rmul__(self, other):
    return self * other

  def __rsub__(self, other):
    # TODO: other - self
    raise NotImplementedError

  def __rtruediv__(self, other):
    # TODO: other / self
    raise NotImplementedError

  # ---- Activation / math functions ----

  def exp(self):
    # TODO: d(e^x)/dx = e^x
    raise NotImplementedError

  def log(self):
    # TODO: d(ln x)/dx = 1/x
    raise NotImplementedError

  def tanh(self):
    # TODO: d(tanh x)/dx = 1 - tanh(x)^2. Reuse the output `t`.
    raise NotImplementedError

  def relu(self):
    # TODO: d(relu x)/dx = 1 if x > 0 else 0
    raise NotImplementedError

  # ---- Autograd ----

  def backward(self):
    # TODO:
    # 1. Build a topological order of all nodes reachable from self (DFS).
    # 2. Set self.grad = 1.0 (seed).
    # 3. Walk the order in reverse and call each node's _backward().
    raise NotImplementedError
