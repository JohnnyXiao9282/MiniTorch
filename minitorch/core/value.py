"""Scalar Value with reverse-mode automatic differentiation.

Rule: no inner functions. Each operation records its inputs (`inputs`)
and op name (`_op`). The method `_backward()` looks at `_op` and applies
the matching gradient rule to the children.
"""


class Value:
    """A scalar that tracks its data, gradient, and the graph that produced it."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        # A tuple (not a set) so that x + x keeps both children.
        self.inputs = tuple(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # ---- Backward rule for ONE node (called by backward()) ----

    def _backward(self):
        if self._op == "+":
            for i in self.inputs:
                i.grad += self.grad
        elif self._op == "*":
            x, y = self.inputs
            x.grad += self.grad * y.data
            y.grad += self.grad * x.data
                
            

    def __add__(self, other):    
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data + other.data, (self, other), "+")
            
        
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data * other.data, (self, other), "*")

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
    

    def relu(self):
       

    # ---- Autograd ----
    def collect_nodes(self, visited, nodes):
        if self in visited:
            return 
        else:
            visited.add(self)
            for i in self.inputs:
                i.collect_nodes(visited, nodes)
            nodes.append(self)

    def backward(self):
        nodes = []
        visited = set()
        self.collect_nodes(visited, nodes)
        self.grad = 1.0
        
        for node in reversed(nodes):
            node._backward()