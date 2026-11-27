import math

class Value:
    """A scalar that tracks its data, gradient, and the graph that produced it."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
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
        elif self._op == "**":
            x = self.inputs[0]
            x.grad += self.exponent * x.data ** (self.exponent - 1) * self.grad
        elif self._op == "exp":
            x = self.inputs[0]
            x.grad += self.data * self.grad
        elif self._op == "log":
            x = self.inputs[0]
            x.grad += (1 / x.data) * self.grad
        elif self._op == "tanh":
            x = self.inputs[0]
            x.grad += (1 - self.data ** 2) * self.grad
        elif self._op == "relu":
            x = self.inputs[0]
            x.grad += (self.data > 0) * self.grad
            
        
                
            

    def __add__(self, other):    
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data + other.data, (self, other), "+")
            
        
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data * other.data, (self, other), "*")

    def __pow__(self, exponent):
        result = Value(self.data ** exponent, (self,), "**")
        result.exponent = exponent
        return result

    def __neg__(self):
        return self * -1
        

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return self + (-other)

    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return self * other ** -1

    # Reflected operators so that `2 + v` and `2 * v` work.
    def __radd__(self, other):  
        if not isinstance(other, Value):
            other = Value(other)
        return self + other

    def __rmul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return self * other

    def __rsub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other + (-self)

    def __rtruediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other * self ** -1

    # ---- Activation / math functions ----

    def exp(self):
        return Value(math.exp(self.data), (self,), "exp")
    
    def log(self):
        return Value(math.log(self.data), (self,), "log")
        

    def tanh(self):
        return Value(math.tanh(self.data), (self,), "tanh")
    

    def relu(self):
        return Value(max(0, self.data), (self,), "relu")
        
       

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