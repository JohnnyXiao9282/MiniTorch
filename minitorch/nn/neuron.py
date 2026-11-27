import random
from minitorch.core.value import Value

class Neuron:
    def __init__(self, num_inputs):
        self.bias = Value(0)
        self.weights = [Value(random.uniform(-1, 1)) for i in range(num_inputs)]
        
    def __call__(self, inputs):
        output = Value(0)
        for w, x in zip(self.weights, inputs):
            output += w * x
        output += self.bias
        return output.tanh()
        
    def parameters(self):
        return self.weights + [self.bias]
        
