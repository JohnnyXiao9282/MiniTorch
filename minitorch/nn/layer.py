from minitorch.nn.neuron import Neuron


class Layer:
    def __init__(self, num_inputs, num_neurons):
        self.neurons = [Neuron(num_inputs) for i in range(num_neurons)]

    def __call__(self, data):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron(data))
        return outputs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params += neuron.parameters()
        return params
