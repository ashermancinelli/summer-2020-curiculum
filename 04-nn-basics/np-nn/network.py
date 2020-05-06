import numpy as np
import matplotlib.pyplot as plt

class Sigmoid:
    def __init__(self, shape):
        self.a = np.zeros(shape)

    def forward(self, z):
        '''
        Take in the result of wx + b of the previous layer.
        Return the activation.
        Save the activation for computing the backprogagation
        '''
        self.a = 1/(1+np.exp(-z))
        return self.a

    def backward(self, upstream_gradient):
        self.da = upstream_gradient * self.a * (1 - self.a)
        return self.da

    def __str__(self):
        return f'-- Sigmoid Layer: {self.a.shape}'

class MSECost:
    def forward(self, y, yhat):
        self.y = y
        self.yhat = yhat
        self.batch_size = y.shape[1]
        cost = (1 / (2 * self.batch_size)) * np.sum(np.square(y - yhat))
        self.cost = np.squeeze(cost)
        return self.cost

    def backward(self):
        self.dyhat = (-1 / self.batch_size) * (self.y - self.yhat)
        return self.dyhat

class Layer:
    def __init__(self, input_shape, nout):
        if isinstance(input_shape, (tuple, list)):
            input_shape = input_shape[0]
        self.w = 2 * np.random.random((nout, input_shape)) - 1
        self.b = 2 * np.random.random((nout, 1)) - 1

    def forward(self, a):
        '''
        parameter 'a' is the activations from the previous layer.
        Save this for future backprogagation calculations.
        '''
        self.a_prev = a
        self.z = (self.w @ a) + self.b
        return self.z

    def backward(self, upstream_gradient):
        self.dw = upstream_gradient @ self.a_prev.T
        self.db = np.sum(upstream_gradient, axis=1, keepdims=True)
        self.da_prev = self.w.T @ upstream_gradient
        return self.da_prev

    def update_parameters(self, lr):
        self.w -= self.dw * lr
        self.b -= self.db * lr

    def __str__(self):
        return f'-- Layer: weights: {self.w.shape} biases: {self.b.shape}'

class Network:
    def __init__(self, layers, lr, cost_fn):
        self.costs = []
        self.accs = []
        self.layers = layers
        self.lr = lr
        self.cost_fn = cost_fn

    def forward(self, input):
        '''
        Pass the input into the network and calculate the forward
        propagation of each layer. Save the final output as yhat.
        '''
        x = input
        for layer in self.layers:
            print(layer)
            x = layer.forward(x)

        self.yhat = x
        return self.yhat

    def backward(self, y):
        '''
        Comput the cost and gradients for each layer based on the output
        of the network and the desired values y.
        '''
        cost = self.cost_fn.forward(y, self.yhat)
        self.costs.append(cost)
        dyhat = self.cost_fn.backward()
        upstream_gradient = dyhat

        for layer in reversed(self.layers):
            upstream_gradient = layer.backward(upstream_gradient)
            if isinstance(layer, Layer):
                layer.update_parameters(self.lr)

        return cost


if __name__ == '__main__':
    n_epochs = 500

    ''' XOR dataset 
    
    Each data point in the dataset has features [a, b, a*b]
    and labels [xor, !xor]
    
    '''
    x = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1]
        ]).T

    y = np.array([
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1]
        ]).T

    model = Network([
        Layer(x.shape, 5),
        Sigmoid(5),
        Layer(5, 8),
        Sigmoid(8),
        Layer(8, 2),
        Sigmoid(2),
    ], cost_fn=MSECost(), lr=1)

    for i in range(n_epochs):

        predictions = model.forward(x)
        cost = model.backward(y)
        print(f'Cost: {cost}')

    print(f'Predictions of the trained network:\ninputs:\n{x.T}\nPredictions:\n{predictions.T}\nLabels:\n{y.T}')
    plt.plot(model.costs)
    plt.show()
