import numpy as np
import matplotlib.pyplot as plt

def binary_crossentropy(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def squared_error(Y, Y_hat):
    '''
    Y: labels
    Y_hat: predictions
    '''
    m = Y.shape[1]
    cost = (1 / (2 * m)) * np.sum(np.square(Y - Y_hat))
    cost = np.squeeze(cost)
    dY_hat = -1 / m * (Y - Y_hat)
    return cost, dY_hat

class Sigmoid:
    def __init__(self, shape):
        self.A = np.zeros(shape)
    def forward(self, z):
        self.A = 1/(1+np.exp(-z))
        return self.A
    def backward(self, dA):
        sig = self.forward(dA)
        self.dZ = dA * self.A * (1 - self.A)
        return self.dZ
    def __str__(self):
        return f'Sigmoid with shape: {self.A.shape}'

class Relu:
    def forward(self, z):
        return np.maximum(0, z)
    def backward(self, dA, z):
        dz = np.array(dA, copy = True)
        dz[z <= 0] = 0;
        return dz;

class Layer:
    def __init__(self, n_inputs, n_outputs, lr=0.001):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.weights = np.random.random((n_outputs, n_inputs))
        self.biases = np.ones(n_outputs)

    def forward(self, inputs):
        assert inputs.shape[0] == self.n_inputs, f'Inputs {inputs.shape} did not match expected shape {self.n_inputs}'
        self.A_prev = inputs
        self.z = (self.weights @ inputs) + np.expand_dims(self.biases, axis=1)
        return self.z

    def backward(self, dA):
        self.dW = np.dot(dA, self.A_prev.T)
        self.db = np.sum(dA, axis=1, keepdims=True)
        self.dA_prev = np.dot(self.weights.T, dA)
        return self.dA_prev

    def update(self):
        self.weights -= self.lr * self.dW
        self.biases -= self.lr * np.sum(self.db, axis=1)

    def __str__(self):
        return f'-- Weights: {self.weights.shape} Biases: {self.biases.shape})'

class Network:
    def __init__(self, layers, lr=0.001):
        self.layers = layers
        self.lr = lr
        for layer in self.layers:
            layer.lr = self.lr

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            print(x.shape)
            print(layer)
            x = layer.forward(x)
        return x

    def backward(self, dA):
        x = dA
        for layer in reversed(self.layers):
            x = layer.backward(x)
        return x

    def update(self):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.update()

    def eval(self, x, y):
        Y_hat = self.forward(x)
        for actual, pred in zip(Y_hat, y):
            print(actual, pred)
            if actual == pred:
                correct += 1

        return correct/len(Y_hat)

    def __str__(self):
        for i in self.layers:
            print(i)

if __name__ == '__main__':
    n_epochs = 20

    model = Network([
        Layer(2, 16),
        Layer(16, 16),
        Sigmoid(16),
        Layer(16, 16),
        Sigmoid(16),
        Layer(16, 1),
    ], lr=0.01)

    ''' XOR dataset '''
    x = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
        ]).T

    y = np.array([
        [0],
        [1],
        [1],
        [0]
        ]).T

    history = []
    for i in range(n_epochs):
        Y_hat = model.forward(x)
        cost, dA = squared_error(y, Y_hat)
        print("Cost at epoch {}: {}".format(i, cost))
        model.backward(dA)
        model.update()
        acc = model.eval(x, y)
        history.append([cost, acc])

    plt.plot(history)
    plt.show()

