import numpy as np
import matplotlib.pyplot as plt

class Sigmoid:
    def forward(self, z):
        return 1/(1+np.exp(-z))
    def backward(self, dA):
        return self.forward(dA) * (1 - self.forward(dA))

class Layer:
    def __init__(self, nin, nout):
        ...
    def forward(self, dA):
        ...
    def backward(self, dA):
        ...

if __name__ == '__main__':
    n_epochs = 5000

    ''' XOR dataset 
    
    Each data point in the dataset has features [a, b, a*b]
    and labels [xor, !xor]
    
    '''
    x = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1]
        ])

    y = np.array([
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1]
        ])

    w0 = np.random.random((3, 5))
    b0 = np.zeros((1, 5))

    w1 = np.random.random((5, 10))
    b1 = np.zeros((1, 10))

    w2 = np.random.random((10, 2))
    b2 = np.zeros((1, 2))

    activation = Sigmoid()
    errors, accs = [], []
    n = 0.1

    for i in range(n_epochs):

        layer0 = x

        layer1 = activation.forward((layer0 @ w0) + b0)

        layer2 = activation.forward((layer1 @ w1) + b1)

        layer3 = activation.forward((layer2 @ w2) + b2)

        #Back propagation using gradient descent
        layer3_error = y - layer3
        layer3_delta = layer3_error * activation.backward(layer3)

        layer2_error = layer3_delta @ w2.T
        layer2_delta = layer2_error * activation.backward(layer2)

        layer1_error = layer2_delta @ w1.T
        layer1_delta = layer1_error * activation.backward(layer1)

        # Update the weights depending on the learning rate and 
        # the derivatives of the layer above
        w2 += (layer2.T @ layer3_delta) * n
        w1 += (layer1.T @ layer2_delta) * n
        w0 += (layer0.T @ layer1_delta) * n

        error = np.mean(np.abs(layer3_error))
        errors.append(error)
        accuracy = (1 - error) * 100
        accs.append(accuracy)

        print(f'Error: {error} Acc: {accuracy}')

    fig, axes = plt.subplots(2)
    fig.suptitle('Training Results')
    axes[0].plot(errors)
    axes[0].set_title('Cost')
    axes[1].plot(accs)
    axes[1].set_title('Accuracy')
    axes[1].set_ylim([0, 100])
    plt.show()
