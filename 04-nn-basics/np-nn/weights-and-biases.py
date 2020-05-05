import numpy as np
import matplotlib.pyplot as plt

'''
Neural Network from scratch.

Common terms:
    w: weights
    b: biases
    x: inputs (or sometimes current value of the network)
    z: output of the weights * input + biases
    a: output of z passed to the activation function
    l, loss, error: The value of the cost function for a given sample

'''

class Sigmoid:
    def forward(self, z):
        return 1/(1+np.exp(-z))
    def backward(self, upstream_gradient):
        return upstream_gradient * (1 - upstream_gradient)

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
    
    X with shape [batch_size, num_features]
    '''
    x = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1]
        ])

    batch_size = x.shape[0]

    y = np.array([
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1]
        ])

    w0 = np.random.random((3, 5))
    b0 = np.zeros((1, 5))

    w1 = np.random.random((5, 2))
    b1 = np.zeros((1, 2))

    activation = Sigmoid()
    errors, accs = [], []
    learning_rate = .001

    for i in range(n_epochs):

        # z = xw + b for first layer
        z0 = (x @ w0) + b0
        a0 = activation.forward(z0)

        # z = xw + b for second layer
        z1 = (a0 @ w1) + b1
        a1 = activation.forward(z1)
        
        # yhat is the common term for the output of the final layer
        # of the network
        yhat = a1

        '''
        That was the easy part. Now we must take the derivative of the cost function L.
        We will sum each element of the network (weights, biases, activations) with its
        partial derivative with respect to the cost function. Feeding forward works
        from input -> layer 0 -> ... -> layer n -> cost function.

        To take the derivative of L w.r.t to each part of the network, we 
        work backwards starting with dL

        First, calculate the cost:
        '''
        L = np.squeeze((1 / (2 * batch_size)) * np.sum(np.square(y - yhat)))
        acc = (1-np.mean(np.abs(y - yhat))) * 100

        '''
        Here we calculate the derivative of the cost function w.r.t the outputs of the network
        (dL/dyhat), as well as the partial of the activation function (dyhat/dz and dL/dz).
        We work backwards through the network. This is called 'backpropagation'.
        '''
        dL_dyhat = (-1 / batch_size) * (yhat - y)
        dyhat_dz = activation.backward(yhat)
        dL_dz = dL_dyhat * dyhat_dz

        '''
        From here we will take the partial derivative with respect to each variable
        that we are able to adjust. We will need dL/dW, dL/db, and dL/da_previous 
        for each layer in order to adjust the weights and biases to decrease
        the cost of the network. dL/da_previous will be used to adjust variables
        for the layers below.

        dL/dB = sum( dL/dz * dz/db )
              = sum( dL/dz * 1 )
              = sum( dL/dz )
        '''
        dL_db = np.sum(dL_dz, axis=0, keepdims=True)

        '''
        dL/dW = dL/dz * dz/dw
              = dL/dz * a_previous^T
        '''
        dL_dw = a0.T @ dL_dz

        '''
        dL/da = dL/dz * dz/dw
              = dL/dz * w^T
        '''
        dL_da = dL_dz @ w1.T

        '''
        Now that was all for one layer. We must now adjust weights and biases for the
        current layer and then continue to compute the same values
        for the layers below.
        '''
        w1 -= dL_dw * learning_rate
        b1 -= dL_db * learning_rate

        '''
        Now for the first layer, we will run the same calculations
        using the upstream gradients:
        '''
        da_dz = activation.backward(a0)
        dL_dz = dL_da * da_dz

        dL_db = np.sum(dL_dz, axis=0, keepdims=True)
        dL_dw = x.T @ dL_dz

        w0 -= dL_dw * learning_rate
        b0 -= dL_db * learning_rate

        print(f'Error: {L} Acc: {acc}')
        errors.append(L); accs.append(acc)

    fig, axes = plt.subplots(2)
    fig.suptitle('Training Results')
    axes[0].plot(errors)
    axes[0].set_title('Cost')
    axes[1].plot(accs)
    axes[1].set_title('Accuracy')
    axes[1].set_ylim([0, 100])
    plt.show()
