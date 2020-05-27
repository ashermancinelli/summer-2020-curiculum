### Neural Network Basics

I have found [this youtube video](https://www.youtube.com/watch?v=aircAruvnKk) to be very helpful in developing some intuition about neural networks.
I suggest watching this before starting the lesson. Don't feel overwhelmed by new terms - I plan to introduce terms before explanation.

We will start our neural network journel with the smallest element of a neural network:
*the perceptron*.

![Perceptron](https://akashsethi24.files.wordpress.com/2017/09/perceptron.png?resize=385%2C254)

The equation to generate the output of a perceptron is as follows:

![stepfunction(\beta + \sum_{i=1}^n{x_i \times w_i})](https://render.githubusercontent.com/render/math?math=stepfunction(%5Cbeta%20%2B%20%5Csum_%7Bi%3D1%7D%5En%7Bx_i%20%5Ctimes%20w_i%7D))

Each input is multiplied by its respective weight, added with a bias, and summed.
The result is then the input to the perceptron's *step function* which scales the output.
The weights of the perceptron are adjusted during training.

In a neural network, each layer is (conceptually) composed of perceptrons.
In reality, the layers are represented as matrices. You will see this in our example.

The first operation of a neural network is forward propagation (or a *forward pass* of the network, where inputs are passed through the network and the activation of the last layer is the final output of the network.
The difference between the outputs of the network and the intended output of the network is the *loss*.
The loss is also refered to as the *cost* or *error* of the network.

The next operation of a neural network when training is backpropagation. Let us first develop some intuition and
motivation for this concept before discussing implementation details.

In order to improve the performance of the network, the weights and biases are adjusted to make the output of the network more closely match the intended output.
It is helpful to think of the network as a point on a 3d surface where the z-axis represents the loss, and the `x` and `y` axes are the parameters of the network.
Adjusting `x` and `y` may increase or decrease the loss.
To optimize the network then, we intend to adjust `x` and `y` such that z is minimized.
We may picture the network as a point on the surface walking back and forth down the surface until it finds a minimum. 

How might we determine the adjustments to `x` and `y` that most efficiently decrease z?
In vector calculus, you may have learned to find the gradient of steepest descent; we will
use this technique here as well. To find the adjustments to `x` and y, we will take the partial
derivate of z with respect to `x` and `y` to find each of their respective adjustments like so:

![x = x - ( n \times \frac{\partial z}{\partial x} )](https://render.githubusercontent.com/render/math?math=x%20%3D%20x%20-%20(%20n%20%5Ctimes%20%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20x%7D%20))

![y = y - ( n \times \frac{\partial z}{\partial y} )](https://render.githubusercontent.com/render/math?math=y%20%3D%20y%20-%20(%20n%20%5Ctimes%20%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20y%7D%20))

Where `n` represents the *learning rate*, or how large of steps we will take during optimization.

In practice, the derivatives of our loss with respect to each of our parameters will be slightly more complex. 
Let:

![w = weights](https://render.githubusercontent.com/render/math?math=w%20%3D%20weights)

![\beta = biases](https://render.githubusercontent.com/render/math?math=%5Cbeta%20%3D%20biases)

![A = activation_of_layer](https://render.githubusercontent.com/render/math?math=A%20%3D%20activation_of_layer)

![\sigma = activation function](https://render.githubusercontent.com/render/math?math=%5Csigma%20%3D%20activation%20function)

![L = loss](https://render.githubusercontent.com/render/math?math=L%20%3D%20loss)

![i = inputs](https://render.githubusercontent.com/render/math?math=i%20%3D%20inputs)

![\hat{y} = network\ \ \ output](https://render.githubusercontent.com/render/math?math=%5Chat%7By%7D%20%3D%20network%5C%20%5C%20%5C%20output)

![network(i) = \sigma ( ( w \times i ) + \beta )](https://render.githubusercontent.com/render/math?math=network(i)%20%3D%20%5Csigma%20(%20(%20w%20%5Ctimes%20i%20)%20%2B%20%5Cbeta%20))

Where `N` represents our 1-layer network.

The loss will be calculated based on the output of the final (and only) layer.
Therefor the gradient with respect to the activation of the final layer will be:

![\frac{\partial L}{\partial A} = \frac{\partial L}{\partial L} \times \frac{\partial L}{\partial A} = 1 \times \frac{\partial L}{\partial A}](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20A%7D%20%3D%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20L%7D%20%5Ctimes%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20A%7D%20%3D%201%20%5Ctimes%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20A%7D)

This is a rather simple calculation, and we will use this to work backwards to calculate the gradients of the rest of the parameters that come before the activation of the final layer in a forward-pass of the network.
Using this gradient, the partial w.r.t. the biases will be:

![\frac{\partial L}{\partial \beta} = \frac{\partial L}{\partial A} \times \frac{\partial A}{\partial \beta}](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cbeta%7D%20%3D%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20A%7D%20%5Ctimes%20%5Cfrac%7B%5Cpartial%20A%7D%7B%5Cpartial%20%5Cbeta%7D)

Likewise, the partial derivate of the loss w.r.t. the weights will be:

![\frac{\partial L}{\partial w} = \frac{\partial L}{\partial A} \times \frac{\partial A}{\partial w}](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20w%7D%20%3D%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20A%7D%20%5Ctimes%20%5Cfrac%7B%5Cpartial%20A%7D%7B%5Cpartial%20w%7D)

This pattern is why the process is called *backpropagation*: the gradients propagate backwards through the network.
If our network were two layers, the gradient of the loss w.r.t. the first layer's gradients would be calculated using the second layer's gradients and so forth.

In `np-nn/network.py`, you will find some classes which allow you to build your own minimal neural network.
Each layer has a method for forward propagation (`forward`) and a method for backpropagation (`backward`).
The network also has `forward` and `backward` methods.
The network's `forward` method passes the inputs through the first layer, and passes subsequent outputs to the `forward` methods of subsequent layers.
The network's `backward` method passes the derivative of the loss to the `backward` method of the *final* layer.
The output of that is passed to the `backward` method of the previous layer and so on.

#### Do not pass this mark until you have read through the entire `np-nn/network.py` file

Now that we have examined the implementation of a neural network, let us examine a library
that will take care of some details for us. We will use PyTorch for this example, but feel
free to look into Keras, TensorFlow, caffe2, or any other library that sounds interesting to you.
They all aim to acomplish roughly the same thing, however PyTorch uses `forward` and `backward` methods just like in `np-nn/network.py` so I figured it would be the most natural choice.

The following is a recreation of our XOR example using PyTorch instead of our own network.
Note that we don't have to define our own `backward` method!
The creators of PyTorch went to great lengths to implement automatic differentiation so that we do not have to calculate a single gradient by hand.

```python
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.lin1 = nn.Linear(3, 5)
        self.lin2 = nn.Linear(5, 8)
        self.lin3 = nn.Linear(8, 2)
    
    def forward(self, x):
        x = self.lin1(x)
        x = F.sigmoid(x)
        x = self.lin2(x)
        x = F.sigmoid(x)
        x = self.lin3(x)
        x = F.sigmoid(x)
        return x

net = Net()

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

yhat = net(x)
```
