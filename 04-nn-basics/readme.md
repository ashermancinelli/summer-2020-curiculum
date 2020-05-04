### Neural Network Basics

I have found [this youtube video](https://www.youtube.com/watch?v=aircAruvnKk) to be
very helpful in developing some intuition about neural networks. I suggest watching this
before starting the lesson. Don't feel overwhelmed by new terms - I plan to introduce terms
before explanation.

We will start our neural network journel with the smallest element of a neural network:
*the perceptron*.

![Perceptron](https://akashsethi24.files.wordpress.com/2017/09/perceptron.png?resize=385%2C254)

The equation to generate the output of a perceptron is as follows:

![stepfunction(\beta + \sum_{i=1}^n{x_i \times w_i})](https://render.githubusercontent.com/render/math?math=stepfunction(%5Cbeta%20%2B%20%5Csum_%7Bi%3D1%7D%5En%7Bx_i%20%5Ctimes%20w_i%7D))

Each input is multiplied by its respective weight, added with a bias, and summed. The result is then the input to the
perceptron's *step function* which scales the output. The weights of the perceptron are adjusted during training.

In a neural network, each layer is (conceptually) composed of perceptrons.

```python
network = models.Sequential()
network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
```
