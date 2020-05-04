### Neural Network Basics

I have found [this youtube video](https://www.youtube.com/watch?v=aircAruvnKk) to be
very helpful in developing some intuition about neural networks. I suggest watching this
before starting the lesson.

We will start our neural network journel with the smallest element of a neural network:
*the perceptron*.

![Perceptron](https://akashsethi24.files.wordpress.com/2017/09/perceptron.png?resize=385%2C254)

```python
network = models.Sequential()
network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
```
