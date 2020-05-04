### Neural Network Basics

I have found [this youtube video](https://www.youtube.com/watch?v=aircAruvnKk) to be
very helpful in developing some intuition about neural networks. I suggest watching this
before starting the lesson.

We will start our neural network journel with the smallest element of a neural network:
*the perceptron*.

![Perceptron](https://www.google.com/search?q=perceptron&sxsrf=ALeKk024K1ac5v7P9VhZN97YbCFE79Nmuw:1588559944169&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjnvLT5lpnpAhWzoFsKHT1_ABUQ_AUoAnoECBUQBA&biw=1280&bih=689#imgrc=S4urUXdXX7jJNM)

```python
network = models.Sequential()
network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(784, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
```
