
Convolutional Neural Networks
---

Now that you have the fundamentals of neural networks down, it is time to move on to more
practical applications.
Convolutional networks are a bit more complex then regular
dense networks (the only kinds you have worked with so far) and they handle a different
variety of problems.

Convolution is a process that gives the nodes in our networks the ability to determine
context.
The activation of one node in a network is dependent on the activations
of the nodes around it, which makes this a particularly useful tool for image problems.

That will be the focus of this lesson: neural networks for image processing.
There is significant crossover between this lesson and the application of neural networks to signal processing problems.

We will be using the CIFAR10 dataset, which has thousands of images broken into 10 classes.
This means that every image in the dataset can be binned into one of 10 categories.

Layers
---

The new layers we will be working with in our convolutional network are convolutional layers and pooling layers.

Conv Layers
---

These layers are much like what you've been exposed to in Physics with Laplace transforms.
Please see the figure below for a diagram of how the convolutions are applied:

![Conv Layer](images/ConvLayer.png)

A convolutional layer is made up of filters, each of which is applied to the input matrix as seen above.
The diagram above is animated at [this link](https://cs231n.github.io/convolutional-networks/).
Please visit that site and watch the algorithm in action, hopefully developing some intuition for the operation.

Pooling Layers
---

Typically after a convolutional layer, a *pooling* layer is applied.
This reduces the size of a matrix by pooling elements together, as seen in the diagram below:

![Pooling Layer](images/PoolingLayer.png)

Pooling is a form a downsampling to reduce the size of the data that the network
must keep in memory.
This does not change the performance of the network much, so it's very useful in increasing the performance of our network.

Please find further instruction in the python scripts in this directory.
The order you ought to look through the scripts is:

1. `./cifar10.py`
2. `./network.py`
3. `./evaluate_network.py`

Problems
---

1. Why do we use pooling layers?
    - Could you trian a network without pooling layers?
    - What parameters of the convolutional layers might you want to adjust if you were not to use pooling layers?
    - Do a bit of research to figure out why almost all convolutional networks used today use pooling layers
2. What would you change about the network if you were to feed it 4-channel images (RBGA for red, blue, gree, and alpha for transparancy)?
3. How would you train a network with a dataset that contains images of different sizes?
4. Train a network on the Fashion-MNIST dataset.
    - This dataset is composed of grascale images (so only one channel) which should make the problem a bit easier
    - There are thousands of images broken up into the following classes:
        - T-shirt/top
        - Trouser
        - Pullover
        - Dress
        - Coat
        - Sandal
        - Shirt
        - Sneaker
        - Bag
        - Ankle boot
    - PyTorch has a datset loader for Fashion-MNIST just like with CIFAR10: `torchvision.datasets.FashionMNIST`
5. If you did not have much trouble with Fashion-MNIST, attempt to train a network on this [architectural dataset](https://old.datahub.io/dataset/architectural-heritage-elements-image-dataset)
    - You're lucky enough to have a datset with images all of the same sizes, however there is also a version of this dataset at the link above which has not normalized the shapes of the images. If you're feeling really ambitious, attempt to create a dataset loader for this image dataset which reshapes all the images to work in your network.
        
