
### Convolutional Neural Networks

Now that you have the fundamentals of neural networks down, it is time to move on to more
practical applications. Convolutional networks are a bit more complex then regular
dense networks (the only kinds you have worked with so far) and they handle a different
variety of problems.

Convolution is a process that gives the nodes in our networks the ability to determine
context. The activation of one node in a network will change depending on the activations
of the nodes around it, which makes this a particularly useful tool for image problems.

That will be the focus of this lesson: neural networks for image processing.

We will be using the CIFAR10 dataset, which has thousands of images and 10 classes. This
means that every image in the dataset can be binned into one of 10 categories.
