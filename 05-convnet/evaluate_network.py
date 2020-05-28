
# import our train and test datasets
from cifar10 import *

# Also import the network we alread defined and trained
from network import Net

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

if __name__ == '__main__':

    # create a network with the topology we defined in network.py
    net = Net()

    # reload the parameters of our trained network
    net.load_state_dict(torch.load('./network-state'))

    '''
    The following block is equivilant to:
    images = []
    labels = []
    for ims, lbls in testloader:
        images = ims
        labels = lbls
        break

    Just grabs the first batch of images and labels from
    the dataloader.
    '''
    images, labels = iter(testloader).next()

    # Pass the batch through the network
    yhat = net(images)

    # Take the max prediction for each image
    _, predictions = torch.max(yhat)

    # Display the image and print the label and prediction
    for index in range(batch_size):
        display_image(images[index])
        print(f'Label: {classes[labels[index]]}'
                f' Prediction: {classes[predictions[index]]}')

    # Now that we've seen a small sample of how well our network
    # performs, let's collect some metrics
    correct = 0
    total = 0

    # This 'with' block is just to ensure that our network
    # does not calculate its gradients since we're no longer training.
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
