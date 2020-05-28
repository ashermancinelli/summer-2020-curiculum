
import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

# Each batch will have 4 images and 4 labels
batch_size = 4

'''
transforms.Compose is a method to encapsulate several operations such that they all
behave as one operation. Here we are normalizing our data with two normalization
methods.
'''
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

'''
Remember when we used scikit-learn's iris dataset?
This works in much the same way.
We could manually download the entire dataset ourselves, however
we would have to shuffle the data and split it up into train and test
datasets ourselves. In addition, pytorch's dataset loader will automatically
apply the transforms we defined above.

Pytorch will do all this for us using torchvision.datasets.
'''

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

'''
Now, we could split our data into lists just like we did with the
dense neural networks in the previous lesson, but using pytorch's Dataloader
class gives us a number of benefits.

When we work with images, the amount of data we are able to hold in memory at
any given time becomes a challange - even more so when we attemt to run
with a graphics card.

To get around this limitation, pytorch's Dataloader class will pull a single
batch into memory which we will use to train our network. Once we are done with
the batch, the Dataloader class will delete the reference to the batch so that it's memory
is freed and we can load the next batch into memory.

I recommend you play with the batch_size parameter below. See if you can increase it
enough to make your machine run out of RAM. Perhaps you have enough ram to hold a very large
batch!

When we run on graphics cards, you'll likely have about 16GiB of GPU memory, so you'll have to
ensure that the batch size is small enough that you never exceed the GPU ram, but you're still
using large enough batch sizes that you're efficently using your time.
'''
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# These are the labels for each class
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''
We're wrapping this in a main block so we can import this file and use
it elsewhere.

Run this a few times to see a few different batches. Each time you run
this, you should see different images and labels, since the dataset
is shuffled.
'''
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    def display_image(image):
        image = image / 2 + 0.5     # undo the normalization we will apply to the images
        npimage = image.numpy()
        plt.imshow(np.transpose(npimage, (1, 2, 0)))
        plt.show()
        
    for features, labels in trainloader:
        for index in range(batch_size):
            print(f'Label: {classes[labels[index]]}')
            display_image(features[index])

        break # We only want to look at one batch of images
