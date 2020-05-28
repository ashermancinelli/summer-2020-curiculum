
'''
Here we import our file to set up the dataset.

The reason we import it like this is that we don't have to refer
to everything we import as cifar10.thing_we_defined_in_that_file
For example, we can now just use 'trainloader' instead of 
cifar10.trainloader
'''
from cifar10 import *

# These are the module s within torch that allow us to define layers
import torch.nn as nn
import torch.nn.functional as F

# This module contains optimization algorithms from which we will choose one.
# More on this later.
import torch.optim as optim

'''
In pytorch, a network is a class which inherits from PyTorch's own Module
class. In other libraries, this is not so. Remember, we're just trying
to express a series of feed-forward operations and another set of
operations used for backpropogation. This is just like the previous
lesson, but we're using a library for three reasons:
convenience, robustness, and performance.
'''
class Net(nn.Module):
    def __init__(self):
        '''
        This constructor will add member variables which are all PyTorch's
        classes. We will than call the member variables in the forward method
        of this class.

        Again, the magic of PyTorch is that we do not have to define a
        backward method, like with our toy NN library. PyTorch will calculate
        the gradients for us.
        '''

        # This helps pytorch set up our class - don't pay too much attention
        # to this. I can explain if you like, but this doesn't really 
        # pertain to anything else we're doing.
        super(Net, self).__init__()

        '''
        Our images will have three channels (each pixel has RGB values),
        which is why the first dimension of our convolutional layer 
        will have a size of 3
        '''
        self.conv1 = nn.Conv2d(3, 6, 5)

        '''
        We pool our values as described in the readme
        We will use this layer more than once
        '''
        self.pool = nn.MaxPool2d(2, 2)

        # convolve again
        self.conv2 = nn.Conv2d(6, 16, 5)

        '''
        This is the fully-connected (aka dense) layer that will
        'plug in' to our last convolutional layer.
        From this layer on, it's just like the networks we used in the
        previous lesson.
        '''
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        # First conv layer
        x = self.conv1(x)

        # First activation
        x = F.relu(x)

        # First pooling
        x = self.pool(x)

        # Second conv layer
        x = self.conv2(x)

        # Second activation
        x = F.relu(x)

        # Second pooling
        x = self.pool(x)

        # Reshape the matrix coming out of the last convolutional layer to
        # have the correct shape to be fed into the first fully-connected layer
        x = x.view(-1, 16 * 5 * 5)

        # First fully connected layer and subsequent activation
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc1(x)
        x = F.relu(x)

        # No activation on final layer
        x = self.fc3(x)

        # Our final x value is our yhat
        return x


net = Net()

# In our toy network, we used mean squared error. Cross entropy is just
# another way to calculate the loss. You can play with other cost
# functions if you like.
loss_fn = nn.CrossEntropyLoss()

'''
We will use Stochastic Gradient Descent (SGD) for our optimization
algorithm. I haven't mentioned this until now, but there are other
optimization algorithms that work much the same way, but may be
better suited to certain problems. When developing your own networks,
I reccommend you start with the Adam optimizer. 

To give you an example of the differences:
sometimes you want to change your learning rate 
dynamically depending on the performance of the network.
Some optimizers handle this differently.

We pass in the parameters of our network that can be optimized.
Each parameter knows how to calculate it's own gradient, and the
network will handle the backward method, connecting the gradients
of all it's layers. The optimizer will handle the changes to make to
the parameters.
'''
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Here we loop over the dataset more than once.
for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients from the previoius training
        # session
        optimizer.zero_grad()

        '''
        calling net(inputs) like this is how we do a forward-pass
        on our network. then, we calculate our loss with our loss function.
        After that, we call backward on the loss, which will then call backward on
        each layer of our network to calculate the gradients.

        We then call .step() on our optimizer, which will apply the gradients
        to the parameters of our network, among other things.
        '''
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item() # .item() grabs the scalar value from a pytorch tensor
        if i % 2000 == 1999:    # print every 2000 mini-batches

            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
