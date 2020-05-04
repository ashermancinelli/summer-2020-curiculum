
import numpy as np
import matplotlib.pyplot as plt
import math

'''
How might we create a distribution of data to work with?

data is readily available online, and we will use publicly available data
later, but for a quick test it is very convenient to be able to generate
our own data.

I'm sure this will be quite easy for you, but take a look and try to generate
some other distributions.
'''

def create_sin_with_noise(num_points, noise):

    # generate num_points data points in the range [0, 100)
    x_values = np.linspace(0, 100, num_points)

    # Generate num_points data points in a 
    # normal distribution with mu=0, std dev=0.1
    norm_dist = np.random.normal(0, noise, num_points)

    # generate a sine function with added noise
    # from the normal distribution
    y_values = [ math.sin(i*math.pi/30) + v for i, v in zip(x_values, norm_dist) ]

    return x_values, y_values

if __name__ == '__main__':

    xs, ys = create_sin_with_noise(
            num_points=300,
            noise=0.1)

    plt.scatter(xs, ys)
    plt.show()
