
### Neural Network from Scratch

[This link](https://medium.com/towards-artificial-intelligence/nothing-but-numpy-understanding-creating-neural-networks-with-computational-graphs-from-scratch-6299901091b0)
provided very helpful and clear explanations for every step of `network.py`, especialy figure 79b.
Please feel free to spend some time going through the article before looking at the files
in this directory.

Every bidirectional neural network will follow roughly the same pseudocode:
```
A := inputs
for epochs in number_epochs:
    for layer in layers:
        A = layer.weights * A + layer.biases
        A = layer.activation_function(A)



### Feed-Forward
