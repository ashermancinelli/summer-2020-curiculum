### Autoregressive Models

Autoregressive models predict unknown data points for a continuous variable
based on known discrete data points.
I found [these slides](http://web.stanford.edu/class/ee269/Lecture12.pdf)
very helpful in understanding the material, though more theoretical than practical.
Please read through the slides and then find `autoreg.py` in this directory.

We will come back to this dataset when we discuss gaussian processes.

### Problems

1. Ensure you have installed all the relevant libraries
2. Read through `autoreg.py` before running anything
  - Ensure you understand the comments. Take frequent breaks to search for
    terms you don't understand.
3. Make sure you understand `matplotlib` well enough to experiment with it yourself.
  Feel free to look through [these examples](https://matplotlib.org/3.2.1/tutorials/introductory/sample_plots.html)
  of `matplotlib`.
4. Refer to `gen_data.py` in assignment 1 to generate your own dataset. Fit an autoregression model
  to these data. Fit the model several times using different parameters to find a 'sweet spot'.
5. Extra: Find a time-series dataset online (see [this link](https://www.kaggle.com/datasets)) and fit
  a model to the data.
6. Extra: Find another library for autoregression online and create a model with it. Note: you may have to
  tune more parameters with the other library.
