
'''
Pandas is a library for working with structured data, much like
an excel sheet. You may manipulate the rows and columns in
interesting ways, but for now we will just use it to read in
a csv file.
'''
import pandas as pd
import matplotlib.pyplot as plt
series = pd.read_csv('daily_temps.csv', header=0, index_col=0)

series.plot()
plt.title('Time-Series plot')
plt.show()

'''
The following plots each datum in relation to the previous
datum. If the data we are working with is random, this plot
should look like a 2d gaussian distribution (totally random).
If there appears to be any sort of structure, then we have some
time-dependence in our data, which we may exploit using
an autoregressive technique.
'''
pd.plotting.lag_plot(series)
plt.title('Lag plot')
plt.show()

'''
Let's examine how strongly each datum is correlated with
the previous:
'''
values = pd.DataFrame(series.values)
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
print(f'Correlation: {dataframe.corr()}')

'''
This is another helpful graph which displays how the correlations
between datum is increasing as time increases.
'''
pd.plotting.autocorrelation_plot(series)
plt.title('Autocorrelation plot')
plt.show()

'''
Let's create a model to fit this data.
Feel free to run this script in a python repl and play around with
the dataset yourself.
'''

# Library that sets some sensible defaults for an
# autoregressive model
from statsmodels.tsa.ar_model import AutoReg

# Converts from pandas dataframe to numpy arrays
data = series.values

# Split into training and validation datasets
train, test = data[:int(len(data)*.7)], data[int(len(data)*.7):]

'''
Create and train model
What happens when you change the 'lags' to be 50? 5000?

Try to find the line between underfitting and overfitting.
At what point do the predictions become too specific to be helpful?
When are the predictions too vague to be helpful?
'''
model = AutoReg(train, lags=500).fit()

# generate predictions based on training data
predictions = model.predict(
        start=len(train),
        end=len(train)+len(test)-1,
        dynamic=False)

plt.plot(train, color='blue')
plt.title('Training data')
plt.show()

plt.plot(test, color='blue', label='test data')
plt.plot(predictions, color='red', label='predictions')
plt.title('Test data with predictions')
plt.legend()
plt.show()
