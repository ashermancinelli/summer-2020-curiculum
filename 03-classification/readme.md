### Classification

Classification is the process of *binning* data. We will train models
to bin new data based on training data. We will primarily use the
[Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set).

The first classifier we will use is a support vector machine. This
method creates 'supporting vectors' which bound each data cluster.
[This link](https://ocw.mit.edu/courses/sloan-school-of-management/15-097-prediction-machine-learning-and-statistics-spring-2012/lecture-notes/MIT15_097S12_lec12.pdf) describes the calculations well.

In `classify.py`, you will find an SVM example using scikit learn. In this library,
there are [many other classifiers](https://scikit-learn.org/stable/supervised_learning.html)
which are better suited to different problem domains. Take a look at the provided link,
as you will have to use another classifier later.

Ensemble methods are often used to classify data when different classifiers will perform
better than others only in certain domains. For example, scikit learn has a voting
classifier which weights the predictions of multiple classifiers depending on how well
they perform in a given domain.
```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Create classifiers to be contained by our voting classifier
clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

eclf1 = VotingClassifier(estimators=[
                             ('lr', clf1),
                             ('rf', clf2),
                             ('gnb', clf3),
                         ],
                         voting='soft')
```
For example, if you know that your dataset lends itself to either a naive
bayes model *or* a linear model, you might simply create a voting classifier with
both of those models. The better model will win out and get the vote most of the time.
You may also tune the voting method. Above I chose 'soft' so that the vote of each
classifier is weighted. You may also choose to make your model predict the majority
vote of its internal classifier. See [this link](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier) for more.

One of the most popular classifiers used professionally and in
[online data science competitions](https://www.kaggle.com/competitions) is
[AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier).
AdaBoost uses `n` weak classifiers and weights their outputs to produce the final prediction.
This is comperable to a voting classifier where every internal model is a small, weak decision
tree classifier.

### Problems

1. Choose another model from [this link](https://scikit-learn.org/stable/supervised_learning.html)
  and fit it to the iris dataset. How does it compare to the SVM? When should you use the model you
  chose over an SVM?
2. In the same way that we fitted a model to the iris dataset, create at least one
  classifier for the dataset constructed like so:
  ```python
  x, y = datasets.make_blobs(
    n_samples=10000,
    n_features=10,
    centers=100,
    random_state=0)
  ```
  Collect metrics and compare your results. Perhaps try an ensemble method and another classifier.
  
3. Can you beat Adaboost? Find another dataset online and construct an ensemble method. Pick classifiers
  that you think would work well together and construct an ensemble classifier out of it. Fit both models
  to the data and record which one performs better. Try tuning the parameters of the models to see what changes.
  [Kaggle's datasets](https://www.kaggle.com/datasets) will prove helpful here.
