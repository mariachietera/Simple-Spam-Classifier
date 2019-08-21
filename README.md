# Simple Spam Classifier

This folder contains a simple implementation of a Multinomial Naive Bayes Spam Classifier and its evaluation obtained printing a confusion matrix and calculating the most common evaluation scores.

## Setup

Create a virtualenv with python 3.* (tested on Python3.7.2) and install `seaborn`, `matplotlib`, `pandas`, `sklearn`.

### Usage:
```
> spam_classifier.py
Dataset loading..

See confusion matrix at /path/mx.png

Model evaluation scores:
Precision: 0.92
Recall: 0.92
F-score: 0.92
Accuracy: 0.92


Enter text and I will label it (Q for exit):
>
```

Test the functions inserting any string or try the following example post:  
```
> (Wanted) I will be moving to Chicago for work around July 1st and am searching for 1 or 2 female roommate(s) to share an apartment close to Printer's Row! Open to other areas if necessary :)
```

Output:
```
ham
```

Another example:
```
> Viber:- +15129816074- BUY 2 GET 1 FREE. These phones are BRAND NEW box never used, sim-free FACTORY UNLOCKED GENUINE 1 YRS WARRANTY sealed in box.
```
Output:
```
spam
```