from sklearn.datasets import load_wine;
from sklearn.metrics import accuracy_score;
from sklearn.naive_bayes import GaussianNB;
import numpy as np;

x, y = load_wine(return_X_y=True);

GNB = GaussianNB();
model = GNB.fit(x, y);

y_pred = model.predict(x);

accuracy = accuracy_score(y, y_pred);

print('The accuracy of the given dataset is :- ', str(accuracy));