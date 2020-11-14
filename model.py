import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('/home/alecwilson/Documents/Birmingham/Kaggle/Fashion/fashion-mnist_train.csv', dtype=int)  # read train data
dft = pd.read_csv('/home/alecwilson/Documents/Birmingham/Kaggle/Fashion/fashion-mnist_test.csv', dtype=int)  # read test data

X_train = df.drop('label', axis=1)
y_train = df['label']
X_test = dft.drop('label', axis=1)
y_test = dft['label']

class classifiers():
    def rfc(self):
        rf = RandomForestClassifier(n_estimators=64, n_jobs=-1)
        rf.fit(X_train, y_train.values.ravel())

        y_pred = rf.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(metrics.classification_report(y_test, y_pred))
    def logistic(self):
        lr = LogisticRegression()
        lr.fit(X_train, y_train.values.ravel())

        y_pred2 = lr.predict(X_test)
        print(accuracy_score(y_test, y_pred2))
        print(metrics.classification_report(y_test, y_pred2))
    def knn(self):
        kn = KNeighborsClassifier(n_neighbors=5)
        kn.fit(X_train, y_train.values.ravel())

        y_pred3 = kn.predict(X_test)
        print(accuracy_score(y_test, y_pred3))
        print(metrics.classification_report(y_test, y_pred3))

def main():
    cl = classifiers()
    cl.rfc()
    cl.logistic()
    cl.knn()

if __name__ == "__main__":
    main()