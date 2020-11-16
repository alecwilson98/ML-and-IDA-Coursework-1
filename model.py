import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import joblib

# read train data
df = pd.read_csv('/home/alecwilson/Documents/Birmingham/Kaggle/Fashion/fashion-mnist_train.csv', dtype=int)
# read test data
dft = pd.read_csv('/home/alecwilson/Documents/Birmingham/Kaggle/Fashion/fashion-mnist_test.csv', dtype=int)

X_train = df.drop('label', axis=1)
y_train = df['label']
X_test = dft.drop('label', axis=1)
y_test = dft['label']

class Classifiers():
    def rfc(self):
        rf = RandomForestClassifier(random_state=1)
        n_estimators = [10, 100, 1000]
        max_features = [range(1, 21, 2)]
        max_depth = [10, 15, 20]
        hyper_rf = dict(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
        grid_rf = GridSearchCV(rf, hyper_rf, scoring='accuracy', n_jobs=-1)
        grid_rf.fit(X_train, y_train)

        y_pred = grid_rf.predict(X_test)
        print(metrics.classification_report(y_test, y_pred))
        print('Best accuracy score: %s ' % y_pred.best_score_)
        print('Best Hyperparameters: %s' % y_pred.best_params_)

        joblib.dump(grid_rf.best_estimator_, 'rf_results.pkl', compress=1)
    def svm(self):
        sv = SVC(random_state=2)
        kernal = ['linear', 'poly', 'rbf', 'sigmoid']
        C = [100, 10, 1, 0.1, 0.01]
        hyper_sv = dict(kernal=kernal, C=C)
        grid_sv = GridSearchCV(sv, hyper_sv, scoring='accuracy', n_jobs=-1)
        grid_sv.fit(X_train, y_train)

        y_pred2 = grid_sv.predict(X_test)
        print(metrics.classification_report(y_test, y_pred2))
        print('Best accuracy score: %s ' % y_pred2.best_score_)
        print('Best Hyperparameters: %s' % y_pred2.best_params_)

        cm_2 = confusion_matrix(y_test, y_pred2)
        cm_2.plot()
        plt.show()

        joblib.dump(grid_sv.best_estimator_, 'sv_results.pkl', compress=1)
    def knn(self):
        kn = KNeighborsClassifier(random_state=3)
        n_neighbors = range(1, 21, 2)
        weights = ['uniform', 'distance']
        metric = ['euclidean', 'manhattan', 'minkowski']
        hyper_kn = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)
        grid_kn = GridSearchCV(kn, hyper_kn, scoring='accuracy', n_jobs=-1)
        grid_kn.fit(X_train, y_train)

        y_pred3 = grid_kn.predict(X_test)
        print(metrics.classification_report(y_test, y_pred3))
        print('Best accuracy score: %s ' % y_pred3.best_score_)
        print('Best Hyperparameters: %s' % y_pred3.best_params_)

        cm_3 = confusion_matrix(y_test, y_pred3)
        cm_3.plot()
        plt.show()

        joblib.dump(grid_kn.best_estimator_, 'kn_results.pkl', compress=1)
def main():
    cl = Classifiers()
    cl.rfc()
    cl.svm()
    cl.knn()

if __name__ == "__main__":
    main()