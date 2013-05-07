# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

class Cluster(object):

    def __init__(self, data, unsup, targets):
        model = KMeans(n_clusters=9)
        model.fit(data + unsup)
        self.model = model
        preds = [self._predict_class(d) for d in data]
        #trn_data = [(d+preds[i]) for i,d in enumerate(data)]

        clf = LogisticRegression(C=1, penalty='l1', tol=0.01) # best performance around 0.4 without autoencoding
        clf.fit(preds, targets)
        self.clf = clf

    def _predict_class(self, data):
        label = self.model.predict(data)
        vect = self._to_binary(label)
        return vect

    def _to_binary(self, label):
        vect = [0] * len(self.model.labels_)
        vect[label] = 1
        return vect

    def predict(self, data):
        vect = self._predict_class(data)
        return self.clf.predict(vect)[0]

    def fit(self, data, targets):
        pass
