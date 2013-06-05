import numpy as np

# sklearn
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.mixture import GMM

# custom models
from net import Net, DeepNetClassifier
from cluster import Cluster
from svm_wrapper import SklearnWrapper

class Classifier(object):

    @classmethod
    def preds(self, data, targets, cv, cv_targets, extra):
        if len(cv) != len(cv_targets): raise Exception("Number of CV data and CV targets must be equal")
        models, weights = self.train(data, targets, cv, cv_targets, extra)
        #preds = self.vote(models, cv, weights, debug=False)
        preds = [models[0].predict(c) for c in cv]
        return preds

    @classmethod
    def train(self, data, targets, cv, cv_targets, extra):
        models = []
        weights = []
        model = DeepNetClassifier(data, targets, cv, cv_targets, extra, [len(data[0]), 100, 9], epochs=80, smoothing=200, new=True)
        models.append(model)
        weights.append(1.)

        for m in models:
            m.fit(data, targets)
        return models, weights

    @classmethod
    def vote(self, models, cv, weights, debug=False):
        preds = []
        for c in cv:
            probs = []
            for m in models:
                p = m.predict_proba(c)#[0]
                if debug: print "Preds from ", m, p
                probs.append(p)
            votes = np.zeros(len(probs[0]))
            for i,p in enumerate(probs):
                w = weights[i]
                weighted_pred = p * w
                votes = votes + weighted_pred
                if debug: print "weighted preds ", weighted_pred
            if debug: print "FINAL ", votes, "\n"
            votes = list(votes)
            _max = max(votes)
            p = votes.index(_max) + 1
            p = str(p) + '.0'
            preds.append(p)
        return preds
