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
    def preds(self, data, targets, cv, extra):
        models, weights = self.train(data, targets, cv, extra)
        preds = self.vote(models, cv, weights, debug=False)
        #preds = [models[0].predict(c) for c in cv]
        return preds

    @classmethod
    def train(self, data, targets, cv, extra):
        models = []
        weights = []
        files = ["data/1000_ex_10_comp/net_epoch_1.txt",
                "data/1000_ex_10_comp/net_epoch_3.txt", 
                "data/1000_ex_10_comp/net_epoch_5.txt", 
                "data/1000_ex_10_comp/net_epoch_10.txt", 
                "data/1000_ex_4_hidden/net_epoch_1.txt", 
                "data/1000_ex_4_hidden/net_epoch_3.txt", 
                "data/1000_ex_4_hidden/net_epoch_5.txt", 
                "data/1000_ex_4_hidden/net_epoch_10.txt"]
        for f in files:
            model = DeepNetClassifier(data, extra, targets, [len(data[0]), 600, 200, 9], epochs=10, smoothing=100, new=False, filename_in=f)
            models.append(model)
            weights.append(1.)

        svc = SVC(C=10, tol=0.01, degree=2, class_weight='auto', gamma=0.)
        model = SklearnWrapper(svc, return_type='list')
        models.append(model)
        weights.append(0.5)
        #model = GMM(n_components=9, covariance_type="spherical", n_iter=100)
        #model.means_ = np.array([np.array(data[targets == (i+1)]).mean(axis=0) for i in xrange(9)])
        #model.means = np.array([
        #model = SGDClassifier(alpha=1., epsilon=0.1, n_iter=10, penalty='l2', power_t=1.)
        #weights.append(1.)

        #models.append(Cluster(data, extra, targets))
        #weights.append(1.)

        #weights.append(1.)
        #models.append(model)

        #models.append(LogisticRegression(C=1.3, penalty='l1', tol=0.05)) # best performance around 0.4 without autoencoding
        #weights.append(1.)

        #models.append(ExtraTreesClassifier())
        #weights.append(1.)

        #models.append(RandomForestClassifier())
        #weights.append(1.)

        #models.append(KNeighborsClassifier())
        #weights.append(1.0)

        #models.append(GradientBoostingClassifier())
        #weights.append(1.0)

        #model = Net(data, targets)

        #model = DeepNetClassifier(data, extra, targets, [len(data[0]), 600, 200, 9], epochs=10, smoothing=100, new=True)
        #models.append(model)
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
