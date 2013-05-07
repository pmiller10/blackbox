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

class Classifier(object):

    @classmethod
    def preds(self, data, targets, cv, extra):
        #model = GMM(n_components=9, covariance_type="spherical", n_iter=100)
        #model.means_ = np.array([np.array(data[targets == (i+1)]).mean(axis=0) for i in xrange(9)])
        #model.means = np.array([
        model = Cluster(data, extra, targets)
        #model = SVC(C=10, tol=0.01, degree=2, class_weight='auto', gamma=0.)
        #model = SGDClassifier(alpha=1., epsilon=0.1, n_iter=10, penalty='l2', power_t=1.)
        #model = LogisticRegression(C=1, penalty='l1', tol=0.01) # best performance around 0.4 without autoencoding
        #model = ExtraTreesClassifier()
        #model = RandomForestClassifier()
        #model = KNeighborsClassifier()
        #model = GradientBoostingClassifier()
        #model = Net(data, targets)
        #model = DeepNetClassifier(data, targets, epochs=200, smoothing=20, new=True)
        print "Model: ", model
        model.fit(data, targets)
        preds = [model.predict(c) for c in cv]
        return preds
