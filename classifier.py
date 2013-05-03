# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# custom models
from net import Net

class Classifier(object):

    @classmethod
    def preds(self, data, targets, cv):
        model = SVC(C=1, tol=0.001, degree=2, class_weight='auto', gamma=0.)
        #model = LogisticRegression(C=1, penalty='l1', tol=0.01)
        #model = ExtraTreesClassifier()
        #model = RandomForestClassifier()
        #model = KNeighborsClassifier()
        #model = GradientBoostingClassifier()
        #model = Net(data, targets)
        print model
        model.fit(data, targets)
        preds = [model.predict(c) for c in cv]
        return preds
