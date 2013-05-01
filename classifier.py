# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

# custom models
from net import Net

class Classifier(object):

    @classmethod
    def preds(self, data, targets, cv):
        #model = LogisticRegression()
        #model = ExtraTreesClassifier()
        #model = RandomForestClassifier()
        #model = KNeighborsClassifier()
        #model = GradientBoostingClassifier()
        model = Net(data, targets)
        model.fit(data, targets)
        preds = [model.predict(c) for c in cv]
        return preds
