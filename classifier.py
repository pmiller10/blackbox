# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# custom models
from net import Net, DeepNetClassifier

class Classifier(object):

    @classmethod
    def preds(self, data, targets, cv):
        #model = SVC(C=1, tol=0.001, degree=2, class_weight='auto', gamma=0.)
        #model = LogisticRegression(C=1, penalty='l1', tol=0.01) # best performance around 0.4 without autoencoding
        #model = ExtraTreesClassifier()
        #model = RandomForestClassifier()
        #model = KNeighborsClassifier()
        #model = GradientBoostingClassifier()
        #model = Net(data, targets)
        model = DeepNetClassifier(data, targets, epochs=1, smoothing=1, new=True)
        print "Model: ", model
        model.fit(data, targets)
        preds = [model.predict(c) for c in cv]
        return preds
