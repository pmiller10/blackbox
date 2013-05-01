# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

# pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer, SoftmaxLayer

def to_binary(target):
    vector = [0,0,0,0,0,0,0,0,0]
    vector[int(target[0])-1] = 1
    return vector

def from_binary(pred):
    return pred.index(1) + 1

class Classifier(object):

    @classmethod
    def preds(self, data, targets, cv):
        #model = LogisticRegression()
        #model = ExtraTreesClassifier()
        #model = RandomForestClassifier()
        #model = KNeighborsClassifier()
        #model = GradientBoostingClassifier()
        model = self.nn(data, targets)

        if hasattr(model, 'predict'):
            model.fit(data, targets)
            preds = [model.predict(c) for c in cv]
        else:
            preds = [model.activate(c) for c in cv]
            preds = [str(p[0]) for p in preds]
            #print preds
            print model.activate(cv[0])
        return preds

    @classmethod
    def nn(self, data, targets):
        ds = ClassificationDataSet(len(data[0]), 1, nb_classes=9)
        print ds.outdim
        #targets = [to_binary(t) for t in targets]
        for i,d in enumerate(data):
            t = targets[i]
            t = int(t[0]) - 1
            ds.addSample(d, t)
        ds._convertToOneOfMany()
        print ds.outdim
        
        net = buildNetwork(len(data[0]), 5, ds.outdim, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer)
        trainer = BackpropTrainer(net, ds)
        for i in range(5):
            trainer.train()
        return net
