from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer, SoftmaxLayer, TanhLayer

class Net(object):

    def __init__(self, data, targets):
        ds = ClassificationDataSet(len(data[0]), 1, nb_classes=9)
        for i,d in enumerate(data):
            t = targets[i]
            t = int(t[0]) - 1
            ds.addSample(d, t)
        ds._convertToOneOfMany()
        
        net = buildNetwork(len(data[0]), 10, ds.outdim, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer)
        trainer = BackpropTrainer(net, ds)
        for i in range(50):
            print trainer.train()
        self.model = net

    def fit(self, data, targets):
        pass

    def predict(self, data):
        pred = self.model.activate(data)
        return self._from_binary(pred)

    # not used anymore
    def _to_binary(self, target):
        vector = [0,0,0,0,0,0,0,0,0]
        vector[int(target[0])-1] = 1
        return vector

    def _from_binary(self, pred):
        best_index = 0
        score = 0.
        for i,prob in enumerate(pred):
            if prob > score:
                score = prob
                best_index = i
        pred = str(best_index + 1.)
        return pred
