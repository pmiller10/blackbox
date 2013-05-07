from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer, SoftmaxLayer, TanhLayer
from my_pickle import save, load

# DNN
import sys
sys.path.append("../DNN")
from dnn import DNNRegressor

class BaseNet(object):

    def _to_binary(self, target):
        if type(target) == list: raise Exception("the target must not have more than one element")
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

    def fit(self, data, targets):
        pass

class Net(BaseNet):

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

    def predict(self, data):
        pred = self.model.activate(data)
        return self._from_binary(pred)

class DeepNetClassifier(BaseNet):

    filename = "data/net.txt"

    def __init__(self, data, targets, epochs=1, smoothing=1, new=True):
        if new:
            targets = [self._to_binary(t) for t in targets]
            end = 500
            net = DNNRegressor(data[:end], targets[:end], [1875, 900, 400, 9], hidden_layer="SigmoidLayer", final_layer="SigmoidLayer", compression_epochs=epochs, bias=True, autoencoding_only=False)
            net = net.fit()
            ds = SupervisedDataSet(1875, 9)
            for i,d in enumerate(data):
                t = targets[i]
                ds.addSample(d, t)
            trainer = BackpropTrainer(net, ds)
            for i in range(smoothing):
                trainer.train()
            self.model = net
            save(self.filename, net)
        else:
            self.model = load(self.filename)

    def predict(self, data):
        pred = self.model.activate(data)
        pred = self._from_binary(pred)
        return pred
