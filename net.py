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

    filename_out = "data/net.txt"
    filename_in = "data/600_200_70_hidden_200_epochs_20_smoothing_2000_examples.txt"
    #filename_in = "data/900_and_400_hidden_200_epochs_and_40_smoothing_1000_examples.txt"
    #filename_in = "data/500_hidden_200_epochs_and_20_smoothing_2000_examples_squared.txt" # score of 0.27 test set

    def __init__(self, data, extra, targets, epochs=1, smoothing=1, new=True):
        if new:
            targets = [self._to_binary(t) for t in targets]
            print "...training the DNNRegressor"
            net = DNNRegressor(data, extra, targets, [len(data[0]), 600, 200, 70, 9], hidden_layer="SigmoidLayer", final_layer="SigmoidLayer", compression_epochs=epochs, bias=True, autoencoding_only=False)
            print "...running net.fit()"
            net = net.fit()
            ds = SupervisedDataSet(len(data[0]), 9)
            for i,d in enumerate(data):
                t = targets[i]
                ds.addSample(d, t)
            trainer = BackpropTrainer(net, ds)
            print "...smoothing for epochs: ", smoothing
            for i in range(smoothing):
                trainer.train()
            self.model = net
            print "...saving the model"
            save(self.filename_out, net)
        else:
            model = load(self.filename_in)
            self.model = model

    def predict(self, data):
        pred = self.model.activate(data)
        pred = self._from_binary(pred)
        #print self.model.params[:10]
        return pred

    def train(self, data, targets, epochs):
        net = self.model.copy()
        net.sortModules()
        net.params[:] = self.model.params
        targets = [self._to_binary(t) for t in targets]
        ds = SupervisedDataSet(1875, 9)
        for i,d in enumerate(data):
            t = targets[i]
            ds.addSample(d, t)
        trainer = BackpropTrainer(net, ds)
        for i in range(epochs):
            trainer.train()
        self.model = net
