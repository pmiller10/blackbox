from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer, SoftmaxLayer, TanhLayer
from my_pickle import save, load
from score import score
import numpy

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

    filename_in = "data/1000_ex_4_hidden/net_epoch_3.txt"
    #filename_in = "data/600_200_70_hidden_200_epochs_20_smoothing_2000_examples.txt" # score of 0.21 on test set
    #filename_in = "data/900_and_400_hidden_200_epochs_and_40_smoothing_1000_examples.txt" # score of 0.35 on test set
    #filename_in = "data/900_and_400_hidden_200_epochs_and_100_smoothing_500_examples.txt"
    #filename_in = "data/600_200_70_hidden_200_epochs_and_20_smoothing_scaled.txt"
    #filename_in = "data/500_hidden_200_epochs_and_20_smoothing_2000_examples_squared.txt" # score of 0.27 test set
    #filename_in = "data/600_200_hidden_10_compression_60_smoothing_1000_examples_score_52.txt" # score of 34.7 test set

    def __init__(self, data, targets, cv_data, cv_targets, extra, layers, epochs=1, smoothing=1, new=True, filename_in=False):
        
        if len(cv_data) != len(cv_targets): raise Exception("Number of CV data and CV targets must be equal")
        if len(data) != len(targets): raise Exception("Number of data and targets must be equal")

        if new:
            class_tr_targets = [str(int(t[0]) - 1) for t in targets] # for pybrain's classification datset
            print "...training the DNNRegressor"
            net = DNNRegressor(data, extra, class_tr_targets, layers, hidden_layer="SigmoidLayer", final_layer="SoftmaxLayer", compression_epochs=epochs, bias=True, autoencoding_only=False)
            print "...running net.fit()"
            net = net.fit()
            ds = ClassificationDataSet(len(data[0]), 1, nb_classes=9)
            bag = 2
            noisy, _ = self.dropout(data, noise=0.2, bag=bag, debug=True)
            bagged_targets = []
            for t in class_tr_targets:
                for b in range(bag):
                    bagged_targets.append(t)
            for i,d in enumerate(noisy):
                t = bagged_targets[i]
                ds.addSample(d, t)
            ds._convertToOneOfMany()

            print "...smoothing for epochs: ", smoothing
            self.model = net
            preds = [self.predict(d) for d in cv_data]
            cv = score(preds, cv_targets, debug=False)
            preds = [self.predict(d) for d in data]
            tr = score(preds, targets, debug=False)
            trainer = BackpropTrainer(net, ds, verbose=True, learningrate=0.001, weightdecay=0.1)
            print "Train score before training: ", tr
            print "CV score before training: ", cv
            for i in range(smoothing):
                trainer.train()
                self.model = net
                preds = [self.predict(d) for d in cv_data]
                cv = score(preds, cv_targets, debug=False)
                preds = [self.predict(d) for d in data]
                tr = score(preds, targets, debug=False)
                print "Train score at epoch ", (i+1), ': ', tr
                print "CV score at epoch ", (i+1), ': ', cv
                #if i == 1:
                    #print "...saving the model"
                    #save("data/1000_ex_4_hidden/net_epoch_1.txt", net)
                #elif i == 3:
                    #print "...saving the model"
                    #save("data/1000_ex_4_hidden/net_epoch_3.txt", net)
                #elif i == 5:
                    #print "...saving the model"
                    #save("data/1000_ex_4_hidden/net_epoch_5.txt", net)
            print "...saving the model"
            #save("data/1000_ex_4_hidden/net_epoch_10.txt", net)
        else:
            model = load(filename_in)
            self.model = model

    def predict(self, data):
        pred = self.model.activate(data)
        pred = self._from_binary(pred)
        #print self.model.params[:10]
        return pred

    def predict_proba(self, data):
        pred = self.model.activate(data)
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

    def dropout(self, data, noise=0., bag=1, debug=False):
        if bag < 1:
            raise Exception("bag must be 1 or greater")
        length = len(data[0])
        zeros = round(length * noise)
        ones = length - zeros
        zeros = numpy.zeros(zeros)
        ones = numpy.ones(ones)
        merged = numpy.concatenate((zeros, ones), axis=1)
        dropped = []
        originals = []
        bag = range(bag) # increase by this amount
        for d in data:
            for b in bag:
                numpy.random.shuffle(merged)
                dropped.append(merged * d)
                originals.append(d)
        if debug:
            print "...number of data: ", len(data)
            print "...number of bagged data: ", len(dropped)
            print "...data: ", data[0][:10]
            print "...noisy data: ", dropped[0][:10]
        return dropped, originals
