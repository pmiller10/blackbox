import sys

# framework
sys.path.append("../handwriting_classification")
from preprocess import Preprocess

# DNN
sys.path.append("../DNN")
from dnn import AutoEncoder

class BlackboxPreprocess(Preprocess):

    @classmethod
    def autoencode(self, data, targets, cv):
        autoencoder = AutoEncoder(data, targets, [1875, 900, 9], hidden_layer="SigmoidLayer", final_layer="SigmoidLayer", compression_epochs=1, smoothing_epochs=0, bias=True, autoencoding_only=True)
        autoencoder = autoencoder.fit()
        data = [autoencoder.activate(d) for d in data]
        cv = [autoencoder.activate(c) for c in cv]
        print data[0]
        return data, cv
