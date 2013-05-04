import sys
from my_pickle import save, load

# framework
sys.path.append("../handwriting_classification")
from preprocess import Preprocess

# DNN
sys.path.append("../DNN")
from dnn import AutoEncoder

filename = "data/net.txt"

class BlackboxPreprocess(Preprocess):

    @classmethod
    def autoencode(self, data, targets, cv, epochs=1, new=True):
        if new:
            end = 1000
            autoencoder = AutoEncoder(data[:end], targets[:end], [1875, 900, 9], hidden_layer="SigmoidLayer", final_layer="SigmoidLayer", compression_epochs=epochs, bias=True, autoencoding_only=True)
            autoencoder = autoencoder.fit()
            save(filename, autoencoder)
        else:
            autoencoder = load(filename)
        data = [autoencoder.activate(d) for d in data]
        cv = [autoencoder.activate(c) for c in cv]
        print data[0][:10]
        return data, cv
