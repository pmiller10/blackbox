from data import Data
from classifier import Classifier
from score import score
from time import time
from blackbox_preprocess import BlackboxPreprocess

def log(info):
    f = file('out.txt', 'w+')
    f.write(str(info))
    f.close()



data, targets = Data.data()
extra = Data.extra()
data = data + extra

# preprocessing
start = time()
matrix = BlackboxPreprocess.to_matrix(data)
print matrix.shape
matrix = BlackboxPreprocess.scale(matrix)
#matrix = BlackboxPreprocess.rbf_kernel(matrix, 1800)
#matrix, _ = BlackboxPreprocess.autoencode(matrix, targets, [], epochs=200, new=True)
#matrix = BlackboxPreprocess.to_matrix(matrix)
#matrix = BlackboxPreprocess.polynomial(matrix, 2)
print matrix.shape
#data = matrix.tolist()

# split training and CV data
cv_data, cv_targets = data[:500], targets[:500]
data, targets, extra = data[500:1000], targets[500:], data[1000:]

# testing
preds = Classifier.preds(data, targets, cv_data, extra)
s = score(preds, cv_targets, debug=False)
print "Score: ", s
log(s)
print "Duration: ", time() - start
