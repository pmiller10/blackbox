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

# preprocessing
start = time()
matrix = BlackboxPreprocess.to_matrix(data)
print matrix.shape
matrix = BlackboxPreprocess.scale(matrix)
#matrix, _ = BlackboxPreprocess.autoencode(matrix, targets, [], epochs=200, new=True)
#matrix = BlackboxPreprocess.to_matrix(matrix)
#matrix = BlackboxPreprocess.polynomial(matrix, 2)
print matrix.shape
data = matrix.tolist()

# split training and CV data
cv_data, cv_targets = data[500:], targets[500:]
data, targets = data[:500], targets[:500]

# testing
preds = Classifier.preds(data, targets, cv_data)
s = score(preds, cv_targets, debug=False)
print s
log(s)
print "Duration: ", time() - start
