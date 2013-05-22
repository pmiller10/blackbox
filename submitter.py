from data import Data
from classifier import Classifier
from score import score
from time import time
from blackbox_preprocess import BlackboxPreprocess

def log(info):
    f = file('submission.txt', 'w+')
    f.write(str(info))
    f.close()

def submission(preds):
    out = ""
    for p in preds:
        out += str(p) + "\n"
    log(out)


data, targets = Data.data()
print "training data: ", len(data)
test = Data.test()
print "test data: ", len(test)
data = data + test
print "all data: ", len(data)

# preprocessing
start = time()
matrix = BlackboxPreprocess.to_matrix(data)
print matrix.shape
matrix = BlackboxPreprocess.scale(matrix)
#matrix = BlackboxPreprocess.polynomial(matrix, 2)
print matrix.shape
data = matrix.tolist()

# split training and test data
test_data = data[1000:]
data, targets = data[:1000], targets[:1000]

# testing
preds = Classifier.preds(data, targets, test_data, [])
print "Duration: ", time() - start

submission(preds)
