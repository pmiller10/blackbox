import sys
sys.path.append("../handwriting_classification")
from preprocess import Preprocess
from data import Data
from classifier import Classifier
from score import score

data, targets = Data.data()

# preprocessing
matrix = Preprocess.to_matrix(data)
print matrix.shape
matrix = Preprocess.scale(matrix)
#matrix = Preprocess.polynomial(matrix, 2)
print matrix.shape
data = matrix.tolist()

# split training and CV data
cv_data, cv_targets = data[500:], targets[500:]
data, targets = data[:500], targets[:500]

# testing
preds = Classifier.preds(data, targets, cv_data)
print score(preds, cv_targets, debug=False)
