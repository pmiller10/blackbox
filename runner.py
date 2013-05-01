import sys
sys.path.append("../handwriting_classification")
from preprocess import Preprocess
from data import Data
from classifier import Classifier
from score import score

data, targets = Data.data()
cv_data, cv_targets = data[500:], targets[500:]
data, targets = data[:500], targets[:500]

preds = Classifier.preds(data, targets, cv_data)
print score(preds, cv_targets)
