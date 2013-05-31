from data import Data
from classifier import Classifier
from score import score
from time import time
from blackbox_preprocess import BlackboxPreprocess
from sklearn import preprocessing

"""
So far, the best performance with logistic regression involves the following preprocessing steps:
1. scale to 0 mean and variance of 1
2. square it
This should leave the mean of the squares at 0 as well. These steps give a score of ~0.4

When I tried normalizing it between 0 and 1, performance suffered. I tried scaling it, squaring and then taking
the norm. However, values that had been below the mean were now above the mean. This made me think that it might
be lognormally distributed. The scaling function was fitting it to a Gaussian distribution. If it fit it to a 
Gaussian distribution, and then when it was normalized between 0 and 1 the long tail of the actual distribution
might push values above or below the mean, depending on whether the long tail was above or below the mean.

I found that the following steps work best when normalizing between 0 and 1:
1. scale to 0 mean and variance of 1 (not including this step decreases CV score to ~0.37)
2. square it
3. take the L2 norm (not including this step decreases CV score to ~0.35)
4. normalize between 0 and 1

When all 4 steps are followed (in order) the score is ~0.4
"""



def log(info):
    f = file('out.txt', 'w+')
    f.write(str(info))
    f.close()

data, targets = Data.data()
extra = Data.test()
data = data + extra

# preprocessing
start = time()
matrix = BlackboxPreprocess.to_matrix(data)
print "(examples, dimensions): ", matrix.shape
matrix = BlackboxPreprocess.scale(matrix)
#matrix = BlackboxPreprocess.polynomial(matrix, 2)
matrix = preprocessing.normalize(matrix, norm='l2')
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1.,1.))
matrix = min_max_scaler.fit_transform(matrix)
#matrix = BlackboxPreprocess.norm(matrix)
print "(examples, dimensions): ", matrix.shape
data = matrix.tolist()

# split training and CV data
cv_data, cv_targets = data[:500], targets[:500]
data, targets, extra = data[:1000], targets, data[1000:]

# testing
preds = Classifier.preds(data, targets, cv_data, extra)
s = score(preds, cv_targets, debug=False)
print "Score: ", s
log(s)
print "Duration: ", time() - start
