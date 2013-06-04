from data import Data
from blackbox_preprocess import BlackboxPreprocess
from sklearn.linear_model import LogisticRegression


data, targets = Data.data()
extra = Data.test()
data = data + extra
originals = data

# preprocessing
matrix = BlackboxPreprocess.to_matrix(data)
print "(examples, dimensions): ", matrix.shape
matrix = BlackboxPreprocess.scale(matrix)
matrix = BlackboxPreprocess.polynomial(matrix, 2)
print "(examples, dimensions): ", matrix.shape
data = matrix.tolist()

# split training and CV data
tr_data = data[:1000]
unlabeled = data[1000:]

# create psuedo labels
model = LogisticRegression(C=1.3, penalty='l1', tol=0.05)
print len(targets)
print targets[:10]
model.fit(tr_data, targets)

labeled = []
for i,u in enumerate(unlabeled):
    orig = originals[i]
    orig = [str(f) for f in orig]
    orig = ','.join(orig)
    pred = model.predict(u)
    pred = str(pred[0])
    label = pred + ',' + orig
    labeled.append(label)
labeled = "\r\n".join(labeled)

f = file('pseudo_labels.txt', 'w+')
f.write(str(labeled))
f.close()
