from sklearn.linear_model import LogisticRegression

class Classifier(object):

    @classmethod
    def preds(self, data, targets, cv):
        model = LogisticRegression()
        model.fit(data, targets)
        preds = [model.predict(c) for c in cv]
        return preds
