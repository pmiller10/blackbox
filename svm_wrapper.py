""" Wraps a Scikit-learn class. 
It takes a an instance of a class as an init argument
and then assigns this class to its model attribute.

to_wrap = ExtraTreesRegressor()
model = SklearnWrapper(to_wrap)
model.fit(data, targets)
model.predict(data)
"""

import numpy

class SklearnWrapper():

    def __init__(self, model, return_type):
        self.model = model
        self.return_type = return_type

    def fit(self, data, targets):
        self.model.fit(data, targets)

    def predict(self, data, debug=False):
        pred = self.model.predict(data)
        if debug: print pred, ' for ', data
        return pred

    def predict_proba(self, data):
        pred = self.predict(data)
        if self.return_type == 'list':
            pred = pred[0]
        elif self.return_type == 'single':
            pass
        else:
            raise Exception("return_type must be either 'list' or 'single'")
        klass = self._to_binary(pred)
        return klass

    def _to_binary(self, label):
        vect = [0.] * 9
        label = int(label[0])
        vect[label-1] = 1.
        return numpy.array(vect)
