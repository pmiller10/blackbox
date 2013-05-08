import re

class Data(object):

    unsup_file = "data/extra_unsupervised_data.csv"
    train_file = "data/train.csv"
    test_file = "data/test.csv"

    @classmethod
    def data(self, data_file=train_file):
        f = file(data_file, 'r')
        lines = f.readlines()[1:] # remove header
        lines = [re.sub("\r\n", '', line).split(',') for line in lines]
        floats = []
        targets = []
        for line in lines:
            targets.append(line[0])
            del line[0] # remove ID
            f = [float(i) for i in line]
            floats.append(f)
        return floats, targets

    @classmethod
    def test(self, data_file=test_file):
        f = file(data_file, 'r')
        lines = f.readlines()[1:] # remove header
        lines = [re.sub("\r\n", '', line).split(',') for line in lines]
        floats = []
        for line in lines[:1000]:
            f = [float(i) for i in line]
            floats.append(f)
        return floats

    @classmethod
    def extra(self, data_file=unsup_file):
        f = file(data_file, 'r')
        lines = f.readlines()[1:1001] # remove header
        lines = [re.sub("\r\n", '', line).split(',') for line in lines]
        floats = []
        for line in lines:
            f = [float(i) for i in line]
            floats.append(f)
        return floats
