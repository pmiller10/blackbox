def score(preds, targets, debug=False):
    if len(preds) != len(targets):
        print len(preds)
        print len(targets)
        raise Exception("Different number of predictions and targets")
    correct = 0.
    total = len(preds)
    for i,p in enumerate(preds):
        t = targets[i]
        if t == p:
            correct += 1.
            if debug: print "right",t,p
        else:
            if debug: print "wrong",t,p

    return correct/total
