def score(preds, targets):
    if len(preds) != len(targets): raise Exception("Different number of predictions and targets")
    correct = 0.
    total = len(preds)
    for i,p in enumerate(preds):
        t = targets[i]
        if t == p:
            correct += 1.
    return correct/total
