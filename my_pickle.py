import pickle
def save(name, out):
	f = open(name, 'w')
	pickle.dump(out, f)
	f.close()

def load(name):
	f = open(name, 'r')
	new_net = pickle.load(f)
	return new_net
