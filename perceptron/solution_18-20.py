import sys
import random
import numpy as np

def load_data(file_path) :
	X = []
	Y = []
	m = []
	with open(file_path) as f :
		for line in f :
			m.append([float(i) for i in line.split()])
	mm = np.array(m)
	row = len(m)
	X = np.c_[np.ones(row), mm[::, :-1]]
	Y = mm[:,-1]
	return X, Y
	
def sign(x) :
	if x > 0 :
		return 1.
	return -1.


def test(X, Y, w) :
	n = len(Y)
	ne = sum([1 for i in range(n) if sign(np.dot(X[i], w)) != Y[i]])
	return ne / float(n)

def train(X, Y, updates = 50, pocket = True) :
	col = len(X[0])
	n = len(X)
	w = np.zeros(col)
	wg = w
	error = test(X, Y, w);

	for k in range(updates) :
		idx = random.sample(range(n), n)
		for i in idx :
			if sign(np.dot(X[i], w)) != Y[i] :
				w = w + Y[i] * X[i]
				e = test(X, Y, w)
				if e < error :
					error = e
					wg = w
				break
	if pocket :
		return wg
	return w

def main() :
	X, Y = load_data('hw1_18_train.dat')
	TX, TY = load_data('hw1_18_test.dat')
	error = 0
	n = 200
	for i in range(n) :
		#w = train(X, Y, updates = 50)
		#w = train(X, Y, updates = 50, pocket = False)
		w = train(X, Y, updates = 100, pocket = True)
		error += test(TX, TY, w)
	print(error / n)

if __name__ == '__main__' :
	main()

