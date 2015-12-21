import sys
import random
import numpy as np

def load_data() :
	X = []
	Y = []
	m = []
	with open('hw1_15_train.dat') as f :
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

def train(X, Y, rand = False, alpha = 1) :
	col = len(X[0])
	n = len(X)
	w = np.zeros(col)
	ans = 0
	idx = range(n)
	if rand :
		idx = random.sample(idx, n)
	k = 0
	update = False
	while True :
		i = idx[k]
		if sign(np.dot(X[i], w)) != Y[i] :
			ans += 1
			w = w + alpha * Y[i] * X[i, :]
			update = True
		k += 1
		if k == n :
			if update == False :
				break
			k = 0
			update = False
	return ans

def naive_cycle() :
	X, Y = load_data()
	ans = train(X, Y)
	print(ans)

def predefined_random(n, alpha = 1) :
	X, Y = load_data()
	cnt = 0
	for i in range(n) :
		cnt += train(X, Y, rand = True, alpha = alpha)
	print(cnt / n)

def main() :
	naive_cycle()
	predefined_random(2000)
	predefined_random(2000, 0.5)

if __name__ == '__main__' :
	main()
 
