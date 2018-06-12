

import movements as ld
import numpy as np
import mlp


filename = '../data/movements_day1-3.dat'

(train, traint), (valid, validt), (test,testt) =ld.load_datafile(filename=filename)
#number of hidden nodes.
hiddennodes = [2,6,8,12]
for i, hnodes in enumerate(hiddennodes): #irerate thro the given number of hidden nodes
	print('Number of hidden nodes = {} '.format(hnodes))
	it = 10
	for i in range(it):
		net = mlp.mlp(train, traint, hnodes, eta = 0.1)
		net.earlystopping(train, traint, valid, validt)
	net.confusion(test, testt)	
