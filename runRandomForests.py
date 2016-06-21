import numpy as np 
import scipy.io as sio
import pandas as pd 
import sys
from mat2py import mat2py 
from sklearn.ensemble import RandomForestClassifier 

def runRandomForests(train,test):
	#train = numpy.load('train.npy'); 
	#test = numpy.load('test.npy');
	#train, test = mat2py.mat2py()
	print ('#2')
	#train = train.fillna(0);
	#test = test.fillna(0);

	trainArr = train[:,1:]; 
	trainRes = train[:,0]; 
	testArr = test[:, 1:];

	rf = RandomForestClassifier(n_estimators = 100)
	rf.fit(trainArr, trainRes)

	print ('#3')
	testRes = rf.predict(testArr)
	#numpy.savetxt('rForestRes', testRes)
	#print (testRes) 
	i = 0
	for res in testRes:
		if res == 1:
			print (i)
		i = i+1
	return testRes

if __name__ == "__main__": 
	train, test = mat2py(sys.argv[1], sys.argv[2], sys.argv[3])
	runRandomForests(train,test)