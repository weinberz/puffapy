from mat2py import mat2py 
from runRandomForests import runRandomForests 
from plotRandomForests import plotRandomForests
import argparse 
import os.path 
import numpy as np

def puffapy(): 
	return 

if __name__ == "__main__": 
	parser = argparse.ArgumentParser(description='stuff')
	parser.add_argument('RFfile')
	parser.add_argument('training')
	parser.add_argument('fields', nargs="+", help="Field(s) to extract from .mat file")
	parser.add_argument('--test_data',dest='testing',default=[])
	args = parser.parse_args()

	if args.testing == []:
		separate_testing = False
	else: 
		separate_testing = True

	if (args.training).endswith('.npy'):
		train = np.load(args.training)
		if separate_testing: 
			test = np.load(args.testing)
		else: 
			test = np.array(train)
	else:
		if len(args.fields) <= 1:
			raise TypeError('Must import at least two parameters (including target variable) for RF classification')
		else: 
			train = mat2py(args.training, args.fields)
			if separate_testing:
				test = mat2py(args.testing, args.fields)
			else:
				test = np.array(train)

	nonpuffs, puffs, maybe, pnames = runRandomForests(train,test, args.RFfile, args.fields)
	plotRandomForests(nonpuffs, puffs, maybe, pnames)


