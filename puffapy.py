from mat2py import mat2py
from runRandomForests import runRandomForests
from plotRandomForests import plotRandomForests
import argparse
import os.path
import numpy as np

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Random Forest classification of tracks')
	parser.add_argument('RFfile', help="Classifier to load or name of file to save classifier in")
	parser.add_argument('training', help="Numpy array or HDF5-encoded MATLAB file")
	parser.add_argument('--fields', default = [], nargs="+", help="Field(s) to extract from .mat file")
	parser.add_argument('--test_data', dest='testing',default=[])
	args = parser.parse_args()

	if (args.training).endswith('.npy'):
		train = np.load(args.training)
		fields = list(train.dtype.names)
	else:
		if not args.fields or len(args.fields) <= 1: 
			raise ValueError('Must import at least two parameters (including labeled variable) for RF classification')
			quit()
		else: 
			fields = args.fields
		train = mat2py(args.training, args.fields)

	if args.testing:
		if (args.testing).endswith('.npy'):
			test = np.load(args.testing)
		else:
			test = mat2py(args.testing, args.fields)
	else:
		test = np.array(train)

	train = np.array(train.tolist())
	test = np.array(test.tolist())

	nonpuffs, puffs, maybe = runRandomForests(train, test, args.RFfile)
	plotRandomForests(nonpuffs, puffs, maybe, fields[1:])
