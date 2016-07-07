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
	parser.add_argument('RFfile', help="Classifier to load or name of file to save classifier in")
	parser.add_argument('training', help="Numpy array or HDF5-encoded MATLAB file")
	parser.add_argument('fields', nargs="+", help="Field(s) to extract from .mat file")
	parser.add_argument('--test_data',dest='testing',default=[])
	args = parser.parse_args()

	#YOU MIGHT NEED TO MOVE ME. WHY?
	if len(args.fields) <= 1:
		# Because here, we want to check if we're loading a file, and only throw this error if there are no fields in the file we're loading
		raise ValueError('Must import at least two parameters (including labeled variable) for RF classification')
		return
	else:
		fields = args.fields

	if (args.training).endswith('.npy'):
		train = np.load(args.training)
	else:
		train = mat2py(args.training, args.fields)

	if args.testing:
		if (args.testing).endswith('.npy'):
			test = np.load(args.testing)
		else:
			test = mat2py(args.testing, args.fields)
	else:
		test = np.array(train)

	nonpuffs, puffs, maybe, pnames = runRandomForests(train,test, args.RFfile, args.fields)
	plotRandomForests(nonpuffs, puffs, maybe, pnames)
