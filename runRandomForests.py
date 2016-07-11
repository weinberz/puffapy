import argparse
import os.path
import sys

import numpy as np
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from mat2py import mat2py

# Assumes first parameter passed is always the target variable
# and sorts between train and test arrays based on that
def runRandomForests(train, test, RFfile):

	# Add the track to training data train_tracks if its been scored
	train_tracks = []
	for feature in train:
		if feature[0] != 0.:
			train_tracks.append(feature)

	train_tracks = np.array(train_tracks)
	test_tracks = np.array(test)

	# Contains parameter values for all tracks part of training data
	trainArr = train_tracks[:,1:]

	# Contains target variables for all tracks part of training data
	trainRes = train_tracks[:,0]

	# Contains parameter values for all tracks part of test data
	# Test data contains the tracks used for training
	testArr = test_tracks[:,1:]

	# Convert all NaNs to 0 for RF to work properly
	trainArr = np.nan_to_num(trainArr)
	trainRes = np.nan_to_num(trainRes)
	testArr = np.nan_to_num(testArr)

	if os.path.isfile(RFfile) == False:
		# Train the classifier and use it to predict the target variables of test data
		rf = RandomForestClassifier(n_estimators = 500, oob_score = True)
		rf.fit(trainArr, trainRes)

		# Save the classifier as RFfile
		joblib.dump(rf, RFfile, compress=True)

	else:
		# Load the classifier
		rf = joblib.load(RFfile)

	testRes = rf.predict(testArr)

	nparams = np.zeros((testArr.shape[1]))
	puffs = [[] for _ in range(len(nparams)+1)]
	nonpuffs = [[] for _ in range(len(nparams)+1)]
	maybe = [[] for _ in range(len(nparams)+1)]

	# For each class,
	# class[1] is a list of track indices classified as that class
	# class[2],[3],[4] (...) are that track's 1st, 2nd and 3rd (...) parameter values by default

	for i, res in enumerate(testRes):
		if res == 2:
			nonpuffs[0].append(i)
			for j, param in enumerate(nparams):
				nonpuffs[j+1].append(testArr[i,j])
		elif res == 1:
			puffs[0].append(i)
			for j, param in enumerate(nparams):
				puffs[j+1].append(testArr[i,j])
		else:
			maybe[0].append(i)
			for j, param in enumerate(nparams):
				maybe[j+1].append(testArr[i,j])

	# Create a dictionary of the track indices (1 indexing) for each label
	nonp= [x+1 for x in nonpuffs[0]]
	p= [x+1 for x in puffs[0]]
	m= [x+1 for x in maybe[0]]

	labels = ['nonpuffs', 'puffs', 'maybe']
	ID = [nonp,p,m]
	idx = dict(zip(labels, ID))

	# Save dictionary as a .mat
	sio.savemat('RFresults', idx)

	return nonpuffs[1:], puffs[1:], maybe[1:]

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

	runRandomForests(train, test, args.RFfile)
