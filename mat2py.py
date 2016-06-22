import argparse
import os
import sys

import h5py
import numpy as np
import scipy.io as sio

# mat2py imports a parameter from a MATLAB struct and returns it as a ndarray
# mfile the struct's name e.g. (test.mat)
# p is the desired parameter e.g. (y_pstd)

def mat2py(mfile, params):
# Finds mfile's path if it's in current dir
	mfilepath = os.path.abspath(mfile);

	# Imports the MATLAB struct and retrieves tracks (HP5 group type)
	try:
		# Imports the MATLAB struct and retrieves tracks (HP5 group type)
		f = h5py.File(mfile,'r')
		gp1 = f.get('tracks')

		for p in params:
			# data is a list of ndarrays for every track of specified parameter p[i]
			data = [gp1[element[0]][:] for element in gp1[p]]
			if p == params[0]:
				arr = data
			else:
				arr = np.hstack((arr,data))

		arr = np.squeeze(arr)

		train = []
		test = []
		i = 0
		while i < len(arr):
			test.append(arr[i])
			if arr[i,0] != 0.:
				train.append(arr[i])
			#else:
				#test.append(arr[i])
			i = i+1

		train = np.array(train)
		test = np.array(test)
		print ('HIHIIIH', len(test))

		f.close()
		print ('Success!')
		return train, test

	except OSError:
		print("Woops, old matlab format. This will fail.")

	# arr is an ndarray of ndarrays
	arr = np.array(data);

	return arr;

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Convert .mat files to numpy arrays for use with randomforest.py")
	parser.add_argument("matfile", help="The file to be parsed")
	parser.add_argument("fields", nargs="+", help="Field(s) to extract from .mat file")
	args = parser.parse_args()
	mat2py(args.matfile,args.fields);
