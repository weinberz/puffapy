import argparse
import os
import sys

import h5py
import numpy as np
import scipy.io as sio

# mat2py imports a parameter from a MATLAB struct and returns it as a ndarray
# mfile the struct's name e.g. (test.mat)
# p is the desired parameter e.g. (y_pstd)

def mat2py(mfile,p):
# Finds mfile's path if it's in current dir
	mfilepath = os.path.abspath(mfile);

	# Imports the MATLAB struct and retrieves tracks (HP5 group type)
	try:
		f = h5py.File(mfile,'r');
		gp1 = f.get('tracks');

		# data is a list of ndarrays for every track of specified parameter p
		data = [gp1[element[0]][:] for element in gp1[p]];
		#t = len(gp1.get('t'));
		#data.append(gp1[element[t]][:] for element in gp1['t'])


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
