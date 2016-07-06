import argparse
import os
import sys
import h5py
import numpy as np

# mat2py imports a parameter from a MATLAB struct and returns it as a ndarray
# mfile the struct's name e.g. (test.at); should always be in the current directory
# params are the parameters used for classification; the first should always be the target variable e.g. (isPuff, pallAdiff, pfallR2, pip)

def mat2py(mfile, params):
	# Finds mfile's path if it's in current dir
	mfilepath = os.path.abspath(mfile)

	try:
		# Imports MATLAB file and retrieves tracks (name of the struct)
		f = h5py.File(mfilepath,'r')
		gp1 = f.get('tracks')
	
		for p in params:
			# data is an ndarray of values of p for all tracks
			data = [gp1[element[0]][:] for element in gp1[p]]
			# arr becomes a ndarray containing all desired parameter values for all tracks
			# Builds arr one parameter at a time
			if p == params[0]:
				arr = np.array(data)
			else:
				arr = np.hstack((arr,data))
		
		#Eliminate any extra, unneeded dimensions
		arr = np.squeeze(arr)
		f.close()

		np.save('ndarray'+os.path.splitext(mfile)[0], arr)
		return arr 

	except OSError:
		print("Woops, old matlab format. This will fail.")
		return 

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Convert .mat files to numpy arrays for use with randomforest.py")
	parser.add_argument("matfile", help="The file to be parsed and trained from")
	parser.add_argument("fields", nargs="+", help="Field(s) to extract from .mat file")
	args = parser.parse_args()
	mat2py(args.matfile, args.fields);