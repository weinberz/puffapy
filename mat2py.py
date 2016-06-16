import numpy as np 
import scipy.io as sio
import h5py 
import os
import sys 

# mat2py imports a parameter from a MATLAB struct into a Python ndarray 
# mfile the struct's name e.g. ('test.mat')
# p is the desired parameter e.g. ('y_pstd')

def mat2py(mfile,p):


# Finds mfile's path if it's in current dir
	mfilepath = os.path.abspath(mfile);

# Imports the MATLAB struct and retrieves tracks (HP5 group type)
	f = h5py.File(mfile); 
	gp1 = f.get('tracks');

# data is a list of ndarrays for every track of specified parameter p
	data = [gp1[element[0]][:] for element in gp1[p]];
	#t = len(gp1.get(p));

	#data = [gp1[element[0]][:] for element in gp1[44]];
	#data.append(gp1[element[t]][:] for element in gp1[p])
	# i = 0;
	# for d in data: 
	# 	if d == None: 
	# 		return i;
	# 	i += 1; 

# arr is an ndarray of ndarrays 
	arr = np.array(data);
	print (arr);
	#return arr;

if __name__ == "__main__": 
	mat2py(sys.argv[1], sys.argv[2]);

# Finds mfile's path, if you don't know where it is
	# for r,d,f in os.walk("C:\\"):
 #    	for files in f:
 #        	 if files == mfile:
 #              	mfilepath == os.path.join(r,files)


# Retrieves list of parameter names for reference 
# (Not super necessary but nice to have)
	# param = [];
	# for item in gp1.keys():
	# 	param = param + [item];

