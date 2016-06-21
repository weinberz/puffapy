import numpy as np 
import scipy.io as sio
import h5py 
import os
import sys 

def mat2py(mfile, a, b):


# Finds mfile's path if it's in current dir
	mfilepath = os.path.abspath(mfile);

# Imports the MATLAB struct and retrieves tracks (HP5 group type)
	f = h5py.File(mfile,'r')
	gp1 = f.get('tracks')

	params = [a,b]

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


if __name__ == "__main__": 
	mat2py(sys.argv[1], sys.argv[2], sys.argv[3]);