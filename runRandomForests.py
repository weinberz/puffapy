import numpy as np 
import scipy.io as sio
import sys
from mat2py import mat2py 
from sklearn.ensemble import RandomForestClassifier 
import argparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import cross_validation

def runRandomForests(train,test):

	# #(TODO) This currently assumes that the first param passed is always isPuff
	# # and sorts between train and test arrays based on that

	# trainArr = []
	# trainRes = []
	# testArr = []
	# #print ('train', train)
	# #print('test', test)
	# for feature in train:
	# 	if feature[0] != 0:
	# 		trainArr.append(feature)
	train_tracks = []
	for feature in train: 
		if feature[0] != 0.: 
			train_tracks.append(feature)

	train_tracks = np.array(train_tracks)
	test_tracks = np.array(test)

	trainRes = train_tracks[:,0]
	trainArr = train_tracks[:,1:]
	testArr = test_tracks[:,1:]

	trainArr = np.nan_to_num(trainArr)
	trainRes = np.nan_to_num(trainRes)
	testArr = np.nan_to_num(testArr)

	rf = RandomForestClassifier(n_estimators = 500)
	# scores = cross_validation.cross_val_score(rf, trainArr, trainRes, cv = 10)
	# print ('mean score', scores)
	rf.fit(trainArr, trainRes)

	testRes = rf.predict(testArr)

	#Split classification results into lists with track indices and respective data
	# *Need to make plotting options customizable
	i = 0
	puffs = []
	puffs_x = []
	puffs_y = []
	puffs_z = []
	nonpuffs = []
	nonpuffs_x = []
	nonpuffs_y = []
	nonpuffs_z = []
	maybe = []
	maybe_x = []
	maybe_y = []
	maybe_z = []
	for res in testRes:
		if res == 2: 
			nonpuffs.append(i)
			nonpuffs_x.append(testArr[i,0]) 
			nonpuffs_y.append(testArr[i,1])
			nonpuffs_z.append(testArr[i,2])
		elif res == 1:
			puffs.append(i) 
			puffs_x.append(testArr[i,0]) 
			puffs_y.append(testArr[i,1]) 
			puffs_z.append(testArr[i,2]) 
		else:
			maybe.append(i)
			maybe_x.append(testArr[i,0]) 
			maybe_y.append(testArr[i,1])
			maybe_z.append(testArr[i,2])
		i = i+1
	
	#Plot in 3D
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	for c, m, xs, ys, zs in [('r','o', puffs_x,puffs_y,puffs_z), 
	('c', 'o', nonpuffs_x, nonpuffs_y,nonpuffs_z),
	('g', 'o', maybe_x, maybe_y,maybe_z)]: 
		ax.scatter(xs,ys,zs,c=c, marker = m)

	#*Needs to be customizable
	ax.set_xlabel('pallAdiff')
	ax.set_ylabel('pfallR2')
	ax.set_zlabel('pip')
	plt.savefig('C:\\Users\\tiffany\\Downloads\\fig3d.jpg')
	plt.show()

# 	i = 0
# 	puffs = []
# 	puffs_x = []
# 	puffs_y = []
# 	puffs_z = []
# 	nonpuffs = []
# 	nonpuffs_x = []
# 	nonpuffs_y = []
# 	nonpuffs_z = []
# 	maybe = []
# 	maybe_x = []
# 	maybe_y = []
# 	maybe_z = []
# 	for res in testRes:
# 		if res == 2: 
# 			nonpuffs.append(i)
# 			nonpuffs_x.append(testArr[i,1]) 
# 			nonpuffs_y.append(testArr[i,0])
# 			nonpuffs_z.append(testArr[i,3])
# 		elif res == 1:
# 			puffs.append(i) 
# 			puffs_x.append(testArr[i,1]) 
# 			puffs_y.append(testArr[i,0]) 
# 			puffs_z.append(testArr[i,3]) 
# 		else:
# 			maybe.append(i)
# 			maybe_x.append(testArr[i,1]) 
# 			maybe_y.append(testArr[i,0])
# 			maybe_z.append(testArr[i,3])
# 		i = i+1
# #Plot 2D
# 	plt.plot(nonpuffs_x, nonpuffs_y, 'c.', label = 'Nonpuffs')
# 	plt.plot(puffs_x, puffs_y,'r.', label = 'Puffs') 
# 	plt.plot(maybe_x, maybe_y, 'g.', label = 'Maybe')
# 	plt.xlabel('posthoc nDiff')
# 	plt.ylabel('# prom. peaks')
# 	plt.title('3 Parameter Model RF')
# 	plt.legend()
# 	#plt.savefig('C:\\Users\\tiffany\\Downloads\\fig.jpg')
# 	plt.show()
	#return testRes
	print ("PUFFS \n", puffs)
	print ("\nNONPUFFS \n", nonpuffs)
	print ("\nMAYBE \n", maybe)

if __name__ == "__main__": 
	parser = argparse.ArgumentParser(description='stuff')
	parser.add_argument("matfile")
	parser.add_argument("fields", nargs="+", help="Field(s) to extract from .mat file")
	parser.add_argument("--train_data",dest='training',default=[])
	args = parser.parse_args()

	if args.training == []:
		separate_training = False
	else: 
		separate_training = True

	if separate_training:
		train = mat2py(args.matfile, args.fields)
		test = mat2py(args.training, args.fields)
	else:
		train = mat2py(args.matfile, args.fields)
		test = np.array(train)

	runRandomForests(train,test)