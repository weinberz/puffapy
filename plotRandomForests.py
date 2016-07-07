import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#nonpuffs, puffs and maybe are the dictionaries returned by runRandomForests.py
#Populate p2D or p3D with indices of parameters as they were entered on the command line, or as they appear in the dictionary 
def plotRandomForests(nonpuffs, puffs, maybe, pnames, p2D = [1,2], save2D = '2Dfig.jpg', p3D = [1,2,3], save3D = '3Dfig.jpg'):
	if len(pnames) < 2:
		raise ValueError('Cannot plot with less than 2 parameters')
		return
	else:
		# Plots 2D scatter plot
		fig = plt.figure()
		plt.plot(nonpuffs[p2D[0]], nonpuffs[p2D[1]], 'c.', label = 'Nonpuffs')
		plt.plot(maybe[p2D[0]], maybe[p2D[1]], 'g.', label = 'Maybe')
		plt.plot(puffs[p2D[0]], puffs[p2D[1]],'r.', label = 'Puffs')
		plt.xlabel(pnames[p2D[0]])
		plt.ylabel(pnames[p2D[1]])
		plt.title('2D Scatter Plot for RF Results')
		plt.legend()
		plt.savefig('C:\\Users\\tiffany\\Downloads\\'+ save2D)

		if len(pnames) > 2:
			#Plot in 3D
			fig = plt.figure()
			ax = fig.add_subplot(111,projection='3d')

			for c, m, xs, ys, zs in [ ('c', 'o', nonpuffs[p3D[0]], nonpuffs[p3D[1]],nonpuffs[p3D[2]]),
			('g', 'o', maybe[p3D[0]], maybe[p3D[1]],maybe[p3D[2]]),
			('r','o', puffs[p3D[0]],puffs[p3D[1]],puffs[p3D[2]]),]:
				ax.scatter(xs,ys,zs,c=c, marker = m)

			ax.set_xlabel(pnames[p3D[0]])
			ax.set_ylabel(pnames[p3D[1]])
			ax.set_zlabel(pnames[p3D[2]])
			plt.savefig('C:\\Users\\tiffany\\Downloads\\' + save3D)

		plt.show()
