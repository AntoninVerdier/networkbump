import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()

nb_neurons = 512

# Dynamic plotting


def scatter3d_time_plotting(data, theta):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	xs = np.tile(np.arange(4200, step=2), nb_neurons)
	ys = np.tile(theta, 2100)
	xs, ys = np.meshgrid(xs, ys)

	zs = data
	print('begin plotting')   
	ax.plot_surface(xs, ys, zs)

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.savefig('scatter3d_time_plotting.png')


def plotting(data, theta):
	for step in data:
		#plt.subplot(311)
		plt.xlim((-180, 180))
		plt.ylim((-5, 20))
		plt.title('Iteration {}'.format(step))
		plt.plot(theta*180/np.pi, data[step][0])
		plt.vlines(data[step][2], 10, 12) 		# Cursor to decode the angle (function unclear)
		
		# Inhibitory neurons 
		# plt.subplot(312)
		# plt.xlim((-180, 180))
		# plt.ylim((-5, 20))
		# plt.title('Iteration {}'.format(step))
		# plt.plot(theta*180/np.pi, data[step][1])
		plt.pause(0.001)

		# plt.subplot(313, projection='polar')


		plt.show()
		plt.clf()

data = pickle.load(open('../Output/data.pkl', 'rb'))

theta = np.array([i/nb_neurons*2*np.pi for i in range(0, nb_neurons)]) - np.pi

#plotting(data, theta)
rE_line = [i for step in data for i in data[step][0]]
print(len(rE_line))

scatter3d_time_plotting(rE_line, theta)


def experimental_3d_plotting(data, theta):
	fig = plt.figure()
	ax = fig.gca(projection='3d')


	# Make data.
	xs = np.cos(theta)
	ys = np.sin(theta)
	X, Y = np.meshgrid(xs, ys)
	Z = data[500][0]
	# Plot the surface.
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
						   linewidth=0, antialiased=False)

	# Customize the z axis.
	ax.set_xlim(-1, 1)
	ax.set_ylim(-1, 1)
	ax.set_zlim(-5, 10)

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()