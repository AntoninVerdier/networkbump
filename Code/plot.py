import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from tqdm import tqdm

plt.ion()

nb_neurons = 512

# Dynamic plotting

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

def scatter3d_time_plotting(data, theta):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	xs = np.tile(np.arange(4200, step=2), nb_neurons)
	ys = np.tile(theta, 2100)
	xs, ys = np.meshgrid(xs, ys, copy=False)

	zs = np.array(data)	
	print('begin plotting')   
	ax.plot_surface(xs, ys, zs)

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.savefig('scatter3d_time_plotting.png')


def plot_angular_diffusion(data, theta, name):

	x = [step for step in data]
	y = [data[step][2] for step in data]
	
	plt.savefig('../Figures/plot_angular_diffusion_{}.png'.format(name))
	plt.plot(x[200:1500], y[200:1500])

def plotting(data, theta):
	for step in tqdm(data):
		#plt.subplot(311)
		plt.xlim((-180, 180))
		plt.ylim((-5, 20))
		plt.title('Iteration {}'.format(step))
		plt.plot(theta*180/np.pi, data[step][0])
		plt.vlines(data[step][2], 10, 12) 
		
		#Inhibitory neurons 
		# plt.subplot(312)
		# plt.xlim((-180, 180))
		# plt.ylim((-5, 20))
		# plt.title('Iteration {}'.format(step))
		# plt.plot(theta*180/np.pi, data[step][1])

		# Angular deviation 
		# plt.subplot(313)
		# x = [step for step in data]
		# y = [data[step][2] for step in data]
		# plt.plot(x, y)
	
		plt.pause(0.001)	
		#plt.savefig('../Figures/MOVIE/{}.png'.format(step))
		plt.show()
		plt.clf()

def heatmap(data, name):

	heatmap = np.array([data[step][0] for step in data]).reshape(len(data), nb_neurons)
	ax = sns.heatmap(heatmap)

	plt.title('Heatmap of activity along the ring')
	plt.savefig('../Figures/heatmap_{}.png'.format(name))

def get_tuning_bias(step, threshold=1.2):
	files_to_keep = []
	for i in tqdm(range(1000)):
		data = pickle.load(open('../Output/data_noisex05_{0}.pkl'.format(i), 'rb'))
		if np.max(data[step][0]) > threshold:
			files_to_keep.append(i)

	pickle.dump(files_to_keep, open('../Output/tmp/files_to_keep.pkl', 'wb'))
	files = pickle.load(open('../Output/tmp/files_to_keep.pkl', 'rb'))

	x = np.array([])
	for i in tqdm(files):
		data = pickle.load(open('../Output/data_noisex05_{0}.pkl'.format(i), 'rb'))
		x = np.append(x, data[1000][2])
	pickle.dump(x, open('x.pkl', 'wb'))
	x = pickle.load(open('x.pkl', 'rb'))
	x = [int(i) for i in x]

	plt.hist(x, bins=15)
	plt.title('Diffusion of the bump (n={})'.format(len(x)))
	plt.savefig('../Figures/histogram_noise05.png')
	plt.close()



def distribution_of_decay_time(threshold=1.2):
	distribution = []
	for i in tqdm(range(1000)):
		data = pickle.load(open('../Output/data_noisex05_{0}.pkl'.format(i), 'rb'))
		for step in range(200, len(data)):
			if np.max(data[step][0]) < threshold:
				distribution.append(step)
				break
	pickle.dump(distribution, open('../Output/tmp/distribution.pkl', 'wb'))

	x = pickle.load(open('../Output/tmp/distribution.pkl', 'rb'))

	plt.hist(x, bins='auto')
	plt.savefig('../Figures/distribution_of_decay_time_noise05.png')

# get_tuning_bias(1500, threshold=0.6)
# distribution_of_decay_time(threshold=0.6)