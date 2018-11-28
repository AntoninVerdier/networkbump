import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg") # Only for macOs config
from matplotlib import pyplot as plt
plt.ion()

nb_neurons = 512

# Dynamic plotting
def plotting(data, theta):
	for step in data:
		plt.xlim((-180, 180))
		plt.ylim((-5, 20))
		plt.title('Iteration {}'.format(step))
		plt.plot(theta*180/np.pi, data[step][0])
		plt.pause(0.001)
		plt.show()
		plt.clf()

data = pickle.load(open('../Output/data.pkl', 'rb'))
theta = np.array([i/nb_neurons*2*np.pi for i in range(0, nb_neurons)]) - np.pi

plotting(data, theta)