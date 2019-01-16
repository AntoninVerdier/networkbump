import os
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
from tqdm import tqdm

def get_tuning_bias(step, threshold=1.2):
	files_to_keep = []
	for i in tqdm(range(1000)):
		data = pickle.load(open('../Output/data_{0}.pkl'.format(i), 'rb'))
		if np.max(data[step][0]) > threshold:
			files_to_keep.append(i)

	print('Fichiers à garder : ', files_to_keep)
	print('Nombre de fichiers : ', len(files_to_keep))

	pickle.dump(files_to_keep, open('../Output/tmp/files_to_keep.pkl', 'wb'))
	files = pickle.load(open('../Output/tmp/files_to_keep.pkl', 'rb'))

	x = np.array([])
	for i in tqdm(files):
		data = pickle.load(open('../Output/data_{0}.pkl'.format(i), 'rb'))
		x = np.append(x, data[1000][2])
	x = pickle.load(open('x.pkl', 'rb'))
	print(np.hstack(x))
	x = [int(i) for i in x]

	plt.hist(x, bins=20)
	plt.savefig('../Figures/histogram_noise1.png')
	plt.title('Diffusion of the bump (n={})'.format(len(x)))
	plt.show()


def distribution_of_decay_time(threshold=1.2):
	distribution = []
	for i in tqdm(range(1000)):
		data = pickle.load(open('../Output/data_{0}.pkl'.format(i), 'rb'))
		for step in range(150, len(data)):
			if np.max(data[step][0]) < threshold:
				distribution.append(step)
				break
	print(distribution)
	pickle.dump(distribution, open('../Output/tmp/distribution.pkl', 'wb'))

	x = pickle.load(open('../Output/tmp/distribution.pkl', 'rb'))

	plt.hist(x, bins=40)
	plt.savefig('../Figures/distribution_of_decay_time_noise1.png')
	plt.show()

get_tuning_bias(1500)
distribution_of_decay_time()