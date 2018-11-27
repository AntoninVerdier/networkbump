# Model bump attractor 
import numpy as np
from scipy.linalg import circulant 
from tqdm import tqdm
import time
import pickle

import argparse

import matplotlib.pyplot as plt
plt.ion()

def generate_stimulus(stimon, stimoff, stim, delayend):
	v = np.exp(kappa*np.cos(theta))
	stimulus = np.array(stim * v).reshape(nb_neurons, 1)
	stimon = np.floor(stimon/dt)
	stimoff = np.floor(stimoff/dt)
	delayend = np.floor(delayend/dt)
	delaywin = np.floor(100/dt)

	return stimulus, stimon, stimoff, delayend, delaywin

def f(x):
	tmp = []
	for i in x:
		if 0 < i < 1:
			tmp.append(i*i)
		elif i >= 1:
			tmp.append(np.sqrt(4 * i -3))
		elif i < 0:
			tmp.append(0)
	return np.array(tmp).reshape(512, 1)


def decode(r, th):
	return np.arctan2(sum(np.multiply(r, np.sin(th))), sum(np.multiply(r, np.cos(th))))


def noise(sigE, sigI):
	noise_E = sigE * np.random.randn(nb_neurons, 1)
	noise_I = sigI * np.random.randn(nb_neurons, 1)

	return noise_E, noise_I

def matrix_product(X, Y):
	Z = np.zeros([512, 1])
	for i in range(X.shape[0]):
		for j in range(Y.shape[1]):
			Z[i, j] = np.sum(X[:1]*np.transpose(Y[:, :1]))

	return Z

# Variable definition
nb_neurons = 512
nb_population = 8

total_time = 4200
dt = 2
nb_steps = int(total_time/dt)
delayPop = np.zeros([nb_neurons, 1])

# Time constants of neurones equations
tauE = 20
tauI = 10

# Connection matrices
kappa = 5
GEE = 6
GIE = 4
GEI = 3.4
GII = 0.85

I0E = 0.2
I0I = 0.5

stim = 200

sigE = 1
sigI = 3

theta = np.array([i/nb_neurons*2*np.pi for i in range(0, nb_neurons)]) # Need to find a more elegant way
v = np.exp(kappa*np.cos(theta))
WE = circulant(v/sum(v))

# Define iterations when stimulus is applied

#### Preliminary calculations #####

# Define matrices initial states
rE = np.zeros([nb_neurons, 1])
rI = np.zeros([nb_neurons, 1])

theta = theta - np.pi
v = np.exp(kappa*np.cos(theta))
v = v/sum(v)

stimulus, stimon, stimoff, delayend, delaywin = generate_stimulus(1000, 1500, 200, 3500)

data = {}
for i in tqdm(range(1, nb_steps), ascii=True):
	# Additive noise
	noise_E, noise_I = noise(sigE, sigI)

	Z = matrix_product(WE, rE)
	
	IE = GEE*Z + (I0E - GIE*np.mean(rI))*np.ones([nb_neurons, 1])
	II = (GEI*np.mean(rE) - GII*np.mean(rI) + I0I)*np.ones([nb_neurons, 1])

	if stimon < i < stimoff:
		IE = IE + stimulus

	if delayend < i < delayend + (stimoff - stimon):
		IE = IE - stim

	if i > delayend - delaywin and i <= delayend:
		delayPop = delayPop + rE/delaywin

	# Integrate with time step dependant
	rE = rE + (f(IE) + noise_E - rE)*dt/tauE
	rI = rI + (f(II) + noise_I - rI)*dt/tauI

	# Get decoded angle from network activity

	# ang = decode(rE, np.transpose(theta))
	# if i < delayend: response=ang
	data[i] = rE

pickle.dump(data, open('../Output/data.pkl', 'wb'))


