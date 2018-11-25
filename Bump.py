# Model bump attractor 
import numpy as np
from scipy.linalg import circulant 
from tqdm import tqdm


import matplotlib.pyplot as plt
plt.ion()

def noise(sigE, sigI):
	noise_E = sigE * np.random.randn(nb_neurons, 1)
	noise_I = sigI * np.random.randn(nb_neurons, 1)

	return noise_E, noise_I

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

# Dynamic plotting
def plotting(data, theta):
	for step in data:
		plt.xlim((-180, 180))
		plt.ylim((-5, 20))
		plt.title('Iteration {}'.format(step))
		plt.plot(theta*180/np.pi, data[step])
		plt.pause(0.001)
		plt.show()
		plt.clf()

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
kappa = 1.5
GEE = 6
GIE = 4
GEI = 3.4
GII = 0.85

I0E = 0.2
I0I = 0.5

sigE = 1
sigI = 3

stimon = 1000
stimoff = 1500
stim = 200
delayend = 3500

theta = np.array([i/nb_neurons*2*np.pi for i in range(0, nb_neurons)]) # Need to find a more elegant way
v = np.exp(kappa*np.cos(theta))
v = v/sum(v)
WE = circulant(v)

# Define iterations when stimulus is applied

#### Preliminary calculations #####

# Define matrices initial states
rE = np.zeros([nb_neurons, 1])
rI = np.zeros([nb_neurons, 1])

theta = theta - np.pi
v = np.exp(kappa*np.cos(theta))
v = v/sum(v)
stimulus = np.array(stim * v).reshape(512, 1)
stimon = np.floor(stimon/dt)
stimoff = np.floor(stimoff/dt)
delayend = np.floor(delayend/dt)
delaywin = np.floor(100/dt)


data = {}
for i in tqdm(range(1, nb_steps), ascii=True):
	# Additive noise
	noise_E, noise_I = noise(sigE, sigI)

	IE = GEE*np.matmul(WE, rE) + (I0E - GIE*np.mean(rI))*np.ones([nb_neurons, 1])
	II = (GEI*np.mean(rE) - GII*np.mean(rI) + I0I)*np.ones([nb_neurons, 1])

	if i > stimon and i < stimoff:
		IE = IE + stimulus

	if i > delayend and i < delayend + (stimoff - stimon):
		IE = IE - stim

	if i > delayend - delaywin and i <= delayend:
		delayPop = delayPop + rE/delaywin

	# Integrate with time step dependant
	rE = rE + (f(IE) + noise_E - rE)*dt/tauE
	rI = rI + (f(II) + noise_I - rI)*dt/tauI

	# Get decoded angle from network activity	plt.plot(theta*180/np.pi, rE)

	# ang = decode(rE, np.transpose(theta))
	# if i < delayend: response=ang
	data[i] = rE

#plotting(data, theta)

