from scipy.linalg import circulant 
from tqdm import tqdm
import numpy as np
import math
import pickle
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Network Bump Model. It has two rings of neurons interconnected')
parser.add_argument('--nb_neurons', '-n', type=int, default=512, 
					help='Number of neurons per ring. Default value is set to 512')
parser.add_argument('--nb_stimulus', '-ns', type=int, default=1,
					help='Define the shape of the stimulus. Number of stimulaions around the ring')
args = parser.parse_args()

def generate_stimulus(stimon, stimoff, stim, delayend):
	stimulus = np.array(stim * v).reshape(args.nb_neurons, 1)
	stimon, stimoff = np.floor(stimon/dt), np.floor(stimoff/dt)
	delayend = np.floor(delayend/dt)

	return stimulus, stimon, stimoff, delayend

def f(a):
	mask = ((a > 0) & (a < 1))
	a *= (mask*a + np.invert(mask))
	a[a < 0] = 0
	mask = a >= 1
	a += -a*mask + np.sqrt(4*(a*mask + 0.75*np.invert(mask)) - 3)

	return a

def decode(r, th):
	return np.arctan2(sum(r*np.sin(th)), sum(r*np.cos(th)))


def noise(sigE, sigI):
	noise_E = sigE * np.random.randn(args.nb_neurons, 1)
	noise_I = sigI * np.random.randn(args.nb_neurons, 1)

	return noise_E, noise_I

# Variable definition
total_time = 4200
dt = 2
nb_steps = int(total_time/dt)

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

stim = 200

theta = np.array([i/args.nb_neurons*2*np.pi for i in range(0, args.nb_neurons)]) # Need to find a more elegant way
v = np.exp(kappa*(np.cos(theta)))
v = v/sum(v)
WE = circulant(v)

# Define matrices initial states
rE = np.zeros([args.nb_neurons, 1])
rI = np.zeros([args.nb_neurons, 1])

theta = theta - np.pi
v = np.exp(kappa*(np.cos(args.nb_stimulus*theta)))
v = v/sum(v)

stimulus, stimon, stimoff, delayend = generate_stimulus(200, 700, 200, 3500)

data = {}
for step in tqdm(range(1, nb_steps), ascii=True):
	
	noise_E, noise_I = noise(sigE, sigI)
	
	IE = GEE*np.dot(WE, rE) + (I0E - GIE*np.mean(rI))*np.ones([args.nb_neurons, 1])
	II = (GEI*np.mean(rE) - GII*np.mean(rI) + I0I)*np.ones([args.nb_neurons, 1])

	# Apply stimulus at a given time
	if stimon < step < stimoff:
		IE = IE + stimulus
	
	# Apply if needed to try to reduce the bump
	if delayend < step < delayend + (stimoff - stimon):
		IE = IE - stim

	# Integrate with time step dependant
	rE = rE + (f(IE) + noise_E - rE)*dt/tauE
	rI = rI + (f(II) + noise_I - rI)*dt/tauI

	# Get decoded angle from network activity and convert it to degres
	ang = decode(rE, theta.reshape(args.nb_neurons, 1))*180/np.pi
	
	# Save data in object
	data[step] = [rE, rI, ang] 

# Save data in dictionnary
pickle.dump(data, open('../Output/data.pkl', 'wb'))


