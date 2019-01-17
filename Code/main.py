from scipy.linalg import circulant 
from tqdm import tqdm
import numpy as np
import math
import pickle
import argparse

import Plot

### Make a class for different stimulus shape
### Plot diffusion fo 

# Parse arguments
parser = argparse.ArgumentParser(description='Network Bump Model. It has two rings of neurons interconnected')
parser.add_argument('--nb_neurons', '-n', type=int, default=512, 
					help='Number of neurons per ring. Default value is set to 512')
parser.add_argument('--nb_stimulus', '-ns', type=int, default=1,
					help='Define the shape of the stimulus. Number of stimulaions around the ring')
parser.add_argument('--plot', '-p', action='store_true',
					help='Plot the data')
parser.add_argument('--name', '-f', type=str, default='unk',
					help='Suffix for plot name')
args = parser.parse_args()



class Stimulus:
	def __init__(self):
		self.kappa = 0.8
		self.theta = np.arange(args.nb_neurons)*2*np.pi/args.nb_neurons
		self.nb_neurons = args.nb_neurons

	def exponential(self, min_val=0.0035, gain=5, roll=0):
		v = np.exp(self.kappa*(np.cos(args.nb_stimulus*(self.theta - np.pi))))
		v = v/sum(v)
		v[v < min_val] = 0

		if roll != 0:
			v2 = np.roll(v, np.int(roll*512/360))
			return (v + v2)*gain
		
		return v

	def squared(self, max_val=0.004, length_stimulus=60):
		v = np.zeros(self.nb_neurons)
		v[int(v.shape[0]/2 - length_stimulus/2):int(v.shape[0]/2 + length_stimulus/2)] = max_val
		return v
	
	def sinusoidal(self, max_val=0.004, length_stimulus=60):
		v = np.zeros(self.nb_neurons)
		v[int(v.shape[0]/2 - length_stimulus/2):int(v.shape[0]/2 + length_stimulus/2)] = np.sin(np.linspace(0, np.pi, num=length_stimulus)) * max_val
		return v


def generate_stimulus(stimulus_type, stimon, stimoff, stim, delayend=3500):
	stimulus = np.array(stim * stimulus_type).reshape(args.nb_neurons, 1)
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
total_time = 5000
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


# Define connexions between neurons
theta = np.arange(args.nb_neurons)*2*np.pi/args.nb_neurons
v = np.exp(kappa*(np.cos(theta)))
v = v/sum(v)
WE = circulant(v)

# Define matrices initial states
rE = np.zeros([args.nb_neurons, 1])
rI = np.zeros([args.nb_neurons, 1])

stimulus_type = Stimulus().exponential(gain=1, roll=0)
stimulus, stimon, stimoff, delayend = generate_stimulus(stimulus_type, 200, 700, 200)

data = {}
for step in tqdm(range(1, nb_steps + 1), ascii=True):
	
	noise_E, noise_I = noise(sigE, sigI)
	
	IE = GEE*np.dot(WE, rE) + (I0E - GIE*np.mean(rI))*np.ones([args.nb_neurons, 1])
	II = (GEI*np.mean(rE) - GII*np.mean(rI) + I0I)*np.ones([args.nb_neurons, 1])

	#Apply stimulus at a given time	
	if stimon < step < stimoff:
		IE = IE + stimulus

	# # Apply if interval is needed
	# if stimoff < step < stimoff + 200:
	# 	IE = IE - stimulus

	# if stimoff + 200 < step < stimoff + 450:
	# 	IE = IE + np.roll(stimulus, 80)
	
	# # Apply if needed to try to reduce the bump
	# if delayend < step < delayend + (stimoff - stimon):
	# 	IE = IE - stimulus

	# Integrate with time step dependant
	rE = rE + (f(IE) + noise_E - rE)*dt/tauE
	rI = rI + (f(II) + noise_I - rI)*dt/tauI

	# Get decoded angle from network activity and convert it to degres
	ang = decode(rE, (theta - np.pi).reshape(args.nb_neurons, 1))*180/np.pi
	
	# Save data in object
	data[step] = [rE, rI, ang] 

# Save data in dictionnary
pickle.dump(data, open('../Output/data_{}.pkl'.format(args.name), 'wb'))

# Plotting part
if args.plot:
	Plot.heatmap(data, args.name)
	data = pickle.load(open('../Output/data_{}.pkl'.format(args.name), 'rb'))
	Plot.plotting(data, theta - np.pi)


