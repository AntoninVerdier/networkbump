# Ring Network Model

## Run the code

The code is available under `Code/` . Please launch `main.py --help` to see options. All options regarding network properties (such as excitation matrix, time constants, ...) are for now hard-coded. An update may be provided to ensure better flexibility.

## Plot data

You will find in `Figures/` and `Output/` example of plots and raw data output , respectively. Each run of main outputs a `.pkl` containing a dictionary. Keys are steps and the value is a list of three objects. The first one is a numpy array of firing rates for all excitatory neurons, the second one is a numpy array of firing rates for all inhibitory neurons  and the last is the angle associated to the center of mass of the bump.

Various functions are available in `Code/plot.py`, please feel free to call them at the end of `Code/main.py`. Some of them generate intermediate pickle files because computing can be long.



This project is still under construction. Please feel free to make suggestions !



