import os
os.chdir( "/home/lucasfuzato/TCC/CODE" )

import numpy as np

# Physics constants -----------------------------------------------
G = 9.8
AIR_DRAG = .05

# Initial conditions ----------------------------------------------
initial_velocities = np.arange( 10 , 31 , 2 , dtype = int )
launch_angles      = np.arange( 15 , 61 , 5 , dtype = int )

# Range kutta -----------------------------------------------------
H_STEP = 1e-3      # range kutta interval
R_STEP = 5*1e-2    # recording step

# Files -----------------------------------------------------------

# information about initial conditions ( launch angle and inital velocity )
# along with max_height , distance traveled , flight duration , computational
# time and if it is in the training or test set of the first network 
COND_FILE = "data/conditions.csv"

# position of the projectile at any moment, for every initial
# condition ( foreign key )
TR_FILE = "data/conditions.csv"