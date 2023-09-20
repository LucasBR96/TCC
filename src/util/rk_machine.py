import numpy as np
import time
from typing import *

class rk_machine:

    '''
    runge kutta of any order machine
    '''

    def __init__( self , X : np.ndarray , h : float, der_fun : Callable , rk_fun : Callable , h_fun : Callable = None ):

        self.X = X              # Initial conditions of the system
        self.X_dot = None       # Derivation of the system
        self.h = h              # Current time resolution
        self.der_fun = der_fun  # X update function
        self.h_fun = h_fun      # Adaptative timestep function
        self.rk_fun = rk_fun    # Runge kutta function

        self.time_elapsed = 0   # Time since the simulation started ( simulation )
        self.clock        = 0   # Time since the simulation started ( real )

    def __call__( self ):

        t = time.time()
        tup = self.step()
        dt = time.time() - t

        self.clock += dt
        return tup

    def step( self ):

        if self.time_elapsed > 0:
            
            #-----------------------------------------------------
            # Updating the state, using the range_kutta 2 method
            X_dot = self.rk_fun( self.X , self.h , self.der_fun )
            self.X = self.X + self.h*X_dot

            #---------------------------------------------
            # Updating the timestep
            if not ( ( self.h_fun is None ) or ( self.X_dot is None ) ):
                self.h = self.h_fun( self.h , self.X_dot , X_dot )
            self.X_dot = X_dot

        tup = ( self.time_elapsed , self.X.copy() )
        self.time_elapsed += self.h
        return tup

