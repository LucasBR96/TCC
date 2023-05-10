import numpy as np
from typing import *

class rk2_machine:

    '''
    runge kutta of degree two machine
    '''

    def __init__( self , X : np.ndarray , h : float, der_fun : Callable , h_fun : Callable ):

        self.X = X              # Initial conditions of the system
        self.X_dot = None       # Derivation of the system
        self.h = h              # Current time resolution
        self.der_fun = der_fun  # X update function
        self.h_fun = h_fun      # Adaptative timestep function

        self.time_elapsed = 0   # Time since the simulation started

    def __call__( self ):

        if self.time_elapsed > 0:
            
            #-----------------------------------------------------
            # Updating the state, using the range_kutta 2 method
            k1 = self.der_fun( self.X )
            k2 = self.der_fun(
                self.X + self.h*k1 
            )
            X_dot = ( k1 + k2 )/2
            self.X = self.X + self.h*X_dot

            #---------------------------------------------
            # Updating the timestep
            if not ( ( self.h_fun is None ) or ( self.X_dot is None ) ):
                self.h = self.h_fun( self.h , self.X_dot , X_dot )
            self.X_dot = X_dot

        tup = ( self.time_elapsed , self.X.copy() )
        self.time_elapsed += self.h
        return tup