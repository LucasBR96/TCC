import numpy as np
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

        self.time_elapsed = 0   # Time since the simulation started

    def __call__( self ):

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

def rk2( X : np.ndarray , h : float, der_fun : Callable ):

    k1 = der_fun( X )
    k2 = der_fun(
        X + h*k1 
    )
    return ( k1 + k2 )/2

def get_rk2_machine( X : np.ndarray , h : float, der_fun : Callable , h_fun : Callable = None ):
    return rk_machine( X , h , der_fun , rk2 , h_fun )

def rk4( X : np.ndarray , h : float, der_fun : Callable ):

    k1 = der_fun( X )
    k2 = der_fun( X + k1*( h/2 ) )
    k3 = der_fun( X + k2*( h/2 ) )
    k4 = der_fun( X + k3*h )

    return ( k1 + 2*k2 + 2*k3 + k4 )/6

def get_rk4_machine( X : np.ndarray , h : float, der_fun : Callable , h_fun : Callable = None ):
    return rk_machine( X , h , der_fun , rk4 , h_fun )