import os
import sys

import itertools as itt
import time
import threading as thrd

import numpy as np

rkpath = os.getcwd() + "/src/classes/RungeKutta"
sys.path.append( rkpath )
from rk_machine import rk_machine , get_rk4_machine

ANGLES = np.linspace( -np.pi/2 , np.pi/2 , 20 )
LENS   = np.linspace( 1 , 2 , 10 )
G = 9.8

def get_omega_dot( theta , l ):
    return -np.sin( theta )*( G/l )

def get_dpend( X : np.ndarray ):

    '''
    Considering X like:

    X = [ l , theta , omega ]

    where theta is the angle, omega is the angualar velocity and l
    is the pendulum lenght.
    '''

    X_hat = np.zeros( 3 )

    # lenght is static, so no derivation for it
    # X_hat[ 0 ] = X[ 0 ]

    # update theta is just omega
    X_hat[ 1 ] = X[ 2 ]

    #update omega follows motion equation
    X_hat[ 2 ] = get_omega_dot( X[ 1 ] , X[ 0 ] )

    return X_hat