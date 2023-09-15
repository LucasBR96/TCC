import os
os.chdir( "/home/lucasfuzato/TCC/CODE" )

import numpy as np
import torch as tc
import torch.utils.data as tdt
import pandas as pd

from typing import *
import random as rd
import matplotlib.pyplot as plt

from src.util.rk_machine import *
from src.util.loader_data import *
from src.util.constantes import *

# runge kutta methods --------------------------------------------------------
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

def rk_iterator( rk : rk_machine, num_iter : int = 10 ):

    # at least one iteration. if it is one, it would be 
    # better to just call the rk machine straight
    num_iter = max( 1 , num_iter )

    while True:

        yield rk()
        for _ in range( num_iter - 1 ):
            rk()


# projectile launch -------------------------------------------------------------------------
def get_xacc( vel_x ):
    return -AIR_DRAG*np.abs( vel_x )*vel_x

def get_yacc( vel_y ):
    return get_xacc( vel_y ) - G

def state_update( state : np.ndarray ):

    state_delta = np.ndarray( 4 )

    _ , _ , vx , vy = state
    state_delta[ 0 ] = vx
    state_delta[ 1 ] = vy
    state_delta[ 2 ] = get_xacc( vx )
    state_delta[ 3 ] = get_yacc( vy )

    return state_delta
