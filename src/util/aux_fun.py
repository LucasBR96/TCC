import os
os.chdir( "/home/lucasfuzato/TCC/CODE" )

import numpy as np
import torch as tc
import torch.nn as tnn
import torch.optim as top
import torch.nn.functional as tfun
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

# training -----------------------------------------------------------------------------------
def make_net( 
        shape : Tuple, 
        activation : tnn.Module = tnn.ReLU,
        omega = 0,
        lr : float = 1e-3
    ):

    # making the mlp regressor
    net = tnn.Sequential()
    n = len( shape )
    for i in range( 1 , n ):
        in_size = shape[ i - 1 ]
        out_size = shape[ i ]
        net.append( tnn.Linear( in_size , out_size , dtype = STD_TYPE , device = DEVICE) )
        if i < n - 1:
            net.append( activation() )
    # net = net.to( device )

    # optimizer
    opm = top.Adam( net.parameters() , lr = lr , weight_decay = omega )

    return net , opm

def fit_fun( net , opm ):

    def update( X : tc.Tensor , y : tc.Tensor ):

        net.zero_grad()

        # ---------------------------------
        # data loss
        y_hat = net( X )
        loss = tfun.mse_loss( y , y_hat )
        loss.backward()
        opm.step()
        
    return update

def eval_fun( net ):

    def eval_f( X : tc.Tensor , y : tc.Tensor ):

        with tc.no_grad():
            y_hat = net( X )
            loss = tfun.mse_loss( y , y_hat )

        return loss.cpu().item()
        
    return eval_f

# ----------------------------------------------------------
# Data manipulation

# copied from stack overflow
# source: https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
def moving_average( a : np.ndarray , n : int = 3 ):
    
    # a   = [ 0 , 1 , 2 , 2 , 5 , 2 , 6 , 4 ] , n = 3
    # ret = [ 0 , 1 , 3 , 5 , 10 , 12 , 18 , 22 ]
    ret = np.cumsum( a , axis = 0, dtype = float )

    # ret[ n: ]  = [ 5 , 10 , 12 , 18 , 22 ]
    # ret[ :-n ] = [ 0 , 1 , 3 , 5 , 10 ]
    #
    # ret[ n: ]  = [ 5 , 9 , 9 , 13 , 12 ]
    ret[n:] = ret[n:] - ret[:-n]
    
    # arr = [ 3 , 5 , 9 , 9 , 13 , 12 ]/3
    # arr = [ 1 , 1.66 , 3 , 3 , 4.33 , 4 ]
    arr = ret[n - 1:] / n
    return arr