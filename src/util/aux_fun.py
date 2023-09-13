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


# Double pendulum -------------------------------------------------------------------------
def get_omega1_dot( theta1 , omega1 , theta2 , omega2 ):

    num1 = -3*np.sin( theta1 )
    num2 = -np.sin( theta1 - 2*theta2 )
    num3 = -2*np.sin( theta1 - theta2 )*( omega2**2 + ( omega1**2 )*np.cos( theta1 - theta2 ) )

    den = 3 - np.cos( 2*theta1 - 2*theta2 )
    return ( num1 + num2 + num3 )/den

def get_omega2_dot( theta1 , omega1 , theta2 , omega2 ):

    num_1 = 2*np.sin( theta1 - theta2 )
    num_2 = 2*( omega1**2 ) + 2*np.cos( theta1 ) + ( omega2**2 )*np.cos( theta1 - theta2 )

    den = 3 - np.cos( 2*theta1 - 2*theta2 )
    return num_1*num_2/den

def dpend_update( X : np.ndarray ):

    '''

    Considering X like:

    X = [ theta_1 , theta_2 , omega_1 , omega_2 ]

    where theta is the angle and omega is the angualar velocity

    G = L1 = L2 = M1 = M2 = 1
    '''

    theta_1 = X[ 0 ]
    theta_2 = X[ 1 ]
    omega_1 = X[ 2 ]
    omega_2 = X[ 3 ]

    X_hat = np.zeros( 4 )
    X_hat[ 0 ] = omega_1
    X_hat[ 1 ] = omega_2 
    X_hat[ 2 ] = get_omega1_dot( theta_1 , omega_1 , theta_2 , omega_2 )
    X_hat[ 3 ] = get_omega2_dot( theta_1 , omega_1 , theta_2 , omega_2 )

    return X_hat


# -------------------------------------------------------------------
# coordinate manipulation
def from_polar( df : pd.DataFrame ):

    theta_1 = df[ "theta_1" ].to_numpy()
    theta_2 = df[ "theta_2" ].to_numpy()

    # ref = 1.5*np.pi
    x1 = np.sin( theta_1 )
    y1 = -np.cos( theta_1 )

    x2 = x1 + np.sin( theta_2 )
    y2 = y1 - np.cos( theta_2 )

    cart_df = pd.DataFrame()
    cart_df[ "t" ] = df[ "t" ]
    cart_df[ "x1" ] = x1
    cart_df[ "y1" ] = y1
    cart_df[ "x2" ] = x2
    cart_df[ "y2" ] = y2

    return cart_df

def sample_coordinates( cart_df : pd.DataFrame, start = 0 , end = 10 ):

    start = max( 0 , start )
    end   = max( 0 , end )
    if end <= start :
        raise ValueError( "only one value between above and below can be equal to zero")

    below = cart_df[ "t" ] < end
    above = cart_df[ "t" ] >= start
    where = above & below

    return cart_df.loc[ where ]

# def plot_path( ax : plt.Axes , cart_df : pd.DataFrame, moment , before : float = 10 , after = 10 ):
#     pass

# -----------------------------------------------------------------
# sampling data
def make_sampler( pos_frame , train_ratio = .5 , batch_size = 50 ):

    t_max = pos_frame[ "t" ].max()
    ceil = train_ratio*t_max

    where = pos_frame[ "t" ] <= ceil
    train_df = pos_frame.loc[ where ]
    train_loader = tdt.DataLoader(
        meu_loader( train_df ),
        batch_size,
        shuffle = True
    )

    test_df = pos_frame.loc[ ~where ]
    test_loader = tdt.DataLoader(
        meu_loader( test_df ),
        batch_size,
        shuffle = True
    )

    return train_loader , test_loader

