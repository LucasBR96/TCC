import os
import sys
from typing import *

import itertools as itt
import time
import threading as thrd

import numpy as np
import pandas as pd

rkpath = os.getcwd() + "/src/classes/RungeKutta"
sys.path.append( rkpath )
from rk_machine import rk_machine , get_rk4_machine

ANGLES = np.linspace( -np.pi/2 , np.pi/2 , 20 )

MAX_H  = 2
MIN_H  = 1
H_TICKS = 10
LENS   = np.linspace( MIN_H , MAX_H , H_TICKS )
G = 9.8

H_STEP = 1e-3
R_STEP = 5*1e-2
MAX_TIME = 300

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

def init_rk( theta , omega , l ) -> rk_machine:

    X = np.array([ l , theta , omega ] )

    rpk = get_rk4_machine(
        X,
        H_STEP,
        get_dpend
    )

    return rpk

def simulate( rkp : rk_machine ) -> Tuple[ pd.DataFrame , float ]:

    total_time = 0
    register_list = []

    num_iter = int( MAX_TIME/H_STEP )
    rec_interval = int( R_STEP/H_STEP )
    for i in range( num_iter ):

        start = time.time()
        t , X = rkp()
        total_time += time.time() - start

        if not i%rec_interval:

            d = {}

            d["t"]       = f"{t:.3f}"
            d[ "l" ]     = f"{X[0]:.4f}"
            d[ "theta" ] = f"{X[1]:.4f}"
            d[ "omega" ] = f"{X[2]:.4f}"

            register_list.append( d )
    
    return pd.DataFrame( register_list ) , total_time

# def mec_energy( l , theta , omega ):

#     grav_energy = ( MAX_H - l*np.cos( theta ) )*G
#     k_energy    = .5*l*( omega**2 )

#     return k_energy + grav_energy

def record_simu( simu_data : pd.DataFrame , path : str ):
    simu_data.to_csv( path , index = False )

def main():

    # where to save each simulations
    path = "data/simulations/single_pendulum"

    # initial conditions
    combs = itt.product( ANGLES, ANGLES, LENS )

    # meta data
    meta_path = "data/meta/single_pendulum.csv"
    meta_lst  = []

    i = 0
    for theta , omega , l in combs:

        # no point into simulating an static pendulum
        if theta == omega == 0:
            continue
        simu_path = f"{path}/case_{i}.csv"
        i += 1
        
        # initializing the runge kutt machine
        rk = init_rk( theta , omega , l )

        # running the simulation
        data , duration = simulate( rk )
        record_simu( data , simu_path )

        # recording meta info
        meta_d = {
            "case" : i - 1,
            "theta": f"{theta:.4f}",
            "omega": f"{omega:.4f}",
            "l":     f"{l:.4f}",
            "duration": duration
        }

        if i == 500:
            break
        
        meta_lst.append( meta_d )
        print( meta_d.items() )
    
    # saving the meta data
    meta_data = pd.DataFrame( meta_lst )
    meta_data.to_csv( meta_path )

if __name__ == "__main__":
    main()
