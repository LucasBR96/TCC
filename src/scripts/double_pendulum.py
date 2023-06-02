import os
import sys
ipath = os.getcwd() + "/src/classes"
sys.path.append( ipath )

# ----------------------------------------------------------------
# Standard library
import itertools as itt
import time
import threading as thrd
import time as t
from typing import *

#---------------------------------------------------------
# 3rd party mods
import numpy as np
import pandas as pd

#----------------------------------------------------
# From this project
from rk_machine import rk_machine , get_rk4_machine

#------------------------------------------------------
# CONSTANTS
THETA_DIVS = 10
THETA_RANGE = np.linspace( -np.pi/2 , np.pi/2 , THETA_DIVS )

H_STEP = 1e-2
R_STEP = 5*1e-2
MAX_TIME = 300

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

def init_omega2( theta1 , theta2 ):

    """
    All simulations must begin with the same
    mechanical energy, wicht is the one when
    both arms of the pendulum are paralel to the
    ground and with angular velocity equal to 
    zero

    for that, we will consider omega one equal
    to zero and calculate omega2 based on theta1
    and theta2
    """

    omega2 = np.sqrt(
        2*np.cos( theta1 ) + np.cos( theta2 )
    )

    #------------------------------------------------
    # If the starts to the right of the y axis, 
    # it should start swinging clockwise
    if( np.sin( theta1 ) + np.sin( theta2 ) >= 0 ):
        omega2 *= -1
    return omega2

def dpend_update( X : np.ndarray ):

    '''

    Considering X like:

    X = [
        [ theta_1 , omega_1 ],
        [ theta_2 , omega_2 ]
    ]

    where theta is the angle and omega is the angualar velocity
    '''

    theta_1 = X[ 0 , 0 ]
    omega_1 = X[ 0 , 1 ]
    theta_2 = X[ 1 , 0 ]
    omega_2 = X[ 1 , 1 ]

    X_hat = np.zeros( ( 2 , 2 ) )
    X_hat[ 0 , 0 ] = omega_1
    X_hat[ 0 , 1 ] = get_omega1_dot( theta_1 , omega_1 , theta_2 , omega_2 )
    X_hat[ 1 , 0 ] = omega_2
    X_hat[ 1 , 1 ] = get_omega2_dot( theta_1 , omega_1 , theta_2 , omega_2 )

    return X_hat

def init_rk( simu_id : int ) -> rk_machine:

    #------------------------------------------------
    # Getting the initial state
    idx_1 = simu_id%THETA_DIVS
    theta_1 = THETA_RANGE[ idx_1 ]

    idx_2 = simu_id//THETA_DIVS
    theta_2 = THETA_RANGE[ idx_2 ]

    omega_2 = init_omega2( theta_1 , theta_2)

    X = np.array([
        [ theta_1 , 0 ],
        [ theta_2 , omega_2]
    ])

    rpk = get_rk4_machine(
        X,
        H_STEP,
        dpend_update
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

            d["t"] = f"{t:.3f}"
            d[ "theta_1" ] = f"{X[0,0]:.4f}"
            d[ "omega_1" ] = f"{X[0,1]:.4f}"
            d[ "theta_2" ] = f"{X[1,0]:.4f}"
            d[ "omega_2" ] = f"{X[1,1]:.4f}"

            register_list.append( d )
    
    return pd.DataFrame( register_list ) , total_time

def record_simu( simu_data : pd.DataFrame , path : str ):
    simu_data.to_csv( path , index = False )

def record_meta( simu_data : pd.DataFrame , simu_id : int , exec_time : float ):
    pass

def main():

    path = "data/simulations/double_pendulum"
    h_step = 1e-2
    max_time = 120

    angles = np.linspace( 0 , np.pi/2 , 11 )
    combs = itt.product( angles , repeat = 4 )
    #------------------------------------------------------
    # Throw away, as value equals 0 0 0 0. returning a 
    # static pendulum
    next( combs )

    i , total = 0 , 0 
    for tup in combs:

        theta_1 , omega_1 , theta_2 , omega_2 = tup
        X = np.array([
            [ theta_1 , omega_1 ],
            [ theta_2 , omega_2 ]
        ])
        
        rpk = get_rk4_machine(
            X,
            h_step,
            dpend_update
        )

        end = f"/case_{i}.csv"
        full_path = path + end
        i += 1

        t = time.time()
        simulate( rpk , max_time , full_path)
        dt = time.time() - t
        total += dt

        print( end + "--------------------------" )
        print( f"theta_1 = {theta_1:.2f} rad")
        print( f"theta_2 = {theta_2:.2f} rad")
        print( f"omega_1 = {omega_1:.2f} rad/s")
        print( f"omega_2 = {omega_2:.2f} rad/s")

        print( f"\ntempo de exec = {1000*dt:.2f} ms\n")
    
    sec , ms = int( total ) , ( 1000*total )%1000
    minium , sec = sec//60 , sec%60 
    print( f"\ntempo total = {minium}:{sec}:{ms} mins")


# if __name__ == "__main__":
#     # main()

#     def print_even( ):
#         for i in range( 10 ):
#             print( 2*i  )
    
#     def print_odd( ):
#         for i in range( 10 ):
#             print( 2*i + 1 )
    
#     t1 = thrd.Thread( target = print_even )
#     t2 = thrd.Thread( target = print_odd )
    
#     t1.start()
#     t2.start()