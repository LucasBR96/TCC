import os
import sys
ipath = os.getcwd() + "/src/classes"
sys.path.append( ipath )

import itertools as itt
import time
import threading as thrd

import numpy as np
from rk_machine import rk_machine , get_rk4_machine

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

def simulate( rkp : rk_machine , max_time , path ):

    with open( path , "a" ) as f:
        f.write( "t,theta_1,omega_1,theta_2,omega_2")

        while True:

            t , X = rkp()
            
            s = f"\n{t:.3f}"
            s += f",{X[0,0]:.4f}" #theta_1
            s += f",{X[0,1]:.4f}" #omega_1
            s += f",{X[1,0]:.4f}" #theta_2
            s += f",{X[1,1]:.4f}" #omega_2
            # print( s )

            f.write( s )
            if t >= max_time:
                break

def init_omega2( theta1 , theta2 )    

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


if __name__ == "__main__":
    main()

    thrd.Thread()
    # path = "data/simulations/double_pendulum/train_data"
    # h_step = 5*( 1e-4 )
    # max_time = 120

    # h = 0.01
    # for i in range( 1 ):

    #     if i == 0:
    #         X = np.array([
    #             [ 1 , 0 ],
    #             [ 1 , 0 ],
    #         ]) 
    #     elif i == 1:
    #         X = np.array([
    #             [ 1 , 0 ],
    #             [ 1 + h , 0 ]
    #         ])
    #     elif i == 2:
    #         X = np.array([
    #             [ 1 , 0 ],
    #             [ 1 - h , 0 ]
    #         ])
    #     X = X*( np.pi/2 )

    #     rpk = rk2_machine(
    #         X,
    #         h_step,
    #         dpend_update,
    #         None
    #     )

    #     full_path = path + f"/train_{i}.csv"
    #     simulate( rpk , max_time , full_path )