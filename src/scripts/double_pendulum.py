import os
import sys
ipath = os.getcwd() + "/src/classes"
sys.path.append( ipath )

import numpy as np
from rk_machine import rk2_machine

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

def simulate( rkp : rk2_machine , max_time , path ):

    with open( path , "a" ) as f:
        f.write( "t,theta_1,omega_1,theta_2,omega_2")

        while True:

            t , X = rkp()
            
            s = f"\n{t}"
            s += f",{X[0,0]}" #theta_1
            s += f",{X[0,1]}" #omega_1
            s += f",{X[1,0]}" #theta_2
            s += f",{X[1,1]}" #omega_2
            # print( s )

            f.write( s )
            if t >= max_time:
                break
    

def main():

    path = "data/simulations/double_pendulum"
    h_step = 5*( 1e-4 )
    max_time = 120

    Num_train = 90
    Num_test  = 10

    h = 0.01
    for i in range( Num_train + Num_test):

        if i == 0:
            X = np.array([
                [ 1 , 0 ],
                [ 1 , 0 ],
            ]) 
        elif i == 1:
            X = np.array([
                [ 1 , 0 ],
                [ 1 + h , 0 ]
            ])
        elif i == 2:
            X = np.array([
                [ 1 , 0 ],
                [ 1 - h , 0 ]
            ])
        X = X*( np.pi/2 )
        
        rpk = rk2_machine(
            X,
            h_step,
            dpend_update,
            None
        )

        if i < Num_train:
            end = f"/train_data/train_{i}.csv"
        else:
            end = f"/test_data/test_{ i - Num_train }.csv"
        full_path = path + end

        simulate( rpk , max_time , full_path)

if __name__ == "__main__":
    main()

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