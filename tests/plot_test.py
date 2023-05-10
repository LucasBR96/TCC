import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import List

def correct_pos( theta_1 , theta_2 ):

    ref = 1.5*np.pi
    x1 = np.cos( ref + theta_1 )
    y1 = np.sin( ref + theta_1 )

    x2 = x1 + np.cos( ref + theta_2 )
    y2 = y1 + np.sin( ref + theta_2 )

    return x2 , y2 

def plot_simu( dfs : pd.DataFrame , df_name : str ):

    theta_1 : float = df[ "theta_1" ].to_numpy()
    theta_2 : float = df[ "theta_2" ].to_numpy()
    x , y = correct_pos( theta_1 , theta_2 )

    plt.xlim( -3 , 3 )
    plt.ylim( -3 , 3 )
    plt.plot( x , y , "r-" )
    plt.title( df_name )

    plt.show()

if __name__ == "__main__":

    df = pd.read_csv( "lol.csv" )
    theta_1 : float = df[ "theta_1" ].to_numpy()
    theta_2 : float = df[ "theta_2" ].to_numpy()
    x , y = correct_pos( theta_1 , theta_2 )
    plt.plot( x , y , "r-" )

    df = pd.read_csv( "lob.csv" )
    theta_1 : float = df[ "theta_1" ].to_numpy()
    theta_2 : float = df[ "theta_2" ].to_numpy()
    x , y = correct_pos( theta_1 , theta_2 )
    plt.plot( x , y , "b--" )

    plt.xlim( -3 , 3 )
    plt.ylim( -3 , 3 )
    plt.gca().set_aspect('equal')
    
    plt.show()
