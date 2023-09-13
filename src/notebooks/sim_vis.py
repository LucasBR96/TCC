import os
os.chdir( "/home/lucasfuzato/TCC/CODE" )
# print( os.getcwd() )

import matplotlib.animation as anm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np


################################################################

# FOR SOME REASON ANIMANTIONS DON'T WORK IN JUPYTER NOTEBOOKS
#

###############################################################3

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

def draw_simulation( simu_df , tail_size = 50 , max_time = 20 ):

    xy_df = from_polar( simu_df )
    fig , ax = plt.subplots( figsize = ( 8 , 8 ) )

    def draw( i ):
        
        ax.clear()
        ax.set_xlim( -2.1 , 2.1 )
        ax.set_ylim( -2.1 , 1.5 )
        ax.set_aspect( "equal" )

        # getting the correct positions
        ser = xy_df.iloc[ i ]
        x1 = ser[ "x1" ] 
        y1 = ser[ "y1" ] 
        x2 = ser[ "x2" ] 
        y2 = ser[ "y2" ]

        #-----------------------------------------
        # Drawing the pendulum

        #first bar
        ax.plot(
            [ 0 , x1 ],
            [ 0 , y1 ],
            "--k"
        )

        #secondond bar
        ax.plot(
            [ x1 , x2 ],
            [ y1 , y2 ],
            "--k"
        )

        #tip
        ax.plot(
            x2,
            y2,
            "*k"
        )

        #----------------------------------------------------------
        # Drawing the tail
        if not i:
            return
    
        tail_end = max( 0 , i - tail_size )
        tail = xy_df.iloc[ tail_end : i ]
        ax.plot(
            tail[ "x2" ].to_numpy(),
            tail[ "y2" ].to_numpy(),
            "-r"                   # red solid line as tail
        )
    
    n = len( xy_df.index )
    anim = anm.FuncAnimation( fig , draw , frames = n  , interval = 25 )
    plt.show( )

if __name__ == "__main__":
    simu_df = pd.read_csv( "data/pend_sim.csv" ).iloc[ : : 5 ]
    draw_simulation( simu_df )