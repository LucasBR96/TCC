import matplotlib.animation as anm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import sys
from collections import deque
from random import randint

utilpath = os.getcwd() + "/src/util"
sys.path.append( utilpath )
from aux_fun import interpolate

FPS = 48

def interpolate_FPS( df : pd.DataFrame , rate = FPS ):

    result = list()

    select = df[ [ "t" , "theta_1" , "theta_2" ] ]
    next_t = 0
    n = len( select )
    for i in range( n - 1 ):

        row : pd.Series = select.iloc[ i ]
        nxt_row : pd.Series = select.iloc[ i + 1 ]
        if not( row[ "t" ] <= next_t <= nxt_row[ "t" ] ):
            continue
        
        r_dict = {}
        r_dict[ "t" ] = next_t
        rt , nrt = row[ "t" ] , nxt_row[ "t" ]

        r_dict[ "theta_1" ] = interpolate(
            row[ "theta_1" ], rt,
            nxt_row[ "theta_1" ], nrt,
            next_t
        )

        r_dict[ "theta_2" ] = interpolate(
            row[ "theta_2" ], rt,
            nxt_row[ "theta_2" ], nrt,
            next_t
        )

        result.append( r_dict )
        next_t += 1/rate
    
    return pd.DataFrame( result )

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

class dp_painter:

    def __init__( self , cart_df , tail_color , tail_size = 50 ):

        self.cart_df : pd.DataFrame = cart_df
        self.tail_color = tail_color
        self.tail_size = tail_size

    def _get_ppos( self , frame_idx ):

        row = self.cart_df.iloc[ frame_idx ]
        pos1 = row[ [ "x1" , "y1" ] ].to_numpy()
        pos2 = row[ [ "x2" , "y2" ] ].to_numpy()

        return pos1 , pos2
    
    def _get_tail_pos( self , frame_idx ):

        bottom = max( 0 , frame_idx - self.tail_size )
        row = self.cart_df.iloc[ bottom : frame_idx ]
        tail_pos = row[ [ "x2" , "y2" ] ].to_numpy()
        return tail_pos

    def draw( self , ax : plt.Axes , frame_idx ):

        #-----------------------------------------
        # Drawing the pendulum
        pos1 , pos2 = self._get_ppos( frame_idx )
        x1 , y1 = pos1
        x2 , y2 = pos2

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
        if not frame_idx:
            return
        tail = self._get_tail_pos( frame_idx )
        ax.plot(
            tail[ : , 0 ],
            tail[ : , 1 ],
            color = self.tail_color
        )

def print_simulation( sim_list ):

    colors = [ "red" , "green" , "blue" , "purple" , "cyan" , "orange" ] 

    dps = []
    for sim_num in sim_list:
        path = f"data/simulations/double_pendulum/case_{sim_num}.csv"
        df = pd.read_csv( path )
        idf = interpolate_FPS( df )
        cdf = from_polar( idf )

        color = colors[ sim_num%( len( colors ) ) ]
        dp = dp_painter( cdf , color , 50 )
        dps.append( dp )

    fig , ax = plt.subplots()

    def draw( i ):

        ax.clear()
        for dp in dps:
            dp.draw( ax , i )

        ax.set_xlim( -2.1 , 2.1 )
        ax.set_ylim( -2.1 , 2.1 )
        ax.set_aspect( "equal" )
    
    anim = anm.FuncAnimation( fig , draw , frames = len( cdf ) , interval = 50/FPS )
    plt.show()

if __name__ == "__main__":

    argl = sys.argv[ 1: ]
    if not( argl ):
        sim_list = [ 0 , 1 , 2 ]
    else:
        sim_list = [ int( x ) for x in argl ]
    print_simulation( sim_list )