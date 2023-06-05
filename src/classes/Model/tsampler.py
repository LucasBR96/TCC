import torch as tc
import torch.utils.data as tdt
import pandas as pd
import numpy as np

from typing import *
import os
import sys

utilpath = os.getcwd() + "/src/util"
sys.path.append( utilpath )
from aux_fun import interpolate , edge_search

class TimeStateDset( tdt.Dataset ):

    def __init__( self , df : pd.DataFrame , step : float , conversion : Callable ) -> None:

        super().__init__()

        self.df : pd.DataFrame = df
        self.step : float = step
        self.conversion : Callable = conversion

        #---------------------------------------
        # timepoints of each simulation
        self._time_poins : np.ndarray = df[ "t" ].to_numpy()

        #------------------------------------------
        # Defining the lenght of the data_set
        tmax : float = max( self._time_poins ) - self.step
        self.n = edge_search( self._time_poins , tmax )
    
    def __len__( self ):
        return self.n
    
    def __getitem__(self, index) -> Tuple[ tc.Tensor , tc.Tensor ]:
        
        #----------------------------------------------------
        # finding the base state
        row : pd.Series = self.df.iloc[ index ]
        S : tc.Tensor = self.conversion( row )

        t_prime = self.step + row[ "t" ]
        next_idx : int = edge_search( self._time_poins , t_prime )

        if self._time_poins[ next_idx ] == t_prime:
            nxt_row : pd.Series = self.df.iloc[ next_idx ]
            S_prime = self.conversion( nxt_row )

        else:
            nxt_row_1 : pd.Series = self.df.iloc[ next_idx ]
            S_prime_1 = self.conversion( nxt_row_1 )
            t_1 = nxt_row_1[ "t" ]

            nxt_row_2 : pd.Series = self.df.iloc[ next_idx + 1]
            S_prime_2 = self.conversion( nxt_row_2 )
            t_2 = nxt_row_2[ "t" ]

            S_prime = interpolate( S_prime_1 , t_1 , S_prime_2 , t_2 , t_prime )
        
        return S , S_prime

def get_dp_sampler( simmu_id : int , step : float = 0.5 ):

    def conversion( row : pd.DataFrame ):

        S = tc.zeros( ( 2 , 2 ) )
        S[ 0 , 0 ] = row[ "theta_1" ]
        S[ 0 , 1 ] = row[ "omega_1" ]
        S[ 1 , 0 ] = row[ "theta_2" ]
        S[ 1 , 1 ] = row[ "omega_2" ]

        return S
    
    path = f"data/simulations/double_pendulum/case_{simmu_id}.csv"
    df = pd.read_csv( path )
    
    return TimeStateDset( df , step , conversion )

if __name__ == "__main__":

    g = get_dp_sampler( 71 )
    print( g[ 15 ] )