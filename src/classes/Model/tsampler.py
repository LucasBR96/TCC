import torch as tc
import torch.utils.data as tdt

import pandas as pd
import numpy as np

from typing import *

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
        
        row : pd.Series = self.df.iloc[ index ]
        S : tc.Tensor = self.conversion( row )

        t_prime = self.step + row[ "t"]
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
            t_2 = nxt_row_1[ "t" ]

            S_prime = interpolate( S_prime_1 , t_1 , S_prime_2 , t_2 )
        
        return S , S_prime
