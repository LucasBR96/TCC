import numpy as np
import torch as tc
import torch.utils.data as tdt
import pandas as pd

class meu_loader( tdt.Dataset ):

    def __init__( self , dpend_df : pd.DataFrame ) -> None:
        super().__init__()

        self.n = len( dpend_df )
        self.t = dpend_df[ "t" ].to_numpy()
        self.thetas = dpend_df[ [ "theta_1" , "theta_2" ] ].to_numpy()

    def __len__( self ):
        return self.n
    
    def __getitem__(self, index):

        t = self.t[ index ]
        thetas = tc.from_numpy( self.thetas[ index ] )
        return t , thetas
    