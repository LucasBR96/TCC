import os
os.chdir( "/home/lucasfuzato/TCC/CODE" )

import numpy as np
import torch as tc
import torch.utils.data as tdt
import pandas as pd

from src.util.constantes import *

class meu_loader( tdt.Dataset ):

    def __init__( self , test_data = False , max_time = 1. ) -> None:
        super().__init__()

        # data_set with initial conditions, separated
        # filtered by if its a training or test set
        cond_df = pd.read_csv( COND_FILE )
        where = cond_df[ "test_set" ] == test_data
        self.cond_df = cond_df.loc[ where ]

        # loading the trajectory points
        tr_df = pd.read_csv( TR_FILE )

        # setting data points that belong to initial conditions
        # that will be used
        init_cond = set( self.cond_df.index )
        foo = lambda x : x in init_cond
        where_ic = tr_df[ "simu_id"].apply( foo )

        # setting data points with time values lower than max_time
        where_time = tr_df[ "t" ] <= max_time

        # filtering trajectory points
        where = where_ic & where_time
        self.tr_df = tr_df.loc[ where ]

        # number of points
        self.n = len( self.tr_df.index )

    def __len__( self ):
        return self.n
    
    def __getitem__( self, index ):

        tr_ser = self.tr_df.iloc[ index ]
        x = tr_ser[ "x" ]
        y = tr_ser[ "y" ]
        t = tr_ser[ "t" ]

        simu_id = tr_ser[ "simu_id" ]
        tr_cond = self.cond_df.loc[ simu_id ]
        theta_0 = tr_cond[ "theta_0" ]
        v_0  = tr_cond[ "v_0" ]

        X = tc.tensor( [ theta_0, v_0 , t ] , device = DEVICE ,dtype = tc.float64 )
        pos = tc.tensor( [ x , y ] , device = DEVICE , dtype = tc.float64)
        return X , pos 