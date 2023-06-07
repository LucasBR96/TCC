from train_core import tr_core , make_core
from typing import *

import sys
import os

modelpath = os.getcwd() + "/src/Model"
sys.path.append( modelpath )
from tsampler import get_dp_sampler

class train_app:

    def __init__( self , _tr_core : tr_core , batch_size = 500 , num_iter = 10000 , eval_interval = 50 , timestep = .1 ):

        # does the bulk of the training
        self._core : tr_core = _tr_core

        self.batch_size = batch_size
        self.num_iter = num_iter
        self.eval_interval = eval_interval
        self.timestep = timestep
    
    def run( self , training_set , test_set ):

        train_data = get_dp_sampler( training_set , self.batch_size , self.timestep )
        test_data = get_dp_sampler( test_set , self.batch_size , self.timestep )

        i = 0
        while i < self.num_iter:
    
    def to_dict( self ):

        d : Dict = dict()

        # core information
        d.update( self._core.to_dict() )

        return d
    
    def __str__(self) -> str:
        
        d = self.to_dict()
        s = ""
        for k , v in d.items():
            s += f"{k} : {v}\n"
        
        return s

def make_app( **kwargs ):

    # making core 
    core = make_core( **kwargs )
    return train_app( core )

print( make_app( lr = 1e-2 ) )

