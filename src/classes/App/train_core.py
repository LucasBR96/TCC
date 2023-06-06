import torch as tc
import torch.optim as top
import torch.nn as tnn

import sys
import os
from typing import *

class tr_core:

    def __init__( self , net : tnn.Module , opm : top.Optimizer , loss_fn : tnn.Module , lr : float ):

        self.net = net
        self.opm = opm
        self.loss_fn = loss_fn
        self.lr = lr
    
    def __call__( self , X , X_prime ) -> Tuple[ tc.Tensor , tc.Tensor]:

        X_hat = self.net( X )
        loss_val = self.loss_fn( X_hat , X_prime )

        return X_hat , loss_val
    
    def fit ( self , X , X_prime ):

        self.opm.zero_grad()

        _ , loss_val = self( X , X_prime )

        loss_val.backward()

        self.opm.step()
    
    def eval( self , X , X_prime ):

        with tc.no_grad():
            
            X_hat , loss_val = self( X , X_prime )
            X_arr = X_hat.numpy()
            loss_item = loss_val.item()
        
        return X_arr , loss_item