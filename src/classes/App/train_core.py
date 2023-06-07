import torch as tc
import torch.optim as top
import torch.nn as tnn

import sys
import os
from typing import *

rkpath = os.getcwd() + "/src/classes/Model"
sys.path.append( rkpath )
from state_net import get_dp_mlp

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
    
    def to_dict( self ):

        d = {}

        # learning rate
        d["lr"] = self.lr

        # loss function name
        d["loss_fun"] = str( self.loss_fn.__class__ )

        # opm name
        d[ "opm" ] = str( self.opm.__class__ )

        # neural net details
        d[ "num_layers" ] = self.net.num_layers
        d[ "layer_w" ] = self.net.layer_w

        return d

def make_core( **kwargs ) -> tr_core:

    #--------------------------------------------
    # initing the neural network
    num_layers = kwargs.get( "num_layers" , 5 )
    layer_w = kwargs.get( "layer_w" , 20 )
    net = get_dp_mlp( num_layers , layer_w)

    #---------------------------------------------
    # opm
    lr = kwargs.get( "lr" , 1e-3 )
    opm_type = kwargs.get( "opm" , top.Adam )
    opm = opm_type( net.parameters() , lr = lr )

    #--------------------------------------------
    # loss
    loss_type = kwargs.get( "loss_fn" , tnn.L1Loss )
    loss_fn = loss_type()

    return tr_core(
        net, opm , loss_fn , lr
    )