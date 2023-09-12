import torch as tc
import torch.nn as tnn
import torch.nn.functional as tfun

from typing import *

class diamond_mlp( tnn.Module ):

    def __init__( self , num_layers : int , layer_w : int , in_size : int , act : type , out_size = None):

        super().__init__( )

        # total amount of layers excluding output and input
        self.num_layers = num_layers

        # width of hidden layers 
        self.layer_w = layer_w

        # width of input layer
        self.in_size = in_size

        # width of output layer
        if out_size is None:
            out_size = in_size
        self.out_side = out_size
        
        # actual network
        self.seq : tnn.Sequential = tnn.Sequential()
        self.seq.append( tnn.Linear( in_size , layer_w ) )
        self.seq.append( act() )
        for _ in range( num_layers ):
            self.seq.append( tnn.Linear( layer_w , layer_w ) )
            self.seq.append( act() )
        self.seq.append( tnn.Linear( layer_w , out_size ) )
    
    def forward( self , X ):

        return self.seq( X )

class state_mlp( tnn.Module ):

    def __init__( self , nrow : int , ncol : int , num_layers : int , layer_w : int , act : type  ):

        super().__init__()

        self.num_layers = num_layers
        self.layer_w = layer_w
        self.nrow = nrow
        self.ncol = ncol

        out_size = nrow*ncol
        self.inner_net = diamond_mlp( num_layers , layer_w , out_size , act )
    
    def forward( self , S : tc.Tensor ):

        single_sample : bool = ( S.ndim == 2 )
        if single_sample:
            S = tc.unsqueeze( S , 0 )
        
        S_flat : tc.Tensor = tc.flatten( S , start_dim = 1 )
        Z : tc.Tensor = self.inner_net( S_flat )

        if not single_sample:
            S_hat = tc.reshape( Z , ( len( S ) , self.nrow , self.ncol ) )
        else:
            S_hat = tc.reshape( Z , ( self.nrow , self.ncol ) )
        return S_hat

def get_dp_mlp( num_layers : int , layer_w : int  ) -> state_mlp:
    return diamond_mlp( num_layers , layer_w , 4 , tnn.ReLU )

def get_pend_mlp( num_layers : int , layer_w : int  ) -> state_mlp:
    return diamond_mlp( num_layers , layer_w , 3 , tnn.ReLU, 2 )

def get_reg_mlp( shape : int , inner_fun = tnn.ReLU ) -> state_mlp:
    
    net = tnn.Sequential()

    i , n = 1 , len( shape )
    while i < n:

        s0 = shape[ i - 1 ]
        s1 = shape[ i ]
        net.append(
            tnn.Linear( s0 , s1 )
        )

        if i < n - 1:
            net.append( inner_fun() )
        
        i += 1
    
    return net