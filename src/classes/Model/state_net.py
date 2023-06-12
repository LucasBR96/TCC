import torch as tc
import torch.nn as tnn
import torch.nn.functional as tfun


class diamond_mlp( tnn.Module ):

    def __init__( self , num_layers : int , layer_w : int , out_size : int , act : type ):

        super().__init__( )

        # total amount of layers excluding output and input
        self.num_layers = num_layers

        # width of hidden layers 
        self.layer_w = layer_w

        # width of both output and input layers
        self.out_side = out_size
        
        # actual network
        self.seq : tnn.Sequential = tnn.Sequential()
        self.seq.append( tnn.Linear( out_size , layer_w ) )
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
    return diamond_mlp( num_layers , layer_w , 2 , tnn.ReLU )