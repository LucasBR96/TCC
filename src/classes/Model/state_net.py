import torch as tc
import torch.nn as tnn
import torch.nn.functional as tfun

class state_mlp( tnn.Module ):

    def __init__( self , nrow : int , ncol : int , num_layers : int , layer_w : int , act : type  ):

        super().__init__()

        self.nrow = nrow
        self.ncol = ncol
        in_size = nrow*ncol

        seq : tnn.Sequential = tnn.Sequential()
        seq.append( tnn.Linear( in_size , layer_w ) )
        seq.append( act() )

        for _ in range( num_layers ):
            seq.append( tnn.Linear( layer_w , layer_w ) )
            seq.append( act() )
        
        seq.append( tnn.Linear( layer_w , in_size ) )
        self.inner_net = seq
    
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