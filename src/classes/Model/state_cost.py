import torch as tc
import torch.nn as tnn
import torch.nn.functional as tfun

class state_distance( tnn.Module ):

    def __init__( self ):
        
        super().__init__()
    
    def forward( self , S : tc.Tensor , S_hat : tc.Tensor ):
        
        single_sample : bool = ( S.ndim == 2 )
        if single_sample:
            S = tc.unsqueeze( S , 0 )
            S_hat = tc.unsqueeze( S_hat , 0 )

        r = ( ( S - S_hat )**2 ).mean( dim = 2 )
        w = tfun.softmax( r , dim = 1 )
        c = ( r*w ).sum( dim = 1 )
        return c.mean()
        
