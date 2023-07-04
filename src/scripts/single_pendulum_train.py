import sys
import os
import argparse

import torch as tc
import torch.nn as tnn
import torch.optim as top
import torch.utils.data as tdt

import pandas as pd

utilpath = os.getcwd() + "/src/classes"
sys.path.append( utilpath )
from App.train_core import tr_core
from Model.tsampler import get_single_sampler
from Model.state_net import get_pend_mlp

utilpath = os.getcwd() + "/src/util"
sys.path.append( utilpath )
from aux_fun import basic_k_fold

# Constants -------------------------------------------
DEVICE = "cuda" if tc.cuda.is_available() else "cpu"
NUM_SIMUS = 500

def make_core( **kwargs ) -> tr_core:

    #--------------------------------------------
    # initing the neural network
    num_layers = kwargs.get( "num_layers" , 5 )
    layer_w = kwargs.get( "layer_w" , 20 )
    net = get_pend_mlp( num_layers , layer_w ).to( DEVICE )

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

def train( core , test_simu , train_simu , batch_size = 500 , step = 0.5 , num_iters = 10**4 , eval_step = 100 , verbose = True ):

    #-------------------------------------------------------------------
    # data for testing and training, step is the time distance
    # between the X and X_prime
    test_data = get_single_sampler( test_simu , batch_size , step )
    train_data = get_single_sampler( train_simu , batch_size , step )

    result = pd.DataFrame( 
        columns = [ "tr_cost" , "ts_cost" ],
        index = list( range( 0 , num_iters , eval_step ) ) 
    )

    for i in range( num_iters + 1 ):

        #--------------------------------------------
        # fetching training data
        smm , ( tr_X , tr_X_prime ) = next( train_data )
        tr_X = tr_X.to( DEVICE )
        tr_X_prime = tr_X_prime[ : , 1: ].to( DEVICE ) # removing pendulum len from output

        #--------------------------------------------
        # fit the data, if necessary
        if i%eval_step:
            core.fit( tr_X , tr_X_prime )
            continue
        
        #-------------------------------------------
        # evaluating training data
        _ , tr_loss = core.eval( tr_X , tr_X_prime )

        #--------------------------------------------
        # fetching training data and evaluating
        _ , ( ts_X , ts_X_prime ) = next( test_data )
        ts_X = ts_X.to( DEVICE )
        ts_X_prime = ts_X_prime[ : , 1: ].to( DEVICE )
        X_arr , ts_loss = core.eval( ts_X , ts_X_prime )

        #-----------------------------------------------
        # printing the results
        if verbose:

            print( f"\niter {i} " + "-"*50 )

            print( f"\ntrain_loss: {tr_loss:.5f}" )
            print( f"test_loss: {ts_loss:.5f}\n" )

            eval_frame = pd.DataFrame( index = [ 'predicted' , 'real' ] , columns = [ 'theta' , 'omega' ] )
            eval_frame.loc[ 'predicted' ] = X_arr[ 0 ]
            eval_frame.loc[ 'real' ] = ts_X_prime[ 0 ].cpu().numpy()
            print( eval_frame )

        #----------------------------------------------------
        # updating the evolution frame
        result.loc[ i ] = [ tr_loss , ts_loss ]
    return result

if __name__ == "__main__":
    
    import matplotlib
    import matplotlib.pyplot as plt

    #data_sets
    test_data , train_data = next( basic_k_fold( NUM_SIMUS , 10 ) )
    colors = [ "red" , "green" , "purple" , "black" ]
    sizes  = [ .05 , .1 , .5 , 1. ]

    for i in range( 4 ):
        #training device

        print( f"step = {sizes[ i ]}")
        core = make_core( lr = 1e-3  , layer_w = 128 , num_layers = 6 )
        result = train( core , test_data , train_data , step = sizes[ i ], verbose = True, num_iters = 4*( 10**3 ) , batch_size = 5000 )

        plt.plot( result.index , result[ "ts_cost" ] , linestyle = "-" , color = colors[ i ] , label = f"t = {sizes[i]}" )
        plt.plot( result.index , result[ "tr_cost" ] , linestyle = "--" , lw = .3 , color = colors[ i ] )


    plt.yscale( 'log' )
    plt.legend()
    plt.show()