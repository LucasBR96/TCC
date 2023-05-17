import os
import sys

rkpath = os.getcwd() + "/src/classes/Model"
sys.path.append( rkpath )
from state_net import state_mlp

import unittest
import torch as tc
import torch.nn as tnn

def test1():

    A = state_mlp( 4 , 4 , 4 , 10 , tnn.ReLU )
    t = tc.rand( ( 10 , 4 , 4 ) )

    print( t[ 0 ] )

    t_hat = A( t )
    print( t_hat[ 0 ] )

def test2():

    A = state_mlp( 4 , 4 , 4 , 10 , tnn.ReLU )
    t = tc.rand( ( 4 , 4 ) )

    print( t )

    t_hat = A( t )
    print( t_hat )

test2()