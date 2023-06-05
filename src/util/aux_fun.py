import numpy as np
import torch as tc
import torch.utils.data as tdt

from typing import *

def basic_k_fold( n : int , k : int ):

    arr = np.arange( n )
    for i in range( k ):
        to_test = ( arr%k == i )
        
        test_fold = arr[ to_test ]
        train_fold = arr[ ~to_test ]
        yield test_fold , train_fold

def interpolate( x1 , t1 , x2 , t2 , t_mid ):

    alpha = ( t2 - t_mid )/( t2 - t1 )
    return x1*alpha + x2*( 1 - alpha )

def edge_search( arr : np.ndarray | tc.Tensor , x : Any ):
    
    '''
    binary search, if x is not in arr then returns the
    index of the biggest number in arr
    '''

    low = 0
    high = len(arr) - 1
    result = -1

    while low <= high:
        mid = (low + high) // 2

        if arr[mid] == x:
            return mid

        if arr[mid] < x:
            result = mid
            low = mid + 1
        else:
            high = mid - 1

    return result

def data_generate( sim_list : List[ int ], load_fun , batch_size = 500  ):

    d_loader : Iterator = iter([])

    i = 0
    while True:

        nxt_batch = next( d_loader , None )

        if not ( nxt_batch is None):
            # yield i , nxt_batch
            yield nxt_batch
            continue
        
        if i < len( sim_list ):

            d_loader = iter(
                tdt.DataLoader(
                    load_fun( sim_list[ i ] ),
                    batch_size = batch_size,
                    shuffle = True
                )
            )
            i += 1
            continue
        
        break
