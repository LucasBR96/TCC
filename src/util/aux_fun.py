import numpy as np
import torch as tc

from typing import *

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
