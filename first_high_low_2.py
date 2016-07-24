import numpy as np

def first_high(vec,target):
    for i in range(vec.shape[0]):
        if (vec[i] > target):
            return i
    return -1
    
def first_low(vec,target):
    for i in range(vec.shape[0]):
        if (vec[i] < target):
            return i
    return -1