import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import mymath as Mmath

def nsqNeigh(xy,rneigh,nmaxneigh):
    dist = genDistanceMatrix(xy)
    n = np.shape(xy)[0]
    neigh = np.zeros((n,nmaxneigh),dtype=int)
    for i, _ in enumerate(xy):
        count = 0
        for j, _ in enumerate(xy):
            if (i!=j) and (dist[i,j] < rneigh):
                neigh[i,count] = j
                count = count + 1
    return neigh

def genDistanceMatrix(xy):
    n = np.shape(xy)[0]
    dist = np.zeros((n,n))
    for i, acurr in enumerate(xy):
        for j, bcurr in enumerate(xy):
            dist[i,j] = Mmath.get_dist(acurr,bcurr)
    return dist

def genSimpleArray(xsize,ysize,r0=1):
    xmin = -xsize/2
    xmax = xsize/2
    ymin = -ysize/2
    ymax = ysize/2
    a1 = np.array([r0,0])
    a2 = np.array([r0/2,r0*np.sqrt(3)/2])
    n2 = int(round(1.2*ysize/a2[0]))
    n1 = n2
    orig = np.array([xmin,ymax])-n2*a2
    xvec, yvec = simpleArray(orig,a1,a2,n1,n2)
    xvec, yvec = enforceArrayBounds(xvec,yvec,[xmin,xmax,ymin,ymax])
    xy = np.column_stack((xvec,yvec))
    return xy

def enforceArrayBounds(xvec,yvec,bounds):
    xmin, xmax, ymin, ymax = bounds
    indexx = (xmin < xvec) & (xvec < xmax)
    indexy = (ymin < yvec) & (yvec < ymax)
    index = indexx & indexy
    return xvec[index], yvec[index]

def simpleArray(orig,a1,a2,n1,n2,tol=1.e-6):
    # create lattice
    xmat = np.empty((n1,n2))
    ymat = np.empty((n1,n2))
    iter = itertools.product(range(n1),range(n2))
    for i, j in iter:
        xmat[i,j], ymat[i,j] = orig + a1*i + a2*j
    n = xmat.size
    xvec = np.reshape(xmat,n)
    yvec = np.reshape(ymat,n)
    return xvec, yvec
    
        
    