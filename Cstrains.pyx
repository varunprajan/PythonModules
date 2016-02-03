import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def CGenericFilter(double[:,::1] data,
                    double[:,::1] kernel,
                    int kerneloption):
    # kerneloption:
        # 1 - convolutes filter with data, and returns nan if any of data values are nan
        # 2 - convolutes filter with data, ignoring nan values (i.e. not added to sum). Returns nan if center data point is nan
        # 3 - like 2, but reweights (better for averaging, since weights should add up to 1)
    # edge handling - not sophisticated. produces nan if filter extends past edge, just for kerneloption 1. Other kerneloptions ignore edge data, which should be fine.
    cdef int m, n, mkernel, nkernel
    (mkernel, nkernel) = np.shape(kernel)
    (m,n) = np.shape(data)
    cdef:
        double[:,::1] datafiltered = np.zeros((m,n))
        int sizeleft, sizeright, sizelow, sizehigh
        int i, j, i2, j2, i2kernel, j2kernel
        int indexlow, indexhigh, indexleft, indexright
        int nanflag
        double res, kernelweight
        double datacurr, kernelcurr
        double mynan = np.nan
        
    cdef extern from "numpy/npy_math.h":
        bint npy_isnan(double x)

    sizeleft = -int(nkernel/2)
    sizeright = nkernel + sizeleft
    sizelow = -int(mkernel/2)
    sizehigh = mkernel + sizelow
    datafiltered = np.zeros((m,n))
    for i in range(m):
        indexlow = max(0,i+sizelow)
        indexhigh = min(m,i+sizehigh)
        for j in range(n):
            indexleft = max(0,j+sizeleft)
            indexright = min(n,j+sizeright)
            nanflag = 0
            if (i + sizelow < 0) or (i + sizehigh > m) or (j + sizeleft < 0) or (j + sizeright > n):
                nanflag = 1
            if npy_isnan(data[i,j]): # if center data point is bad
                datafiltered[i,j] = mynan
            else:
                res = 0.0
                kernelweight = 0.0
                for i2 in range(indexlow,indexhigh):
                    i2kernel = i2-i-sizelow
                    for j2 in range(indexleft,indexright):
                        j2kernel = j2-j-sizeleft
                        kernelcurr = kernel[i2kernel,j2kernel]
                        datacurr = data[i2,j2]
                        if npy_isnan(datacurr): # if data point is bad
                            nanflag = 1
                        else:
                            res += datacurr*kernelcurr
                            kernelweight += kernelcurr
                if kerneloption == 3: # if center data point is good, guarantees nonzero kernelweight
                    datafiltered[i,j] = res/kernelweight
                elif kerneloption == 2:
                    datafiltered[i,j] = res
                elif kerneloption == 1:
                    if nanflag == 1: # check if any of the filter values were nan
                        res = mynan
                    datafiltered[i,j] = res
    return datafiltered