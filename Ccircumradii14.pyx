import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs

@cython.boundscheck(False)
@cython.wraparound(False)
def CGetCircumradii(double[:,::1] xy,
                    int[:,::1] triangles,
                    int n):
    cdef:
        unsigned int i, j, k
        double x1, y1, x2, y2, x3, y3
        double Ax, Ay, Bx, By, Cx, Cy, A, B, C
        double xc, yc
        double invareatw, third, areatw
        double[:,::1] props = np.zeros((n,2))

    third = 1.0/3.0
    for i in range(n):
        props[i,0] = i
        x1 = xy[triangles[i,0],0]
        y1 = xy[triangles[i,0],1]
        x2 = xy[triangles[i,1],0]
        y2 = xy[triangles[i,1],1]
        x3 = xy[triangles[i,2],0]
        y3 = xy[triangles[i,2],1]
        xc = (x1 + x2 + x3)*third
        yc = (y1 + y2 + y3)*third
        Ax = x3 - x2
        Ay = y3 - y2
        Bx = x1 - x3
        By = y1 - y3
        Cx = x2 - x1
        Cy = y2 - y1
        A = Ax*Ax + Ay*Ay
        B = Bx*Bx + By*By
        C = Cx*Cx + Cy*Cy
        areatw = fabs((x3-xc)*Cy + (x1-xc)*Ay + (x2-xc)*By)
        if areatw > 1.0e-10:
            props[i,1] = 0.5*sqrt(A*B*C)/areatw
    return props