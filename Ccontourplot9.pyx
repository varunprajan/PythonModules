import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs
from cython cimport bool

@cython.boundscheck(False)
@cython.wraparound(False)
def CWalkAndMarchAll(double[:,::1] xy,
                     double[:] z,
                     int[:,::1] triangles,
                     int[:,::1] neighbors,
                     double[:,::1] xmat,
                     double[:,::1] ymat,
                     double radius):
    
    cdef unsigned int m,n
    (m,n) = np.shape(xmat)
    
    cdef:
        double x1,y1,z1,x2,y2,z2,x3,y3,z3
        double[:,::1] zmat = np.zeros((m,n))
        double mynan = np.nan
        unsigned int k1, k2, k3
        unsigned int ansprev = 0
        int trianswer, vertexcurr
        int rowansprev = -1
        double xpoint, ypoint

    for k1 in range(m):
        if rowansprev != -1:
            ansprev = rowansprev
        rowansprev = -1
        for k2 in range(n):
            xpoint = xmat[k1,k2]
            ypoint = ymat[k1,k2]
            trianswer = CWalkAndMarch(xy,triangles,neighbors,xpoint,ypoint,ansprev)
            if trianswer < 0:
                zmat[k1,k2] = mynan
            else:
                vertexcurr = triangles[trianswer,0]
                x1 = xy[vertexcurr,0]
                y1 = xy[vertexcurr,1]
                z1 = z[vertexcurr]
                vertexcurr = triangles[trianswer,1]
                x2 = xy[vertexcurr,0]
                y2 = xy[vertexcurr,1]
                z2 = z[vertexcurr]
                vertexcurr = triangles[trianswer,2]
                x3 = xy[vertexcurr,0]
                y3 = xy[vertexcurr,1]
                z3 = z[vertexcurr]
                zmat[k1,k2] = CInterp(x1,x2,x3,y1,y2,y3,z1,z2,z3,xpoint,ypoint,radius)
                ansprev = trianswer
                if rowansprev == -1:
                    rowansprev = trianswer
    return zmat

@cython.boundscheck(False)
@cython.wraparound(False)    
cdef CInterp(double x1,
             double x2,
             double x3,
             double y1,
             double y2,
             double y3,
             double z1,
             double z2,
             double z3,
             double xpoint,
             double ypoint,
             double radius):
    cdef:
        double side1x, side1y, side2x, side2y, side3x, side3y
        double side1, side2, side3
        double invareatw, circumradius
        double A, B, C
        
    side3x = x2 - x1
    side2x = x1 - x3
    side1x = x3 - x2
    side3y = y2 - y1
    side2y = y1 - y3
    side1y = y3 - y2
    side3 = side3x*side3x + side3y*side3y
    side2 = side2x*side2x + side2y*side2y
    side1 = side1x*side1x + side1y*side1y
    invareatw = 1.0/(x3*side3y + x1*side1y + x2*side2y)
    circumradius = 0.5*fabs(invareatw)*sqrt(side1*side2*side3)
    if circumradius > radius:
        return np.nan
    else:
        A = side1y*z1 + side2y*z2 + side3y*z3
        B = side1x*z1 + side2x*z2 + side3x*z3
        C = (x3*y2 - x2*y3)*z1 + (x1*y3 - x3*y1)*z2 + (x2*y1 - x1*y2)*z3
        return (A*xpoint - B*ypoint + C)*invareatw

@cython.boundscheck(False)
@cython.wraparound(False)        
cdef CWalkAndMarch(double [:,::1] xy,
                   int [:,::1] triangles,
                   int [:,::1] neighbors,
                   double xpoint,
                   double ypoint,
                   int guess):
    
    cdef:
        double pt0x,pt0y,pt1x,pt1y,pt2x,pt2y
        int res
        unsigned int count
        unsigned int badflip, proceed
        
    for count in range(1000):
        pt0x = xy[triangles[guess,0],0]
        pt0y = xy[triangles[guess,0],1]
        pt1x = xy[triangles[guess,1],0]
        pt1y = xy[triangles[guess,1],1]
        pt2x = xy[triangles[guess,2],0]
        pt2y = xy[triangles[guess,2],1]
        badflip = 0
        proceed = 1
        if proceed == 1:
            if SameSide(pt2x,pt2y,xpoint,ypoint,pt0x,pt0y,pt1x,pt1y):
                proceed = 1
            else:
                res = neighbors[guess,2]
                if res < 0:
                    proceed = 1
                    badflip = 1
                else:
                    proceed = 0
                    guess = res
        if proceed == 1:
            if SameSide(pt0x,pt0y,xpoint,ypoint,pt1x,pt1y,pt2x,pt2y):
                proceed = 1
            else:
                res = neighbors[guess,0]
                if res < 0:
                    proceed = 1
                    badflip = 1
                else:
                    proceed = 0
                    guess = res
        if proceed == 1:
            if SameSide(pt1x,pt1y,xpoint,ypoint,pt2x,pt2y,pt0x,pt0y):
                proceed = 1
            else:
                res = neighbors[guess,1]
                if res < 0:
                    proceed = 1
                    badflip = 1
                else:
                    proceed = 0
                    guess = res
        if proceed == 1:
            if badflip == 1:
                return -1
            else:
                return guess
                        
@cython.boundscheck(False)
@cython.wraparound(False)
cdef SameSide(double pt1x,
              double pt1y,
              double pt2x,
              double pt2y,
              double vert1x,
              double vert1y,
              double vert2x,
              double vert2y):
    
    cdef:
        double subx, suby, res1, res2
        
    subx = vert2x - vert1x
    suby = vert1y - vert2y
    res1 = suby*(pt1x - vert1x) + subx*(pt1y - vert1y)
    res2 = suby*(pt2x - vert1x) + subx*(pt2y - vert1y)
    return res1*res2 >= 0