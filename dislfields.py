import numpy as np
import mymath as Mmath

def getuglobal(b,mu,nu,pos,posd,thetad):
    A = Mmath.get_rot_mat_2d(thetad)
    Areverse = Mmath.get_rot_mat_2d(-thetad)
    dpos = pos - posd
    dposrot = np.einsum('ij,j',A,dpos)
    urot = getu(b,mu,nu,dposrot)
    return np.einsum('ij,j',Areverse,urot)
    
def getu(b,mu,nu,dpos):
    dx1, dx2 = dpos
    return np.array([getu1(b,mu,nu,dx1,dx2),getu2(b,mu,nu,dx1,dx2)])
    
def getu1(b,mu,nu,dx1,dx2):
    prefac = b/(2*np.pi*(1-nu))
    firstterm = 1/2*(dx1*dx2/(dx1**2 + dx2**2))
    secondterm = (1-nu)*np.arctan2(dx2,dx1)
    return prefac*(firstterm + secondterm)
    
def getu2(b,mu,nu,dx1,dx2):
    prefac = b/(2*np.pi*(1-nu))
    firstterm = 1/2*(dx2**2/(dx1**2 + dx2**2))
    secondterm = 1/4*(1-2*nu)*np.log((dx1**2 + dx2**2)/b**2)
    return prefac*(firstterm - secondterm)
    
def gets11(b,mu,nu,dx1,dx2):
    prefac = mu*b/(2*np.pi*(1-nu))
    rest = dx2*(3*dx1**2 + dx2**2)/(dx1**2 + dx2**2)**2
    return -prefac*rest
    
def gets22(b,mu,nu,dx1,dx2):
    prefac = mu*b/(2*np.pi*(1-nu))
    rest = dx2*(dx1**2 - dx2**2)/(dx1**2 + dx2**2)**2
    return prefac*rest
    
def gets12(b,mu,nu,dx1,dx2):
    prefac = mu*b/(2*np.pi*(1-nu))
    rest = dx1*(dx1**2 - dx2**2)/(dx1**2 + dx2**2)**2
    return prefac*rest
    
def adjustDxn(dx1,dx2,rcore,tol=1e-10):
    r = np.sqrt(dx1**2 + dx2**2)     
    if (r**2 < rcore**2*tol):
        rcorefac = rcore/np.sqrt(2.0)
        dx1 = np.copysign(rcorefac,dx1)
        dx2 = np.copysign(rcorefac,dx2)
    elif (r**2 < rcore**2):
        rfac = rcore/r
        dx1 = dx1*rfac
        dx2 = dx2*rfac
    return dx1, dx2
 
