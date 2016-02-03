import numpy as np
import mymath as Mmath

def get_u_global(b,nu,pos,posdisl,thetadisl,degoption=True):
    A = Mmath.get_rot_mat_2d(thetadisl,degoption=degoption)
    Areverse = Mmath.get_rot_mat_2d(-thetadisl,degoption=degoption)
    dpos = pos - posd
    dposrot = np.einsum('ij,j',A,dpos)
    urot = getu(b,nu,dposrot)
    return np.einsum('ij,j',Areverse,urot)
    
def get_u(b,nu,dpos):
    dx1, dx2 = dpos
    prefac = b/(2*np.pi*(1-nu))
    rsq = dx1**2 + dx2**2
    u1 = get_u1(prefac,rsq,dx1,dx2,nu)
    u2 = get_u2(prefac,rsq,dx1,dx2,nu)
    return np.array([u1,u2])
    
def get_u1(prefac,rsq,dx1,dx2,nu):
    u1first = 1/2*(dx1*dx2/rsq)
    u1second = (1-nu)*np.arctan2(dx2,dx1)
    return prefac*(u1first + u1second)
    
def get_u2(prefac,rsq,dx1,dx2,nu):
    u2first = 1/2*(dx2**2/rsq)
    u2second = 1/4*(1-2*nu)*np.log(rsq/b**2)
    return prefac*(u2first - u2second)

def get_stress(b,mu,nu,dpos):
    dx1, dx2 = dpos
    prefac = mu*b/(2*np.pi*(1-nu))
    rsq = dx1**2 + dx2**2
    u1 = get_u1(prefac,rsq,dx1,dx2,nu)
    u2 = get_u2(prefac,rsq,dx1,dx2,nu)
    return np.array([u1,u2])
    
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
 
