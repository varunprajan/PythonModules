import numpy as np

def getKIDisp(KI,mu,nu,x,y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    u1 = KI/mu*np.sqrt(r/(2*np.pi))*(1 - 2*nu + np.sin(theta/2)**2)*np.cos(theta/2)
    u2 = KI/mu*np.sqrt(r/(2*np.pi))*(2 - 2*nu - np.cos(theta/2)**2)*np.sin(theta/2)
    return u1, u2
    
def getKIIDisp(KII,mu,nu,x,y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    u1 = KII/mu*np.sqrt(r/(2*np.pi))*(2 - 2*nu + np.cos(theta/2)**2)*np.sin(theta/2)
    u2 = KII/mu*np.sqrt(r/(2*np.pi))*(-1 + 2*nu + np.sin(theta/2)**2)*np.cos(theta/2)
    return u1, u2