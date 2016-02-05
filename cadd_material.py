import numpy as np
import crystallography as cr

def stiffness_2d_voigt(E,nu):
    prefac = E/(1+nu)/(1-2*nu)
    C11 = (1-nu)*prefac
    C12 = nu*prefac
    C44 = (1-2*nu)*prefac/2
    stiffness3dvoigt = cr.cubicVoigt({'11': C11, '12': C12, '44': C44})
    return stiffness3dvoigt[[[0],[1],[5]],[0,1,5]]
    