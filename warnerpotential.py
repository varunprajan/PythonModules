import numpy as np
import mymath as Mmath
import scipy.optimize as spo
from tabularpotential import SplinePotential

r0def = 2**(1/6) # equilibrium distance for nearest neighbor

def genWarnerPotential(dimensions,r3bar,r0=r0def,**kwargs):
    return WarnerPotential(dimensions,r0,r3bar,**kwargs)
    
class WarnerPotential(SplinePotential):
    rnnndef = {2: np.sqrt(3) - 0.02, 3: np.sqrt(2) - 0.02}
    alphabardef = {2: 3.5*2**(1/6), 3: 6.5*2**(1/6)}

    def __init__(self,dimensions,r0,r3bar,usf=None,r1bar=1.05,alphabar=None,offset=0.0,spline=None):
        self.dimensions = dimensions
        self.r0 = r0
        self.usf = usf
        self.r3bar = r3bar
        # if r3bar > WarnerPotential.rnndef[dimensions]:
            # raise ValueError('r3bar is greater than 2nd nearest neighbor distance!')
        if alphabar is None:
            self.alphabar = WarnerPotential.alphabardef[dimensions]
        self.r1bar = r1bar
        self.offset = offset
        super().__init__(spline)
        
    def gen_splines(self):
        def getResidual(coeff):
            self._set_splines_coeff_(coeff)
            return self._residual_()
        coeffsol = spo.fsolve(getResidual,np.zeros(4))
        self._set_splines_coeff_(coeffsol)
    
    def _set_splines_coeff_(self,coeff):
        self.spline[0] = coeff[0:4] # cubic spline
        self._get_dspline_()
    
    def _residual_(self):
        f = np.zeros(4)
        f[0] = self._morse_energy_(self.r1bar) - self._spline_energy_(self.r1bar,0)
        f[1] = self._morse_force_(self.r1bar) - self._spline_force_(self.r1bar,0)
        f[2] = self._spline_energy_(self.r3bar,0) - 0
        f[3] = self._spline_force_(self.r3bar,0) - 0
        return np.array(f)
            
    def get_energy(self,r):
        rbar = r/self.r0
        if rbar < self.r1bar:
            return self._morse_energy_(rbar)
        elif rbar < self.r3bar:
            return self._spline_energy_(rbar,0)
        else:
            return 0
            
    def get_force(self,r):
        rbar = r/self.r0
        if rbar < self.r1bar:
            return self._morse_force_(rbar)/self.r0
        elif rbar < self.r3bar:
            return self._spline_force_(rbar,0)/self.r0
        else:
            return 0
            
    def set_default(self):
        self.r1bar = 1.05
        self.offset = 0.0
        if self.dimensions == 2:
            self.r2bar = np.sqrt(2)
            self.alphabar = 3.5*self.r0
        elif self.dimensions == 3:
            self.r2bar = np.sqrt(3/2)
            self.alphabar = 6.5*self.r0
    