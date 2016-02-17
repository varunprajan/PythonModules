import numpy as np
import mymath as Mmath
import scipy.optimize as spo
from tabularpotential import SplinePotential

r0def = 1 # equilibrium distance for nearest neighbor
# note: these values were originally calculated for r0 = 2**(1/6),
# so they must be adjusted for other values of r0
epvec2d = np.array([0.102,0.160,0.214,0.264,0.308,0.346,0.378])
usfvec2d = np.array([0.6333,0.5265,0.4309,0.3422,0.2641,0.1966,0.1397])
epvec3d = np.array([0.009,0.077,0.143,0.207,0.266,0.318])
usfvec3d = np.array([0.8818,0.7344,0.6024,0.4816,0.3758,0.2824])

def gen_std_potential(ductilityindex,dimensions,r0=r0def):
    if dimensions == 2:
        epvec, usfvec = epvec2d, usfvec2d*2**(1/6)/r0
    elif dimensions == 3:
        epvec, usfvec = epvec3d, usfvec3d*2**(1/3)/r0**2
    return NewPotential(dimensions,r0,ep=epvec[ductilityindex],usf=usfvec[ductilityindex])
    
def gen_usf_potential(usftarget,dimensions,r0=r0def):
    ep = estimate_ep(dimensions,r0,usftarget)
    return NewPotential(dimensions,r0,ep=ep)
    
def gen_non_std_potential(dimensions,r0=r0def,**kwargs):
    return NewPotential(dimensions,r0,**kwargs)
    
class NewPotential(SplinePotential):
    r2bardef = {2: np.sqrt(2), 3: np.sqrt(3/2)}
    rnnndef = {2: np.sqrt(3) - 0.02, 3: np.sqrt(2) - 0.02}
    alphabardef = {2: 3.5*2**(1/6), 3: 6.5*2**(1/6)}

    def __init__(self,dimensions,r0,ep,usf=None,r1bar=1.05,r2bar=None,r3bar=None,rnnn=None,alphabar=None,offset=0.0,spline=None):
        self.dimensions = dimensions
        self.r0 = r0
        self.ep = ep
        self.usf = usf
        self.r1bar = r1bar
        if r2bar is None:
            self.r2bar = NewPotential.r2bardef[dimensions]
        if rnnn is None:
            self.rnnn = NewPotential.rnnndef[dimensions]
        if alphabar is None:
            self.alphabar = NewPotential.alphabardef[dimensions]
        self.r3bar = r3bar
        self.offset = offset
        super().__init__(spline)
        
    def gen_splines(self,r3barfac=0.98,step=0.01,tol=1.e-8):
        while r3barfac > 0:
            self.r3bar = self.r2bar + (self.rnnn - self.r2bar)*r3barfac
            def getResidual(coeff):
                self._set_splines_coeff_(coeff)
                return self._residual_()
            coeffsol = spo.fsolve(getResidual,np.zeros(8))
            self._set_splines_coeff_(coeffsol)
            # 2nd spline can be non-attractive; if so, reduce r3bar
            maxval2 = Mmath.max_poly_root(self.spline[1])
            if maxval2 > tol:
                r3barfac = r3barfac - step
            else:
                return
        raise ValueError('Bad input')
    
    def _set_splines_coeff_(self,coeff):
        self.spline[0] = coeff[0:4] # cubic spline
        self.spline[1] = coeff[4:8] # cubic spline
        self._get_dspline_()
    
    def _residual_(self):
        f = np.zeros(8)
        f[0] = self._morse_energy_(self.r1bar) - self._spline_energy_(self.r1bar,0)
        f[1] = self._morse_force_(self.r1bar) - self._spline_force_(self.r1bar,0)
        f[2] = self._spline_energy_(self.r2bar,0) - (-self.ep)
        f[3] = self._spline_force_(self.r2bar,0) - self._spline_force_(self.r2bar,1)
        f[4] = self._spline_energy_(self.r2bar,1) - (-self.ep)
        f[5] = self._spline_energy_(self.r3bar,1) - 0
        f[6] = self._spline_force_(self.r3bar,1) - 0
        # unnecessary, but nice to have
        ddspline0, ddspline1 = np.polyder(self.dspline[0]), np.polyder(self.dspline[1])
        f[7] = np.polyval(ddspline0,self.r2bar) - np.polyval(ddspline1,self.r2bar) 
        return np.array(f)
            
    def get_energy(self,r):
        rbar = r/self.r0
        if rbar < self.r1bar:
            return self._morse_energy_(rbar)
        elif rbar < self.r2bar:
            return self._spline_energy_(rbar,0)
        elif rbar < self.r3bar:
            return self._spline_energy_(rbar,1)
        else:
            return 0
            
    def get_force(self,r):
        rbar = r/self.r0
        if rbar < self.r1bar:
            return self._morse_force_(rbar)/self.r0
        elif rbar < self.r2bar:
            return self._spline_force_(rbar,0)/self.r0
        elif rbar < self.r3bar:
            return self._spline_force_(rbar,1)/self.r0
        else:
            return 0
    
    # used for computing Kic, Kie, etc.    
    def get_K_props(self,lattice):
        alpha = self.alphabar/self.r0
        # for closed-packed slip
        if self.usf is None:
            self.usf = self.estimate_usf()
        # lattice-dependent properties
        if lattice == 'hex':
            symmetry = 'cubic'
            C11 = 3/2*np.sqrt(3)*alpha**2
            Velastic = {'11': C11, '12': C11/3, '44': C11/3}
            Vsurface = {'112': {'surf': 1/self.r0}}
        elif lattice == 'fcc':
            symmetry = 'cubic'
            C11 = 2*np.sqrt(2)*alpha**2/self.r0
            Velastic = {'11': C11, '12': C11/2, '44': C11/2}
            Vsurface = {'111': {'surf': np.sqrt(3)/self.r0**2}, '001': {'surf': 2/self.r0**2}, '011': {'surf': np.sqrt(9/2)/self.r0**2}}
        elif lattice == 'hcp':
            symmetry = 'hexagonal'
            C11 = 5/np.sqrt(2)*alpha**2/self.r0
            Velastic = {'11': C11, '33': C11*16/15, '13': C11*4/15, '44': C11*4/15, '66': C11/3}
            Vsurface = {'001': {'surf': 1.458*2**(1/3)/self.r0**2}}
        return {'elastic': Velastic, 'surface': Vsurface, 'unstable': {'full': self.usf, 'partial': self.usf}, 'symmetry': symmetry, 'rho': rho}
        
    def estimate_ep(self,usftarget):
        """
        Estimates phi(r2) = ep required to achieve
        a desired gamma_us
        """
        return -((usftarget - self.offsetusf)*self.area - 1)/2

    def estimate_usf(self,ep):
        """
        Estimates gamma_us for the potential, using phi(r2) = ep
        and the offset value (obtained from molecular statics)
        """
        return (1 - 2*np.abs(ep))/self.area + self.offset_usf

    def elasticdict(self,lattice):
        alpha = self.alphabar/self.r0
        if self.dimensions == 2:
            if lattice == 'hex':
                C11 = 3/2*np.sqrt(3)*alpha**2
                return {'11': C11, '33': C11, '12': C11/3, '13': C11/3, '44': C11/3, '66': C11/3} # 3d constants are dummy               
        elif self.dimensions == 3:
            if lattice == 'fcc':
                C11 = 2*np.sqrt(2)*alpha**2/self.r0
                return {'11': C11, '12': C11/2, '44': C11/2}
            elif lattice == 'hcp':
                C11 = 5/np.sqrt(2)*alpha**2/self.r0
                return {'11': C11, '33': C11*16/15, '13': C11*4/15, '44': C11*4/15, '66': C11/3}
        raise ValueError('Undefined lattice')
    
    @property
    def surfacedict(self,lattice):
        if lattice == 'hex':
            return {'112': 1/self.r0}
        elif lattice == 'fcc':
            return {'111': np.sqrt(3)/self.r0**2, '001': 2/self.r0**2, '011': np.sqrt(9/2)/self.r0**2}
        elif lattice == 'hcp':
            return {'001': 1.458*2**(1/3)/self.r0**2}       
        
    @property
    def rho(self):
        if (self.dimensions == 2):
            return 2/(np.sqrt(3)*self.r0**2)
        elif (self.dimensions == 3):
            return np.sqrt(2)/self.r0**3        
        
    @property
    def area(self):
        """
        Returns the area of the plane along which slip occurs
        (for calculation of gamma_us)
        """
        if self.dimensions == 2:
            return self.r0
        elif self.dimensions == 3:
            return np.sqrt(3)/2*self.r0**2
    
    @property
    def offset_usf(self):
        """
        Returns the offset value for gamma_us
        I.e. the difference between the analytical estimate and the
        actual (molecular statics) value.
        """
        if self.dimensions == 2:
            return -0.078*2**(1/6)/self.r0
        elif self.dimensions == 3:
            return -0.052*2**(2/6)/self.r0**2
        
# obsolete stuff
# 2d
# smoothed potential, cutoff -0.05 (new potential #2)
# epsvecnew22d = [0.113,0.169,0.218,0.262,0.301,0.348,0.39]
# usfvecnew22d = [0.6121,0.5067,0.4116,0.3261,0.2504,0.1844,0.1479]
# original potential
# epsvecold2d = [0.1,0.15,0.22,0.27,0.31,0.35,0.38]
# usfvecold2d = [0.6371,0.5087,0.4211,0.3334,0.2631,0.1927,0.1399]
# 3d
# potential with incorrect r2bar = sqrt(17/12)
# epsvecnew23d = [0.01,0.17,0.247,0.313,0.372,0.422]
# usfvecnew23d = [0.9165,0.7667,0.6299,0.5061,0.3952,0.3013]
    