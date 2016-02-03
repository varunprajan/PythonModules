import numpy as np
import scipy.optimize as spo

r0def = 1 # equilibrium distance for nearest neighbor
# tabulated values of phi(r2) = ep and gamma_us (see tables in paper) for 2D and 3D
# note: these values were originally calculated for r0 = 2**(1/6),
# so they must be adjusted for other values of r0
epvec2d = np.array([0.102,0.160,0.214,0.264,0.308,0.346,0.378])
usfvec2d = np.array([0.6333,0.5265,0.4309,0.3422,0.2641,0.1966,0.1397])
epvec3d = np.array([0.009,0.077,0.143,0.207,0.266,0.318])
usfvec3d = np.array([0.8818,0.7344,0.6024,0.4816,0.3758,0.2824])

def gen_std_potential(ductilityindex,dimensions,r0=r0def):
    """
    Creates a potential (2D or 3D) corresponding to the various
    potentials in the paper. Varying the ductility index
    from 0 to 6 corresponds to potentials G to A in 2D.
    Varying the ductility index from 0 to 5 corresponds to
    potentials F to A in 3D.
    """
    if dimensions == 2:
        epvec, usfvec = epvec2d, usfvec2d*2**(1/6)/r0
    elif dimensions == 3:
        epvec, usfvec = epvec3d, usfvec3d*2**(1/3)/r0**2
    return NewPotential(dimensions,r0,ep=epvec[ductilityindex],usf=usfvec[ductilityindex])
    
def gen_usf_potential(usftarget,dimensions,r0=r0def):
    """
    Creates a potential (2D or 3D) with a desired
    gamma_us equal to usftarget
    """
    ep = estimate_ep(dimensions,r0,usftarget)
    return NewPotential(dimensions,r0,ep=ep)

def max_poly_root(coeff):
    """
    Auxiliary math function.
    Finds maximum of polynomial given its coefficients.
    Requires at least one real root.
    """
    roots = np.roots(np.polyder(coeff))
    return np.max([np.polyval(coeff,root) if not np.iscomplex(root) else -np.Inf for root in roots]) # returns -inf for complex roots, so these are ignored

class TabularPotential:
    def __init__(self):
        pass
        
    def get_energy(self,r):
        pass
        
    def get_force(self,r):
        pass
    
    def get_energy_table(self,rvec):
        """
        Construct a table, with two columns, of r and phi(r) (energy).
        Useful for plotting purposes.
        """
        return np.array([[r,self.get_energy(r)] for r in rvec])
    
    def get_force_table(self,rvec):
        """
        Construct a table, with two columns, of r and -phi'(r) (force).
        Useful for plotting purposes.
        """
        return np.array([[r,self.get_force(r)] for r in rvec])
        
    def write_file_lammps(self,filename,rvec,writeoption='w',extend=False,name='dummy',fac=0.1):
        """
        Writes tabular potential file for LAMMPS
        """
        if extend:
            rvec = np.insert(rvec,0,rvec[0]*fac)
        with open(filename,writeoption) as f:
            npoints = np.shape(rvec)[0]
            self._write_header_lammps_(f,npoints,name)
            self._write_table_lammps_(f,rvec)
            
    def _write_header_lammps_(self,f,npoints,name,comment='#'):
        """
        Writes header for tabular potential file for LAMMPS
        """
        f.write('{} \n\n'.format(comment))
        f.write('{} Tabulated potential \n'.format(comment))
        for key, val in self.__dict__.items():
            f.write('{2} {0} = {1} \n'.format(key,val,comment))
        f.write('\n\n')
        f.write('{0} \n'.format(name))
        f.write('N {0} \n\n'.format(npoints))
        
    def _write_table_lammps_(self,f,rvec,fmt='{:d} {:10.8f} {:10.8f} {:10.8f} \n'):
        """
        Writes data for tabular potential file for LAMMPS
        """
        for i, r in enumerate(rvec):
            f.write(fmt.format(i+1,r,self.get_energy(r),self.get_force(r)))
            
class SplinePotential(TabularPotential):
    # note on units:
    # internal operations (with prefix '_') are defined in normalized space (e.g. rbar = r/r0)
    # therefore, forces have units of energy/rbar (not energy/distance)
    # external operations are defined in conventional (unnormalized) space
    def __init__(self,spline):
        """
        Creates/initializes member of spline potential class
        (actual potential inherits from this class)
        """
        super().__init__
        if spline is None:
            self.spline = [None]*2
            self.dspline = [None]*2
            self.gen_splines()
        else:
            self._get_dspline_()

    def _gen_splines_(self):
        pass

    def _morse_energy_(self,rbar):
        """
        Returns energy for morse potential at interatomic separation rbar = r/r0
        """
        return (np.exp(self.alphabar*(1 - rbar)) - 1)**2 - 1 - self.offset
    
    def _morse_force_(self,rbar):
        """
        Returns force for morse potential at interatomic separation rbar = r/r0
        """
        return 2*self.alphabar*np.exp(self.alphabar*(1 - rbar))*(np.exp(self.alphabar*(1 - rbar)) - 1)

    def _get_dspline_(self):
        """
        Gets coefficients of spline derivatives from spline coefficients.
        """
        for i, spline in enumerate(self.spline):
            if spline is None:
                self.dspline[i] = None
            else:
                self.dspline[i] = np.polyder(spline)
    
    def _spline_energy_(self,rbar,splinenum):
        """
        Returns energy for a spline potential by evaluating spline
        """
        return np.polyval(self.spline[splinenum],rbar)
            
    def _spline_force_(self,rbar,splinenum):
        """
        Returns force for a spline potential by evaluating spline derivative
        """
        return -np.polyval(self.dspline[splinenum],rbar)
    
class NewPotential(SplinePotential):
    r2bardef = {2: np.sqrt(2), 3: np.sqrt(3/2)}
    rnnndef = {2: np.sqrt(3) - 0.02, 3: np.sqrt(2) - 0.02}
    alphabardef = {2: 3.5*2**(1/6), 3: 6.5*2**(1/6)}

    def __init__(self,dimensions,r0,ep,usf=None,r1bar=1.05,r2bar=None,r3bar=None,rnnn=None,alphabar=None,offset=0.0,spline=None):
        """
        Creates/initializes instance of member of class,
        which represents a potential. Required arguments
        are the dimensionality, the equilibrium separation
        (which can be set to 1 without loss of generality),
        and ep = phi(r2). Optional arguments (e.g. alphabar)
        are set using defaults if not specified.
        """
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
        """
        Constructs splines for potential by solving continuity equations
        and phi(r2) = ep
        """
        while r3barfac > 0:
            self.r3bar = self.r2bar + (self.rnnn - self.r2bar)*r3barfac
            def get_residual(coeff):
                self._set_splines_coeff_(coeff)
                return self._residual_()
            coeffsol = spo.fsolve(get_residual,np.zeros(8))
            self._set_splines_coeff_(coeffsol)
            # 2nd spline can be non-attractive; if so, reduce r3bar
            maxval2 = max_poly_root(self.spline[1])
            if maxval2 > tol:
                r3barfac = r3barfac - step
            else:
                return
        raise ValueError('Bad input')
    
    def _set_splines_coeff_(self,coeff):
        """
        Sets cubic spline coefficients from coeff vector (length 8)
        """
        self.spline[0] = coeff[0:4] # cubic spline
        self.spline[1] = coeff[4:8] # cubic spline
        self._get_dspline_()
    
    def _residual_(self):
        """
        Returns energy for the potential (phi)
        given the interatomic separation, r
        """
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
        """
        Returns energy for the potential (phi)
        given the interatomic separation, r
        """
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
        """
        Returns force for the potential (-dphi/dr)
        given the interatomic separation, r
        """
        rbar = r/self.r0
        if rbar < self.r1bar:
            return self._morse_force_(rbar)/self.r0
        elif rbar < self.r2bar:
            return self._spline_force_(rbar,0)/self.r0
        elif rbar < self.r3bar:
            return self._spline_force_(rbar,1)/self.r0
        else:
            return 0
            
def estimate_ep(dimensions,r0,usftarget):
    """
    Estimates phi(r2) = ep required to achieve
    a desired gamma_us
    """
    offsetusf = get_offset_usf(dimensions,r0)
    area = get_area(dimensions,r0)
    return -((usftarget - offsetusf)*area - 1)/2

def estimate_usf(dimensions,r0,ep):
    """
    Estimates gamma_us for the potential, using phi(r2) = ep
    and the offset value (obtained from molecular statics)
    """
    offsetusf = get_offset_usf(dimensions,r0)
    area = get_area(dimensions,r0)
    return (1 - 2*np.abs(ep))/area + offsetusf 

def get_area(dimensions,r0):
    """
    Returns the area of the plane along which slip occurs
    (for calculation of gamma_us)
    """
    if dimensions == 2:
        return r0
    elif dimensions == 3:
        return np.sqrt(3)/2*r0**2
    
def get_offset_usf(dimensions,r0):
    """
    Returns the offset value for gamma_us
    I.e. the difference between the analytical estimate and the
    actual (molecular statics) value.
    """
    if dimensions == 2:
        return -0.078*2**(1/6)/r0
    elif dimensions == 3:
        return -0.052*2**(2/6)/r0**2

    