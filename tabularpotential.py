import numpy as np

class TabularPotential:
    def __init__(self):
        pass
        
    def get_energy(self,r):
        pass
        
    def get_force(self,r):
        pass
    
    def get_energy_table(self,rvec):
        return np.array([[r,self.get_energy(r)] for r in rvec])
    
    def get_force_table(self,rvec):
        return np.array([[r,self.get_force(r)] for r in rvec])
        
    def get_energy_force_table(self,rvec):
        return np.array([[r,self.get_energy(r),self.get_force(r)] for r in rvec])  
    
    def write_file_lammps(self,filename,rvec,writeoption='w',extend=False,name='dummy',fac=0.1):
        if extend:
            rvec = np.insert(rvec,0,rvec[0]*fac)
        with open(filename,writeoption) as f:
            npoints = np.shape(rvec)[0]
            self._write_header_lammps_(f,npoints,name)
            self._write_table_lammps_(f,rvec)
            
    def _write_header_lammps_(self,f,npoints,name,comment='#'):
        f.write('{} \n\n'.format(comment))
        f.write('{} Tabulated potential \n'.format(comment))
        for key, val in self.__dict__.items():
            f.write('{2} {0} = {1} \n'.format(key,val,comment))
        f.write('\n\n')
        f.write('{0} \n'.format(name))
        f.write('N {0} \n\n'.format(npoints))
        
    def _write_table_lammps_(self,f,rvec,fmt='{:d} {:10.8f} {:10.8f} {:10.8f} \n'):
        for i, r in enumerate(rvec):
            f.write(fmt.format(i+1,r,self.get_energy(r),self.get_force(r)))

class SplinePotential(TabularPotential):
    # note on units:
    # internal operations (with prefix '_') are defined in normalized space (e.g. rbar = r/r0)
    # therefore, forces have units of energy/rbar (not energy/distance)
    # external operations are defined in conventional (unnormalized) space
    def __init__(self,spline):
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
        return (np.exp(self.alphabar*(1 - rbar)) - 1)**2 - 1 - self.offset
    
    def _morse_force_(self,rbar):
        return 2*self.alphabar*np.exp(self.alphabar*(1 - rbar))*(np.exp(self.alphabar*(1 - rbar)) - 1)

    def _get_dspline_(self):
        for i, spline in enumerate(self.spline):
            if spline is None:
                self.dspline[i] = None
            else:
                self.dspline[i] = np.polyder(spline)
    
    def _spline_energy_(self,rbar,splinenum):
        return np.polyval(self.spline[splinenum],rbar)
            
    def _spline_force_(self,rbar,splinenum):
        return -np.polyval(self.dspline[splinenum],rbar)
            
