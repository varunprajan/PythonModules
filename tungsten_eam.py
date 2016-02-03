import numpy as np
import datetime

class EAM:
    def __init__(self):
        pass
        
    def write_to_lammps_setfl(self,file,Nrho,drho,Nr,dr,suffix='.eam.alloy'):
        def write_comments(f):
            f.write('EAM potential in LAMMPS setfl format \n')
            f.write('#\n')
            f.write('Made on: {0} \n'.format(datetime.datetime.now()))
        def write_header(f):
            f.write('{0} {1} \n'.format(1,self.element))
            f.write('{0} {1} {2} {3} {4} \n'.format(Nrho,drho,Nr,dr,self.cutoff))
            f.write('{0} {1} {2} {3} \n'.format(self.atomicnum,self.atomicmass,self.latconst,self.lattype))
        def write_data(f):
            self.write_array(f,Nrho,drho,'get_F')
            self.write_array(f,Nr,dr,'get_rho')
            self.write_array(f,Nr,dr,'get_Phir')
        with open(file+suffix,'w') as f:
            write_comments(f)
            write_header(f)
            write_data(f)
                
    def write_array(self,f,N,dx,methodstr):
        for i in range(N):
            method = getattr(self,methodstr)
            val = method(i*dx)
            f.write('{0:.15e} \n'.format(val))
            
    def get_F(self,rho):
        return 2*rho
        
    def get_rho(self,r):
        return r**2
        
    def get_Phir(self,r):
        return r + 2        
    

class EAMMarinica(EAM):
    # units: lengths - Angstroms, energies - eV
    def __init__(self,name,element='W',atomicnum=74,atomicmass=183.84,latconst=3.1439,lattype='bcc',cutoff=5.5,rinnercutoff=2.002970124727):
        self.element = element
        self.name = name
        self.atomicnum = atomicnum
        self.atomicmass = atomicmass
        self.latconst = latconst
        self.lattype = lattype
        self.cutoff = cutoff
        self.rinnercutoff = rinnercutoff
        self.Fdata = self.load_data('F')
        self.rhodata = self.load_data('rho')
        self.Phidata = self.load_data('Phi')
        
    def load_data(self,suff,ext='.txt',subdir='Data Files/'):
        file = subdir + self.name + '_' + suff + ext
        return np.loadtxt(file)
        
    def get_F(self,rho):
        a1, a2 = self.Fdata
        return a1*np.sqrt(rho) + a2*rho**2
        
    def get_rho(self,r):
        r = max([r,self.rinnercutoff])
        return self.eval_cubic_knots(r,self.rhodata)
        
    def get_Phi(self,r):
        return self.eval_cubic_knots(r,self.Phidata)
        
    def get_Phir(self,r):
        return r*self.get_Phi(r)
        
    def eval_cubic_knots(self,r,data):
        val = 0
        for ai, deltai in data:
            if r < deltai:
                val += ai*(deltai - r)**3
        return val