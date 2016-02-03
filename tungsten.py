import numpy as np
import mdutilities_energy as mdue
import ddplot

class WPotential(object):
    def __init__(self,a,simdir):
        self.a = a
        self.simdir = simdir
        
    def burgers_vec(self):
        return np.array([0,0,self.burgers_mag()])
        
    def burgers_mag(self):
        return self.a*np.sqrt(3)/2
    
    def lattice_period(self):
        return self.a*np.array([np.sqrt(2),np.sqrt(6),np.sqrt(3)/2])
    
    def disl_center(self):
        return self.a*np.array([1/np.sqrt(6),1/np.sqrt(8)])
    
    def get_filename_elastic(self,subdir='Elasticity/',simname='elastic',suffix='.log'):
        return self.simdir + subdir + simname + suffix
    
    def get_filename_surface(self,surfname,subdir='Surface Energy/',simname='surf_tungsten',suffix='.log'):
        return '{0}{1}{3}/{2}_{3}{4}'.format(self.simdir,subdir,simname,surfname,suffix)
    
    def get_filename_core(self,xyztype,size='',subdir='Dislocation/Core_Structure',simname='screw_core_tungsten',suffix='.xyz'):
        return '{0}{1}{5}/{2}_{3}{4}'.format(self.simdir,subdir,simname,xyztype,suffix,size)
    
    def get_surface_energy(self,surfnames=['100','110','111','211']):
        return self.get_energy_both(surfnames,'surface')
    
    def get_filename_gsf(self,surfname,subdir='GSF_Energy/',simname='gsf_tungsten',suffix='.log',direction='111'):
        return '{0}{1}{3}_{5}/{2}_{3}_{5}{4}'.format(self.simdir,subdir,simname,surfname,suffix,direction)
        
    def get_filename_decohesion(self,surfname,subdir='Decohesion/',simname='decohesion_tungsten',suffix='.log'):
        return '{0}{1}{3}/{2}_{3}{4}'.format(self.simdir,subdir,simname,surfname,suffix)
    
    def get_gsf_energy(self,surfnames=['110','211']):
        return self.get_energy_both(surfnames,'gsf')
    
    def get_decohesion_energy(self,surfnames=['110','211']):
        return self.get_curve_data(surfnames,'decohesion')
    
    def get_energy_both(self,surfnames,option):
        return self.get_curve_data(surfnames,option), self.get_energy_data(surfnames,option)
    
    def get_curve_data(self,surfnames,option):
        curvedata = []
        filefun, curvefun = self.get_filename_fun(option), self.get_curve_fun(option)
        for surfname in surfnames:
            filename = filefun(surfname)
            curvedatanew = curvefun(filename,subdir='')
            curvedata.append(curvedatanew)
        return curvedata
    
    def get_energy_data(self,surfnames,option):
        energydict = {}
        filefun, energyfun = self.get_filename_fun(option), self.get_energy_fun(option)
        for surfname in surfnames:
            filename = filefun(surfname)
            energydict[surfname] = energyfun(filename,subdir='')
        return energydict
    
    def get_filename_fun(self,option):
        if option == 'surface':
            return self.get_filename_surface
        elif option == 'gsf':
            return self.get_filename_gsf
        elif option == 'decohesion':
            return self.get_filename_decohesion
        
    def get_curve_fun(self,option):
        if option == 'surface':
            return mdue.get_surface_energy_curve
        elif option == 'gsf':
            return mdue.get_gsf_energy_curve
        elif option == 'decohesion':
            return mdue.get_surface_energy_curve # not a typo --- decohesion is similar to surface energy calc.!
        
    def get_energy_fun(self,option):
        if option == 'surface':
            return mdue.get_surface_energy
        elif option == 'gsf':
            return mdue.get_usf_energy
    
    def gen_chopped_xyz(self,xyztype,xlims,ylims,zlims,size=''):
        xyz = xyzData()
        filename = self.get_filename_core(xyztype,size=size)
        xyz.read_from_file(filename)
        xyz.chop_data(xlims,ylims,zlims)
        return xyz
        
    def gen_chop_both(self,xlims,ylims,zlims,size=''):
        zlimsfake = [-np.Inf,np.Inf]
        xyzbefore = self.gen_chopped_xyz('before',xlims,ylims,zlimsfake,size=size)
        idx = (xyzbefore.xyz[:,2] > zlims[0]) & (xyzbefore.xyz[:,2] < zlims[1])
        xyzbefore.adjust_using_idx(idx)
        xyzafter = self.gen_chopped_xyz('after',xlims,ylims,zlimsfake,size=size)
        xyzafter.adjust_using_idx(idx)
        return xyzbefore, xyzafter
        
    def gen_dddata(self,xyzbefore,xyzafter,tol=1e-4):
        burgersvec = self.burgers_vec()
        cutoff = (1-tol)*self.a
        return ddplot.DDData(xyzbefore.xyz,xyzafter.xyz,cutoff,burgersvec)

def write_array_sub(array,f):
    fmt = ['{0} ']*array.shape[1]
    for line in array.tolist():
        for num, fmtnum in zip(line, fmt):
            try:
                f.write(fmtnum.format(num))
            except ValueError:
                f.write(fmtnum.format(int(num)))
        f.write('\n')
        
class xyzData(object):
    def __init__(self,atomtype=None,xyz=None,natoms=None):
        self.atomtype = atomtype
        self.xyz = xyz
        self.natoms = natoms
    
    def read_from_file(self,filename):
        data = self.parse_xyzfile(filename)
        self.parse_data(data)

    def parse_xyzfile(self,filename,nheaderlines=2):
        with open(filename,'r') as f:
            for i, line in enumerate(f):
                if i >= nheaderlines - 1:
                    return np.loadtxt(f)
    
    def parse_data(self,data):
        self.atomtype = data[:,0].astype(int)
        self.xyz = data[:,1:]
        self.natoms = data.shape[0]
                    
    def chop_data(self,xlims,ylims,zlims):
        idx = np.ones((self.natoms,),dtype=bool)
        for colidx, lims in enumerate([xlims,ylims,zlims]):
            idx = idx & self.get_idx(lims,colidx)
        self.adjust_using_idx(idx)
        
    def adjust_using_idx(self,idx):
        self.natoms = np.sum(idx)
        self.atomtype = self.atomtype[idx]
        self.xyz = self.xyz[idx,:]
        
    def get_idx(self,lims,colidx):
        coord = self.xyz[:,colidx]
        return (coord > lims[0]) & (coord < lims[1])
    
    def write_to_file(self,filename):
        with open(filename,'w') as f:
            self.write_header(f)
            self.write_data(f)
            
    def write_data(self,f):
        for i in range(self.natoms):
            f.write('{:d} {:.6f} {:.6f} {:.6f} \n'.format(self.atomtype[i],*self.xyz[i,:]))
    
    def write_header(self,f):
        f.write('{} \n'.format(self.natoms))
        f.write('blah \n')

def write_to_file_dd(xyzbefore,xyzafter,subdir,simname,period,center):
    filename = get_filename(subdir,simname,'dd','dd')
    with open(filename,'w') as f:
        f.write('CSYS \n')
        for i in range(3):
            f.write('0.0 0.0 0.0 \n')
        f.write('PERIOD \n')
        f.write('{:.6f} {:.6f} {:.6f} \n'.format(*period))
        f.write('DISLO CENTER \n')
        f.write('{:.6f} {:.6f} \n'.format(*center))
        f.write('NUM_UNREL \n')
        f.write('{} \n'.format(xyzbefore.natoms))
        f.write('COOR_UNREL \n')
        write_to_file_dd_sub(xyzbefore,f,pad=True)
        f.write('NUM_REL \n')
        f.write('{} \n'.format(xyzafter.natoms))
        f.write('COOR_REL \n')
        write_to_file_dd_sub(xyzafter,f,pad=False)
        
def write_to_file_dd_sub(obj,f,pad):
    for i in range(obj.natoms):
        if pad:
            f.write('{1:.6f} {2:.6f} {3:.6f} {0} \n'.format(0,*obj.xyz[i,:]))
        else:
            f.write('{:.6f} {:.6f} {:.6f} \n'.format(*obj.xyz[i,:]))        