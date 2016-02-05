import pandas as pd
import numpy as np
import myio as Mio
import mymath as Mmath
import os

directiondict = {'x': 0, 'y': 1, 'z': 2}
energyfacdict = {'metal': 1.60217657e-19, 'LJ': 1, None: 1}
lengthfacdict = {'metal': 1.e-10, 'LJ': 1, None: 1}
        
class LammpsFile(object):
    def __init__(self,filename,subdir,ndim=3,units=None):
        self.filepath = os.path.join(subdir,filename)
        self.ndim = ndim
        self.units = units
        self.boxdims = self.box_dimensions()
        
    def box_dimensions(self):
        pass
     
    def area(self,plane='xz'): # default: xz area
        boxdims = self.box_dimensions()
        L = boxdims[:,1] - boxdims[:,0]
        L1, L2 = (L[directiondict[char]] for char in plane) # e.g. 'xz' -> Lx, Lz
        lengthfac = lengthfacdict[self.units]
        if self.ndim == 2:
            return L1*lengthfac
        elif self.ndim == 3:
            return L1*L2*lengthfac**2 

class DumpFile(LammpsFile):
    def __init__(self,filename,bounds=None,subdir='Dump Files/',ndim=3,units=None,postype='unscaled'):
        super().__init__(filename,subdir,ndim=ndim,units=units)
        self.bounds = self.recover_bounds(bounds,postype)
        self.headerlines = self.header_lines()
    
    def header_lines(self,key='ITEM: ATOMS'):
        with open(self.filepath,'r') as f:
            for i, line in enumerate(f):
                if line.startswith(key):
                    return i
    
    # read
    def read_file(self,postype):
        res = self.cut_down_file(self.bounds)
        if postype == 'scaled':
            res = self.unscale_coords(res) # back to real coordinates
        return res

    def cut_down_file(self,bounds,chunksize=1000000):
        reader = pd.read_csv(self.filepath,sep=' ',skiprows=self.headerlines,iterator=True,chunksize=chunksize)
        readernew = self.yield_cut_chunk(reader)
        return np.concatenate(list(readernew),axis=0)
        
    def yield_cut_chunk(self,reader,indices=range(2,5)):
        for chunk in reader:
            values = chunk.values
            indexall = np.ones((values.shape[0],), dtype=bool)
            if self.bounds is not None:
                for [posmin, posmax], index in zip(self.bounds,indices):
                    valuescurr = values[:,index]
                    indexall = indexall & (posmin <= valuescurr) & (valuescurr <= posmax)
            yield values[indexall,:]

    # scaling, unscaling
    def box_dimensions(self,key='ITEM: BOX'):
        with open(self.filepath,'r') as f:
            for line in f:
                if line.startswith(key):
                    lines = [next(f) for i in range(3)]
                    return np.loadtxt(lines)  
    
    def recover_bounds(self,bounds,postype):
        if postype == 'scaled':
            bounds = self.scale_bounds(bounds)
        return bounds
    
    def unscale_coords(self,dumparray,indexstart=2):
        for i in range(3):
            dumparray[:,i+indexstart] = Mmath.rescale_coords(dumparray[:,i+indexstart],[0,1],self.boxdims[i,:])
        return dumparray

    def scale_bounds(self,bounds):
        if bounds is None:
            return None
        else:
            boundsnew = np.empty(np.shape(bounds))
            for i in range(3): # dimensions
                boundsnew[i,:] = Mmath.rescale_coords(bounds[i,:],self.boxdims[i,:],[0,1])
            return boundsnew        

class LogFile(LammpsFile):
    def __init__(self,filename,simtype=None,subdir='Log Files/',ndim=3,units=None):
        super().__init__(filename,subdir,ndim=ndim,units=units)
        self.read_data(simtype)
        
    def read_data(self,simtype):
        if simtype == 'energy' or 'stressstrain':
            self.minconfig = self.optimized_config()
        elif simtype == 'elastic':
            self.elastic = self.elastic_constants()
    
    @classmethod
    def LJ_2d(cls,filename):
        return cls(filename,ndim=2,units='LJ')
        
    @classmethod
    def LJ_3d(cls,filename):
        return cls(filename,ndim=3,units='LJ')
    
    def box_dimensions(self,key='Created orthogonal box'):
        with open(self.filepath,'r') as f:
            for line in f:
                if line.startswith(key):
                    indicesstart = Mio.findCharIndices(line,'(')
                    indicesend = Mio.findCharIndices(line,')')
                    bounds = [line[istart+1:iend] for istart, iend in zip(indicesstart,indicesend)]
                    return np.transpose(np.loadtxt(bounds))
                    
    def optimized_config(self,starttag='Step',endtag='Loop time'):
        """Returns a numpy array of the optimized configuration in an MS run:
        i.e., the last line from each minimization"""
        with open(self.filepath,'r') as f:
            chunkfun = lambda: Mio.yield_last_item(Mio.yield_lines_between(f,starttag,endtag))
            return np.loadtxt(iter(chunkfun,None))
      
    def all_lines(self,starttag='Step',endtag='Loop time'): # get all lines of MS/MD run (output: list of numpy arrays)
        """Returns a list of numpy arrays of each minimization in an MS run"""
        with open(self.filepath,'r') as f:
            chunkfun = lambda: Mio.np_array(Mio.yield_lines_between(f,starttag,endtag))
            return list(iter(chunkfun,None))

    # elastic constants        
    def elastic_constants(tag='Elastic Constant'):
        """Generates dictionary of elastic constants from log file from elastic constants simulation
        E.g. dict['44'] = blah"""
        elconstdict = {}
        with open(self.filepath,'r') as f:
            for line in Mio.yield_tagged_lines(f,tag=tag):
                linesplit = line.split(' ')
                idxeq = linesplit.index('=')
                name, val = linesplit[idxeq-1], linesplit[idxeq+1]
                elconstdict[name] = float(val)
        return elconstdict
    
    # stress strain
    def stress_strain_curve(self,stressindex,lengthindex):
        data = self.minconfig
        stressfac = energyfacdict[self.units]/lengthfacdict[self.units]**3
        strain = (data[:,lengthindex] - data[0,lengthindex])/data[0,lengthindex]
        stress = -data[:,stressindex]*stressfac # -1 because LAMMPS outputs pressures
        return np.column_stack((strain,stress))
    
    # various energies
    def block_force(self,energyindex=2):
        res = self.energy_sub(energyindex)
        xvec, energyvec = res[:,0],res[:,1]
        forcevec = -Mmat.get_diff_vec(xvec,energyvec) # force = -d(energy)/d(pos)
        return np.column_stack(xvec,forcevec)

    def surface_energy_curve(self,energyindex=2):
        res = self.energy_sub(energyindex)
        res[:,1] = res[:,1]/2
        return res
        
    def surface_energy(self,energyindex=2):
        res = self.energy_sub(energyindex)
        idxenergy = -1 # last data point
        convergedenergy = res[idxenergy,1]
        return convergedenergy/2 # two surfaces

    def gsf_energy_curve(self,energyindex=2):
        return self.energy_sub(energyindex)
        
    def sf_energy(self,energyindex=2):
        res = self.energy_sub(energyindex)
        npoints = res.shape[0]
        idxenergy = (npoints+1)//2 # midpoint
        return res[idxenergy,1]
        
    def usf_energy(self,energyindex=2):
        res = self.energy_sub(energyindex)
        return np.max(res[:,1])
    
    def energy_sub(self,energyindex,plane='xz'):
        data = self.minconfig
        area = self.area(plane=plane)
        energyfac = energyfacdict[self.units]
        energyvec = energyfac*(data[:,energyindex] - data[0,energyindex])/area
        npoints = np.shape(energyvec)[0]
        xvec = np.arange(npoints)
        return np.column_stack((xvec,energyvec))
