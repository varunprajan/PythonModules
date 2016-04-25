import numpy as np
import myio as Mio
import mymath as Mmath
import itertools
import os

filetagtry = 'files_'

# get subroutines/functions
def gener_funcs_in_file(filename):
    with open(filename,'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(('function','subroutine')):
                yield parse_func_line(line,1)
            elif line.startswith('recursive subroutine'):
                yield parse_func_line(line,2)
                
def parse_func_line(line,idxword):
    words = line.split()
    wordwithname = words[idxword]
    idx = wordwithname.find('(')
    return wordwithname[:idx]

# compare fortran input files
def are_all_files_equivalent(fidir,restartdir,suffix='.0.restart'):
    for fname1 in os.listdir(fidir):
        fpath1 = os.path.join(fidir,fname1)
        fpath2 = os.path.join(restartdir,fname1+suffix)
        if not are_fortran_files_equivalent(fpath1,fpath2):
            return False
    return True

def are_fortran_files_equivalent(fname1,fname2,tol=1e-7):
    f1 = gener_fortran_file(fname1)
    f2 = gener_fortran_file(fname2)
    for num1, num2 in itertools.zip_longest(f1,f2):
        if (num1 != num2):
            try:
                if abs(num1 - num2) > tol: # floating point
                    return False
            except TypeError:
                return False
    return True

def gener_fortran_file(file):
    with open(file,'r') as f:
        for line in f:
            line = line.strip()
            if line:
                strnums = line.split()
                for strnum in strnums:
                    yield Mio.coerce_string(strnum)                

# write arrays
def write_array(array,file):
    with open(file,'w') as f:
        write_array_sub(array,f)

def write_array_sub(array,f,padblank=True):
    fmt = ['{0} ']*array.shape[1]
    for line in array.tolist():
        for num, fmtnum in zip(line, fmt):
            try:
                f.write(fmtnum.format(num))
            except ValueError:
                f.write(fmtnum.format(int(num)))
        f.write('\n')
    if padblank:
        f.write('\n')
    
def write_array_sub_dump(array,f,arrayname,endtag='end'):
    f.write(arrayname + ':')
    f.write('\n')
    write_array_sub(array,f,padblank=False)
    f.write('end')
    f.write('\n')
    
def reshape_for_writing(array):
    """Reshape 1D array into 2D array with a single column"""
    if len(array.shape) == 1:
        array = array[:,np.newaxis]
    return array

# read files
def read_from_dump(dumpfile,filetag='',endtag='end'):
    datadict = {}
    with open(dumpfile,'r') as f:
        for key, value, _ in yield_file_items(f,filetag): # filetag is not used here
            genernew = Mio.yield_until(f,endtag)
            value = Mio.my_read_array(genernew)
            datadict[key] = value
    return datadict

def yield_file_items(f,filetag,commentmarker='#'):
    for line in Mio.yield_no_comments(f,comment=commentmarker):
        words = line.strip().split(':')
        key, value = words[0].strip(), words[1].strip()
        isfile = key.startswith(filetag)
        if isfile: # strip filetag
            key = Mio.findKey(key,filetag,'')
        yield (key, value, isfile)

def read_input(filename,filetag=filetagtry,endtag='end',subdir=''):
    datadict = {}
    with open(subdir+filename,'r') as f:
        for key, value, isfile in yield_file_items(f,filetag):
            if not value:
                genernew = Mio.yield_until(f,endtag)
                if isfile: # read list of structures
                    value = [read_input(file,filetag,endtag,subdir) for file in genernew]
                else: # read array
                    value = Mio.my_read_array(genernew)
            else:
                if isfile:
                    try: # read file with array
                        value = Mio.my_read_array(subdir+value)
                    except ValueError: # read structure file
                        value = read_input(value,filetag,endtag,subdir)
                else:
                    value = Mio.coerce_string(value)
            datadict[key] = value
    return datadict
        
# python dictionary to user input files
def write_input(datadict,filename,filetag=filetagtry,endtag='end',subdir=''):
    endwithspace = endtag + '\n\n'
    with open(subdir+filename,'w') as f:
        filepref = Mio.getFilePrefix(filename)
        for key, val in sorted(datadict.items()):
            if isinstance(val,np.ndarray):
                f.write('{0}: \n'.format(key))
                Mio.my_write_array(f,val)
                f.write(endwithspace)
            elif isinstance(val,dict):
                dictfilename = key + '.' + filepref
                f.write('{0}{1}: {2}\n\n'.format(filetag,key,dictfilename))
                write_input(val,dictfilename,filetag,endtag,subdir)
            elif isinstance(val,list):
                f.write('{0}{1}: \n'.format(filetag,key))
                for i, subdict in enumerate(val):
                    dictfilename = '{0}_{1}.{2}'.format(key,i,filepref)
                    f.write(dictfilename + '\n')
                    write_input(subdict,dictfilename,filetag,endtag,subdir)
                f.write(endwithspace)
            else:
                f.write('{0}: {1}\n\n'.format(key,val))
    
# python dictionary to lammps dump file
def writeLammpsDump(datadict,filename,timestep):
    xyarray = datadict['deformed_positions']
    typesarray = datadict['types']
    natoms = xyarray.shape[0]
    with open(filename,'w') as f:
        f.write('ITEM: TIMESTEP\n')
        f.write('{0}\n'.format(timestep))
        f.write('ITEM: NUMBER OF ATOMS\n')
        f.write('{0}\n'.format(natoms))
        f.write('ITEM: BOX BOUNDS ss ss pp\n') # may need to change this, if PBCS are implemented
        boxbounds = getBoxBounds(xyarray)
        Mio.my_write_array(f,boxbounds)
        f.write('ITEM: ATOMS id type xs ys zs\n')
        arrayout = getArrayOut(xyarray,typesarray,boxbounds)
        Mio.my_write_array(f,arrayout,fmt=['{:d} ']*2+['{:f} ']*3)
        # Mio.my_write_array(f,filename,arrayout,fmt='%0i %0i %0f %0f %0f')
        
def getBoxBounds(xyarray):
    boxbounds = np.empty((3,2))
    for i in range(2):
        vec = xyarray[:,i]
        boxbounds[i,:] = [np.min(vec), np.max(vec)]
    boxbounds[-1,:] = [-0.5, 0.5] # z-bounds (irrelevant)
    return boxbounds
    
def getArrayOut(xyarray,typesarray,boxbounds):
    natoms = xyarray.shape[0]
    xyscaled = np.empty((natoms,3))
    xyarraynew = np.zeros((natoms,3))
    xyarraynew[:,:2] = xyarray[:,:2]
    for i in range(3):
        vec = xyarraynew[:,i]
        bounds = boxbounds[i,:]
        xyscaled[:,i] = Mmath.rescale_coords(vec,bounds,[0,1])
    atomvec = np.arange(1,natoms+1,1)
    return np.column_stack((atomvec,typesarray[:,0],xyscaled))
       