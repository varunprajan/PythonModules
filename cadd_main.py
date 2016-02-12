import numpy as np
import myio as Mio
import mymath as Mmath
import cadd_io as cdio
import cadddatadump as cddump
import my_plot as myplot
import os
import itertools

filetag = 'files_'
endtag = 'end'

class Simulation(object):
    """Top-level class for CADD simulation. Contains simname, simtype ('cadd','cadd_nodisl','fe', etc.)
    and directories for inputs and outputs."""
    def __init__(self,simtype,simname,userpath='',fortranpath='',dumppath='',readinput=True,simfile=None,nfematerials=None):
        self.simtype = simtype
        self.simname = simname
        self.userpath = userpath
        self.fortranpath = fortranpath
        self.dumppath = dumppath
        if readinput:
            self.read_user_input_data(simfile,nfematerials)
    
    # read inputs
    def read_user_input_data(self,simfile=None,nfematerials=None):
        """Populate data by reading it from a set of *user* simulation files in directory userpath."""
        if simfile is None:
            simfile = self.simfile
        self.data = CADDData(self.simtype,nfematerials=nfematerials) # (re-)initialize
        self.data.read_user_inputs(simfile,subdir=self.userpath)
    
    @property
    def simfile(self,suffix='.inp'):
        """Name of main user input file"""
        return self.simname + suffix

    # write inputs (fortran)
    def write_fortran_all(self):
        """Write data to a set of *fortran* simulation files in directory fortranpath."""
        for structname, val in self.data.__dict__.items():
            filepath = self.fortran_input_file_path(structname)
            with open(filepath,'w') as f:
                val.write_fortran(f)
    
    def fortran_input_file_path(self,structname):
        """Returns path of fortran input file corresponding to structure name"""
        return os.path.join(self.fortranpath,self.fortran_input_file_name(structname))
    
    def fortran_input_file_name(self,structname):
        """Returns name of fortran input file corresponding to structure name
        E.g. 'materials' -> '[simname]_materials'"""
        return '{}_{}'.format(self.simname,structname)
            
    # plot dump
    def plot_dump_from_increment(self,increment,fignum=1,**kwargs):
        """Plots dump file corresponding to a particular increment number"""
        filepath = self.dump_file_path(increment)
        return self.plot_dump_from_file(filepath,fignum=fignum,**kwargs)
        
    def plot_dump_from_file(self,filepath,fignum=1,pretty=True,**kwargs):
        """Plots dump file corresponding to a particular dump file path"""
        dumpdict = cdio.read_from_dump(filepath) # read in dump dictionary from file
        cadddump = cddump.CADDDataDump(dumpdict) # initialize cadddump object
        objs = self.objects_to_plot() # figure out objects to plot (atoms, elements, etc.), based on simtype
        cadddump.gen_all_plot(objs) # generate plot objects for these attributes
        fig = myplot.my_plot(cadddump,fignum=fignum) # plot them!
        if pretty:
            myplot.pretty_figure(fig,aspect=1,ticksize=None,**kwargs)
        return fig

    def objects_to_plot(self):
        """Returns objects (atoms, feelements, disl, etc.) to plot, based on simulation type"""
        objs = ['atoms'] # if not present, there is no issue: they will be empty arrays, not plotted
        if self.simtype in ['fe','dd','cadd','cadd_nodisl']:
            objs.extend(['feelements'])
        if self.simtype in ['dd','cadd']:
            objs.extend(['disl','sources','obstacles'])
        if self.simtype in ['cadd_nodisl', 'cadd']:
            objs.extend(['pad','interface'])
        return objs       
    
    def dump_file_path(self,increment):
        """Returns path of dump file corresponding to increment"""
        return os.path.join(self.dumppath,self.dump_file_name(increment))
    
    def dump_file_name(self,increment,suffix='.dump'):
        """Returns name of dump file corresponding to increment
        E.g. increment = 100 -> '[simname].100.dump'"""
        return '{}.{}{}'.format(self.simname,increment,suffix)

class Struct(object):
    """Generic structure container. Can perform reads from dictionary, writes to fortran file,
    and checks of data validity"""
    def __init__(self):
        pass

    def __repr__(self):
        return 'Struct, with name {0}'.format(self.__class__.__name__)
    
    # checks
    def check_data(self):
        """Checks that data is 'correct' (correct type, dimensions) by performing a check on self, as well
        as on each member in self"""
        self.struct_check()
        for key, val in self.__dict__.items():
            val.check_data()
        
    def struct_check(self):
        pass

    def check_equal_rows(self,attributelist):
        """Checks if selected attributes of self (of type ArrayData) all have values
        with the same number of rows. For instance, for dislocations, the arrays containing dislocation positions,
        signs, cuts, etc. should all have the same number of rows: ndisl
        If all arrays have zero rows, this is declared explicitly."""
        nrowslist = [getattr(self,attr).nrows for attr in attributelist]
        if len(set(nrowslist)) != 1: # i.e. dissimilar entries
            raise ValueError('Nrows is not consistent across arrays for structure {0}'.format(self))

    # read
    def read_from_dict(self,structdict):
        """Read items from dictionary"""
        for attr, val in structdict.items():
            item = getattr(self,attr)
            item.read_from_dict(val)
    
    # write    
    def write_fortran(self,f):
        """Write attributes to fortran file with descriptor f, in order of lower case key
        (in CADD code, items are read-in in order of lower case key)"""
        for key, val in sorted(self.__dict__.items(),key=lambda s: s[0].lower()): # sort by lower case key
            val.write_fortran(f)
        
class CADDData(Struct):
    """Second-level class for CADD simulation. Contains all
    of the various structures (nodes, materials, compute, etc.)
    for the specific simulation"""            
    def __init__(self,simtype,nfematerials=None):        
        # general
        self.nodes = Nodes()
        self.materials = ListStruct(Material)
        self.misc = Misc()
        self.groups = ListStruct(Group)
        self.compute = Compute()
            
        # atomistic
        if simtype in ['atomistic','cadd','cadd_nodisl']:
            self.potentials = ListStruct(Potential)
            self.interactions = Interactions()
            self.neighbors = Neighbors()
            self.damping = Damping()
                
        # fe
        if simtype in ['fe','dd','cadd','cadd_nodisl']:
            self.feelements = ListStruct(FEElement)
                
        # dd
        if simtype in ['dd','cadd']:
            self.dislmisc = DislMisc(nfematerials=nfematerials) # if not None, will use default constants for nmaxdisl, etc.
            self.disl = ListStruct(Dislocations)
            self.escapeddisl = ListStruct(EscapedDislocations)
            self.ghostdisl = ListStruct(GhostDislocations)
            self.obstacles = ListStruct(Obstacles)
            self.sources = ListStruct(Sources)
            self.slipsys = ListStruct(SlipSystem)
                
        # cadd
        if simtype == 'cadd':
            self.detection = Detection()
    
    def read_user_inputs(self,mainuserinputfile,subdir):
        datadict = cdio.read_input(mainuserinputfile,subdir=subdir) # read user inputs into dictionary
        self.read_from_dict(datadict) # read dictionary into self.data
        self.check_data() # checks validity of inputs
    
    def struct_check(self):
        self.check_interactions()
        self.check_nfematerials()
        
    def check_interactions(self):
        try:
            nmaterials = self.materials.num_structs
            npotentials = self.potentials.num_structs
            self.interactions.check_all(nmaterials,npotentials)
        except AttributeError:
            pass
            
    def check_nfematerials(self):
        n = self.get_nfematerials()
        if len(n) > 1:
            raise ValueError('Inconsistent number of fematerials across structures')
            
    def get_nfematerials(self):
        n = set()
        for attr in ['feelements','disl','escapeddisl','ghostdisl','obstacles','sources','slipsys']:
            try:
                n.add(getattr(self,attr).num_structs)
            except AttributeError:
                pass
        try:
            n.add(self.dislmisc.nfematerials)
        except AttributeError:
            pass
        return n   

class ListStruct(object):
    """Third-level class for CADD simulation. Contains information corresponding
    to a list of structures (e.g. material information, where there is a structure
    for each material)"""
    def __init__(self,subclass):
        """Subclass is the class of the individual structure"""
        self.structlist = []
        self.subclass = subclass
    
    def __repr__(self):
        return 'ListStruct with subclass {0}'.format(self.subclass)
    
    @property
    def num_structs(self):
        return len(self.structlist)

    def add_struct(self,newstruct):
        self.structlist.append(newstruct)
    
    # check
    def check_data(self):
        for struct in self.structlist:
            struct.check_data()
    
    # read
    def read_from_dict(self,data):
        """Read items from dictionary, then check inputs"""
        for structdict in data:
            newinstance = self.subclass()
            newinstance.read_from_dict(structdict)
            self.add_struct(newinstance)
    
    def write_fortran(self,f):
        """Write data in all structures in structlist to fortran file"""
        f.write('{} \n'.format(self.num_structs))
        for struct in self.structlist:
            struct.write_fortran(f)
            f.write('\n')
            
class Data(object):
    """Contains "simple" data (floats, integers, strings). For instance, to store a finite element name
    such as 'CPE4', we would use:
    self.val = 'CPE4'
    self.name = 'FE element name'
    self.desiredtype = str"""
    def __init__(self,val,name,desiredtype):
        self.val = val
        self.name = name
        self.desiredtype = desiredtype

    def __repr__(self):
        return '{0}: {1}'.format(self.name,self.val)
        
    def check_data(self):
        self.check_type()
        
    def check_type(self):
        """Check if data value is of the desired type"""
        if not isinstance(self.val,self.desiredtype):
            raise ValueError('{} must be of type {}'.format(self,self.desiredtype))
    
    def write_fortran(self,f):
        """Write data to fortran file, with file descriptor f"""
        f.write('{} \n'.format(self.val))
    
    def read_from_dict(self,val):
        self.val = val
 
class ArrayData(object):
    """Contains array data. For instance, to store the finite element connectivity, we would use:
    self.val = [connectarray]
    self.desiredtype = int
    self.name = 'FE element connectivity'
    self.desiredshape = [None,[3,4]] # for triangular or rectangular elements"""
    def __init__(self,val,name,desiredtype,desiredshape):
        self.val = val
        self.desiredtype = desiredtype
        self.name = name
        self.desiredshape = desiredshape

    def __repr__(self):
        return 'Numpy array "{0}"'.format(self.name,self.val)
    
    @property
    def shape(self):
        return self.val.shape
        
    @property
    def nrows(self):
        return self.val.shape[0]
    
    # checks
    def check_data(self):
        self.ensure_type()
        self.check_shape()
    
    def ensure_type(self):
        """Attempt to convert array to desired type (e.g. int to float)
        If this attempt fails, throw an error"""
        try:
            self.val = self.val.astype(self.desiredtype)
        except AttributeError:
            message1 = 'Type conversion failed\n'
            message2 = '{0} must be numpy array of type {1}'.format(self,self.desiredtype)
            raise ValueError(message1+message2)
        
    def check_shape(self):
        """Check that array has desired shape.
        If not, throw an error"""
        dimname = {0: 'rows', 1: 'columns'}
        shapeactual = self.shape
        shapedesired = self.desiredshape
        for dim, (m, mdesired) in enumerate(zip(shapeactual,shapedesired)):
            if not self.has_desired_m(m,mdesired):
                dimnamestr = ' or '.join([str(m) for m in mdesired])
                message = '{0} should have {1} {2}'.format(self,dimnamestr,dimname[dim])
                raise ValueError(message)
                
    def has_desired_m(self,m,mdesired):
        if mdesired is not None:
            return m in mdesired
        else:
            return True
    
    # read
    def read_from_dict(self,val):
        """Read value of array; coerce dimensionality if necessary.
        (This makes check_shape logic much simpler)"""
        self.val = self.coerce_dimensionality(val)
        
    def coerce_dimensionality(self,val):
        """Coerce array to have correct dimensionality:
        1) Empty array -> 1D or 2D arrays with zero rows
        2) Single number -> 1D array with a single entry
        3) Single row -> 2D array with a single row"""
        if not val.size: # no entries
            val = self.zero_row_array()
        if not val.shape: # single entry
            val = np.array([val])
        if len(self.desiredshape) == 2: # single row
            if len(val.shape) == 1:
                val = val[np.newaxis,:]
        return val
        
    def zero_row_array(self):
        """If array is empty, create 1D or 2D empty array with zero rows, according to desired shape"""
        try:
            ncolslist = self.desiredshape[1]
            n = ncolslist[0] # irrelevant which entry of desiredshape we use
            return np.zeros((0,n)).astype(self.desiredtype)
        except IndexError: # 1D arrays
            return np.zeros((0,)).astype(self.desiredtype)
    
    # write
    def write_fortran(self,f):
        """Write array to be read by fortran: first, the header, containing the size;
        second, the array itself. 1D arrays are written in column format, so they are first reshaped."""
        array = cdio.reshape_for_writing(self.val)
        m, n = array.shape
        if n == 1: # 1D array
            f.write('{0} \n'.format(m))
        else:
            f.write('{0} {1} \n'.format(m,n))
        cdio.write_array_sub(array,f)
            
class Nodes(Struct):
    _nposcheck = [3,7]
    _ntypescheck = [3]
    
    def __init__(self,posn=None,types=None):
        self.posn = ArrayData(posn,'Node positions',float,[None,Nodes._nposcheck])
        self.types = ArrayData(types,'Node types',int,[None,Nodes._ntypescheck])
            
    def struct_check(self):
        self.check_equal_rows(['posn','types'])
        self.pad_zeros()
        
    def pad_zeros(self):
        """If only xyz are supplied, set columns 4 - 7 (displacements, velocities)
        of posn equal to zero"""
        posnarray = self.posn.val
        mpos, npos = posnarray.shape
        if npos == Nodes._nposcheck[0]:
            nposextra = Nodes._nposcheck[1] - Nodes._nposcheck[0]
            zeropadarray = np.zeros((mpos,nposextra))
            self.posn.val = np.column_stack((posnarray,zeropadarray))

class FEElement(Struct):
    _nconnectcheck = {'CPE3': [3], 'CPE4': [4]}
    
    def __init__(self,elname=None,mnum=None,connect=None):
        self.elname = Data(elname,'FE element name',str)
        self.mnum = Data(mnum,'Material number',int)
        self.connect = ArrayData(connect,'FE element connectivity',int,[None,None])
        
    def struct_check(self):
        self.check_elements()

    def check_elements(self):
        """Checks if # nodes/element in connect matches # expected based on element name"""
        try:
            nelnodesdesired = FEElement._nconnectcheck[self.elname.val]
            if self.connect.val.shape[1] not in nelnodesdesired:
                raise ValueError('Inconsistent nodes per element for elname {0}'.format(self.elname.val))
        except KeyError:
            raise ValueError('Undefined element type')
            
class Material(Struct):
    _melconstcheck = [3]
    _nelconstcheck = [3]

    def __init__(self,burgers=None,disldrag=None,dislvmax=None,elconst=None,lannih=None,lattice=None,mass=None,mname=None,rho=None):
        self.burgers = Data(burgers,'Burgers vector',float)
        self.disldrag = Data(disldrag,'Dislocation drag coefficient',float)
        self.dislvmax = Data(dislvmax,'Max dislocation velocity',float)
        self.elconst = ArrayData(elconst,'Elastic constants',float,[Material._melconstcheck,Material._nelconstcheck])
        self.lannih = Data(lannih,'Annihilation distance',float)
        self.lattice = Data(lattice,'Lattice name',str)
        self.mass = Data(mass,'Atomic mass',float)
        self.mname = Data(mname,'Material name',str)
        self.rho = Data(rho,'Density',float)
        
class Potential(Struct):
    _npotcheck = [3]

    def __init__(self,forcecutoff=None,pname=None,pottable=None):
        self.forcecutoff = Data(forcecutoff,'Potential force cutoff',float)
        self.pname = Data(pname,'Potential name',str)
        self.pottable = ArrayData(pottable,'Potential table',float,[None,Potential._npotcheck])
        
class Group(Struct):
    def __init__(self,gname=None,members=None):
        self.gname = Data(gname,'Group name',str)
        self.members = ArrayData(members,'Group members',int,[None])
        
class Interactions(Struct):
    _ntablecheck = [3]

    def __init__(self,table=None):
        self.table = ArrayData(table,'Interaction table',int,[None,Interactions._ntablecheck])
    
    def check_all(self,nmaterials,npotentials):
        self.check_missing_interactions(nmaterials)
        self.check_wrong_potential(npotentials)
    
    def check_missing_interactions(self,nmaterials):
        """Checks if all interactions between materials i and j are present"""
        interactions = self.table.val[:,:2]
        for i, j in itertools.product(range(1,nmaterials+1),repeat=2):
            if not (Mmath.row_in_array([i,j],interactions) or
                    Mmath.row_in_array([j,i],interactions)):
                raise ValueError('Missing interaction between {0} and {1}'.format(i,j))
                
    def check_wrong_potential(self,npotentials):
        """Checks whether potentials exist for all interactions"""
        potentials = self.table.val[:,2]
        if np.any(potentials > npotentials) or np.any(potentials <= 0):
            raise ValueError('Missing potential')
        
class Neighbors(Struct):
    def __init__(self,checkdisp=None,delay=None,every=None,images=None,Lz=None,skin=None):
        self.checkdisp = Data(checkdisp,'Increments for reneighboring check',int)
        self.delay = Data(delay,'Increments for reneighboring delay',int)
        self.every = Data(every,'Increments for reneighboring every',int)
        self.images = Data(images,'Images in z-direction',int)
        self.Lz = Data(Lz,'Lz, out-of-plane distance',float)
        self.skin = Data(skin,'Skin distance',float)

class Dislocations(Struct):
    _nposncheck = [2]
    _nlocalposcheck = [2]

    def __init__(self,cut=None,posn=None,sgn=None,slipsys=None):
        self.cut = ArrayData(cut,'Dislocation branch cut',int,[None])
        self.posn = ArrayData(posn,'Dislocation positions',float,[None,Dislocations._nposncheck])
        self.sgn = ArrayData(sgn,'Dislocation signs',int,[None])
        self.slipsys = ArrayData(slipsys,'Dislocation slip system',int,[None])
    
    def struct_check(self):
        self.check_equal_rows(['cut','posn','sgn','slipsys'])      
        
class GhostDislocations(Struct):
    _nposncheck = [2]

    def __init__(self,cut=None,posn=None,sgn=None,slipsys=None):
        self.cut = ArrayData(cut,'Ghost dislocation branch cut',int,[None])
        self.posn = ArrayData(posn,'Ghost dislocation positions',float,[None,GhostDislocations._nposncheck])
        self.sgn = ArrayData(sgn,'Ghost dislocation sign',int,[None])
        self.slipsys = ArrayData(slipsys,'Ghost dislocation slip system',int,[None])
    
    def struct_check(self):
        self.check_equal_rows(['cut','posn','sgn','slipsys'])
        
class EscapedDislocations(Struct):
    _nposncheck = [2]

    def __init__(self,cut=None,posn=None,region=None,sgn=None,slipsys=None):
        self.cut = ArrayData(cut,'Escaped dislocation branch cut',int,[None])
        self.posn = ArrayData(posn,'Escaped dislocation positions',float,[None,EscapedDislocations._nposncheck])
        self.region = ArrayData(region,'Region of escaped dislocation',int,[None])
        self.sgn = ArrayData(sgn,'Escaped dislocation sign',int,[None])
        self.slipsys = ArrayData(slipsys,'Escaped dislocation slip system',int,[None])
    
    def struct_check(self):
        self.check_equal_rows(['cut','posn','region','sgn','slipsys'])
        
class Obstacles(Struct):
    _nposncheck = [2]

    def __init__(self,posn=None,slipsys=None,taucr=None):
        self.posn = ArrayData(posn,'Obstacle positions',float,[None,Obstacles._nposncheck])
        self.slipsys = ArrayData(slipsys,'Obstacle slip system',int,[None])
        self.taucr = ArrayData(taucr,'Obstacle critical shear stress',float,[None])
    
    def struct_check(self):
        self.check_equal_rows(['posn','slipsys','taucr'])
        
class Sources(Struct):
    _nposncheck = [2]

    def __init__(self,posn=None,slipsys=None,taucr=None,tnuc=None):
        self.posn = ArrayData(posn,'Source positions',float,[None,Sources._nposncheck])
        self.slipsys = ArrayData(slipsys,'Source slip system',int,[None])
        self.taucr = ArrayData(taucr,'Source critical shear stress',float,[None])
        self.tnuc = ArrayData(tnuc,'Source nucleation time',float,[None])
    
    def struct_check(self):
        self.check_equal_rows(['posn','slipsys','taucr','tnuc'])
        
class SlipSystem(Struct):
    _norigincheck = [2]

    def __init__(self,nslipplanes=None,origin=None,space=None,theta=None):
        self.nslipplanes = ArrayData(nslipplanes,'Number of planes in slip system',int,[None])
        self.origin = ArrayData(origin,'Origin of slip system',float,[None,SlipSystem._norigincheck])
        self.space = ArrayData(space,'Spacing of slip planes',float,[None])
        self.theta = ArrayData(theta,'Angle of slip system',float,[None])
    
    def struct_check(self):
        self.check_equal_rows(['origin','nslipplanes','space','theta'])
        
class DislMisc(Struct):
    _nmaxdisl = 1000
    _nmaxdislslip = 40
    _nmaxescapeddisl = 1000
    _nmaxghostdisl = 100
    _nmaxobsslip = 20
    _nmaxsrcslip = 20

    def __init__(self,gradientcorrection=1,nmaxdisl=None,nmaxdislslip=None,nmaxescapeddisl=None,
                 nmaxghostdisl=None,nmaxobsslip=None,nmaxsrcslip=None,nfematerials=None):
        if nfematerials is not None:
            if nmaxdisl is None:
                nmaxdisl = DislMisc._nmaxdisl*np.ones((nfematerials,)).astype(int)
            if nmaxdislslip is None:
                nmaxdislslip = DislMisc._nmaxdislslip*np.ones((nfematerials,)).astype(int)
            if nmaxescapeddisl is None:
                nmaxescapeddisl = DislMisc._nmaxescapeddisl*np.ones((nfematerials,)).astype(int)
            if nmaxghostdisl is None:
                nmaxghostdisl = DislMisc._nmaxghostdisl*np.ones((nfematerials,)).astype(int)
            if nmaxobsslip is None:
                nmaxobsslip = DislMisc._nmaxobsslip*np.ones((nfematerials,)).astype(int)
            if nmaxsrcslip is None:
                nmaxsrcslip = DislMisc._nmaxsrcslip*np.ones((nfematerials,)).astype(int)
        self.gradientcorrection = Data(gradientcorrection,'Is there a gradient correction for the DD velocity?',int)
        self.nmaxdisl = ArrayData(nmaxdisl,'Maximum number of dislocations per fe material',int,[None])
        self.nmaxdislslip = ArrayData(nmaxdislslip,'Maximum number of dislocations per slip plane',int,[None])
        self.nmaxescapeddisl = ArrayData(nmaxescapeddisl,'Maximum number of escaped dislocations per fe material',int,[None])
        self.nmaxghostdisl = ArrayData(nmaxghostdisl,'Maximum number of ghost dislocations per fe material',int,[None])
        self.nmaxobsslip = ArrayData(nmaxobsslip,'Maximum number of obstacles per slip plane',int,[None])
        self.nmaxsrcslip = ArrayData(nmaxsrcslip,'Maximum number of sources per slip plane',int,[None])
    
    @property
    def nfematerials(self):
        return self.nmaxdisl.size
        
    def struct_check(self):
        self.check_equal_rows(['nmaxdisl','nmaxdislslip','nmaxescapeddisl','nmaxghostdisl','nmaxobsslip','nmaxsrcslip'])
    
class Misc(Struct):
    def __init__(self,dumpincrement=0,incrementcurr=0,increments=None,iscrackproblem=0,potstyle='null',restartincrement=0,timestep=None):
        self.dumpincrement = Data(dumpincrement,'Increments for dump write',int)
        self.incrementcurr = Data(incrementcurr,'Current increment',int)
        self.increments = Data(increments,'Increments for simulation',int)
        self.iscrackproblem = Data(iscrackproblem,'Are we simulating a crack problem?',int)
        self.potstyle = Data(potstyle,'Potential style',str)
        self.restartincrement = Data(restartincrement,'Increments for restart write',int)
        self.timestep = Data(timestep,'Timestep',float)
        
class Damping(Struct):
    def __init__(self,flag=None,gamma=0.0,gname=None):
        self.flag = Data(flag,'Damping flag',int)
        self.gamma = Data(gamma,'Damping coefficient',float)
        self.gname = Data(gname,'Damping group name',str)
        
class Detection(Struct):
    _nedgescheck = [2]
    _gname = 'temp'

    def __init__(self,bandtype=None,damp=None,interfaceedges=None,mdnincrements=None,mdtimestep=None,mnumfe=None,params=None,passdistance=None,ycrack=None):
        self.bandtype = Data(bandtype,'Detection band type',str)
        if damp is None:
            damp = Damping(flag=1,gname=Detection._gname)
        self.damp = damp
        self.interfaceedges = ArrayData(interfaceedges,'Detection interface edges',int,[None,Detection._nedgescheck])
        self.mdnincrements = Data(mdnincrements,'Number of increments for damped MD after passing',int)
        self.mdtimestep = Data(mdtimestep,'Time step for damped MD after passing',float)
        self.mnumfe = Data(mnumfe,'FE material adjacent to detection',int)
        if params is None:
            params = np.empty((0,))
        self.params = ArrayData(params,'Parameters for detection band',float,[None])
        self.passdistance = Data(passdistance,'Dislocation pass distance',float)
        
class Compute(Struct):
    def __init__(self,centro=None):
        if centro is None:
            centro = ComputeData(np.empty((0,)))
        self.centro = centro
        # add more computes here...

class ComputeData(Struct):
    def __init__(self,params,active=0,gname='all'):
        self.active = Data(active,'Is compute active?',int)
        self.gname = Data(gname,'Group for compute',str)
        self.params = ArrayData(params,'Parameters for compute',float,[None])           
