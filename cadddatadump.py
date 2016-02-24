import numpy as np
import cadd_plot as cdplot
import dislocation_marker as dm
import itertools
from collections import OrderedDict

atomradius = 0.3
dislobjsize = 4000
keydict = {'def': 'deformed_positions', 'undef': 'undeformed_positions', 'disp': 'displacements',
           'connect': 'fe_elements', 'types': 'types', 'dislpos': 'dislocation_positions', 'centrosymmetry': 'centro',
           'disltypes': 'dislocation_attributes', 'sources': 'source_positions', 'obstacles': 'obstacle_positions',
           'theta': 'slipsys_angles'}
typedict = {'atoms': 1, 'fenodes': 0, 'pad': -1, 'interface': 2}
typecol = 1
isyscol = 0
bsgncol = 1
dislobjcolor = 'b'
atomcolor = 'k'

class CADDDataDump(object):
    def __init__(self,datadict):
        self.datadict = datadict
        self.currentplot = {}
    
    # properties
    def from_key(self,key):
        keycurr = keydict[key]
        try:
            return self.datadict[keycurr]
        except KeyError:
            return None
    
    def array_from_key(self,key):
        val = self.from_key(key)
        if val is not None:
            if len(val.shape) == 0:
                val = np.array([val])
            elif len(val.shape) == 1: # convert vector to 2d array
                val = val[np.newaxis,:]
        return val
    
    @property
    def connect(self):
        connect = self.array_from_key('connect')
        if connect is not None: 
            return connect - 1 # off by one issue for indexing (Python vs. Fortran)
    
    @property
    def types(self):
        return self.array_from_key('types')
    
    @property
    def dislpos(self):
        return self.array_from_key('dislpos')
        
    @property
    def disltypes(self):
        return self.array_from_key('disltypes')
    
    @property
    def sources(self):
        return self.array_from_key('sources')
    
    @property
    def obstacles(self):
        return self.array_from_key('obstacles')
        
    @property
    def slipsystheta(self):
        return self.array_from_key('theta').ravel()       
    
    def theta_from_isys(self,isys):
        return self.slipsystheta[isys-1] # off by one issue for indexing (Python vs. Fortran)
        
    def all_positions(self,deformed):
        if deformed:
            return self.deformedpos
        else:
            return self.undeformedpos
    
    @property
    def undeformedpos(self):
        res = self.from_key('undef')
        if res is None:
            try:
                res = self.from_key('def')[:,:2] - self.from_key('disp')
            except TypeError:
                res = None
        return res
    
    @property
    def deformedpos(self):
        res = self.from_key('def')
        if res is None:
            try:
                res = self.from_key('undef') + self.from_key('disp')
            except TypeError:
                res = None
        return res[:,:2]
    
    def segments_from_elements(self,deformed):
        """Generate line segments that constitute fe elements"""
        allpositions = self.all_positions(deformed)
        segments = []
        connect = self.connect
        nodesperelement = connect.shape[1]
        for element in connect:
            for i, node in enumerate(element):
                posnode = tuple(allpositions[node,:2])
                idxnext = (i+1)%nodesperelement
                nodenext = element[idxnext]
                posnodenext = tuple(allpositions[nodenext,:2])
                segments.append([posnode,posnodenext])
        return segments
    
    # dislocation positions, markers
    def gen_disl_pos_and_markers(self,aspectratio,width=1):
        """Generate dislocation positions and markers by looping over each
        combination of slip system and sign"""
        isysvec = np.unique(self.disltypes[:,isyscol])
        bsgnvec = np.unique(self.disltypes[:,bsgncol])
        markers, pos = [], []
        for (isys,bsgn) in itertools.product(isysvec,bsgnvec):
            theta = self.theta_from_isys(isys)
            marker = dm.gen_marker(theta,bsgn,width,aspectratio,degoption=False)
            markers.append(marker)
            xy = self.get_relevant_disl(isys,bsgn)
            pos.append(xy)
        return pos, markers
        
    def get_relevant_disl(self,isys,bsgn):
        idx = (self.disltypes[:,isyscol] == isys) & (self.disltypes[:,bsgncol] == bsgn)
        return self.dislpos[idx,:]
            
    def positions_from_label(self,deformed,label):
        """From label (e.g. 'atoms'), get positions of that object"""
        allpositions = self.all_positions(deformed)
        typenum = typedict[label]
        idxtype = (self.types[:,typecol] == typenum)
        return allpositions[idxtype,:]
    
    def gen_nodes(self,label,size,edgecolor,facecolor,deformed):
        positions = self.positions_from_label(deformed,label)
        return cdplot.Points(positions,size,edgecolor,facecolor)
            
    def gen_nodes_fixed(self,label,radius,edgecolor,facecolor,deformed):
        positions = self.positions_from_label(deformed,label)
        return cdplot.PointsFixed(positions,radius,edgecolor,facecolor)
    
    def gen_atoms_plot(self,radius=atomradius,edgecolor=atomcolor,facecolor='b',deformed=True): # solid circles
        return self.gen_nodes_fixed('atoms',radius,edgecolor,facecolor,deformed)
        
    def gen_pad_plot(self,radius=atomradius,edgecolor=atomcolor,facecolor='w',deformed=True): # empty circles
        return self.gen_nodes_fixed('pad',radius,edgecolor,facecolor,deformed)
        
    def gen_interface_plot(self,radius=atomradius,edgecolor=atomcolor,facecolor=atomcolor,deformed=True): # solid circles
        return self.gen_nodes_fixed('interface',radius,edgecolor,facecolor,deformed)
        
    def gen_fe_elements_plot(self,linewidth=3,linecolor='r',deformed=True):        
        segments = self.segments_from_elements(deformed)
        return cdplot.Edges(segments,linewidth,linecolor)
        
    def gen_disl_plot(self,aspectratio=1,size=dislobjsize,color=dislobjcolor):
        if self.dislpos.size:
            pos, markers = self.gen_disl_pos_and_markers(aspectratio)
            return cdplot.MultipleMarkers(pos,markers,color=color,size=size)

    def gen_sources_plot(self,size=dislobjsize,edgecolor=dislobjcolor,facecolor='w'):
        if self.sources.size:
            return cdplot.Points(self.sources,size,edgecolor,facecolor)

    def gen_obstacles_plot(self,size=dislobjsize,edgecolor=dislobjcolor,facecolor=dislobjcolor):
        if self.obstacles.size:
            return cdplot.Points(self.obstacles,size,edgecolor,facecolor)
        
    def gen_all_plot(self,attrlist):
        attrdict = {'atoms': self.gen_atoms_plot,
                    'pad': self.gen_pad_plot,
                    'interface': self.gen_interface_plot,
                    'feelements': self.gen_fe_elements_plot,
                    'sources': self.gen_sources_plot,
                    'obstacles': self.gen_obstacles_plot,
                    'disl': self.gen_disl_plot}
        res = []
        for attr in attrlist:
            res.append(attrdict[attr]())
        self.currentplot = res
        
    def add_to_axes(self,ax):
        for obj in self.currentplot:
            if obj is not None:
                obj.plot(ax)