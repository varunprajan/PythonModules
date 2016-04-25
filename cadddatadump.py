import numpy as np
import cadd_plot as cdplot
import dislocation_marker as dm
import itertools

class PlotStyle(object):
    def __init__(self,obj):
        self.atoms = obj.gen_atoms_plot
        self.pad = obj.gen_pad_plot
        self.interface = obj.gen_interface_plot
        self.feelements = obj.gen_fe_elements_plot
        self.sources = obj.gen_sources_plot
        self.obstacles = obj.gen_obstacles_plot
        self.disl = obj.gen_disl_plot
        
    @classmethod
    def centro(cls,obj):
        instance = cls(obj)
        instance.atoms = obj.gen_atoms_centro_plot
        return instance

class CADDDataDump(object):
    # default plot styles/objects
       
    # atoms
    _ATOMRADIUS = 0.3
    _INTATOMCOLOR = 'k'
    _ATOMCOLOR = 'b'
    _ATOMPOINT = cdplot.Point(radius=_ATOMRADIUS,edgecolor=_INTATOMCOLOR,facecolor=_ATOMCOLOR,zorder=0) # solid circle, different border
    _PADPOINT = cdplot.Point(radius=_ATOMRADIUS,edgecolor=_INTATOMCOLOR,facecolor='w',zorder=0) # empty circle
    _INTERFACEPOINT = cdplot.Point(radius=_ATOMRADIUS,edgecolor=_INTATOMCOLOR,facecolor=_INTATOMCOLOR,zorder=0) # solid circle
    
    # fe
    _MESHLINECOLOR = 'r'
    _MESHLINEWIDTH = 2
    _MESHLINEZORDER = -0.5
    _MESHLINE = cdplot.Edge(width=_MESHLINEWIDTH,color=_MESHLINECOLOR,zorder=_MESHLINEZORDER)
    
    # dd stuff 
    _DISLZORDER = 1 # (show above fe mesh)
    _DISLOBJCOLOR = 'b'
    _DISLOBJSIZE = 400
    _DISLPOINT = cdplot.Point(color=_DISLOBJCOLOR,size=_DISLOBJSIZE,zorder=_DISLZORDER)
    _SOURCEPOINT = cdplot.Point(facecolor='w',edgecolor=_DISLOBJCOLOR,size=_DISLOBJSIZE,zorder=_DISLZORDER) # empty circle
    _OBSPOINT = cdplot.Point(facecolor=_DISLOBJCOLOR,edgecolor=_DISLOBJCOLOR,size=_DISLOBJSIZE,zorder=_DISLZORDER) # solid circle

    # centrosymmetry     
    _CENTROCMAPNAME = 'OrRd'
    _CENTROCRANGE = [0,0.6]
    _CENTROCMAP = cdplot.ColorMap(name=_CENTROCMAPNAME,crange=_CENTROCRANGE)

    def __init__(self,datadict):
        self.datadict = datadict
        self.currentplot = {}
    
    # properties
    def from_key(self,key):
        keydict = {'def': 'deformed_positions', 'undef': 'undeformed_positions', 'disp': 'displacements',
                   'connect': 'fe_elements', 'types': 'types', 'dislpos': 'dislocation_positions', 'centrosymmetry': 'centro',
                   'disltypes': 'dislocation_attributes', 'sources': 'source_positions', 'obstacles': 'obstacle_positions',
                   'theta': 'slipsys_angles'}
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
    def centro(self):
        return self.from_key('centrosymmetry')
    
    @property
    def types(self):
        return self.array_from_key('types')

    def type_num(self,label):
        typedict = {'atoms': 1, 'fenodes': 0, 'pad': -1, 'interface': 2}
        return typedict[label]
    
    @property
    def nodetypes(self):
        return self.types[:,1]
        
    @property
    def dislpos(self):
        return self.array_from_key('dislpos')
        
    @property
    def disltypes(self):
        return self.array_from_key('disltypes')
        
    @property
    def dislisys(self):
        return self.disltypes[:,0]
    
    @property
    def dislbsgn(self):
        return self.disltypes[:,1]
    
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
                res = self.from_key('def')[:,:2] - self.from_key('disp') # just x and y, not z
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
        return res[:,:2] # just x and y, not z
    
    # fe line segments
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
        isysvec = np.unique(self.dislisys)
        bsgnvec = np.unique(self.dislbsgn)
        markers, pos = [], []
        for (isys,bsgn) in itertools.product(isysvec,bsgnvec):
            theta = self.theta_from_isys(isys)
            marker = dm.gen_marker(theta,bsgn,width,aspectratio,degoption=False)
            markers.append(marker)
            xy = self.get_relevant_disl(isys,bsgn)
            pos.append(xy)
        return pos, markers
        
    def get_relevant_disl(self,isys,bsgn):
        idx = (self.dislisys == isys) & (self.dislbsgn == bsgn)
        return self.dislpos[idx,:]
            
    def positions_from_label(self,deformed,label):
        """From label (e.g. 'atoms'), get positions of that object"""
        allpositions = self.all_positions(deformed)
        idxtype = (self.nodetypes == self.type_num(label))
        return allpositions[idxtype,:]
    
    # plots
    
    # atoms
    def gen_nodes_fixed(self,label,radius,edgecolor,facecolor,zorder,deformed):
        positions = self.positions_from_label(deformed,label)
        return cdplot.PointsFixed(positions,radius,edgecolor,facecolor,zorder)
    
    def gen_atoms_centro_plot(self,point=None,cmap=None,deformed=True):
        cmap = self._CENTROCMAP if cmap is None else cmap
        colors = cdplot.gen_cmap(self.centro,cmap)
        idxatoms = self.nodetypes != self.type_num('fenodes')
        idxrealatoms = self.nodetypes[idxatoms] == self.type_num('atoms')
        colors = colors[idxrealatoms,:]
        point = self._ATOMPOINT if point is None else point
        return self.gen_nodes_fixed('atoms',radius=point.radius,edgecolor=point.edgecolor,facecolor=colors,zorder=point.zorder,deformed=deformed)
    
    def gen_atoms_plot_sub(self,label,point,deformed):
        return self.gen_nodes_fixed(label,radius=point.radius,edgecolor=point.edgecolor,facecolor=point.facecolor,zorder=point.zorder,deformed=deformed)
    
    def gen_atoms_plot(self,point=None,deformed=True):
        point = self._ATOMPOINT if point is None else point
        return self.gen_atoms_plot_sub('atoms',point,deformed)
        
    def gen_pad_plot(self,point=None,deformed=True):
        point = self._PADPOINT if point is None else point
        return self.gen_atoms_plot_sub('pad',point,deformed)
        
    def gen_interface_plot(self,point=None,deformed=True):
        point = self._INTERFACEPOINT if point is None else point
        return self.gen_atoms_plot_sub('interface',point,deformed)
    
    # fe elements
    def gen_fe_elements_plot(self,meshline=None,deformed=True):        
        segments = self.segments_from_elements(deformed)
        line = self._MESHLINE if meshline is None else meshline
        return cdplot.Edges(segments,linewidth=line.width,linecolor=line.color,zorder=line.zorder)
    
    # dd stuff
    def gen_disl_plot(self,point=None,aspectratio=1):
        if self.dislpos.size:
            pos, markers = self.gen_disl_pos_and_markers(aspectratio)
            point = self._DISLPOINT if point is None else point
            return cdplot.MultipleMarkers(pos,markers,color=point.color,size=point.size,zorder=point.zorder)

    def gen_disl_obj_plot_sub(self,positions,point):
        return cdplot.Points(positions,size=point.size,edgecolor=point.edgecolor,facecolor=point.facecolor,zorder=point.zorder) # empty circles
            
    def gen_sources_plot(self,point=None):
        if self.sources.size:
            point = self._SOURCEPOINT if point is None else point
            return self.gen_disl_obj_plot_sub(self.sources,point)

    def gen_obstacles_plot(self,point=None):
        if self.obstacles.size:
            point = self._OBSPOINT if point is None else point
            return self.gen_disl_obj_plot_sub(self.obstacles,point)
    
    # all
    def gen_all_plot(self,attrlist,style=None):
        plotstyle = self.plot_style(style)
        self.currentplot = {}
        for attr in attrlist:
            self.currentplot[attr] = getattr(plotstyle,attr)()
        
    def plot_style(self,style=None):
        if style is None:
            return PlotStyle(self)
        elif style == 'centro':
            return PlotStyle.centro(self)
        else:
            raise ValueError('Undefined plot style')          
        
    def add_to_axes(self,ax):
        for key, obj in self.currentplot.items():
            if obj is not None:
                obj.plot(ax)
                