import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mymath as Mmath
from mymath import threewise, pairwise
import cadd_io as cdio

TOL = 1e-10

class Line(object):    
    def __init__(self,startpoint,endpoint):
        self.startpoint = startpoint
        self.endpoint = endpoint
        col = self.determine_direction()
        self.col = col
        self.colother = 1 - col
        self.linelims = self.gen_line_lims()
        self.nodenums = None
        
    def determine_direction(self,tol=TOL):
        """Determine which index (0 or 1 --- i.e. x or y)
        is constant for line.
        I.e. if 0, line is given by x = const."""
        for col in range(2):
            startcoord, endcoord = self.startpoint[col], self.endpoint[col]
            if np.abs(startcoord - endcoord) < tol:
                return col
        raise Exception('Line is not parallel to axes')
    
    def gen_line_lims(self):
        linelims = [self.startpoint[self.colother],self.endpoint[self.colother]]
        return np.array(sorted(linelims)) # linelims[0] < linelims[1]
    
    def get_nnodes(self):
        if self.nodenums is not None:
            return len(self.nodenums)
    
    def line_length(self):
        return self.linelims[1] - self.linelims[0]
        
    def gen_n_points(self,n):
        """Generate n evenly spaced points along line"""
        res = np.empty((n,2))
        for col in range(2):
            startcoord, endcoord = self.startpoint[col], self.endpoint[col]
            res[:,col] = np.linspace(startcoord,endcoord,n)
        return res
        
    def is_backwards(self):
        return self.startpoint[self.colother] > self.endpoint[self.colother]
        
    def stack_endpoints(self,pts):
        return np.vstack((self.startpoint,pts,self.endpoint))
        
    def dilate_points(self,pts):
        colother = self.colother
        coordsold = pts[:,colother]
        linelimsold = min(coordsold), max(coordsold)
        linelimsnew = self.linelims
        coordsnew = Mmath.rescale_coords(coordsold,linelimsold,linelimsnew)
        ptsnew = np.copy(pts)
        ptsnew[:,colother] = coordsnew
        return ptsnew
        
    def shift_points(self,pts):
        col = self.col
        newvalue = self.startpoint[col]
        ptsnew = np.copy(pts)
        ptsnew[:,col] = newvalue
        return ptsnew
    
class Box(object):
    def __init__(self,bounds):
        """Box comprises four lines: right, top, left, bottom
        bounds is [xmin,xmax,ymin,ymax]"""
        self.bounds = bounds
        self.lines = self.gen_box_lines()
    
    @classmethod
    def init_with_nodes(cls,bounds,nnodesvec,nodes):
        obj = cls(bounds)
        obj.create_nodes(nnodesvec,nodes)
        return obj      
        
    def create_nodes(self,nnodesvec,nodes):
        for n, line in zip(nnodesvec,self.lines):
            line.nodenums = nodes.gen_fe_nodes_along_line(line,n)
            
    def populate_nodes(self,nodes):
        for line in self.lines:
            line.nodenums = nodes.search_for_nodes_along_line(line)
    
    @property
    def all_nodes(self):
        res = []
        for line in self.lines:
            res.extend(line.nodenums)
        return list(set(res)) # remove uniques
            
    def box_dimensions(self):
        xmin, xmax, ymin, ymax = self.bounds
        return [xmax - xmin, ymax - ymin]
        
    def node_spacing(self):
        '''Assumes nodes are spaced evenly'''
        Lx, Ly = self.box_dimensions()
        nnodesx = self.lines[1].get_nnodes()
        nnodesy = self.lines[0].get_nnodes()
        return [Lx/(nnodesx-1),Ly/(nnodesy-1)] # offset of 1 to find dist. *between* nodes
    
    def gen_box_lines(self):
        """Create the four lines that the box comprises"""
        bounds = self.bounds
        bottomright = bounds[[1,2]]
        topright = bounds[[1,3]]
        topleft = bounds[[0,3]]
        bottomleft = bounds[[0,2]]
        return [Line(bottomright,topright),Line(topright,topleft),Line(topleft,bottomleft),Line(bottomleft,bottomright)]
        
    def previous_line(self,edgenum):
        idx = (edgenum+3)%4
        return self.lines[idx]
        
    def next_line(self,edgenum):
        idx = (edgenum+1)%4
        return self.lines[idx]
    
    @classmethod
    def from_boxold(cls,boxold):
        boundsnew = boxold.gen_boundsnew_unif()
        return cls(boundsnew)
    
    def gen_boundsnew(self,xfac,yfac):
        xlim = self.bounds[0:2]
        ylim = self.bounds[2:4]
        xlimnew = [xfac*x for x in xlim]
        ylimnew = [yfac*y for y in ylim]
        return np.array(xlimnew + ylimnew)

    def gen_boundsnew_unif(self):
        xmin, xmax, ymin, ymax = self.bounds
        sx, sy = self.node_spacing()
        return np.array([xmin-sx,xmax+sx,ymin-sy,ymax+sy])
    
class Mesh(object):
    fenodetype = np.array([1,0,0])

    def __init__(self,nodesperel=3,nodes=None):
        self.elements = np.zeros((0,nodesperel)).astype(int)
        if nodes is None:
            nodes = Nodes()
        self.nodes = nodes
    
    def add_element_from_nodes(self,nodelist):
        if not self.nodes.is_element_ccw(nodelist):
            nodelist = nodelist[::-1]
        newelement = np.array(nodelist)
        self.elements = np.vstack((self.elements,newelement))

    def add_elements_from_list(self,elementlist):
        for element in elementlist:
            self.add_element_from_nodes(element)

    # element/mesh generation
    def add_ring(self,ring):
        ring.populate_box_new(self)
        elements = self.generate_elements(ring)
        self.add_elements_from_list(elements)
          
    def generate_elements(self,ring):
        elements = []
        for lineold, linenew in zip(ring.boxold.lines,ring.boxnew.lines):
            nodesold, nodesnew = lineold.nodenums, linenew.nodenums
            elements += ring.elements_between(self,nodesold,nodesnew)
        elements += ring.elements_spare(self)
        return elements
    
    # input/output
    @property
    def elements_dump(self):
        return self.elements + 1 # Python vs. Fortran indexing issue
    
    def write_user_inputs_all(self,elpref,nodespref,subdir=''):
        self.write_user_inputs_elements(elpref,subdir)
        self.nodes.write_user_inputs_all(nodespref,subdir)
    
    def write_user_inputs_elements(self,elpref,subdir):
        filename = '{}.connect'.format(subdir+elpref)
        cdio.write_array(self.elements_dump,filename)
        
    def write_dump_all(self,filename):
        with open(filename,'w') as f:
            cdio.write_array_sub_dump(self.elements_dump,f,'fe_elements')
            cdio.write_array_sub_dump(self.nodes.types,f,'types')
            cdio.write_array_sub_dump(self.nodes.xy,f,'undeformed_positions')
            disp = np.zeros(self.nodes.xy.shape)
            cdio.write_array_sub_dump(disp,f,'displacements')
            
class MeshQuad(Mesh):
    def __init__(self,nodesperel=4,nodes=None):
        super().__init__(nodesperel,nodes)
        
    def elements_extend(self,nodesold,nodesnew):
        elements = []
        for nodesoldchunk, nodesnewchunk in zip(pairwise(nodesold),pairwise(nodesnew[1:-1])): # first and last will be taken care of in spare elements
            node1old, node2old = nodesoldchunk
            node1new, node2new = nodesnewchunk
            elements.append([node1new,node2new,node2old,node1old])
        return elements
    
    def elements_std(self,nodesold,nodesnew):
        elements = []
        for nodesoldchunk, nodesnewchunk in zip(pairwise(nodesold),pairwise(nodesnew)):
            node1old, node2old = nodesoldchunk
            node1new, node2new = nodesnewchunk
            elements.append([node1new,node2new,node2old,node1old])
        return elements
            
class MeshTri(Mesh):
    def __init__(self,nodesperel=3,nodes=None):
        super().__init__(nodesperel,nodes)
        
    def elements_collapse(self,nodesold,nodesnew):
        elements = []
        for nodesoldchunk, nodenew in zip(threewise(nodesold),nodesnew[1:-1]):
            node1old, node2old, node3old = nodesoldchunk
            elements.append([node1old,node2old,nodenew])
            elements.append([node2old,node3old,nodenew])
        for nodeold, nodesnewchunk in zip(nodesold[::2],pairwise(nodesnew)):
            node1new, node2new = nodesnewchunk
            elements.append([node1new,node2new,nodeold])
        return elements
        
    def elements_std(self,nodesold,nodesnew):
        elements = []
        for nodesoldchunk, nodesnewchunk in zip(pairwise(nodesold),pairwise(nodesnew)):
            node1old, node2old = nodesoldchunk
            node1new, node2new = nodesnewchunk
            elements.append([node1new,node2new,node1old])
            elements.append([node2new,node2old,node1old])
        return elements
        
    def elements_extend(self,nodesold,nodesnew):
        elements = []
        for nodesoldchunk, nodesnewchunk in zip(pairwise(nodesold),pairwise(nodesnew[1:-1])):
            node1old, node2old = nodesoldchunk
            node1new, node2new = nodesnewchunk
            elements.append([node1new,node2new,node1old])
            elements.append([node2new,node2old,node1old])
        # corners
        elements.append([nodesnew[0],nodesnew[1],nodesold[0]])
        elements.append([nodesnew[-2],nodesnew[-1],nodesold[-1]])
        return elements
            
class MeshRectangle(MeshQuad):
    def __init__(self,bounds,nnodesvec,nodesperel=4,nodes=None,build=True):
        super().__init__(nodesperel,nodes)
        if build:
            self.gen_rectangle_mesh(bounds,nnodesvec)
            
    @classmethod
    def init_from_spacing(cls,bounds,spacing,nodesperel=4,nodes=None,build=True):
        xspacing, yspacing = spacing
        nx = int((bounds[1] - bounds[0])/xspacing) + 1
        ny = int((bounds[3] - bounds[2])/yspacing) + 1
        return cls(bounds,[nx,ny],nodesperel,nodes,build)
        
    def gen_rectangle_mesh(self,bounds,nnodesvec):
        xmin, xmax, ymin, ymax = bounds
        nx, ny = nnodesvec
        lines = []
        for ycoord in np.linspace(ymin,ymax,ny):
            startpoint = np.array([xmin,ycoord])
            endpoint = np.array([xmax,ycoord])
            line = Line(startpoint,endpoint)
            line.nodenums = self.nodes.gen_fe_nodes_along_line(line,nx)
            lines.append(line)
        for lineprev, linenext in pairwise(lines): # iterate over prev, current
            elements = self.elements_std(lineprev.nodenums,linenext.nodenums)
            self.add_elements_from_list(elements)

class Ring(object):
    def __init__(self,boxold,boxnew=None):
        self.boxold = boxold
        if boxnew is None:
            boxnew = Box.from_boxold(boxold)
        self.boxnew = boxnew
       
    def populate_box_new(self,mesh):
        for lineold, linenew in zip(self.boxold.lines,self.boxnew.lines):
            linenew.nodenums = self.gen_nodes_new(mesh.nodes,lineold,linenew)
        
    def elements_spare(self,mesh):
        return []
    
    def elements_between(self,mesh,nodesold,nodesnew):
        return mesh.elements_std(nodesold,nodesnew) # default method, can be overridden

class RingUniform(Ring):        
    def gen_nodes_new(self,nodes,lineold,linenew):
        n = lineold.get_nnodes()
        return nodes.gen_fe_nodes_along_line(linenew,n)
        
class RingExtend(Ring):
    def gen_nodes_new(self,nodes,lineold,linenew):
        ptsold = nodes.get_points_from_line(lineold)
        ptsshifted = linenew.shift_points(ptsold)
        ptsnew = linenew.stack_endpoints(ptsshifted)
        return nodes.gen_fe_nodes_from_points(ptsnew)

    def elements_between(self,mesh,nodesold,nodesnew):
        return mesh.elements_extend(nodesold,nodesnew)
        
    def elements_spare(self,mesh):
        elements = []
        if isinstance(mesh,MeshQuad):
            for i, (lineold, linenew) in enumerate(zip(self.boxold.lines,self.boxnew.lines)):
                nodesold = lineold.nodenums
                nodesnew = linenew.nodenums
                nodesnextnew = self.boxnew.next_line(i).nodenums
                elements.append([nodesnew[-2],nodesnew[-1],nodesnextnew[1],nodesold[-1]])
        return elements

class RingDilate(Ring):        
    def gen_nodes_new(self,nodes,lineold,linenew):
        ptsold = nodes.get_points_from_line(lineold)
        ptsshifted = linenew.shift_points(ptsold)
        ptsnew = linenew.dilate_points(ptsshifted)
        return nodes.gen_fe_nodes_from_points(ptsnew)

class RingCollapse(Ring):        
    def gen_nodes_new(self,nodes,lineold,linenew):
        ptsold = nodes.get_points_from_line(lineold)
        ptsshifted = linenew.shift_points(ptsold)
        nold = ptsshifted.shape[0]
        if Mmath.is_odd(nold):
            ptsshifted = ptsshifted[1::2,:] # every other node
            ptsnew = linenew.stack_endpoints(ptsshifted)
            return nodes.gen_fe_nodes_from_points(ptsnew)
        else:
            raise ValueError('Cannot apply collapse with even number of nodes')

    def elements_between(self,mesh,nodesold,nodesnew):
        return mesh.elements_collapse(nodesold,nodesnew)
            
class RingCollapseMixed(Ring):            
    def gen_nodes_new(self,nodes,lineold,linenew):
        ptsold = nodes.get_points_from_line(lineold)
        ptsshifted = linenew.shift_points(ptsold)
        nold = ptsshifted.shape[0]
        nnew = (nold + 1 + 2)//2
        if Mmath.is_odd(nnew): # collapse
            ptsshifted = ptsshifted[1::2,:] # every other node
        ptsnew = linenew.stack_endpoints(ptsshifted)
        return nodes.gen_fe_nodes_from_points(ptsnew)

    def elements_between(self,mesh,nodesold,nodesnew):
        if len(nodesnew) == len(nodesold) + 2:
            return mesh.elements_extend(nodesold,nodesnew)
        else:
            return mesh.elements_collapse(nodesold,nodesnew)
            
class Nodes(object):
    defaultfetype = [1,0,0] # material 1, fe node, free
    
    def __init__(self,xy=None,types=None):
        if xy is None:
            self.xy = np.zeros((0,2)).astype(float)
        else:
            self.xy = xy
        if types is None:
            self.types = np.zeros((self.xy.shape[0],3)).astype(int)
            self.types[:,:] = [1,1,0] # material 1, atom, free
    
    def write_user_inputs_all(self,nodespref,subdir=''):
        self.write_position(nodespref,subdir)
        self.write_types(nodespref,subdir)
        
    def write_position(self,nodespref,subdir):
        filename = '{}.xy'.format(subdir+nodespref)
        n = self.xy.shape[0]
        xyz = np.zeros((n,3))
        xyz[:,:2] = self.xy
        cdio.write_array(xyz,filename)        

    def write_types(self,nodespref,subdir):
        filename = '{}.types'.format(subdir+nodespref)
        cdio.write_array(self.types,filename)    
    
    @property
    def nnodes(self):
        return self.xy.shape[0]
        
    def get_points(self,nodelist):
        return self.xy[nodelist,:]
        
    def get_points_from_line(self,line):
        return self.get_points(line.nodenums)
        
    def add_node(self,coords,nodetype):
        self.xy = np.vstack((self.xy,coords))
        self.types = np.vstack((self.types,nodetype))
        return self.nnodes-1 # count starts at zero
    
    def set_node_bc(self,nodelist,bcnum):
        self.types[nodelist,2] = bcnum
    
    def set_node_type(self,nodelist,typenum):
        self.types[nodelist,1] = typenum
       
    def set_material_number(self,nodelist,mnum):
        self.types[nodelist,0] = mnum
        
    def set_atoms(self,nodelist,typenum):
        self.set_node_type(nodelist,1)
        
    def set_pad(self,nodelist):
        self.set_node_type(nodelist,-1)
        self.set_node_bc(nodelist,3) # pad atoms are completely fixed for atomistic step
        
    def set_interface(self,nodelist):
        self.set_node_type(nodelist,2)
        
    def set_fenodes(self,nodelist):
        self.set_node_type(nodelist,0)
    
    def closest_point(self,pt):
        nodenum = self.closest_node(pt)
        if nodenum is not None:
            return self.xy[nodenum,:]
    
    def closest_node(self,pt):
        if self.nnodes > 0:
            dist = np.linalg.norm(self.xy - pt, axis=1)
            return np.argmin(dist)

    def closest_point_on_line(self,pt,line):
        nodenum = self.closest_node_on_line(pt,line)
        return self.xy[nodenum,:]
        
    def closest_node_on_line(self,pt,line):
        nodenums = self.search_for_nodes_along_line(line)
        dist = np.linalg.norm(self.xy[nodenums,:] - pt, axis=1)
        idx = np.argmin(dist)
        return nodenums[idx]
    
    def check_fe_node_existence(self,pt,tol=TOL):
        '''Returns two values: already, nodeclosestnum
        already indicates whether point is already in nodes
        nodenum is number of node, if point is already in nodes'''
        nodenum = self.closest_node(pt)
        if nodenum is not None:
            ptclosest = self.xy[nodenum,:]
            nodeatomtype = self.types[nodenum,1]
            already = (np.linalg.norm(pt - ptclosest) < tol) & (nodeatomtype in [0,2])
        else:
            already = False
        return already, nodenum

    def gen_fe_nodes_from_points(self,pts,nodetype=None):
        '''Given a list of points, return node numbers corresponding to those points
        If node does not exist, create it'''
        nodes = []
        if nodetype is None:
            nodetype = Nodes.defaultfetype
        for pt in pts:
            already, nodenum = self.check_fe_node_existence(pt)
            if not already: # add node to list
                nodenum = self.add_node(pt,nodetype)
            nodes.append(nodenum)
        return nodes
    
    def gen_fe_nodes_along_line(self,line,n):
        '''Return node numbers for n equally spaced nodes along line,
        generating nodes if they do not already exist'''
        pts = line.gen_n_points(n)
        return self.gen_fe_nodes_from_points(pts)

    def search_for_nodes_along_line(self,line,tol=TOL):
        '''Return numbers of nodes (already existing) along line
        Nodes are sorted in direction of travel'''
        dist = np.abs(self.xy[:,line.col] - line.startpoint[line.col])
        idx = dist < tol
        if not np.isinf(line.linelims[0]):
            idx = idx & ((self.xy[:,line.colother] - line.linelims[0]) > -tol)
        if not np.isinf(line.linelims[1]):
            idx = idx & ((self.xy[:,line.colother] - line.linelims[1]) < tol)
        nodenums = np.where(idx)[0]
        idxsort = np.argsort(self.xy[nodenums,line.colother])
        nodenums = nodenums[idxsort]
        if line.is_backwards():
            nodenums = nodenums[::-1]
        return nodenums
    
    def is_element_ccw(self,nodelist):
        pts = self.get_points(nodelist)
        return Mmath.get_signed_area(pts) > 0
        
    # generate interface, pad
    def gen_interface_and_pad(self,proposedbounds):
        interfaceboxactual = self.gen_interface(proposedbounds)
        self.gen_pad_atoms(interfaceboxactual)
        interfaceboxactual.populate_nodes(self)
        return interfaceboxactual
    
    def gen_atom_box_bounds(self,proposedbounds):
        xdist = proposedbounds[1] - proposedbounds[0]
        ydist = proposedbounds[3] - proposedbounds[2]
        xvec = np.array([xdist,0.])
        yvec = np.array([0.,ydist])
        bottomleft = proposedbounds[[0,2]]
        bottomright = proposedbounds[[1,2]]
        topleft = proposedbounds[[0,3]]
        bottomleftactual = self.closest_point(bottomleft)
        bottomrightguess = bottomleftactual + xvec
        topleftguess = bottomleftactual + yvec
        linebottom = Line(bottomleftactual,bottomrightguess + xvec) # make line long enough
        lineleft = Line(bottomleftactual,topleftguess + yvec) # make line long enough
        bottomrightactual = self.closest_point_on_line(bottomrightguess,linebottom)
        topleftactual = self.closest_point_on_line(topleftguess,lineleft)
        actualbounds = [bottomleftactual[0],bottomrightactual[0],
                        bottomleftactual[1],topleftactual[1]]
        return np.array(actualbounds)
        
    def gen_interface(self,proposedbounds):
        actualbounds = self.gen_atom_box_bounds(proposedbounds)
        actualbox = Box(actualbounds)
        for line in actualbox.lines:
            nodenums = self.search_for_nodes_along_line(line)
            self.set_interface(nodenums)
        return actualbox
    
    def nodes_outside_box(self,box,tol=TOL):
        xmin, xmax, ymin, ymax = box.bounds
        idx1 = self.xy[:,0] < xmax + tol
        idx2 = self.xy[:,0] > xmin - tol
        idx3 = self.xy[:,1] < ymax + tol
        idx4 = self.xy[:,1] > ymin - tol
        idx = np.all([idx1,idx2,idx3,idx4],axis=0)
        idx = np.logical_not(idx)
        return np.where(idx)[0]
    
    def gen_pad_atoms(self,interfaceboxactual):
        nodenums = self.nodes_outside_box(interfaceboxactual)
        self.set_pad(nodenums)
            