import numpy as np
import cadd_mesh2 as cdmesh
import cadd
import mymath as Mmath
import copy

class CrackMesh(object):
    def __init__(self,ferings,lx,ly,padx,pady,r0,ydisp=None):
        self.ferings = ferings
        self.lx = lx
        self.ly = ly
        self.padx = padx
        self.pady = pady
        self.r0 = r0
        if ydisp is None:
            ydisp = r0*np.sqrt(3)/2 # seems to work for hex mesh, but the appropriate value of ydisp might in general depend on the mesh
        self.ydisp = ydisp
        self.mesh = self.gen_atoms()
        self.interfacebox = self.mesh.nodes.gen_interface_and_pad(np.array([-self.lx/2,self.lx/2,-self.ly/2,self.ly/2]))
        self.outerbox = self.gen_mesh_approx()
        self.ycrack = self.get_ycrack()
        self.cracknodes = self.get_cracknodes()
    
    def gen_atoms(self):
        lxtot = self.lx + self.padx # make box slightly larger to accommodate pad
        lytot = self.ly + self.pady
        xy = cadd.simple_hex_lattice(lxtot,lytot,r0=self.r0) 
        meshnodes = cdmesh.Nodes(xy)
        return cdmesh.MeshTri(nodes=meshnodes)
    
    def gen_mesh_approx(self):
        boxold = copy.copy(self.interfacebox)
        for i in range(self.ferings):
            nnodesvec = [line.get_nnodes() for line in boxold.lines]
            if i < 10:
                if i%3 == 0:
                    ring = cdmesh.RingCollapseMixed(boxold)
                else:
                    ring = cdmesh.RingExtend(boxold)
            else:
                ring = cdmesh.RingCollapseMixed(boxold)
            self.mesh.add_ring(ring)
            boxold = ring.boxnew
        return boxold
    
    def get_ycrack(self):
        ypos = self.mesh.nodes.xy[:,1]
        idx = np.argmin(np.abs(ypos))
        return ypos[idx]
    
    def get_cracknodes(self):
        crackline = cdmesh.Line([-np.inf,self.ycrack],[0.0,self.ycrack])
        return self.mesh.nodes.search_for_nodes_along_line(crackline)
    
    def modify_mesh(self):
        self.mapping = self.modify_nodes()
        self.correct_elements()
        self.fudge_nodes()
    
    def modify_nodes(self):
        """Duplicate crack face nodes, moving one set up by ydisp"""
        mapping = {}
        ydispvec = np.array([0,self.ydisp])
        ydispvecatom = np.array([self.r0/2,self.ydisp])
        defaultfetype = cdmesh.Nodes.defaultfetype
        for nodenum in self.cracknodes:
            posn = self.mesh.nodes.xy[nodenum,:]
            nodetype = self.mesh.nodes.types[nodenum,:]
            isinterfaceatom = nodetype[1] == 2
            isfenode = nodetype[1] == 0
            if isinterfaceatom:
                posnnew = posn + ydispvecatom
                nodenew = self.mesh.nodes.closest_node(posnnew)
            elif isfenode:
                posnnew = posn + ydispvec
                nodenew = self.mesh.nodes.add_node(posnnew,defaultfetype)
            if isinterfaceatom or isfenode:
                self.mesh.nodes.types[nodenew,:] = nodetype # retain type and bc of old node
                mapping[nodenum] = nodenew
        return mapping
    
    def correct_elements(self):
        """Correct the fe elements on the crack plane"""
        # this modifies the element while it's being looped over, which is dangerous
        for i, element in enumerate(self.mesh.elements):
            common = np.intersect1d(element,self.cracknodes,assume_unique=True)
            if common.size: # nodenum belongs to element
                pts = self.mesh.nodes.get_points(element)
                yelementcenter = np.mean(pts,axis=0)[1]
                if Mmath.same_sign(yelementcenter-self.ycrack,self.ydisp): # only create a new element if the element center is on one side of ycrack
                    for nodeold in common:
                        nodenew = self.mapping[nodeold]
                        np.place(element,element==nodeold,nodenew) # replace old num with new num
                    
    def fudge_nodes(self,fac=1/20):
        """Fudge nodes slightly to ensure pad atoms are within an element"""
        yfudge = self.ydisp*fac
        ydispvec = np.array([0,np.copysign(yfudge,self.ydisp)])
        for nodeold, nodenew in self.mapping.items():
            isfenodeold = self.mesh.nodes.types[nodeold,1] == 0
            if isfenodeold: # interface nodes remain fixed
                self.mesh.nodes.xy[nodeold,:] += ydispvec
                self.mesh.nodes.xy[nodenew,:] -= ydispvec
                
    def crack_atom_edges(self,fac):
        idxinterface = self.mesh.nodes.types[self.cracknodes,1] == 2
        interfacenodeold = self.cracknodes[np.where(idxinterface)[0]][0]
        interfacenodenew = self.mapping[interfacenodeold]
        def crack_edges(interfacenode):            
            def atom_edges():
                pos = self.mesh.nodes.xy[interfacenode,:]
                crackline = cdmesh.Line(pos,[0.0,pos[1]])
                return self.mesh.nodes.search_for_nodes_along_line(crackline)
                
            def fe_edges():
                xposint, yposint = self.mesh.nodes.xy[interfacenode,:]
                xposfe = self.outerbox.bounds[0]
                xposfe, yposfe = self.mesh.nodes.closest_point(np.array([xposfe,yposint]))
                xposleft = xposint - fac*self.r0
                crackline = cdmesh.Line([xposleft,yposfe],[xposint,yposfe])
                return self.mesh.nodes.search_for_nodes_along_line(crackline)
                
            cracknodes = np.append(fe_edges(),atom_edges()) + 1 # offset by 1 (Python vs. Fortran indexing)
            return np.array(list(Mmath.pairwise(cracknodes)))

        return np.vstack((crack_edges(interfacenodeold),crack_edges(interfacenodenew)))
        
    
