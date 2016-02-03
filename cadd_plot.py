import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mplc
import matplotlib.patches as mplp
import numpy as np

def z_scaled(z,zbounds=None):
    if zbounds is None:
        zbounds = np.min(z), np.max(z)
    zmin, zmax = zbounds
    zscaled = (z - zmin)/(zmax - zmin)
    return np.clip(zscaled,0,1)

def gen_cmap(z,name,zbounds=None):
    zscaled = z_scaled(z,zbounds)
    cmap = plt.get_cmap(name)
    return cmap(zscaled)

class Points(object):
    def __init__(self,positions,size,edgecolor,facecolor):
        self.offsets = list(zip(positions[:,0],positions[:,1]))
        self.sizes = len(self.offsets)*[size] # all have same size
        self.edgecolor = edgecolor
        self.facecolor = facecolor
        
    def gen_collection(self):
        return mplc.CircleCollection(transOffset=ax.transData,offsets=self.offsets,sizes=self.sizes,edgecolor=self.edgecolor,facecolor=self.facecolor)
        
    def plot(self,ax):
        col = self.gen_collection()
        ax.add_collection(col)
        
class PointsFixed(object):
    def __init__(self,positions,radius,edgecolor,facecolor):
        self.patches = self.construct_patches(positions,radius)
        self.edgecolor = edgecolor
        self.facecolor = facecolor
    
    def gen_collection(self):
        return mplc.PatchCollection(patches=self.patches,edgecolor=self.edgecolor,facecolor=self.facecolor)
    
    def plot(self,ax):
        col = self.gen_collection()    
        ax.add_collection(col)
        
    def construct_patches(self,positions,radius):
        return [mplp.Circle(position,radius) for position in positions]
        
class Edges(object):
    def __init__(self,segments,linewidth,linecolor):
        self.segments = segments
        self.linewidth = linewidth
        self.colors = linecolor
        
    def gen_collection(self,ax):
        return mplc.LineCollection(transOffset=ax.transData,segments=self.segments,linewidth=self.linewidth,colors=self.colors)
        
    def plot(self,ax):
        col = self.gen_collection(ax)
        ax.add_collection(col)
        
class MultipleMarkers(object):
    def __init__(self,xy,marker,color,size):
        self.xy = xy
        self.marker = marker
        self.color = color
        self.size = size
        
    def plot(self,ax):
        for xy, marker in zip(self.xy,self.marker):
            ax.scatter(x=xy[:,0],y=xy[:,1],marker=marker,s=self.size,color=self.color)