import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.spatial as spsp
import Ccontourplot9 as C

def contourPlot(xmat,ymat,datamat,contourspacing=None,axislims=None,datalims=None,scaletype=None,filloption=1,tickoption=1,figuresize=[8,6],font='Arial',extend='both',nticks=5,ncontours=25,fignum=1):
    plt.figure(fignum,figsize=figuresize)
    plt.clf()
    preProcessPlot(font)
    if scaletype is not None:
        datamat = scaleData(scaletype,datamat)
    if datalims is None:
        datalims = [np.nanmin(datamat), np.nanmax(datamat)]
    if contourspacing is not None:
        ncontours = round((datalims[1]-datalims[0])/contourspacing) + 1
    levels = np.linspace(datalims[0], datalims[1], ncontours)
    if axislims is not None:
        (xmat,ymat,datamat) = enforceContourBounds(xmat,ymat,datamat,axislims)
    if filloption == 1:
        plt.contourf(xmat,ymat,datamat,levels,extend=extend)
    else:
        plt.contour(xmat,ymat,datamat,levels,extend=extend)
    drawCbar(datalims,nticks)
    postProcessPlot()

def preProcessPlot(font):
    mpl.rcParams['font.sans-serif']=font
    mpl.rcParams['pdf.fonttype'] = 42
    
def drawCbar(datalims,nticks):
    ticks = np.linspace(datalims[0],datalims[1],nticks)
    cbar = plt.colorbar()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks)
    
def postProcessPlot():
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.autoscale(tight=True)
    plt.show()
    
def scaleData(scaletype,datamat):
    if scaletype[0] == 'mean': # subtract off mean 
        a = np.nanmean(datamat)
    else: # don't subtract anything
        a = 0
    b = scaletype[1]
    return b*(datamat - a)

def enforceContourBounds(xmat,ymat,datamat,bounds):
    xvec, yvec = xmat[0,:], ymat[:,0]
    indexx = bounds[0] < xvec < bounds[1]
    indexy = bounds[2] < yvec < bounds[3]
    return (xmat[indexy,indexx], ymat[indexy,indexx], datamat[indexy,indexx])

def interpolateScattered(xvec,yvec,zvec,spacing,radius=None):
    indexgood = getIndexGood(zvec)
    tri = getTriangulation(xvec,yvec)
    xmat, ymat = regridMat(xvec,yvec,spacing)
    zmat = interpolateScatteredTri(tri,zvec[indexgood],xmat,ymat,radius)
    return (xmat, ymat, zmat)
    
def getTriangulation(xvec,yvec): # possibly with bad values
    indexgood = getIndexGood(xvec)
    tri = spsp.Delaunay(np.column_stack((xvec[indexgood],yvec[indexgood])))
    return tri
    
def interpolateScatteredTri(tri,zvec,xmat,ymat,radius=None): # assumes good values
    if radius is None:
        radius = (xmat.max() - xmat.min())*10 # sufficiently large number
    return np.asarray(C.CWalkAndMarchAll(tri.points,zvec,tri.simplices,tri.neighbors,xmat,ymat,radius))

def regridMat(xvec,yvec,spacing): # regrids vectors, possibly with bad values, using meshgrid
    xvecregrid = regridVec(xvec,spacing[0])
    yvecregrid = regridVec(yvec,spacing[1])
    return np.meshgrid(xvecregrid,yvecregrid)
    
def regridVec(vec, spacing): # regrids vector, possibly with bad values
    vmin, vmax = np.nanmin(vec), np.nanmax(vec)
    pts = round((vmax - vmin)/spacing) + 1
    return np.linspace(vmin, vmax, pts)

def getIndexGood(data):
    return np.invert(np.isnan(data))
        