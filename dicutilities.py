import numpy as np
import contourplot as cp
import strains
import mymath as Mmath
import myio as Mio
import mydictionaries as Mdict
import scipy.spatial as spsp

def contourPlot(filename,variable,contourspacing=[],datalims=[],regrid=True,spacing=None,radius=None,strainoption=True,filterlength=None,override=False,extend='both'):
    dicdict = getData(filename,override)
    magnification = getMagnification(dicdict)
    if regrid: # regrid using real space points
        if spacing is None:
            spacing = magnification*np.array(getPixSpacing(dicdict))
        (Xmat,Ymat) = getXYGrid(dicdict,spacing)
        tri = getXYTriangulation(filename,dicdict,override)			
        datamat = getDataInterpolate(tri,dicdict,Xmat,Ymat,variable,strainoption,radius,spacing,filterlength)
    else: # use (approximate) pixel space points
        Xmat = magnification*dicdict['x']
        Ymat = -magnification*dicdict['y'] # Y increases in upward direction
        datamat = dicdict[variable]
    cp.contourPlot(xmat,ymat,datamat,contourspacing,datalims=datalims,scaletype=getScale(variable),extend=extend)

def getDataInterpolate(tri,dicdict,xmat,ymat,variable,strainoption,radius,spacing,filterlength):
    indexgood = cp.getIndexGood(dicdict['X'].ravel())
    if strainoption and variable[0] == 'e': # compute strains from displacements
        Uvec = dicdict['U'].ravel()[indexgood]
        Vvec = dicdict['V'].ravel()[indexgood]
        Umat = cp.interpolateScatteredTri(tri,Uvec,xmat,ymat,radius)
        Vmat = cp.interpolateScatteredTri(tri,Vvec,xmat,ymat,radius)
        return strains.getStrain(Umat,Vmat,spacing,variable,1,filterlength)
    else:
        datavec = dicdict[variable].ravel()[indexgood]
        return cp.interpolateScatteredTri(tri,datavec,xmat,ymat,radius)
    
def getPixSpacing(dicdict):
    xmat, ymat = dicdict['x'].astype('int16'), dicdict['y'].astype('int16')
    xspacing, yspacing = xmat[0,1] - xmat[0,0], ymat[1,0] - ymat[0,0]
    return xspacing, yspacing
    
def getMagnification(dicdict):
    XYgood = np.column_stack((dicdict['X'].ravel(), dicdict['Y'].ravel()))
    xygood = np.column_stack((dicdict['x'].ravel(), dicdict['y'].ravel()))
    XYmag = np.sum(XYgood,axis=1)
    indexsmall, indexlarge = np.nanargmin(XYmag), np.nanargmax(XYmag)
    distpix = Mmath.get_dist(xygood[indexsmall,:].astype('float64'), xygood[indexlarge,:].astype('float64'))
    distreal = Mmath.get_dist(XYgood[indexsmall,:], XYgood[indexlarge,:])
    return distreal/distpix

def getScale(variable):
    if variable == 'U' or variable == 'V':
        return ['mean', 1000] # mean subtraction, mm -> um
    elif variable == 'exx' or variable == 'eyy':
        return ['none', 100] # no mean subtraction, percent strain
    elif variable == 'exy':
        return ['none', 200] # engineering strain
    
def getData(filename,override):
    return Mio.getAndStore(loadAndCleanFile,Mio.getFilePrefix,override,filename=filename)

def getXYTriangulation(filename,dicdict,override):
    return Mio.getAndStore(getXYTriangulationSub,getDICTestName,override,filename=filename,dicdict=dicdict)
    
def getXYTriangulationSub(dicdict,**kwargs):
    Xvec, Yvec = dicdict['X'].ravel(), dicdict['Y'].ravel()
    return cp.getTriangulation(Xvec,Yvec)

def getXYGrid(dicdict,spacing):
    Xvec, Yvec = dicdict['X'].ravel(), dicdict['Y'].ravel()
    return cp.regridMat(Xvec,Yvec,spacing)

def loadAndCleanFile(filename,subdir='DIC_Files/'):
    dicdict = Mio.getMatlabObject(subdir + filename)
    return cleanUpData(dicdict)

def cleanUpData(dicdict):
    indexbad = dicdict['sigma'] == -1
    for variable, data in dicdict.items():
        try:
            data[indexbad] = np.nan
            dicdict[variable] = data
        except ValueError: # e.g. x, y
            pass
        except TypeError: # weird keys read in by spio
            pass
    return dicdict
    
def getDICTestName(filename,**kwargs): # get everything before '-' (after is image number, extension)
    return Mio.getFilePrefixSub(filename,'-')
    