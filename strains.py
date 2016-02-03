import numpy as np
import mymath as Mmath
import Cstrains as C

def slipMain(umat,vmat,spacing,e1,e2):
    strainmat = slipSub(umat,vmat,spacing)
    slip = np.abs(resolvedStrain(e1,e2,strainmat))
    return np.reshape(slip,np.shape(umat))

def slipSub(umat,vmat,spacing):
    exx = getStrain(umat,vmat,spacing,'exx',2)
    eyy = getStrain(umat,vmat,spacing,'eyy',2)
    exy = getStrain(umat,vmat,spacing,'exy',2)
    inter1 = np.column_stack((exx.ravel(),exy.ravel()))
    inter2 = np.column_stack((exy.ravel(),eyy.ravel()))
    return np.dstack((inter1,inter2))

def resolvedStrain(e1,e2,strainmat):
    # assumes n x 2 x 2 array
    return np.einsum('i, kij, j',e1,strainmat,e2)

def getStrain(umat,vmat,spacing,component,diffoption,filterlength=None):
    strain = getStrainUnfiltered(umat,vmat,spacing,component,diffoption)
    if filterlength is not None:
        Xfilternum = Mmath.roundtoOdd(filterlength/spacing[0])
        Yfilternum = Mmath.roundtoOdd(filterlength/spacing[1])
        gaussiankernelX = getGaussianKernel1D(Xfilternum)
        gaussiankernelY = getGaussianKernel1D(Yfilternum)
        strain = kernelFilter(strain,gaussiankernelX,kerneloption=3,axis='x')
        strain = kernelFilter(strain,gaussiankernelY,kerneloption=3,axis='y')
    return strain
    
def getStrainUnfiltered(umat,vmat,spacing,component,diffoption):
    if component == 'exx':
        return (1/spacing[0])*derivX(umat,diffoption)
    if component == 'eyy':
        return (1/spacing[1])*derivY(vmat,diffoption)
    if component == 'exy':
        halfexy = (0.5/spacing[1])*derivY(umat,diffoption)
        halfeyx = (0.5/spacing[0])*derivX(vmat,diffoption)
        return halfexy + halfeyx

def derivX(data,diffoption): # unnormalized
    kernel = derivKernel(diffoption)
    return kernelFilter(data,kernel,kerneloption=1,axis='x')
    
def derivY(data,diffoption): # unnormalized
    kernel = derivKernel(diffoption)
    return kernelFilter(data,kernel,kerneloption=1,axis='y')
    
def derivKernel(diffoption): # unnormalized
    if diffoption == 1: # 2-point difference
        return np.array([-1,1])
    elif diffoption == 2: # central difference
        return np.array([-1,0,1])
    
def kernelFilter(data,kernel,kerneloption,axis):
    if kernel.ndim == 1: # vector
        n = kernel.shape[0]
        if axis.lower() == 'x':
            kernel.shape = (1,n)
        elif axis.lower() == 'y':
            kernel.shape = (n,1)
    return np.asarray(C.CGenericFilter(data,kernel.astype('float64'),kerneloption))

def getGaussianKernel1D(oddinteger,sigmafac=0.4):
    # Choice of sigmafac = 3*sigmafac_MATLAB -> agree at oddinteger = 3; sigmafac = 2*sigmafac_MATLAB -> agree at oddinteger = infinity
    x = np.linspace(-1,1,oddinteger)
    g = np.exp(-0.5*(x/sigmafac)**2)
    return g/g.sum()