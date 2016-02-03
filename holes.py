import numpy as np
import myio as Mio

Dvec = np.logspace(-5,5,100) # should cover range

def getLSBFit(notchtype,clawtype,a0,subdir='Mat_Files/'):
    if notchtype == 'circle':
        data = Mio.getMatlabObject(subdir + 'maimi_circle.mat')
    elif notchtype == 'crack':
        data = Mio.getMatlabObject(subdir + 'suo_crack.mat')
    datanew = data[clawtype + notchtype]
    datanew[:,1] = datanew[:,1]/a0
    return datanew
    
def getLSBFormula(notchtype,clawtype,a0):
    fdict = {'crackrectilinear': crackCottrellRectilinear, 'cracklinear': crackSuoLinear}
    return formulaEval(fdict[notchtype+clawtype],a0)
    
def getWhitneyFormula(notchtype,crittype,a0): 
    fdict = {'crackavg': crackWhitneyAvg, 'crackpoint': crackWhitneyPoint, 'circleavg': circleWhitneyAvg, 'circlepoint': circleWhitneyPoint}
    return formulaEval(fdict[notchtype+crittype],a0)
    
def formulaEval(f,a0):
    yvec = [f(D/a0) for D in Dvec]
    return np.column_stack((Dvec,yvec))

def crackWhitneyAvg(a):
    return 1/np.sqrt(1 + 2*a)
    
def crackWhitneyPoint(a):
    return np.sqrt(1 + 2*a)/(1 + a)
    
def circleWhitneyPoint(a):
    return (2*(1 + a)**4)/(2 + 8*a + 13*a**2 + 10*a**3 + 6*a**4)
    
def circleWhitneyAvg(a):
    return (2*(1 + a)**3)/((1 + 2*a)*(2 + 4*a + 3*a**2))
        
def crackCottrellRectilinear(a):
    return 2/np.pi*np.arccos(np.exp(-np.pi/(8*a)))
    
def crackSuoLinear(a):
    return (1 + np.pi*a)**(-1/2)