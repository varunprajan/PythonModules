import numpy as np
import scipy.interpolate as spip
import scipy.linalg as spla
import scipy.optimize as spo
import mymath as Mmath

def getHillStressNoComp(svec,Avec):
    s1, s2, s6 = svec
    A11, A12, A66 = Avec
    s1 = rampFun(s1)
    s2 = rampFun(s2)
    return np.sqrt(A11*(s1**2 + s2**2) + 2*A12*s1*s2 + A66*s6**2)

def rampFun(x):
    return 0.5*(np.abs(x) + x)
    
def getShearData(strainmat,stressmat):
    return np.column_stack((strainmat[:,2],stressmat[:,2]))

def getTransverseData(strainmat,stressmat):
    return np.column_stack((strainmat[:,1],stressmat[:,0]))

def getAxialData(strainmat,stressmat):
    return np.column_stack((strainmat[:,0],stressmat[:,0]))
    
def getTensileData(strainmat,stressmat):
    return getAxialData(strainmat,stressmat)
    
def getTensile45Data(strainmat,stressmat):
    return getTensileThetaData(strainmat,stressmat,45)
    
def getTensileThetaData(strainmat,stressmat,theta,degoption=True):
    strainmatnew = rotateStrain2D(strainmat,theta,degoption)
    stressmatnew = rotateStress2D(stressmat,theta,degoption)
    return getAxialData(strainmatnew,stressmatnew)
    
def rotateStrain2D(strainmat,phi,degoption=False):
    A = Mmath.getRotMatA(phi,degoption)
    R = Mmath.getR()
    RAR = np.einsum('ij,jk,kl',R,A,spla.inv(R))
    return np.einsum('kj,ij',RAR,strainmat)
    
def rotateStress2D(stressmat,phi,degoption=False):
    A = Mmath.getRotMatA(phi,degoption)
    return np.einsum('kj,ij',A,stressmat)
    
class EffectiveStressData:
    def __init__(self,option,filepref,Ci=None,Ai=None,f0=None,f0T=None,fs=None,f45=None,f45T=None):
        self.option = option
        self.filepref = filepref
        self.Ci = Ci
        self.Ai = Ai
        self.f0 = f0
        self.f0T = f0T
        self.fs = fs
        self.f45 = f45
        self.f45T = f45T
        self.init_option()
        self.fill_missing_AiCi()
        self.read_fi()
        self.fill_missing_fi()
        
    def init_option(self):
        if self.option == 1: # Hill with no compression
            self.fun = getHillStressNoComp
            
    def fill_missing_AiCi(self):
        if self.Ci is None:
            self.Ci = self.convert_Ai_to_Ci()
        if self.Ai is None:
            self.Ai = self.convert_Ci_to_Ai()
    
    def convert_Ai_to_Ci(self):
        C = np.zeros(3)
        C[0] = self.get_effective_stress([1,0,0])
        C[1] = self.get_effective_stress([0.5,0.5,0.5])
        C[2] = self.get_effective_stress([0,0,1])
        return C
        
    def convert_Ci_to_Ai(self):
        def getResidual(Ai):
            self.Ai = Ai
            C = self.convert_Ai_to_Ci()
            return C - self.Ci
        return spo.fsolve(getResidual,np.zeros(3))
    
    def read_fi(self):
        for attr in ['f0','f0T','fs','f45','f45T']:
            if getattr(self,attr) is None:
                filename = self.get_filename(attr)
                try:
                    tabulardata = np.loadtxt(filename)
                    fobject = StrainStressCurve(table=tabulardata)
                    setattr(self,attr,fobject)
                except FileNotFoundError:
                    pass
                
    def fill_missing_fi(self):
        if self.fs is None:
            self.fs = self.fill_missing_fs()
            
    def fill_missing_fs(self,n=100,fac=0.99):
        s45max = min(self.f45.smax,self.f45T.smax)
        _, C45, Cs = self.Ci
        Cfac = C45/Cs
        staumax = fac*s45max*Cfac
        sigmavec = np.linspace(0,staumax,n)
        gammavec = np.zeros(n)
        for i, sigma in enumerate(sigmavec):
            gammavec[i] = 2*Cfac*(self.f45.get_strain(sigma/Cfac) - self.f45T.get_strain(sigma/Cfac))
        table = np.column_stack((sigmavec,gammavec))
        return StrainStressCurve(table=table)
        
    def get_effective_stress(self,svec):
        return self.fun(svec,self.Ai)
    
    def run_0_test(self,smax,n=200):
        return self.run_proportional_loading(np.array([1,0,0]),smax,n)
        
    def run_shear_test(self,smax,n=200):
        return self.run_proportional_loading(np.array([0,0,1]),smax,n)
        
    def run_45_test(self,smax,n=200):
        return self.run_proportional_loading(np.array([0.5,0.5,0.5]),smax,n)
    
    def run_proportional_loading(self,svecincr,smax,n=200):
        svecnorm = spla.norm(svecincr)
        svecincr = svecincr/svecnorm*smax/n
        stressmat = np.zeros((n,3))
        for i in range(n):
            stressmat[i,:] = svecincr*i
        strainmat = np.zeros((n,3))
        strainvec = np.zeros(3)
        for i in range(1,n):
            svec = stressmat[i,:]
            compliance = self.get_compliance_matrix(svec)
            dstrainvec = np.einsum('ij,j',compliance,svecincr)
            strainvec = strainvec + dstrainvec
            strainmat[i,:] = strainvec
        return strainmat, stressmat
    
    def get_compliance_matrix(self,svec):
        seff = self.get_effective_stress(svec)
        C0, _, Cs = self.Ci
        f0comp = self.f0.get_compliance(seff/C0)
        f0Tcomp = self.f0T.get_compliance(seff/C0)
        fscomp = self.fs.get_compliance(seff/Cs)
        compliance = np.zeros((3,3))
        compliance[0,0] = compliance[1,1] = f0comp
        compliance[1,0] = compliance[0,1] = f0Tcomp
        compliance[2,2] = fscomp
        return compliance
        
    def get_filename(self,attr):
        return self.filepref + '_' + attr + '.txt'
        
class StrainStressCurve:
    def __init__(self,table=None,interp=None):
        self.table = table
        self.interp = interp
        self.smax = self.get_smax()
        if self.interp is None:
            self.interp = self.gen_interp()
    
    def get_smax(self):
        if self.table is not None:
            return self.table[-1,0]
        else:
            return self.interp.x[-1]
            
    def gen_interp(self):
        return spip.interp1d(self.table[:,0],self.table[:,1])
    
    def get_strain(self,stress):
        return self.interp(stress)
        
    def get_compliance(self,stress):
        return self.get_compliance_from_table(stress)
        
    def get_compliance_from_table(self,stress):
        stressvec = self.table[:,0]
        index = np.searchsorted(stressvec,stress)
        if (index == 0):
            index = 1
        elif (index == len(stressvec)):
            index = len(stressvec) - 1
        return (self.table[index,1] - self.table[index-1,1])/(stressvec[index] - stressvec[index-1])
        
        