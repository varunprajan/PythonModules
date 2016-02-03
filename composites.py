import numpy as np
import scipy.optimize as spo

def transformation_matrix(angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    return np.matrix([[c**2, s**2, 2*s*c], [s**2, c**2, -2*s*c], [-s*c, s*c, c**2 - s**2]])

def reuters_matrix():
    return np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 2]])

def get_multiplier(straintype='engineering'):
    if straintype.lower() == 'engineering':
        return 1
    else:
        return 2

def get_el_from_stiffness(stiffnessmatrix,straintype='engineering'):
    multiplier = get_multiplier(straintype)
    E11 = 1/stiffnessmatrix[0, 0]
    E22 = 1/stiffnessmatrix[1, 1]
    G12 = 1/(multiplier*stiffnessmatrix[2, 2])
    v12 = -1*stiffnessnatrix[1, 0]*E11
    return {'E11': E11, 'E22': E22, 'G12': G12, 'v12': v12}
    
def get_mat_from_gscs(fiber,matrix,Vf):
    attrnamelist = ['Ea','Et','Ga','Gt','va','vt','k']
    
    # Fiber Properties
    Eaf, Etf, Gaf, Gtf, vaf, vtf, kf = [getattr(fiber,attrname) for attrname in attrnamelist]
    cf = Vf
    etaf = fiber.get_eta()

    # Matrix Properties
    Eam, Etm, Gam, Gtm, vam, vtm, km = [getattr(matrix,attrname) for attrname in attrnamelist]
    cm = 1 - Vf
    etam = matrix.get_eta()

    # Axial Properties (Hashin)
    Eac = (Eam * cm + Eaf * cf + 4 * (vaf - vam) ** 2 * cm * cf / (cm / kf + cf / km + 1 / Gtm))
    vac = (vam * cm + vaf * cf + (vaf - vam) * (1 / km - 1 / kf) * cm * cf / (cm / kf + cf / km + 1 / Gtm))
    Gac = (Gam * (Gam * cm + Gaf * (1 + cf)) / (Gam * (1 + cf) + Gaf * cm))
    kc = ((km * (kf + Gtm) * cm + kf * (km + Gtm) * cf) / ((kf + Gtm) * cm + (km + Gtm) * cf))
    Gtr = (Gtf / Gtm)
    mc = (1 + 4 * kc * vac ** 2 / Eac)

    # Transverse Properties (GSCS)
    Achr = (3 * cf * cm ** 2 * (Gtr - 1) * (Gtr + etaf) + (
        Gtr * etam + etaf * etam - (Gtr * etam - etaf) * cf ** 3) * (
                        cf * etam * (Gtr - 1) - (Gtr * etam + 1)))
    Bchr = (-3 * cf * cm ** 2 * (Gtr - 1) * (Gtr + etaf) + 1 / 2 * (etam * Gtr + (Gtr - 1) * cf + 1) * (
        (etam - 1) * (Gtr + etaf) - 2 * (Gtr * etam - etaf) * cf ** 3) + cf / 2 * (etam + 1) * (Gtr - 1) * (
                            Gtr + etaf + (Gtr * etam - etaf) * cf ** 3))
    Cchr = (3 * cf * cm ** 2 * (Gtr - 1) * (Gtr + etaf) + (etam * Gtr + (Gtr - 1) * cf + 1) * (
        Gtr + etaf + (Gtr * etam - etaf) * cf ** 3))

    sol = spo.fsolve(lambda x: Achr * x ** 2 + 2 * Bchr * x + Cchr, 1)

    Gtc = Gtm * sol[0]
    vtc = (kc - mc * Gtc) / (kc + mc * Gtc)
    Etc = 2 * (1 + vtc) * Gtc
    
    return TransverseIsoMaterial(Eac,Etc,Gac,Gtc,vac)
    
def gen_ply_using_gscs(fiber,matrix,Vf):
    material = get_mat_from_gscs(fiber,matrix,Vf)
    return gen_ply_from_material(material)
        
def gen_ply_from_material(mat):
    return Ply(mat.Eac,mat.Etc,mat.Gac,mat.vac)
    
class TransverseIsoMaterial(object):
    def __init__(self,Ea,Et,Ga,Gt,va):
        self.Ea = Ea
        self.Et = Et
        self.Ga = Ga
        self.Gt = Gt
        self.va = va
        self.vt = self.Et/(2*self.Gt) - 1
        self.k = self.get_k()
        
    def get_k(self):
        num = self.Ea*self.Et
        denom = 2*self.Ea - 4*self.Et*self.va**2 - 2*self.Ea*self.vt
        return num/denom
        
    def get_n(self):
        return self.Ea + 4*self.k*self.va**2
        
    def get_m(self):
        return self.get_n()/self.Ea
        
    def get_l(self):
        return 2*self.k*self.va
        
    def get_nu_eff(self):
        return 1/2*(1 - self.Gt/self.k)
        
    def get_E_eff(self):
        return self.Gt*(3 - self.Gt/self.k)
        
    def get_eta(self):
        return 3 - 4*self.get_nu_eff()
        
class IsoMaterial(TransverseIsoMaterial):
    def __init__(self,E=None,G=None,v=None):
        assert [E,G,v].count(None) <= 1, "Must supply two of three inputs"

        if E is None:
            E = G*(2*(1+v))
        elif v is None:
            v = E/(2*G) - 1
        elif G is None:
            G = E/(2*(1+v))
        super().__init__(E, E, G, G, v)

class Ply(object):
    def __init__(self, E11=None, E22=None, G12=None, v12=None):
        self.E11 = E11
        self.E22 = E22
        self.G12 = G12
        self.v12 = v12

    def get_compliance(self,straintype='engineering'):
        multiplier = get_multiplier(straintype)
        return np.matrix([[1/self.E11, -self.v12/self.E11, 0], [-self.v12/self.E11, 1/self.E22, 0],
                          [0, 0, 1/(multiplier*self.G12)]])
        
    def get_compliance_rot(self,theta=0,straintype='engineering'):
        compliance = self.get_compliance(straintype)
        T = transformation_matrix(theta)
        if straintype.lower() == 'engineering':
            R = reuters_matrix()
            TI = R*T.I*R.I
        else:
            TI = T.I
        return TI*compliance*T

    def get_stiffness(self,straintype='engineering'):
        return self.get_compliance(straintype).I
        
    def get_stiffness_rot(self,theta=0,straintype='engineering'):
        return self.get_compliance_rot(theta,straintype).I

    def gen_woven_ply(self,orientation=(0,90)):
        Q_weave = (self.get_Q(orientation[0]) + self.get_Q(orientation[1])) / 2
        S_weave = Q_weave.I
        return Ply(**get_elastic_constants(S_weave))

class Laminate(object):
    def __init__(self,plies,theta):
        assert len(plies) == len(theta), "Must supply the same number of plies and angles."
        self.layup = list(zip(plies,theta))
        self.get_laminate_stiffness()

    def get_laminate_stiffness(self):
        l = len(self.layup)
        h = 1
        def z(plynum):
            return h / l * (plynum - l / 2)
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.D = np.zeros((3, 3))
        for k, (ply, theta) in enumerate(self.layup):
            Q_Bar = ply.get_Q(theta, strain_type=strain_type)
            self.A += (z(k + 1) - z(k)) * Q_Bar
            self.B += (1 / 2) * (z(k + 1) ** 2 - z(k) ** 2) * Q_Bar
            self.D += (1 / 3) * (z(k + 1) ** 3 - z(k) ** 3) * Q_Bar
            
    def get_elastic_constants(self,straintype='engineering'):
        return get_elastic_constants(self.A.I,straintype=straintype)