import numpy as np
import fractions
import mymath as Mmath
import scipy.linalg as spla

class Material(object):
    def __init__(self,crystalstructure,elasticprops):
        self.structure = crystalstructure
        self.elasticprops = elasticprops
        self.symmetry = self.get_symmetry()
        self.stiffness = self.get_stiffness()
        
    def get_symmetry(self):
        if self.structure == 'bcc' or self.structure == 'fcc':
            return 'cubic'
        elif self.structure == 'hcp' or self.structure == 'hex':
            return 'hex'
            
    def get_stiffness(self):
        if self.symmetry == 'cubic':
            return cubic_voigt(self.elasticprops)
        elif self.symmetry == 'hexagonal':
            return hex_voigt(self.elasticprops)
            
    def rotated_stiffness(self,aold,anew):
        stiffnesstensor = voigt_to_tensor(self.stiffness)
        stiffnesstensorrot = rotate_tensor(stiffnesstensor,aold,anew)
        return tensor_to_voigt(stiffnesstensorrot)  

    def rotated_compliance(self,aold,anew):
        rotatedstiffness = self.rotated_stiffness(aold,anew)
        return spla.inv(rotatedstiffness)
        
    def plane_strain_compliance(self,aold,anew,symmoption=False):
        rotatedstiffness = self.rotated_stiffness(aold,anew)
        planestrainstiffness = rotatedstiffness[[[0],[1],[5]],[0,1,5]]
        if symmoption: # eliminate shear-normal coupling terms
            for i, j in zip([0,1],[2,2]):
                planestrainstiffness[i,j] = 0
                planestrainstiffness[j,i] = 0
        return spla.inv(planestrainstiffness)
        
    def get_aniso_k_const(self,aold,anew,symmoption=False):
        compliance = self.plane_strain_compliance(aold,anew,symmoption)
        return get_lekh_eigenvalues(compliance)
        
    def get_surface_energy(self,plane):
        pass
      
    def get_usf_energy(self,plane,direction):
        pass

    def compute_A(self,aold,anew):
        compliance = self.plane_strain_compliance(aold,anew)
        return get_aniso_fac(compliance)
    
    def compute_kic(self,aold,anew):
        crackplane = anew[1,:]
        crackplanekey = convert_to_string(crackplane,1)
        gammasurf = self.get_surface_energy(crackplanekey)
        A = self.compute_A(aold,anew)
        return np.sqrt(2*gammasurf/A)
        
    def compute_kie(self,aold,anew,burgersvector,symmoption=False,printoption=False):
        crackfront = anew[2,:]
        crackplane = anew[1,:]
        slipplane = plane_spanned_by_vecs(crackfront,burgersvector)
        slipplanet = plane_spanned_by_vecs(slipplane,crackfront)
        anewslip = np.vstack((slipplanet,slipplane,crackfront))
        theta = Mmath.compute_angle(crackplane,slipplane)
        phi = Mmath.compute_angle(burgersvector,slipplanet)
        if printoption:
            print('Crack Plane: {0}'.format(crackplane))
            print('Slip Plane: {0}'.format(slipplane))
            print('Theta: {0} (degrees)'.format(Mmath.degfac*theta))
            print('Phi: {0} (degrees)'.format(Mmath.degfac*phi))
        compliance = self.plane_strain_compliance(aold,anewslip,symmoption)
        (nu, mu) = isotropic_constants(compliance)
        slipplanekey = convert_to_string(slipplane,1)
        burgerskey = convert_to_string(burgersvector,1)
        gammaus = self.get_usf_energy(slipplanekey,burgerskey)
        try:
            return np.sqrt(2*mu/(1-nu)*(1 + (1-nu)*np.tan(phi)**2)*gammaus)/fI(theta)
        except ZeroDivisionError:
            return np.inf
            
    def compute_k_both(self,aold,anew,burgersvector,symmoption=False,printoption=False):
        kic = self.compute_kic(aold,anew)
        kie = self.compute_kie(aold,anew,burgersvector,symmoption,printoption)
        if printoption:
            print('K_ic: {0}'.format(kic))
            print('K_ie: {0}'.format(kie))
        return kic, kie

class FCC(Material):
    def __init__(self,elasticprops,surfaceenergies,usfenergy):
        self.surfaceenergies = surfaceenergies
        self.usfenergy = usfenergy
        super().__init__('fcc',elasticprops)

    def get_surface_energy(self,plane):
        return self.surfaceenergies[plane]
        
    def get_usf_energy(self,plane,direction):
        if direction == '211' and plane == '111': # partial
            return self.usfenergy       

class BCC(Material):
    def __init__(self,elasticprops,surfaceenergies,usfenergies):
        self.surfaceenergies = surfaceenergies
        self.usfenergies = usfenergies
        super().__init__('bcc',elasticprops)
        
    def get_surface_energy(self,plane):
        return self.surfaceenergies[plane]
        
    def get_usf_energy(self,plane,direction):
        if direction == '111':
            return self.usfenergies[plane]
            
def rotate_tensor(tensor,aold,anew):
    rt = rotation_tensor(aold,anew)
    return np.einsum('ip,jq,kr,ls,ijkl',rt,rt,rt,rt,tensor)

def rotation_tensor(aold,anew):
    return np.einsum('ij,kj',Mmath.normalize_vec_all(aold,1),Mmath.normalize_vec_all(anew,1))

def cubic_voigt(dict):
    # creates stiffness/compliance matrix (Voigt notation) assuming cubic symmetry
    # also works for hexagonal (2d) if T44 = 1/2*(T11 - T12)
    T11, T12, T44 = dict['11'], dict['12'], dict['44']
    tensor = np.zeros([6,6])
    for i in range(3):
        for j in range(3):
            if i == j:
                tensor[i,j] = T11
            else:
                tensor[i,j] = T12
    for i in range(3,6):
        tensor[i,i] = T44
    return tensor
    
def hex_voigt(dict):
    # creates stiffness/compliance matrix (Voigt notation) assuming hexagonal (3d) symmetry; 1-2 is the isotropy plane
    tensor = np.zeros([6,6])
    tensor[0,0] = tensor[1,1] = dict['11']
    tensor[2,2] = dict['33']
    tensor[0,2] = tensor[2,0] = tensor[1,2] = tensor[2,1] = dict['13']
    tensor[3,3] = tensor[4,4] = dict['44']
    tensor[5,5] = dict['66']
    tensor[0,1] = tensor[1,0] = dict['11'] - 2*dict['66']
    return tensor
    
def voigt_to_tensor(matrix):
    tensor = np.zeros([3,3,3,3])
    for i in range(6):
        [l,m] = voigt_to_tensor_index(i)
        for j in range(6):
            [n,p] = voigt_to_tensor_index(j)
            tensor[l,m,n,p] = tensor[m,l,n,p] = tensor[l,m,p,n] = tensor[m,l,p,n] = matrix[i,j]
    return tensor
            
def tensor_to_voigt(tensor):
    tensornew = np.zeros([6,6])
    for i in range(3):
        for j in range(3):
            index1 = tensor_to_voigt_index([i,j])
            for k in range(3):
                for l in range(3):
                    index2 = tensor_to_voigt_index([k,l])
                    tensornew[index1,index2] = tensor[i,j,k,l]
    return tensornew
    
def voigt_to_tensor_index(val):
    if val < 3:
        return [val,val]
    else:
        return Mmath.set_diff(range(3),set([val-3]))
        
def tensor_to_voigt_index(vals):
    if vals[0] == vals[1]:
        return vals[0]
    else:
        sol = Mmath.set_diff(range(3),set(vals))
        return sol[0] + 3
    
def get_aniso_fac(mat):
    a11, a12, a22, a66 = mat[0,0], mat[0,1], mat[1,1], mat[2,2]
    return np.sqrt(a11*a22/2)*np.sqrt((2*a12+a66)/(2*a11) + np.sqrt(a22/a11))

def isotropic_constants(mat): # assumes isotropy
    ratio = -mat[0,1]/mat[0,0]
    nu = ratio/(1 + ratio)
    mu = 1/mat[2,2]
    return (nu, mu)

def plane_spanned_by_vecs(vec1,vec2):
    res = np.cross(vec1,vec2)
    return res//np.abs(gcd_many(list(res)))

def convert_to_string(direction,option):
    if option == 1:
        direction = sorted(abs(direction),reverse=True)
    return ''.join(map(str,direction))

def fI(theta):
    return np.cos(theta/2)**2*np.sin(theta/2)
    
def gcd_many(mylist):
    if len(mylist) > 1:
        return gcd_many([fractions.gcd(mylist[0],mylist[1])]+mylist[2:])
    else:
        return mylist[0]

# Sih/Leibowitz anisotropic crack fields
def get_lekh_eigenvalues(mat):
    a11, a12, a16, a22, a26, a66 = mat[0,0], mat[0,1], mat[0,2], mat[1,1], mat[1,2], mat[2,2]
    poly = [a11,-2*a16,2*a12+a66,-2*a26,a22]
    eigs = np.roots(poly)
    s1, s2 = eigs[np.imag(eigs) > 0]
    p1 = a11*s1**2 + a12 - a16*s1
    p2 = a11*s2**2 + a12 - a16*s2
    q1 = a12*s1 + a22/s1 - a26
    q2 = a12*s2 + a22/s2 - a26
    return {'s1': s1, 's2': s2, 'p1': p1, 'p2': p2, 'q1': q1, 'q2': q2}
    
def print_lekh_const(lekhconst):
    for var in ['s1','s2','p1','p2','q1','q2']:
        for char, fun in zip(['r','i'],[np.real,np.imag]):
            val = fun(lekhconst[var])
            print('variable {0}{1} equal {2}'.format(var,char,val))
        
# Stroh Formalism
        
def getQRT(C):
    Q = np.zeros((3,3))
    R = np.zeros((3,3))
    T = np.zeros((3,3))
    for i in range(3):
        for k in range(3):
            Q[i,k] = C[i,1,k,1]
            R[i,k] = C[i,1,k,2]
            T[i,k] = C[i,2,k,2]
    return (Q, R, T)
    
def getEigp(Q,R,T):
    RT = np.transpose(R)
    eigvals, eigvecs = Mmath.quadEig(Q,R+RT,T)
    indices = np.imag(eigvals) > 0
    pvec = eigvals[indices]
    alphamat = eigvecs[:,indices]
    betamat = np.copy(alphamat)
    for i in range(3):
        betamat[:,i] = (RT + pvec[i]*T).dot(alphamat[:,i])
        # normalize
        kron = spla.norm(2*betamat[:,i].dot(alphamat[:,i]))
        alphamat[:,i] = alphamat[:,i]/np.sqrt(kron)
        betamat[:,i] = betamat[:,i]/np.sqrt(kron)
    return pvec, alphamat, betamat
            
    