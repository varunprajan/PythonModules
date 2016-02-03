import numpy as np
import fractions
import mymath as Mmath
import scipy.linalg as spla

def computeK(materialprops,aold,anew,burgersvector,dislocationtype='partial'):
    Kic, Gic, A, compliancemat = computeKic(aold,anew,materialprops)
    Kie = computeKie(aold,anew,burgersvector,dislocationtype,materialprops)
    Gie = A*Kie**2
    return (Kic, Kie, Gie/Gic, A, Gic)

def computeKic(aold,anew,materialprops):
    crackplane = anew[1,:]
    crackkey = convertToString(crackplane,1)
    gammasurf = materialprops['surface'][crackkey]['surf']
    compliancemat = getPlaneStrainComplianceMat(materialprops['elastic'],materialprops['symmetry'],aold,anew)
    A = computeA(compliancemat)
    return np.sqrt(2*gammasurf/A), 2*gammasurf, A, compliancemat

def computeKie(aold,anew,burgersvector,dislocationtype,materialprops):
    crackfront = anew[2,:]
    crackplane = anew[1,:]
    slipplane = computePlane(crackfront,burgersvector)
    slipplanekey = convertToString(slipplane,1)
    slipplanet = computePlane(slipplane,crackfront)
    anewslip = np.vstack((slipplanet,slipplane,crackfront))
    compliancemat = getPlaneStrainComplianceMat(materialprops['elastic'],materialprops['symmetry'],aold,anewslip)
    theta = Mmath.compute_angle(crackplane,slipplane)
    phi = Mmath.compute_angle(burgersvector,slipplanet)
    (nu, mu) = getElPECompliance(compliancemat)
    gammaus = materialprops['unstable'][dislocationtype]
    printStuff(crackplane,slipplane,theta,phi,nu,mu)
    try:
        return np.sqrt(2*mu/(1-nu)*(1 + (1-nu)*np.tan(phi)**2)*gammaus)/fI(theta)
    except ZeroDivisionError:
        return np.inf

def printStuff(crackplane,slipplane,theta,phi,nu,mu):
    print('Crack Plane: {0}'.format(crackplane))
    print('Slip Plane: {0}'.format(slipplane))
    print('Theta: {0} (degrees)'.format(Mmath.degfac*theta))
    print('Phi: {0} (degrees)'.format(Mmath.degfac*phi))
    print('Nu: {0}, Mu: {1}'.format(nu, mu))
        
def getPlaneStrainComplianceMat(elasticprops,symmetry,aold,anew):
    if symmetry == 'cubic':
        stiffnessmat = cubicVoigt(elasticprops)
    elif symmetry == 'hexagonal':
        stiffnessmat = hexVoigt(elasticprops)
    stiffnesstensor = voigtToTensor(stiffnessmat)
    stiffnesstensorrot = rotateTensor(stiffnesstensor,aold,anew)
    stiffnessmatrot = tensorToVoigt(stiffnesstensorrot)
    return planeStrainComplianceVoigt(stiffnessmatrot)
    
def rotateTensor(tensor,aold,anew):
    rt = rotationTensor(aold,anew)
    return np.einsum('ip,jq,kr,ls,ijkl',rt,rt,rt,rt,tensor)

def rotationTensor(aold,anew):
    return np.einsum('ij,kj',Mmath.normalize_vec_all(aold,1),Mmath.normalize_vec_all(anew,1))
    
def planeStrainComplianceVoigt(matrix,symmoption=True):
    matrixnew = matrix[[[0],[1],[5]],[0,1,5]]
    if symmoption: # eliminate shear-normal coupling terms
        matrixnew[0,2] = matrixnew[1,2] = matrixnew[2,0] = matrixnew[2,1] = 0
    return spla.inv(matrixnew)

def cubicVoigt(dict):
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
    
def hexVoigt(dict):
    # creates stiffness/compliance matrix (Voigt notation) assuming hexagonal (3d) symmetry; 1-2 is the isotropy plane
    tensor = np.zeros([6,6])
    tensor[0,0] = tensor[1,1] = dict['11']
    tensor[2,2] = dict['33']
    tensor[0,2] = tensor[2,0] = tensor[1,2] = tensor[2,1] = dict['13']
    tensor[3,3] = tensor[4,4] = dict['44']
    tensor[5,5] = dict['66']
    tensor[0,1] = tensor[1,0] = dict['11'] - 2*dict['66']
    return tensor
    
def voigtToTensor(matrix):
    tensor = np.zeros([3,3,3,3])
    for i in range(6):
        [l,m] = voigtToTensorIndex(i)
        for j in range(6):
            [n,p] = voigtToTensorIndex(j)
            tensor[l,m,n,p] = tensor[m,l,n,p] = tensor[l,m,p,n] = tensor[m,l,p,n] = matrix[i,j]
    return tensor
            
def tensorToVoigt(tensor):
    tensornew = np.zeros([6,6])
    for i in range(3):
        for j in range(3):
            index1 = tensorToVoigtIndex([i,j])
            for k in range(3):
                for l in range(3):
                    index2 = tensorToVoigtIndex([k,l])
                    tensornew[index1,index2] = tensor[i,j,k,l]
    return tensornew
    
def voigtToTensorIndex(val):
    if val < 3:
        return [val,val]
    else:
        return Mmath.set_diff(range(3),set([val-3]))
        
def tensorToVoigtIndex(vals):
    if vals[0] == vals[1]:
        return vals[0]
    else:
        sol = Mmath.set_diff(range(3),set(vals))
        return sol[0] + 3
    
def computeA(mat):
    a11, a12, a22, a66 = mat[0,0], mat[0,1], mat[1,1], mat[2,2]
    return np.sqrt(a11*a22/2)*np.sqrt((2*a12+a66)/(2*a11) + np.sqrt(a22/a11))

def getElPECompliance(mat): # assumes isotropy
    ratio = -mat[0,1]/mat[0,0]
    nu = ratio/(1 + ratio)
    mu = 1/mat[2,2]
    return (nu, mu)

def computePlane(vec1,vec2):
    res = np.cross(vec1,vec2)
    return res//np.abs(gcdMany(list(res)))

def convertToString(direction,option):
    if option == 1:
        direction = sorted(abs(direction))
    return ''.join(map(str,direction))

def fI(theta):
    return np.cos(theta/2)**2*np.sin(theta/2)
    
def gcdMany(mylist):
    if len(mylist) > 1:
        return gcdMany([fractions.gcd(mylist[0],mylist[1])]+mylist[2:])
    else:
        return mylist[0]

# Sih/Leibowitz anisotropic crack fields
def getLekhEigenvalues(mat):
    a11, a12, a16, a22, a26, a66 = mat[0,0], mat[0,1], mat[0,2], mat[1,1], mat[1,2], mat[2,2]
    poly = [a11,-2*a16,2*a12+a66,-2*a26,a22]
    eigs = np.roots(poly)
    s1, s2 = eigs[np.imag(eigs) > 0]
    p1 = a11*s1**2 + a12 - a16*s1
    p2 = a11*s2**2 + a12 - a16*s2
    q1 = a12*s1 + a22/s1 - a26
    q2 = a12*s2 + a22/s2 - a26
    return s1, s2, p1, p2, q1, q2
        
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
    
# EAM properties

def getMaterialProps(materialname): # units - energy [J/m^2]; stiffness [N/m^2]
    nickelelastic = {'11': 247.862330908453e9,
                     '12': 147.828379827956e9,
                     '44': 124.838117598312e9}
    nickelsurface111 = {'surf': 1629.0e-3}
    nickelsurface110 = {'surf': 2049.0e-3}
    nickelsurface100 = {'surf': 1878.0e-3}
    nickelsurface = {'111': nickelsurface111,'011': nickelsurface110,'001': nickelsurface100}
    nickelunstable = {'partial': 366.0e-3, 'full': 366.0e-3}
    nickelprops = {'elastic': nickelelastic,'symmetry': 'cubic',
                   'surface': nickelsurface,
                   'unstable': nickelunstable}
    props = {'nickel': nickelprops}
    return props[materialname.lower()]
            
    