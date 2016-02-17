import numpy as np
import scipy.linalg as spla
import itertools

degfac = 180/np.pi

def same_sign(t1,t2):
    return not((t1 > 0) != (t2 > 0))
    
def product(iterable):
    product = 1
    for x in iterable:
        product *= x
    return product

def is_odd(num):
    return (num+1)%2 == 0

def get_signed_area(points):
    signedarea = 0
    n = points.shape[0]
    for i, point in enumerate(points):
        x1, y1 = point
        x2, y2 = points[np.mod(i+1,n),:]
        signedarea += (x1*y2 - x2*y1)
    return signedarea
    
def compute_angle(vec1,vec2):
	res = np.dot(normalize_vec(vec1),normalize_vec(vec2))
	res = np.clip(res,-1.0,1.0) # roundoff issues
	return np.arccos(res)

def normalize_vec_all(a,axis): # 0 for columns, 1 for rows
	return np.apply_along_axis(normalize_vec,axis,a)

def normalize_vec(vec):
	return vec/spla.norm(vec)
    
def get_dist(vec1, vec2):
    return spla.norm(vec1 - vec2)
    
def get_area_triangle(a,b,c):
    ab = b - a
    ac = c - a
    onevec = np.ones((3,1))
    mat = np.column_stack((ab,ac,onevec))
    return np.abs(1/2*spla.det(mat))

def round_to_even(number):
    return round_to_int_offset(number,2,0)
    
def round_to_odd(number):
    return roundto_int_offset(number,2,1)
    
def round_to_int_offset(number,integer,offset):
    return integer*round((number + offset)/integer) - offset
    
def get_rot_mat_2d(phi,degoption=False):
    if degoption: # convert to radians
        phi = phi/degfac
    return np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]])
    
def get_rot_mat_A(phi,degoption=False):
    if degoption: # convert to radians
        phi = phi/degfac
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c**2, s**2, 2*s*c],[s**2, c**2, -2*s*c],[-s*c, s*c, c**2-s**2]])
    
def get_R():
    R = np.identity(3)
    R[2,2] = 2
    return R
    
def rescale_coords(data,boundsorig,boundsnew):
    scale = 1/(boundsorig[1] - boundsorig[0])
    numer = data*(boundsnew[1] - boundsnew[0]) 
    return scale*(numer + boundsorig[1]*boundsnew[0] - boundsorig[0]*boundsnew[1])
    
def get_diff_vec(vec1,vec2):
    return np.gradient(vec2)/np.gradient(vec1)
    
def max_poly_root(coeff): # find maximum of polynomial given its coefficients. Requires at least one real root
    roots = np.roots(np.polyder(coeff))
    return np.max([np.polyval(coeff,root) if not np.iscomplex(root) else -np.Inf for root in roots]) # returns -inf for complex roots, so these are ignored
    
def set_diff(set1,set2):
    return [x for x in set1 if x not in set2]
    
def row_in_array(row,array):
    return any((array[:]==row).all(1))
    
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def threewise(iterable):
    a = iterable[::2]
    b = iterable[1::2]
    c = iterable[2::2]
    return zip(a, b, c)
    
def quad_eig(K,C,M):
    # solves quadratic eigenvalue problem (K + lambda*C + lambda^2*M)*a = 0 by reducing
    # the problem to a generalized eigenvalue problem A*x = lambdanew*B*x
    dim = np.shape(K)[0]
    Arow1 = np.hstack((-C,-K))
    Arow2 = np.hstack((np.eye(dim),np.zeros((dim,dim))))
    A = np.vstack((Arow1,Arow2))
    Brow1 = np.hstack((M,np.zeros((dim,dim))))
    Brow2 = np.hstack((np.zeros((dim,dim)),np.eye(dim)))
    B = np.vstack((Brow1,Brow2))
    eigvals, eigvecs = spla.eig(A, B)
    eigvecsnew = normalize_vec_all(eigvecs[:dim,:],0) # convert gen. eig. solution back to quad. eig. problem
    return eigvals, eigvecsnew
        