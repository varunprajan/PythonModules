import numpy as np
import mymath as Mmath
from matplotlib.path import Path

def gen_marker(angle,sgn,width,aspectratio=1,degoption=True):
    if degoption:
        angle = np.radians(angle)
    if sgn == -1:
        angle = angle + np.pi # rotate by 180
    verts, codes = base_marker_pieces(width,height=width*aspectratio)
    rotmat = Mmath.get_rot_mat_2d(angle,degoption=False)
    verts = np.einsum('ij,jk',verts,rotmat)
    return Path(verts,codes)
    
def base_marker_pieces(width,height):
    verts = np.array([
                     [-width/2, 0.], # left, bottom
                     [width/2, 0.], # right, bottom
                     [0.,0.], # middle, bottom
                     [0.,height], # middle, top
                     ])
    codes = [Path.MOVETO,Path.LINETO,Path.MOVETO,Path.LINETO]
    return verts, codes