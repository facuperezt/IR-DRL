import numpy as np
import math
# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())
def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.

    >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True

    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def get_angle(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    angle=np.arccos(np.dot(vector_1,vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2)))
    angle_deg = np.rad2deg(angle)
    return angle, angle_deg

import torch
from typing import Union, List

def directionalVectorsFromQuaternion(quaternion: Union[List, torch.Tensor], scale= 1) -> Union[List, torch.Tensor]:
    """
    Returns (scaled) up/forward/left vectors for world rotation frame quaternion.
    """
    x, y, z, w = quaternion
    up_vector = [
        scale*(2* (x*y - w*z)),
        scale*(1- 2* (x*x + z*z)),
        scale*(2* (y*z + w*x)),
    ]
    forward_vector = [
        scale*(2* (x*z + w*y)),
        scale*(2* (y*z - w*x)),
        scale*(1- 2* (x*x + y*y)),
    ]
    left_vector = [
        scale*(1- 2* (x*x + y*y)),
        scale*(2* (x*y + w*z)),
        scale*(2* (x*z - w*y)),
    ]

    if type(quaternion) is torch.Tensor:
        up_vector = torch.tensor(up_vector)
        forward_vector = torch.tensor(forward_vector)
        left_vector = torch.tensor(left_vector)

    return up_vector, forward_vector, left_vector


def add_list(a: List, b: List, factor: int = 1) -> List:
    """
    adds lists "a" and "b" as vectors, "factor" is multiplied by b
    """
    return (np.array(a) + factor * np.array(b)).tolist()

def getOrientationFromDirectionalVector(v: Union[List, np.ndarray], v_base = None) -> List:

    """

    Gets world space quaternion orientation for a directional vector v


    :param v: Directional vector to extract world space orientation
    """
    if type(v) is list:
        v = np.array(v)
    v = v/np.linalg.norm(v)

    if v_base is None:
        v_base = np.array([0,0,1])
    if type(v_base) is list:
        v_base = np.array(v_base)
    v_base = v_base/np.linalg.norm(v_base)

    if np.dot(v_base, v) > 0.999999: return [0, 0, 0, 1]
    if np.dot(v_base, v) < -0.999999: return [0, 0, 0 ,-1]

    a = np.cross(v_base, v)
    q = a.tolist()
    q.append(np.sqrt((np.linalg.norm(v_base, 2)**2) * (np.linalg.norm(v, 2)**2)) + np.dot(v_base, v))
    q = q/np.linalg.norm(q, 2)

    return q



    """
    Be aware that this does not handle the case of parallel vectors (both in the same direction or pointing in opposite directions). 
    crossproduct will not be valid in these cases, so you first need to check dot(v1, v2) > 0.999999 and dot(v1, v2) < -0.999999, respectively,
    and either return an identity quat for parallel vectors, or return a 180 degree rotation (about any axis) for opposite vectors.
    """

def conjugateQuaternion(q: Union[List, np.ndarray], style='xyzw'):
    if type(q) is list:
        q = np.array(q)

def rotate_vector(v: Union[List, np.ndarray], q: Union[List, np.ndarray]) -> List:
    """
    Quite literally made with chatGPT
    """
    if type(v) is list:
        v = np.array(v)
    if type(q) is list:
        q = np.array(q)

    # Convert the input vector to a 4D column vector
    v_homogeneous = np.array([[v[0]], [v[1]], [v[2]], [1]])

    # Convert the quaternion to a 4x4 rotation matrix
    q_matrix = np.array([[1 - 2*q[1]**2 - 2*q[2]**2, 2*q[0]*q[1] - 2*q[2]*q[3], 2*q[0]*q[2] + 2*q[1]*q[3], 0],
                        [2*q[0]*q[1] + 2*q[2]*q[3], 1 - 2*q[0]**2 - 2*q[2]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 0],
                        [2*q[0]*q[2] - 2*q[1]*q[3], 2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[0]**2 - 2*q[1]**2, 0],
                        [0, 0, 0, 1]])

    # Rotate the vector using the rotation matrix
    v_rotated = q_matrix @ v_homogeneous

    # Convert the result back to a 3D vector
    return np.array([v_rotated[0,0], v_rotated[1,0], v_rotated[2,0]]).tolist()

    


if __name__ == '__main__':
    q = np.array(getOrientationFromDirectionalVector([0,1,0]))
    print(q)
    
    