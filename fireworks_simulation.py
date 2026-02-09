import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import random
import math
import time
from dataclasses import dataclass
from typing import List


class TransformationMatrix:
    @staticmethod
    def identity():
        """
        Identity Matrix - no transformation
        [1 0 0 0]
        [0 1 0 0]
        [0 0 1 0]
        [0 0 0 1]
        """
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def translation(tx, ty, tz):
        """
        Translation Matrix
        
        Mathematical form:
        [1  0  0  tx]   [x]   [x + tx]
        [0  1  0  ty] × [y] = [y + ty]
        [0  0  1  tz]   [z]   [z + tz]
        [0  0  0  1 ]   [1]   [  1   ]
        """
        return np.array([
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def rotation_x(angle_degrees):
        """
        Rotation around X-axis
        
        Mathematical form:
        [1    0         0      0]
        [0  cos(θ)  -sin(θ)   0]
        [0  sin(θ)   cos(θ)   0]
        [0    0         0      1]
        """
        theta = math.radians(angle_degrees)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        return np.array([
            [1.0, 0.0,        0.0,         0.0],
            [0.0, cos_theta, -sin_theta,   0.0],
            [0.0, sin_theta,  cos_theta,   0.0],
            [0.0, 0.0,        0.0,         1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def rotation_y(angle_degrees):
        """
        Rotation around Y-axis
        
        Mathematical form:
        [ cos(θ)   0  sin(θ)  0]
        [   0      1    0     0]
        [-sin(θ)   0  cos(θ)  0]
        [   0      0    0     1]
        """
        theta = math.radians(angle_degrees)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        return np.array([
            [ cos_theta, 0.0, sin_theta, 0.0],
            [ 0.0,       1.0, 0.0,       0.0],
            [-sin_theta, 0.0, cos_theta, 0.0],
            [ 0.0,       0.0, 0.0,       1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def rotation_z(angle_degrees):
        """
        Rotation around Z-axis
        
        Mathematical form:
        [cos(θ)  -sin(θ)  0  0]
        [sin(θ)   cos(θ)  0  0]
        [  0        0     1  0]
        [  0        0     0  1]
        """
        theta = math.radians(angle_degrees)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        return np.array([
            [cos_theta, -sin_theta, 0.0, 0.0],
            [sin_theta,  cos_theta, 0.0, 0.0],
            [0.0,        0.0,       1.0, 0.0],
            [0.0,        0.0,       0.0, 1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def rotation_arbitrary(angle_degrees, axis):
        """
        Rotation around arbitrary axis
        Uses Rodrigues' rotation formula. Something new, not in our course.
        
        axis: normalized 3D vector [x, y, z]
        """
        theta = math.radians(angle_degrees)
        
        # Normalize axis
        axis_length = math.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
        if axis_length == 0:
            return TransformationMatrix.identity()
        
        ux = axis[0] / axis_length
        uy = axis[1] / axis_length
        uz = axis[2] / axis_length
        
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        one_minus_cos = 1.0 - cos_theta
        
        # Rodrigues' formula components
        return np.array([
            [
                cos_theta + ux*ux*one_minus_cos,
                ux*uy*one_minus_cos - uz*sin_theta,
                ux*uz*one_minus_cos + uy*sin_theta,
                0.0
            ],
            [
                uy*ux*one_minus_cos + uz*sin_theta,
                cos_theta + uy*uy*one_minus_cos,
                uy*uz*one_minus_cos - ux*sin_theta,
                0.0
            ],
            [
                uz*ux*one_minus_cos - uy*sin_theta,
                uz*uy*one_minus_cos + ux*sin_theta,
                cos_theta + uz*uz*one_minus_cos,
                0.0
            ],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def scaling(sx, sy, sz):
        """
        Scaling Matrix
        Scales by factors (sx, sy, sz)
        
        Mathematical form:
        [sx  0   0  0]   [x]   [sx·x]
        [0  sy   0  0] × [y] = [sy·y]
        [0   0  sz  0]   [z]   [sz·z]
        [0   0   0  1]   [1]   [ 1  ]
        """
        return np.array([
            [sx,  0.0, 0.0, 0.0],
            [0.0, sy,  0.0, 0.0],
            [0.0, 0.0, sz,  0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def shearing_xy(shx, shy):
        """
        Shearing in XY plane
        shx: shear X based on Y
        shy: shear Y based on X
        
        Mathematical form:
        [1   shx  0  0]   [x]   [x + shx·y]
        [shy  1   0  0] × [y] = [y + shy·x]
        [0    0   1  0]   [z]   [    z    ]
        [0    0   0  1]   [1]   [    1    ]
        """
        return np.array([
            [1.0, shx, 0.0, 0.0],
            [shy, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def shearing_xz(shx, shz):
        """
        Shearing in XZ plane
        shx: shear X based on Z
        shz: shear Z based on X
        """
        return np.array([
            [1.0, 0.0, shx, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [shz, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def shearing_yz(shy, shz):
        """
        Shearing in YZ plane
        shy: shear Y based on Z
        shz: shear Z based on Y
        """
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, shy, 0.0],
            [0.0, shz, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    @staticmethod
    def apply_to_point(matrix, point):
        """
        Apply 4x4 transformation matrix to a 3D point
        
        Returns: Transformed 3D point as numpy array
        """
        # Convert to homogeneous coordinates
        homogeneous_point = np.array([
            point[0],
            point[1],
            point[2],
            1.0
        ], dtype=np.float32)
        
        # Matrix multiplication: M × P
        transformed = np.dot(matrix, homogeneous_point)
        
        # Convert back to 3D
        if transformed[3] != 0.0:
            return np.array([
                transformed[0] / transformed[3],
                transformed[1] / transformed[3],
                transformed[2] / transformed[3]
            ], dtype=np.float32)
        
        return np.array([
            transformed[0],
            transformed[1],
            transformed[2]
        ], dtype=np.float32)
    
    @staticmethod
    def compose(*matrices):
        """
        Compose multiple transformation matrices to obtain composite transformation
        """
        result = TransformationMatrix.identity()
        
        # Apply in reverse order (right-to-left)
        for matrix in reversed(matrices):
            result = np.dot(result, matrix)
        
        return result
