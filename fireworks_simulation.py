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
    

@dataclass
class Particle:
    """Particle with position, velocity, color, and life"""
    position: np.ndarray
    velocity: np.ndarray
    color: np.ndarray
    life: float
    size: float
    trail: list



class Firework:
    """Firework with rocket and explosion particles"""
    
    def __init__(self, position, firework_type="burst"):
        self.position = position.copy()
        self.particles = []
        self.exploded = False
        self.rocket = None
        self.type = firework_type
        self.age = 0.0
        
        self.launch_rocket()
    
    def launch_rocket(self):
        """Launch rocket particle"""
        velocity = np.array([
            random.uniform(-0.5, 0.5),
            random.uniform(18.0, 25.0),
            random.uniform(-0.5, 0.5)
        ], dtype=np.float32)
        
        self.rocket = Particle(
            position=self.position.copy(),
            velocity=velocity,
            color=np.array([1.0, 1.0, 0.8, 1.0], dtype=np.float32),
            life=1.0,
            size=3.0,
            trail=[]
        )
    
    def explode(self):
        """Create explosion using transformation matrices"""
        if self.exploded:
            return
        
        self.exploded = True
        explosion_position = self.rocket.position.copy()
        
        # Number of particles
        num_particles = random.randint(150, 250)
        
        # Random color
        base_hue = random.random()
        
        # Random rotation for variety (ROTATION TRANSFORMATION)
        rotation_angle = random.uniform(0, 360)
        rotation_axis = np.array([
            random.uniform(-1, 1),
            random.uniform(0.5, 1),
            random.uniform(-1, 1)
        ], dtype=np.float32)
        
        for i in range(num_particles):
            
            # Generate base velocity based on type
            if self.type == "burst":
                # Spherical burst
                theta = random.uniform(0, 2 * math.pi)
                phi = random.uniform(0, math.pi)
                speed = random.uniform(10.0, 16.0)
                
                base_velocity = np.array([
                    speed * math.sin(phi) * math.cos(theta),
                    speed * math.cos(phi),
                    speed * math.sin(phi) * math.sin(theta)
                ], dtype=np.float32)
                
            elif self.type == "ring":
                # Circular ring (SCALING TRANSFORMATION - flatten Y)
                angle = (i / num_particles) * 2 * math.pi
                speed = random.uniform(12.0, 18.0)
                
                base_velocity = np.array([
                    speed * math.cos(angle),
                    random.uniform(-1.0, 1.0),
                    speed * math.sin(angle)
                ], dtype=np.float32)
                
            elif self.type == "fountain":
                # Fountain with shearing (SHEARING TRANSFORMATION)
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(8.0, 14.0)
                vertical_bias = random.uniform(0.6, 0.9)
                
                base_velocity = np.array([
                    speed * math.cos(angle) * (1 - vertical_bias),
                    speed * vertical_bias,
                    speed * math.sin(angle) * (1 - vertical_bias)
                ], dtype=np.float32)
                
            elif self.type == "willow":
                # Willow droop effect
                theta = random.uniform(0, 2 * math.pi)
                phi = random.uniform(0, math.pi / 2)
                speed = random.uniform(7.0, 12.0)
                
                base_velocity = np.array([
                    speed * math.sin(phi) * math.cos(theta),
                    speed * math.cos(phi) * 0.4,
                    speed * math.sin(phi) * math.sin(theta)
                ], dtype=np.float32)
                
            else:
                base_velocity = np.array([
                    random.uniform(-12.0, 12.0),
                    random.uniform(-12.0, 12.0),
                    random.uniform(-12.0, 12.0)
                ], dtype=np.float32)
            
            # APPLY ROTATION TRANSFORMATION
            R = TransformationMatrix.rotation_arbitrary(rotation_angle, rotation_axis)
            rotated_velocity = TransformationMatrix.apply_to_point(R, base_velocity)
            
            # APPLY SCALING TRANSFORMATION
            scale_variation = random.uniform(0.85, 1.15)
            S = TransformationMatrix.scaling(scale_variation, scale_variation, scale_variation)
            scaled_velocity = TransformationMatrix.apply_to_point(S, rotated_velocity)
            
            # APPLY SHEARING TRANSFORMATION (for certain types)
            if self.type in ["fountain", "willow"]:
                shear_factor = random.uniform(-0.4, 0.4)
                SH = TransformationMatrix.shearing_xy(shear_factor, 0.0)
                final_velocity = TransformationMatrix.apply_to_point(SH, scaled_velocity)
            else:
                final_velocity = scaled_velocity
            
            # Generate color with variation
            hue_offset = random.uniform(-0.1, 0.1)
            final_hue = (base_hue + hue_offset) % 1.0
            
            r, g, b = self.hsv_to_rgb(final_hue, 0.9, 1.0)
            
            particle = Particle(
                position=explosion_position.copy(),
                velocity=final_velocity,
                color=np.array([r, g, b, 1.0], dtype=np.float32),
                life=1.0,
                size=random.uniform(2.5, 4.5),
                trail=[]
            )
            
            self.particles.append(particle)
    
    @staticmethod
    def hsv_to_rgb(h, s, v):
        """Convert HSV to RGB"""
        i = int(h * 6.0)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6
        
        if i == 0: return v, t, p
        if i == 1: return q, v, p
        if i == 2: return p, v, t
        if i == 3: return p, q, v
        if i == 4: return t, p, v
        if i == 5: return v, p, q
        return v, v, v
    
    def update(self, dt):
        """Update firework state"""
        self.age += dt
        
        gravity = np.array([0.0, -9.8, 0.0], dtype=np.float32)
        
        # Update rocket
        if not self.exploded and self.rocket:
            # TRANSLATION: Update rocket position using transformation matrix
            displacement = self.rocket.velocity * dt
            T = TransformationMatrix.translation(displacement[0], displacement[1], displacement[2])
            self.rocket.position = TransformationMatrix.apply_to_point(T, self.rocket.position)
            
            # Update velocity
            self.rocket.velocity = self.rocket.velocity + gravity * dt
            
            # Trail
            if len(self.rocket.trail) < 20:
                self.rocket.trail.append(self.rocket.position.copy())
            
            # Decay
            self.rocket.life -= dt * 0.5
            self.rocket.color[3] = self.rocket.life
            
            # Explode when falling or life runs out
            if self.rocket.velocity[1] < 0 or self.rocket.life <= 0:
                self.explode()
        
        # Update particles
        if self.exploded:
            alive_particles = []
            
            for particle in self.particles:
                # TRANSLATION: Update particle position using transformation matrix
                displacement = particle.velocity * dt
                T = TransformationMatrix.translation(displacement[0], displacement[1], displacement[2])
                particle.position = TransformationMatrix.apply_to_point(T, particle.position)
                
                # SCALING: Apply drag to velocity (velocity scaling)
                drag_factor = 0.98
                S = TransformationMatrix.scaling(drag_factor, drag_factor, drag_factor)
                particle.velocity = TransformationMatrix.apply_to_point(S, particle.velocity)
                
                # Update velocity with gravity
                particle.velocity = particle.velocity + gravity * dt
                
                # Trail
                if len(particle.trail) < 10:
                    particle.trail.append(particle.position.copy())
                else:
                    particle.trail.pop(0)
                    particle.trail.append(particle.position.copy())
                
                # Life decay
                particle.life -= dt * 0.5
                particle.color[3] = max(0.0, particle.life)
                
                if particle.life > 0:
                    alive_particles.append(particle)
            
            self.particles = alive_particles
            
            return len(self.particles) > 0
        
        return True
    
    def render(self):
        """Render firework"""
        # Render rocket trail
        if self.rocket and not self.exploded:
            if len(self.rocket.trail) > 1:
                glBegin(GL_LINE_STRIP)
                for i, pos in enumerate(self.rocket.trail):
                    alpha = (i / len(self.rocket.trail)) * self.rocket.color[3]
                    glColor4f(1.0, 1.0, 0.5, alpha * 0.5)
                    glVertex3f(pos[0], pos[1], pos[2])
                glEnd()
            
            # Render rocket
            glPointSize(self.rocket.size)
            glBegin(GL_POINTS)
            glColor4f(self.rocket.color[0], self.rocket.color[1], 
                     self.rocket.color[2], self.rocket.color[3])
            glVertex3f(self.rocket.position[0], self.rocket.position[1], self.rocket.position[2])
            glEnd()
        
        # Render particles
        for particle in self.particles:
            # Trail
            if len(particle.trail) > 1:
                glBegin(GL_LINE_STRIP)
                for i, pos in enumerate(particle.trail):
                    alpha = (i / len(particle.trail)) * particle.color[3] * 0.3
                    glColor4f(particle.color[0], particle.color[1], particle.color[2], alpha)
                    glVertex3f(pos[0], pos[1], pos[2])
                glEnd()
            
            # Glow (outer)
            glPointSize(particle.size * 2.5)
            glBegin(GL_POINTS)
            glColor4f(particle.color[0], particle.color[1], particle.color[2], particle.color[3] * 0.2)
            glVertex3f(particle.position[0], particle.position[1], particle.position[2])
            glEnd()
            
            # Core
            glPointSize(particle.size)
            glBegin(GL_POINTS)
            glColor4f(particle.color[0], particle.color[1], particle.color[2], particle.color[3])
            glVertex3f(particle.position[0], particle.position[1], particle.position[2])
            glEnd()

