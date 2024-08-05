import numpy as np

def kinetic_energy(mass, velocity):
    """
    Calculate the kinetic energy of an object.
    
    Args:
    mass (float): Mass of the object in kg.
    velocity (float or np.array): Velocity of the object in m/s.
    
    Returns:
    float: Kinetic energy in Joules.
    """
    if isinstance(velocity, np.ndarray):
        v_squared = np.dot(velocity, velocity)
    else:
        v_squared = velocity ** 2
    return 0.5 * mass * v_squared

def potential_energy_gravitational(mass, height, gravity=9.8):
    """
    Calculate the gravitational potential energy of an object.
    
    Args:
    mass (float): Mass of the object in kg.
    height (float): Height of the object above the reference point in m.
    gravity (float): Acceleration due to gravity in m/s^2. Default is 9.8 m/s^2.
    
    Returns:
    float: Gravitational potential energy in Joules.
    """
    return mass * gravity * height

def force_gravitational(mass1, mass2, distance, G=6.67430e-11):
    """
    Calculate the gravitational force between two objects.
    
    Args:
    mass1 (float): Mass of the first object in kg.
    mass2 (float): Mass of the second object in kg.
    distance (float): Distance between the centers of the objects in m.
    G (float): Gravitational constant. Default is 6.67430e-11 N(m/kg)^2.
    
    Returns:
    float: Gravitational force in Newtons.
    """
    return G * mass1 * mass2 / (distance ** 2)

def momentum(mass, velocity):
    """
    Calculate the momentum of an object.
    
    Args:
    mass (float): Mass of the object in kg.
    velocity (float or np.array): Velocity of the object in m/s.
    
    Returns:
    float or np.array: Momentum in kg*m/s.
    """
    return mass * velocity

def work_done(force, displacement):
    """
    Calculate the work done by a constant force.
    
    Args:
    force (float or np.array): Force applied in Newtons.
    displacement (float or np.array): Displacement in meters.
    
    Returns:
    float: Work done in Joules.
    """
    return np.dot(force, displacement)

def simple_harmonic_motion(amplitude, angular_frequency, time, phase=0):
    """
    Calculate the displacement of an object in simple harmonic motion.
    
    Args:
    amplitude (float): Maximum displacement from equilibrium position in m.
    angular_frequency (float): Angular frequency of oscillation in rad/s.
    time (float or np.array): Time in seconds.
    phase (float): Phase angle in radians. Default is 0.
    
    Returns:
    float or np.array: Displacement from equilibrium position in m.
    """
    return amplitude * np.cos(angular_frequency * time + phase)
