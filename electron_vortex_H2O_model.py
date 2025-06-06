# Space Vortex Theory Inspired: Multi-Vortex Interaction Simulation (H2O Analog)
# Includes time-evolution, sinusoidal spin variation, and nucleus-mass sink modeling

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation grid
N = 300
x = np.linspace(-5, 5, N)
y = np.linspace(-5, 5, N)
X, Y = np.meshgrid(x, y)

# Base parameters
vortex_base_strength = 10
vortex_sep = 2.5
nucleus_strength = -30  # attractive mass sink

# Vortex positions (angle ~104.5° for H2O)
theta = np.radians(52.25)  # half of 104.5°
r = vortex_sep / 2
x1, y1 = -r * np.cos(theta), r * np.sin(theta)
x2, y2 = r * np.cos(theta), r * np.sin(theta)
x3, y3 = 0, -1.0  # third vortex to form triangle

# Oxygen nucleus at center
xO, yO = 0, 0

# Function to create vortex vector field
def vortex_field(x0, y0, strength):
    dx = X - x0
    dy = Y - y0
    r2 = dx**2 + dy**2 + 0.1
    u = -strength * dy / r2
    v = strength * dx / r2
    return u, v

# Function to create attractive mass sink
def mass_sink_field(x0, y0, strength):
    dx = X - x0
    dy = Y - y0
    r2 = dx**2 + dy**2 + 0.1
    u = strength * dx / r2
    v = strength * dy / r2
    return u, v

# Time evolution
frames = 60
fig, ax = plt.subplots(figsize=(7, 7))

def update(t):
    ax.clear()
    ax.set_title(f"Multi-Vortex Field – Frame {t}")
    ax.axis('equal')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axis('off')

    # Sinusoidally varying vortex strength
    strength = vortex_base_strength * (1 + 0.2 * np.sin(2 * np.pi * t / frames))

    # Compute vortex fields
    u1, v1 = vortex_field(x1, y1, strength)
    u2, v2 = vortex_field(x2, y2, strength)
    u3, v3 = vortex_field(x3, y3, strength)
    
    # Nucleus (oxygen) pull field
    uO, vO = mass_sink_field(xO, yO, nucleus_strength)

    # Superpose all fields
    total_u = u1 + u2 + u3 + uO
    total_v = v1 + v2 + v3 + vO
    energy = total_u**2 + total_v**2

    # Plot field
    ax.streamplot(X, Y, total_u, total_v, color=np.log(energy + 1), density=1.2, cmap='inferno')
    ax.contourf(X, Y, np.log(energy + 1), levels=40, cmap='inferno', alpha=0.5)

ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)
plt.show()
