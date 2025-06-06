# Space Vortex Theory Inspired: Two-Vortex Interaction Simulation
# Purpose: Model two electron-like rotating space vortices in 2D
# to observe field interaction, pressure zones, and stable configurations

import numpy as np
import matplotlib.pyplot as plt

# Simulation grid
N = 300
x = np.linspace(-5, 5, N)
y = np.linspace(-5, 5, N)
X, Y = np.meshgrid(x, y)

# Parameters
vortex_sep = 2.5  # distance between two vortices
vortex_strength = 10  # angular momentum proxy

# Define position of two vortices
x1, y1 = -vortex_sep/2, 0
x2, y2 = vortex_sep/2, 0

def vortex_field(x0, y0):
    dx = X - x0
    dy = Y - y0
    r2 = dx**2 + dy**2 + 0.1  # avoid division by 0
    u = -vortex_strength * dy / r2
    v = vortex_strength * dx / r2
    return u, v

# Superimpose two vortices
u1, v1 = vortex_field(x1, y1)
u2, v2 = vortex_field(x2, y2)

total_u = u1 + u2
total_v = v1 + v2

# Energy density (proxy)
energy_density = total_u**2 + total_v**2

# Plot vector field and energy
plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, total_u, total_v, color=np.log(energy_density + 1), cmap='plasma')
plt.contourf(X, Y, np.log(energy_density + 1), levels=40, alpha=0.6, cmap='plasma')
plt.colorbar(label='Log Energy Density')
plt.title('2-Vortex Interaction Field (Space Vortex Analog)')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.show()
