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
fig, ax = plt.subplots(figsize=(8, 8))
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])  # Fixed position for colorbar
cbar = [None]  # Mutable reference to current colorbar

# Overlay comparison field (placeholder for 2p orbital shape)
def overlay_reference(ax):
    ax.plot([-1.5, 1.5], [0, 0], 'w--', lw=1.0, label='Approx. 2p Orbital Axis')
    ax.annotate('O nucleus', xy=(0, 0), xytext=(0.3, 0.3), color='white')
    ax.annotate('Electron lobes', xy=(1.5, 0), xytext=(1.6, 0.5), color='white')
    ax.legend(loc='upper right', facecolor='black', framealpha=0.5)

def update(t):
    ax.clear()
    ax.set_title(f"Multi-Vortex Field – Frame {t}", fontsize=12)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

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
    contour = ax.contourf(X, Y, np.log(energy + 1), levels=40, cmap='inferno', alpha=0.5)

    # Update colorbar using fixed axis
    cbar_ax.cla()
    cbar[0] = fig.colorbar(contour, cax=cbar_ax)
    cbar[0].set_label('Log Energy Density')

    overlay_reference(ax)

ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False, repeat=False)
plt.subplots_adjust(right=0.9)
plt.show()
