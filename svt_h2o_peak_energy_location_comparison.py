# Space Vortex Theory Inspired: Multi-Vortex Interaction Simulation (H2O Analog)
# Includes time-evolution, sinusoidal spin variation, and nucleus-mass sink modeling
# Plus: Extracts peak energy data, radial profile, and integrated energy for reporting

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
nucleus_strength = -30

# Vortex positions for H2O
theta = np.radians(52.25)
r = vortex_sep / 2
x1, y1 = -r * np.cos(theta), r * np.sin(theta)
x2, y2 = r * np.cos(theta), r * np.sin(theta)
x3, y3 = 0, -1.0
xO, yO = 0, 0

# Vortex and sink field generators
def vortex_field(x0, y0, strength):
    dx = X - x0
    dy = Y - y0
    r2 = dx**2 + dy**2 + 0.1
    u = -strength * dy / r2
    v = strength * dx / r2
    return u, v

def mass_sink_field(x0, y0, strength):
    dx = X - x0
    dy = Y - y0
    r2 = dx**2 + dy**2 + 0.1
    u = strength * dx / r2
    v = strength * dy / r2
    return u, v

# Trackers for report
total_energy_log = []
peak_distance_log = []
radial_profile_log = []

# Setup plot
frames = 60
fig, ax = plt.subplots(figsize=(8, 8))
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = [None]

# Overlay 2p orbital axis reference
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

    strength = vortex_base_strength * (1 + 0.2 * np.sin(2 * np.pi * t / frames))
    u1, v1 = vortex_field(x1, y1, strength)
    u2, v2 = vortex_field(x2, y2, strength)
    u3, v3 = vortex_field(x3, y3, strength)
    uO, vO = mass_sink_field(xO, yO, nucleus_strength)

    total_u = u1 + u2 + u3 + uO
    total_v = v1 + v2 + v3 + vO
    energy = total_u**2 + total_v**2

    # Plot field
    ax.streamplot(X, Y, total_u, total_v, color=np.log(energy + 1), density=1.2, cmap='inferno')
    contour = ax.contourf(X, Y, np.log(energy + 1), levels=40, cmap='inferno', alpha=0.5)

    cbar_ax.cla()
    cbar[0] = fig.colorbar(contour, cax=cbar_ax)
    cbar[0].set_label('Log Energy Density')
    overlay_reference(ax)

    # --- Analysis ---
    total_energy_log.append(np.sum(energy))
    peak_index = np.unravel_index(np.argmax(energy), energy.shape)
    peak_x, peak_y = X[peak_index], Y[peak_index]
    peak_dist = np.sqrt(peak_x**2 + peak_y**2)
    peak_distance_log.append(peak_dist)
    radial_profile_log.append(energy[N//2])  # Middle row (y=0)

ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False, repeat=False)
plt.subplots_adjust(right=0.9)
plt.show()

# === REPORT ===
print("\n--- SUMMARY REPORT ---")
print(f"Average Total Energy (arb): {np.mean(total_energy_log):.2f}")
print(f"Average Peak Distance from Center: {np.mean(peak_distance_log):.2f} units")
print("Compare with expected O–H bond length ≈ 0.96 Å")

# Plot final radial profile
plt.figure(figsize=(8, 4))
plt.plot(x, radial_profile_log[-1], label='Vortex Radial Energy')
plt.title("Radial Energy Profile vs 2p Orbital")
plt.xlabel("X-axis (cross-section)")
plt.ylabel("Energy Density")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()