# Re-import libraries due to code execution reset
import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
days = 365
G = 1.0     # Simplified gravitational constant
M_sun = 1.0 # Mass of the Sun (scaled)
dt = 1.0    # Time step (1 day)

# Define planetary initial conditions (semi-major axis in AU, velocity for circular orbit)
planets = {
    "Mercury": {"r0": 0.39, "v0": 9.94},
    "Venus":   {"r0": 0.72, "v0": 7.35},
    "Earth":   {"r0": 1.00, "v0": 6.28},
    "Mars":    {"r0": 1.52, "v0": 5.08}
}

# Helper: Newtonian gravity (SVT-inspired radial acceleration)
def svt_force(pos):
    r2 = np.dot(pos, pos) + 0.01  # Avoid div by 0
    r = np.sqrt(r2)
    force_mag = G * M_sun / r2
    return -pos / r * force_mag

# Trajectory integration using Velocity Verlet
trajectories = {}
for name, data in planets.items():
    r = np.array([data["r0"], 0.0])
    v = np.array([0.0, data["v0"]])
    traj = [r.copy()]
    
    for _ in range(days):
        a = svt_force(r)
        r_next = r + v * dt + 0.5 * a * dt**2
        a_next = svt_force(r_next)
        v_next = v + 0.5 * (a + a_next) * dt
        r, v = r_next, v_next
        traj.append(r.copy())
    
    trajectories[name] = np.array(traj)

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))
for name, traj in trajectories.items():
    ax.plot(traj[:, 0], traj[:, 1], label=name)
ax.plot(0, 0, 'yo', label="Sun")
ax.set_aspect("equal")
ax.set_title("SVT-Inspired Planetary Orbits (365 Days)")
ax.set_xlabel("X (AU)")
ax.set_ylabel("Y (AU)")
ax.grid()
ax.legend()

# Save
output_dir = os.path.join(os.getcwd(), "svt_orbit_year")
os.makedirs(output_dir, exist_ok=True)
path = os.path.join(output_dir, "svt_vs_classical_orbits.png")
plt.savefig(path)
plt.close()
