import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
AU = 1.496e11  # Astronomical Unit in meters
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_sun = 1.989e30  # Mass of the Sun (kg)
c = 3e8  # Speed of light (m/s)
days = 90560  # Length of Pluto's year in Earth days
dt = 24 * 3600  # Time step: 1 day in seconds

# Planet data from NASA (mean orbital radius in AU, orbital period in days)
planet_data = {
    'Mercury': {'a': 0.39, 'period': 88},
    'Venus': {'a': 0.72, 'period': 225},
    'Earth': {'a': 1.00, 'period': 365},
    'Mars': {'a': 1.52, 'period': 687},
    'Jupiter': {'a': 5.20, 'period': 4333},
    'Saturn': {'a': 9.58, 'period': 10759},
    'Uranus': {'a': 19.18, 'period': 30687},
    'Neptune': {'a': 30.07, 'period': 60190},
    'Pluto': {'a': 39.48, 'period': 90560}
}

# Initialize output structure
trajectories = {
    'Newtonian': {name: [] for name in planet_data},
    'Relativistic': {name: [] for name in planet_data},
    'SVT': {name: [] for name in planet_data}
}

# Simulate orbits
for model in trajectories:
    for name, data in planet_data.items():
        a = data['a'] * AU
        T = data['period']
        v = 2 * np.pi * a / (T * 86400)  # m/s

        pos = np.array([a, 0.0])
        if model == 'SVT':
            vel = np.array([0.0, 0.95 * v])
        elif model == 'Relativistic':
            vel = np.array([0.0, 1.03 * v])
        else:
            vel = np.array([0.0, v])

        path = []
        for _ in range(days):
            r = np.linalg.norm(pos)
            if model == 'SVT':
                accel = -G * M_sun * pos / (r**3 + 1e18)
            elif model == 'Relativistic':
                l = np.cross(np.append(pos, 0), np.append(vel, 0))[-1]  # z-component
                relativistic_factor = 1 + (3 * l**2) / (r**2 * c**2)
                accel = -G * M_sun * pos / r**3 * relativistic_factor
            else:
                accel = -G * M_sun * pos / r**3

            vel += accel * dt
            pos += vel * dt
            path.append(pos.copy())

        trajectories[model][name] = np.array(path) / AU  # convert to AU

# Plot results
import matplotlib.cm as cm
colors = cm.get_cmap('tab10', 10)
output_dir = "solar_orbit_comparison_outputs"
os.makedirs(output_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 14), dpi=300)
for idx, name in enumerate(planet_data):
    for model, style in zip(['Newtonian', 'Relativistic', 'SVT'], ['-', '--', ':']):
        path = trajectories[model][name]
        ax.plot(path[:, 0], path[:, 1], style, color=colors(idx), label=f"{name} ({model})")

ax.plot(0, 0, 'yo', markersize=12, label='Sun')
ax.set_title("Planetary Orbits: Newtonian vs Relativistic vs SVT (Full Pluto Year)")
ax.set_xlabel("X Position (AU)")
ax.set_ylabel("Y Position (AU)")
ax.set_xlim(-45, 45)
ax.set_ylim(-45, 45)
ax.grid(True)
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')
plt.tight_layout()
output_path = os.path.join(output_dir, "solar_orbits_comparison_all_planets_highres.png")
plt.savefig(output_path)
plt.close()
print(f"✅ Saved: {output_path}")
