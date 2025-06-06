import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os

# Setup
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

R = 1.0   # Major radius
r0 = 0.3  # Base minor radius
frames = []
ev_list = []

# Frame generation
for t in range(60):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-1.5, 1.5])
    ax.axis('off')
    ax.set_box_aspect([1, 1, 0.75])

    # Sinusoidal compression
    r = r0 * (1 + 0.2 * np.sin(2 * np.pi * t / 50))
    X = (R + r * np.cos(theta)) * np.cos(phi)
    Y = (R + r * np.cos(theta)) * np.sin(phi)
    Z = r * np.sin(theta)

    # Simulated energy field
    energy = (np.sin(theta)**2 + np.cos(phi)**2) * (1 + 0.2 * np.sin(2 * np.pi * t / 50))**2
    ev_list.append(np.sum(energy))

    ax.plot_surface(X, Y, Z, facecolors=plt.cm.inferno(energy / np.max(energy)),
                    rstride=2, cstride=2, antialiased=True, alpha=0.9)

    filename = f"torus3d_frame_{t:03d}.png"
    plt.savefig(filename, dpi=80, bbox_inches='tight')
    plt.close()
    frames.append(imageio.v2.imread(filename))

# Save animated GIF
imageio.mimsave('torus_vortex_3d.gif', frames, fps=15)

# Energy plot
plt.figure(figsize=(8, 4))
plt.plot(ev_list, color='darkorange')
plt.xlabel("Time Step")
plt.ylabel("Energy Density (EV)")
plt.title("3D Toroidal Energy Density Over Time")
plt.grid(True)
plt.savefig("ev_3d_over_time.png", bbox_inches='tight')
plt.close()

# Optional cleanup
for t in range(60):
    os.remove(f"torus3d_frame_{t:03d}.png")

print("✅ GIF saved as torus_vortex_3d.gif")
print("✅ Plot saved as ev_3d_over_time.png")
