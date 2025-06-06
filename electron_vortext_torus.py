import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import imageio

# Parameters
R = 1
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

frames = []

# Generate frames
for frame in range(60):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.0, 1.0])
    ax.axis('off')
    ax.set_box_aspect([1,1,0.66])

    scale = 1 + 0.2 * np.sin(2 * np.pi * frame / 50)
    r = 0.3 * scale

    X_ = (R + r * np.cos(theta)) * np.cos(phi)
    Y_ = (R + r * np.cos(theta)) * np.sin(phi)
    Z_ = r * np.sin(theta)

    energy = np.sin(theta)**2 + np.cos(phi)**2
    ax.plot_surface(X_, Y_, Z_, facecolors=plt.cm.plasma(energy / np.max(energy)),
                    rstride=2, cstride=2, antialiased=True, alpha=0.9)

    # Save frame to buffer
    filename = f"frame_{frame:03d}.png"
    plt.savefig(filename, dpi=80, bbox_inches='tight')
    plt.close()
    frames.append(imageio.imread(filename))

# Export to GIF
imageio.mimsave('toroidal_vortex.gif', frames, fps=20)

# Clean up frames
import os
for frame in range(60):
    os.remove(f"frame_{frame:03d}.png")
