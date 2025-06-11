import numpy as np
import pyvista as pv
import os

# Create output directory
output_dir = os.path.join(os.getcwd(), "svt_em_animated_outputs")
os.makedirs(output_dir, exist_ok=True)

# === Domain setup ===
radius = 1.0
height = 4.0
res = 30

x, y, z = np.meshgrid(
    np.linspace(-1.5 * radius, 1.5 * radius, res),
    np.linspace(-1.5 * radius, 1.5 * radius, res),
    np.linspace(-0.5 * height, 0.5 * height, res),
    indexing="ij"
)

R = np.sqrt(x**2 + y**2)
Theta = np.arctan2(y, x)

# Constants
I = 65 / 12  # Approx. 5.4 A current (from 65W @ 12V)
mu0 = 4e-7 * np.pi

# SVT magnetic field with animation over time
def generate_fields(t_frac):
    B_mag = np.where(
        R < radius,
        mu0 * I * R / (2 * np.pi * radius**2),
        mu0 * I / (2 * np.pi * R)
    )

    # Time-evolving inward vortex
    svt_factor = 1 + 0.8 * np.exp(-10 * R**2) * np.sin(2 * np.pi * t_frac)
    B_mag *= svt_factor

    Bx = -B_mag * np.sin(Theta)
    By = B_mag * np.cos(Theta)
    Bz = -0.1 * svt_factor  # slight Z-vortex component

    Vx = np.zeros_like(Bx)
    Vy = np.zeros_like(By)
    Vz = (1 - (R / radius)**2) * (R < radius)  # inward mercury flow

    return np.stack((Bx, By, Bz), axis=-1), np.stack((Vx, Vy, Vz), axis=-1)

# Create animation
plotter = pv.Plotter(off_screen=True)
plotter.open_gif(os.path.join(output_dir, "svt_em_fields_animation.gif"))

cylinder = pv.Cylinder(radius=radius, height=height, direction=(0, 0, 1))

for frame in range(40):
    t_frac = frame / 40
    B_field, Flow = generate_fields(t_frac)

    grid = pv.StructuredGrid(x, y, z)
    grid["B_field"] = B_field.reshape(-1, 3)
    grid["Flow"] = Flow.reshape(-1, 3)

    glyphs_B = grid.glyph(orient="B_field", scale=False, factor=0.25)
    glyphs_V = grid.glyph(orient="Flow", scale=False, factor=0.15)

    plotter.clear()
    plotter.add_mesh(glyphs_B, color="blue")
    plotter.add_mesh(glyphs_V, color="red")
    plotter.add_mesh(cylinder, color="gray", style="wireframe", opacity=0.2)
    plotter.set_background("white")
    plotter.camera_position = 'xy'
    plotter.write_frame()

plotter.close()
print("✅ Animation saved to:", os.path.join(output_dir, "svt_em_fields_animation.gif"))
