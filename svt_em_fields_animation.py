import numpy as np
import pyvista as pv
import os

# === Settings ===
output_dir = "svt_em_3d_outputs"
os.makedirs(output_dir, exist_ok=True)

# === Domain ===
radius = 1.0
height = 4.0
res = 40

x, y, z = np.meshgrid(
    np.linspace(-1.5 * radius, 1.5 * radius, res),
    np.linspace(-1.5 * radius, 1.5 * radius, res),
    np.linspace(-0.5 * height, 0.5 * height, res),
    indexing="ij"
)

R = np.sqrt(x**2 + y**2)
Theta = np.arctan2(y, x)

# === Parameters ===
I = 65 / 12  # Current ≈ 5.4 A
mu0 = 4e-7 * np.pi

# === Magnetic Field (Classical + SVT) ===
B_mag = np.where(
    R < radius,
    mu0 * I * R / (2 * np.pi * radius**2),
    mu0 * I / (2 * np.pi * R)
)

# SVT central vortex enhancement
svt_factor = 1 + 0.8 * np.exp(-10 * R**2)
B_mag *= svt_factor

# Convert to vector field
Bx = -B_mag * np.sin(Theta)
By = B_mag * np.cos(Theta)
Bz = -0.1 * svt_factor  # inward twisting Z

# === Mercury Flow (along Z-axis parabolic) ===
Vx = np.zeros_like(Bx)
Vy = np.zeros_like(By)
Vz = (1 - (R / radius)**2) * (R < radius)

# === PyVista 3D Render ===
grid = pv.StructuredGrid(x, y, z)
grid["B_field"] = np.stack((Bx, By, Bz), axis=-1).reshape(-1, 3)
grid["Flow"] = np.stack((Vx, Vy, Vz), axis=-1).reshape(-1, 3)

# Save VTK for further analysis
grid.save(os.path.join(output_dir, "svt_em_fields.vtk"))

# Plot
plotter = pv.Plotter()
plotter.add_mesh(grid.glyph(orient="B_field", scale=False, factor=0.3), color="blue", label="Magnetic Field")
plotter.add_mesh(grid.glyph(orient="Flow", scale=False, factor=0.2), color="red", label="Mercury Flow")
plotter.add_mesh(pv.Cylinder(radius=radius, height=height, direction=(0, 0, 1)), color='gray', style='wireframe', opacity=0.3)
plotter.set_background("white")
plotter.view_xy()
plotter.show(screenshot=os.path.join(output_dir, "svt_em_fields_render.png"))
