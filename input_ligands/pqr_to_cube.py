import numpy as np
import sys
from pathlib import Path

def parse_pqr(pqr_path):
    atoms = []
    with open(pqr_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                parts = line.split()
                x, y, z = map(float, parts[5:8])
                charge = float(parts[8])
                atoms.append((x, y, z, charge))
    return np.array(atoms)

def generate_cube(atoms, spacing=0.5, padding=5.0):
    coords = atoms[:, :3]
    charges = atoms[:, 3]
    min_corner = coords.min(axis=0) - padding
    max_corner = coords.max(axis=0) + padding
    grid_dims = np.ceil((max_corner - min_corner) / spacing).astype(int)
    nx, ny, nz = grid_dims

    grid = np.zeros((nx, ny, nz), dtype=np.float32)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                point = min_corner + np.array([i, j, k]) * spacing
                r = np.linalg.norm(coords - point, axis=1)
                r[r < 1e-6] = 1e-6
                potential = np.sum(charges / r)
                grid[i, j, k] = potential

    origin = min_corner
    return origin, spacing, grid

def write_cube(filename, origin, spacing, grid, atoms):
    nx, ny, nz = grid.shape
    with open(filename, 'w') as f:
        f.write("ESP DNN cube file\nGenerated from .pqr\n")
        f.write(f"{len(atoms):5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}\n")
        f.write(f"{nx:5d} {spacing:12.6f} 0.000000 0.000000\n")
        f.write(f"{ny:5d} 0.000000 {spacing:12.6f} 0.000000\n")
        f.write(f"{nz:5d} 0.000000 0.000000 {spacing:12.6f}\n")
        for atom in atoms:
            f.write(f" 0 0.000000 {atom[0]:12.6f} {atom[1]:12.6f} {atom[2]:12.6f}\n")

        flat_grid = grid.transpose(2, 1, 0).flatten()
        for i in range(0, len(flat_grid), 6):
            line = " ".join(f"{val:13.5e}" for val in flat_grid[i:i+6])
            f.write(line + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pqr_to_cube.py molecule.pqr")
        sys.exit(1)

    pqr_file = Path(sys.argv[1])
    atoms = parse_pqr(pqr_file)
    origin, spacing, grid = generate_cube(atoms)
    cube_file = pqr_file.with_suffix(".cube")
    write_cube(cube_file, origin, spacing, grid, atoms)
    print(f"Saved cube file to: {cube_file}")
