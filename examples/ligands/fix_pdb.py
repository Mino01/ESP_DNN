from rdkit import Chem
from rdkit.Chem import AllChem

# Load original MOL file (preserving hydrogens)
mol = Chem.MolFromMolFile("lig1_charged.mol", removeHs=False)

# Add hydrogens if not already explicit
mol = Chem.AddHs(mol)

# Generate 3D coordinates
AllChem.EmbedMolecule(mol, AllChem.ETKDG())
AllChem.UFFOptimizeMolecule(mol)

# Save fixed PDB
Chem.MolToPDBFile(mol, "lig1_charged_fixed.pdb")

# Print atom count
print(f"Number of atoms in fixed molecule: {mol.GetNumAtoms()}")

