# Copyright 2019 Astex Therapeutics Ltd.
#
# Licensed under the Apache License, 

import os
import glob
import logging
import subprocess
import argparse
import numpy as np

try:
    from rdkit import Chem
except ImportError:
    raise ImportError("RDKit is required to run this script. Please install it with 'conda install -c rdkit rdkit'.")

from esp_dnn.model_factory import custom_load_model
from esp_dnn.featurize import Featurize

log = logging.getLogger("predict")


def convert_mol_to_pdb_rdkit(mol_file, pdb_file):
    mol = Chem.MolFromMolFile(mol_file, removeHs=False)
    if mol is None:
        log.warning("MOL2PDB: Error converting %s", mol_file)
        return None
    Chem.MolToPDBFile(mol, pdb_file)
    return pdb_file


def convert_mol2_to_pdb_obabel(mol2_file, pdb_file):
    result = subprocess.run(["obabel", mol2_file, "-O", pdb_file], capture_output=True)
    if result.returncode != 0:
        log.warning("MOL2PDB: OpenBabel failed to convert %s: %s", mol2_file, result.stderr.decode())
        return None
    return pdb_file


def process_input_files(input_dir, output_dir):
    pdb_files = []

    for mol_file in glob.glob(os.path.join(input_dir, "*.mol")):
        pdb_file = os.path.join(output_dir, os.path.basename(mol_file) + ".pdb")
        result = convert_mol_to_pdb_rdkit(mol_file, pdb_file)
        if result:
            pdb_files.append(pdb_file)

    for mol2_file in glob.glob(os.path.join(input_dir, "*.mol2")):
        pdb_file = os.path.join(output_dir, os.path.basename(mol2_file) + ".pdb")
        result = convert_mol2_to_pdb_obabel(mol2_file, pdb_file)
        if result:
            pdb_files.append(pdb_file)

    return pdb_files


def write_pqr(mol, charges, output_file, all_atoms):
    """
    Writes a PQR file with coordinates, predicted charges, and placeholder radii.
    Includes all atoms to match ESP_DNN prediction output.
    """
    conf = mol.GetConformer()

    with open(output_file, "w") as f:
        for i, atom in enumerate(all_atoms):
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            pos = conf.GetAtomPosition(idx)
            charge = charges[i]
            radius = 1.5  # Placeholder; could use a real van der Waals lookup

            f.write(
                f"ATOM  {i+1:5d} {symbol:<4s} MOL     1    "
                f"{pos.x:8.3f} {pos.y:8.3f} {pos.z:8.3f}"
                f"{charge:8.4f} {radius:8.4f}"
            )


def predict_charges(model, featurizer, pdb_file):
    with open(pdb_file, 'r') as f:
        pdb_block = f.read()

    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, proximityBonding=False)
    if mol is None:
        log.error("Failed to parse molecule from %s", pdb_file)
        return

    features_array, neighbours_array = featurizer.get_mol_fetaures(mol)
    all_atoms = list(mol.GetAtoms())
    all_atoms = list(mol.GetAtoms())

    charges = model.predict(
        [np.expand_dims(features_array, axis=0), np.expand_dims(neighbours_array, axis=0)]
    )[0]
    if len(charges) != len(all_atoms):
        raise ValueError("Mismatch between atoms and predicted charges")

    pqr_file = pdb_file.replace(".pdb", ".pqr")
    write_pqr(mol, charges, pqr_file, all_atoms)
    log.info("Wrote: %s", pqr_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input ligand directory")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-m", "--model", required=True, help="Path to model file")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output or args.input)

    log.setLevel(logging.INFO)
    logging.basicConfig(format='[predict:%(levelname)s] %(message)s')

    log.info("Running in ligand mode")
    log.info("Input dir is %s", input_dir)
    log.info("Output dir is %s", output_dir)

    pdb_files = process_input_files(input_dir, output_dir)

    if not pdb_files:
        log.error("No input files found in %s", input_dir)
        return

    model = custom_load_model(args.model)
    featurizer = Featurize()

    for pdb_file in pdb_files:
        try:
            predict_charges(model, featurizer, pdb_file)
        except Exception as e:
            log.error("Error processing %s: %s", pdb_file, str(e))


if __name__ == "__main__":
    main()
