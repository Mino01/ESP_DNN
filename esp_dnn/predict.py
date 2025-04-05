import os
import glob
import logging
import pickle
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

from .data_processing import normalize
from .featurize import Featurize
from .model_factory import custom_load_model


log = logging.getLogger("predict")


class AIChargeError(Exception):
    pass


class MolChargePredictor:
    def __init__(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_file = os.path.join(script_dir, "model", "trained_model.h5")
        features_file = os.path.join(script_dir, "model", "feature_list.dat")
        norm_params_file = os.path.join(script_dir, "model", "norm_params.pkl")

        self.model = custom_load_model(model_file)
        self.featurizer = Featurize(features_file=features_file, pad_value=0.0)
        self.skip_norm_mask = np.array(
            [v.startswith("is_") for v in self.featurizer.features]
        )
        with open(norm_params_file, "rb") as f:
            self.norm_params_dict = pickle.load(f, encoding="latin1")

    def predict_dqs_from_pdb_block(self, pdb_block):
        try:
            mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, proximityBonding=False)
            if mol is None:
                raise AIChargeError("Failed to parse PDB block with RDKit.")

            features_array, neighbours_array = self.featurizer.get_mol_fetaures(mol)
            features_array = np.expand_dims(features_array, axis=0)
            neighbours_array = np.expand_dims(neighbours_array, axis=0)

            features_array, _, _ = normalize(
                features_array,
                skip_norm_mask=self.skip_norm_mask,
                params_dict=self.norm_params_dict,
            )

            charges = self.model.predict([features_array, neighbours_array]).flatten()
            charges -= charges.sum() / len(charges)

            heavy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
            if len(charges) != len(heavy_atoms):
                raise AIChargeError("Mismatch between atoms and predicted charges")

            dqs = []
            charge_idx = 0
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 1:
                    continue
                res_info = atom.GetPDBResidueInfo()
                if res_info is None:
                    raise AIChargeError("Missing residue info in atom.")
                dqs.append((res_info.GetSerialNumber(), charges[charge_idx]))
                charge_idx += 1

            return dqs
        except Exception as e:
            raise AIChargeError(f"Error during prediction: {str(e)}")

    def dqs2pqr(self, pdb_block, dqs):
        output_lines = []
        atom_index = 0
        for line in pdb_block.splitlines():
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                try:
                    charge = float(dqs[atom_index][1])
                except IndexError:
                    raise AIChargeError("Mismatch while writing PQR: index out of range.")
                radius = 1.5
                newline = (
                    f"{line[:30]}"
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{charge:8.4f}{radius:7.2f}"
                )
                output_lines.append(newline)
                atom_index += 1
            else:
                output_lines.append(line)
        return "\n".join(output_lines) + "\n"

    def pdb_block2pqr_block(self, pdb_block):
        dqs = self.predict_dqs_from_pdb_block(pdb_block)
        return self.dqs2pqr(pdb_block, dqs)

    def pdb_file2pqr_block(self, pdb_file):
        with open(pdb_file) as f:
            return self.pdb_block2pqr_block(f.read())

    def pdb_file2pqr_file(self, pdb_file, pqr_file):
        with open(pqr_file, "w") as f:
            f.write(self.pdb_file2pqr_block(pdb_file))


def run(mode, input_dir, output_dir, stop_on_error):
    logging.basicConfig(level=logging.INFO, format="[%(name)s:%(levelname)s] %(message)s")
    log.info("Running in %s mode", mode)

    input_dir = os.path.abspath(input_dir)
    output_dir = input_dir if output_dir is None else os.path.abspath(output_dir)

    log.info("Input dir is %s", input_dir)
    log.info("Output dir is %s", output_dir)

    if mode == "ligand":
        for mol_file in glob.glob(os.path.join(input_dir, "*.mol")):
            mol = Chem.MolFromMolFile(mol_file, removeHs=False)
            if mol is None:
                log.warning("MOL2PDB: Error converting %s", mol_file)
                continue
            pdb_file = os.path.join(output_dir, os.path.basename(mol_file) + ".pdb")
            log.info("MOL2PDB: %s -> %s", mol_file, pdb_file)
            Chem.MolToPDBFile(mol, pdb_file)

    all_pdb_files = sorted(Path(input_dir).glob("*.pdb"))
    if not all_pdb_files:
        log.error("No input files found in %s", input_dir)
        return

    predictor = MolChargePredictor()

    for pdb_file in all_pdb_files:
        pqr_file = str(pdb_file) + ".pqr"
        log.info("%s -> %s", pdb_file, pqr_file)
        try:
            predictor.pdb_file2pqr_file(pdb_file, pqr_file)
        except AIChargeError as e:
            log.error("Error processing %s: %s", pdb_file, str(e))
            if stop_on_error:
                raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["ligand", "protein"], default="ligand")
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", default=None)
    parser.add_argument("-e", "--stop_on_error", action="store_true")

    args = parser.parse_args()
    run(args.mode, args.input_dir, args.output_dir, args.stop_on_error)
