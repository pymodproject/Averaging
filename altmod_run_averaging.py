import os
import argparse
from modeller import *
from modeller.automodel import *
from altmod_averaging import Automodel_averaging, run_averaging
from Bio import SeqIO
from shutil import copyfile
import modeller
from Bio.SeqRecord import SeqRecord
from Bio import PDB
import subprocess
from Bio.PDB import PDBParser, NeighborSearch
from Bio.SVDSuperimposer import SVDSuperimposer
import numpy as np
import shutil


def extract_ligand(pdb_file, output_file):
    parser = PDB.PDBParser()
    structure = parser.get_structure("structure", pdb_file)
    io = PDB.PDBIO()

    class LigandSelect(PDB.Select):
        def accept_residue(self, residue):
            return residue.id[0] != " "

    io.set_structure(structure)
    io.save(output_file, LigandSelect())

def remove_ligand(pdb_file, output_file):
    parser = PDB.PDBParser()
    structure = parser.get_structure("structure", pdb_file)
    io = PDB.PDBIO()

    class NotLigandSelect(PDB.Select):
        def accept_residue(self, residue):
            return residue.id[0] == " "

    io.set_structure(structure)
    io.save(output_file, NotLigandSelect())

def convert_to_sdf(input_file, output_file):

    #"obabel -ipdb 1bcd.B99990020_ligand.pdb -osdf -O 1bcd.B99990020_ligand.sdf"

    subprocess.run(["obabel", "-ipdb", input_file, "-osdf", "-O", output_file])


def extract_titles(file_path):
    titles = []
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) > 1:
                new_title_name = parts[1].replace("_ligand.pdb", ".pdb")
                titles.append(new_title_name)  # Assuming 'Title' is the second column
    return titles

def combine_pdb_files(titles, output_file):
    with open(output_file, 'w') as outfile:
        for model_number, title in enumerate(titles, start=1):
            try:
                with open(title, 'r') as infile:
                    outfile.write(f"MODEL     {model_number}\n")
                    outfile.write(infile.read())
                    outfile.write("ENDMDL\n")
            except FileNotFoundError:
                print(f"File {title} not found.")
            except Exception as e:
                print(f"An error occurred with file {title}: {e}")


def score_ligand_with_smina(receptor, ligand):

    new_name = ligand.replace("_ligand.sdf", "_ligand_scored.sdf")
    command = f"./smina.static --receptor {receptor} --ligand {ligand} --score_only -o {new_name}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    return process.stdout.read()

def process_pdb_files(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".pdb"):
            pdb_file_path = os.path.join(folder_path, file)
            ligand_pdb = pdb_file_path.replace('.pdb', '_ligand.pdb')
            ligand_sdf = pdb_file_path.replace('.pdb', '_ligand.sdf')

            extract_ligand(pdb_file_path, ligand_pdb)
            convert_to_sdf(ligand_pdb, ligand_sdf)
            score = score_ligand_with_smina(pdb_file_path, ligand_sdf)
            print(f"Score for {file}: {score}")

def find_residues_near_ligand(pdb_file, model_index=0, distance_threshold=4.0):
    parser = PDBParser()
    structure = parser.get_structure('structure', pdb_file)

    # Choose the specific model
    try:
        model = structure[model_index]
    except KeyError:
        print(f"Model index {model_index} not found in the structure.")
        return []

    # Separate heteroatoms (ligand) and protein atoms in the selected model
    ligand_atoms = [atom for atom in model.get_atoms() if atom.get_parent().id[0] != ' ']
    protein_atoms = [atom for atom in model.get_atoms() if atom.get_parent().id[0] == ' ']

    # Build a NeighborSearch object for protein atoms
    ns = NeighborSearch(protein_atoms)

    # Find all protein atoms within the distance threshold from any ligand atom
    list_of_residues = []
    close_residues_info = {}
    for atom in ligand_atoms:
        close_atoms = ns.search(atom.get_coord(), distance_threshold)
        for close_atom in close_atoms:
            residue = close_atom.get_parent()
            list_of_residues.append(residue)
            residue_key = (residue.get_full_id(), residue.resname, residue.id[1])
            if residue_key not in close_residues_info:
                close_residues_info[residue_key] = []
            close_residues_info[residue_key].append((close_atom.get_name(), tuple(close_atom.get_coord())))

    # for residue_key, atoms_info in close_residues_info.items():
    #     print(f"Residue: {residue_key}")
    #     for atom_name, coords in atoms_info:
    #         print(f"  Atom: {atom_name}, Coordinates: {coords}")

    return close_residues_info, list_of_residues

def extract_matching_residues_coordinates(pdb_file, residues_info):
    """
    Extracts coordinates of atoms in residues from a PDB file that match
    the residues specified in residues_info.
    :param pdb_file: Path to the PDB file.
    :param residues_info: Dictionary of residues information (from find_residues_near_ligand).
    :return: Dictionary with residue identifiers as keys and lists of (atom_name, atom coordinates) as values.
    """
    parser = PDBParser()
    structure = parser.get_structure("new_structure", pdb_file)

    matching_coordinates = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_key = (chain.id, residue.id)
                # Check if the residue is
                #  in the list from find_residues_near_ligand
                if residue in residues_info:
                    coords = [(atom.get_name(), atom.get_coord()) for atom in residue]
                    matching_coordinates[residue_key] = coords

    return matching_coordinates


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script description here.')
    parser.add_argument('--alnfile', type=str, required=True,
                        help='Path to the alignment file')
    parser.add_argument('--knowns', type=str, required=True,
                        help='Known structures (comma-separated)')
    parser.add_argument('--sequence', type=str, required=True,
                        help='Target sequence')
    parser.add_argument('--lig_rescoring', type=bool, default=False,
                        help='Rescore averaged models (for heteroatomic modeling only)')
    parser.add_argument('--optimization', type=bool, default=True,
                        help='High quality model refinement (slower)')
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Number of parallel jobs')
    parser.add_argument('--n_runs', type=int, default=24,
                        help='Number of MD runs')
    parser.add_argument('--model_name', type=str, default='_averaged_model',
                        help='Name of the output model')
    parser.add_argument('--scoring_func', type=str, default='dope',
                        help='Scoring function for model evaluation')
    parser.add_argument('--select_top', type=float, default=0.1,
                        help='Select top fraction of decoys for averaging')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    job_index = 1
    while os.path.isdir("job_" + str(job_index)):
        job_index += 1 

    job_dir_name = "job_" + str(job_index)
    os.mkdir(job_dir_name)
    shutil.copy("smina.static", job_dir_name)
    shutil.copy("sdsorter.static", job_dir_name)
    shutil.copy(args.alnfile, job_dir_name)

    for know in args.knowns.split(","):
        shutil.copy(know + ".pdb", job_dir_name)
    os.chdir(job_dir_name)
    
    # Set up the environment.
    example_dirpath = os.path.dirname(__file__)
    env = environ()
    env.io.atom_files_directory.append(example_dirpath)

    if args.lig_rescoring:
        env.io.hetatm = True

    # Initialize an 'automodel' object using the 'Automodel_averaging' class.
    a = Automodel_averaging(env, alnfile=args.alnfile,
                            knowns=args.knowns.split(","), sequence=args.sequence)
    
    # Set other parameters based on command-line arguments
    if args.optimization:
        a.md_level = refine.slow

    run_averaging(a, n_jobs=args.n_jobs, n_runs=args.n_runs,
                  model_name=args.model_name, scoring_func=args.scoring_func,
                  select_top=args.select_top)
    
    if args.lig_rescoring:

        files_to_analyze = []
        for file in os.listdir("."):

            if file.startswith(args.sequence + ".B9") and not "_ligand" in file and not "_protein" in file:
                files_to_analyze.append(file)

        for file in files_to_analyze:
            ligand_pdb = file.replace('.pdb', '_ligand.pdb')
            ligand_sdf = file.replace('.pdb', '_ligand.sdf')
            protein_pdb = file.replace('.pdb', '_protein.pdb')

            if not os.path.isfile(ligand_pdb):
                extract_ligand(file, ligand_pdb)
                remove_ligand(file, protein_pdb)
                convert_to_sdf(ligand_pdb, ligand_sdf)

            if not os.path.isfile(file.replace('.pdb', '_ligand_scored.sdf')):
                score = score_ligand_with_smina(protein_pdb, ligand_sdf)
                print(f"Score for {file}: {score}")

        if not os.path.isfile("rescored.txt"):
            command = f"cat *_ligand_scored.sdf > rescored.sdf"
            subprocess.run(command, shell=True, check=True)

            command = (
                f"./sdsorter.static -sort minimizedAffinity rescored.sdf rescored_asc.sdf -print -c > rescored.txt"
            )
            subprocess.run(command, shell=True, check=True)

        titles = extract_titles("rescored.txt")
        combine_pdb_files(titles[:10], 'rescored_combined.pdb')  

    
    os.chdir("..")
