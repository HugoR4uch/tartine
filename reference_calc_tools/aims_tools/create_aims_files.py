import os
import ase
import numpy as np
import ase.io
import ase.io.aims

def extract_lattice_vectors(training_confs_dir, aims_files_dir):
    dirs = os.listdir(training_confs_dir) 
    for directory in dirs:
        name = directory.split('/')[-1].split('.')[0]
        directory = os.path.join(training_confs_dir, directory)
        lattice_vectors = []
        for filename in os.listdir(directory):
            if filename.endswith('.inp'):
                with open(os.path.join(directory, filename), 'r') as file:
                    lines = file.readlines()
                for line in lines:
                    if line.lstrip().startswith('A '):
                        A_vector = np.array(line.split()[1:4], dtype=float)
                        lattice_vectors.append(A_vector)
                    if line.lstrip().startswith('B '):    
                        B_vector = np.array(line.split()[1:4], dtype=float)
                        lattice_vectors.append(B_vector)
                    if line.lstrip().startswith('C '):
                        C_vector = np.array(line.split()[1:4], dtype=float)
                        lattice_vectors.append(C_vector)
        
        lattice_vectors = np.array(lattice_vectors)
        print()
        print(lattice_vectors)
        print(name)
        atoms = ase.io.read(os.path.join(directory, name + '.xyz'), format='xyz')
        atoms.set_cell(lattice_vectors)

        print(atoms.cell)
        atoms.set_pbc((True,True,True))         # ase.io.write('aims_files/'+name + '.in', atoms, format='aims')
        ase.io.aims.write_aims('aims_files/'+name + '.in', atoms)


directory = './training_frames'
aims_dir = './aims_files'

extract_lattice_vectors(directory, aims_dir)


