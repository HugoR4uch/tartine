import sys
import os
import time
import json
import importlib
import copy

import numpy as np
import ase
import ase.io
from ase.constraints import FixAtoms, FixedLine
from ase.optimize import FIRE
from ase.optimize import BFGS
from ase.optimize.minimahopping import MinimaHopping

from typing import Literal

sys.path.append('/home/hr492/michaelides-share/hr492/Projects/tartine_project/software')

import tartines.interface_analysis.interface_analysis_tools as interface_analysis_tools
from tartines.interface_analysis import water_analyser
from tartines.interfaces_building import water_adder

importlib.reload(interface_analysis_tools)
importlib.reload(water_analyser)


from mace.calculators import MACECalculator
from mace.calculators.foundations_models import mace_mp

# === Optional: Plot Configuration ===



class monomer_PES_calculator:
    def __init__(self,
                name,
                substrate,
                model_path,
                E_water = None,
                E_substrate=None,
                savefile_path=None,
                f_max=0.01,
                ):
        

        self.model_path= model_path
        self.name = name
        self.grid_calc_status = 'In Progress'  # Default status. Other value is: 'Completed'
        self.f_max = f_max
        self.substrate = copy.deepcopy(substrate)
        self.substrate.cell[2] = np.array([0,0,100])
        self.calc = MACECalculator(model_path=model_path,device='cuda') 
        self.num_substrate_atoms = len(substrate)

        self.z_interface = np.max(substrate.get_positions()[:,2])
        
        self.data = {'Data': {}, 'Status': self.grid_calc_status}  # Initialize data dictionary

        if savefile_path is None:
            savefile_path = f'ads_energy_surface_{self.name}.json'
        self.save_filepath = savefile_path    
        # format: {(x,y): {'z': [], 'E': [], 'H1': [], 'H2': []}}





        if E_substrate is None:
            self.substrate.set_calculator(self.calc)
            E_substrate = self.substrate.get_potential_energy()
            print(f'Calculated substrate energy: {E_substrate:.2f} eV')
        self.E_substrate = E_substrate


        if E_water is None:
            H2O_r = 0.957 #OH bond length
            H2O_angle = 37.75/180 * np.pi #angle between x-axis and O-H displacemnet 
            H2O_disp = [H2O_r*np.array([-np.cos(H2O_angle),np.sin(H2O_angle),0]),
                        H2O_r*np.array([np.cos(H2O_angle),np.sin(H2O_angle),0]),
                        np.array([0,0,0])]
            water = ase.Atoms('H2O',positions = H2O_disp)
            water.set_cell(np.array([[50,0,0],[0,50,0],[0,0,50]]))  
            water.set_pbc(True)
            water.set_calculator(self.calc)
            E_water = water.get_potential_energy()
            print(f'Calculated water energy: {E_water:.2f} eV')
        self.E_water = E_water





    def get_calculator_grid(self, grid_size=10):
        """
        Generates a grid of points on the substrate surface for water orientation optimization.
        The grid is defined by the primitive unit cell of the substrate.
        """
        primitive_lattice, origin_shift = interface_analysis_tools.get_substrate_primitive_cell_data(self.substrate,symprec=1e-2)
        v1 = primitive_lattice[0]
        v2 = primitive_lattice[1]

        print('Generating grid for water optimization...')
        print(f'Primitive unit cell vectors:\nv1: {v1}\nv2: {v2}')
        print(f'Origin shift: {origin_shift}')
        print(f'Grid size: {grid_size}x{grid_size}')


        coords = np.zeros((2,grid_size,grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                u1 = v1 * i / (grid_size-1) + origin_shift[0]
                u2 = v2 * j / (grid_size-1) + origin_shift[1]
                u = u1 + u2
                coords[0][i, j] = u[0]
                coords[1][i, j] = u[1]

        return coords
    


    def do_grid_calcs(self, grid_size=10, consensus_accept=3, E_error=0.005, max_fails=10,outfile_path=None):
        """
        Performs water orientation optimization for each point in the grid defined by the primitive unit cell of the substrate.
        The results are stored in self.data.
        """

        calc_success = False

        if outfile_path is None:
            outfile_path = f'grid_calculations_{self.name}.out'

        original_stdout = sys.stdout

        with open(outfile_path, 'a') as f:
            
            sys.stdout = f  # Redirect print output to the file

            try:
                if not os.path.exists(self.save_filepath):
                    with open(self.save_filepath, 'w') as f:
                        json.dump({'Data':{},'Status':'In Progress'}, f)
                else:
                    print(f'File {self.save_filepath} already exists. Data will be appended to it.')
                    print(f'Loading existing data from {self.save_filepath}...')
                    status = self.load_existing_data()  # Load existing data if file exists
                    self.grid_calc_status = status

                if self.grid_calc_status == 'Completed':
                    print(f'Grid calculations loaded loaded from {self.save_filepath} already completed! Exiting.')
                    calc_success = True

                if self.grid_calc_status == 'In Progress':
                    print(f'Grid calculations not completed. Continuing with calculations...')
                    coords = self.get_calculator_grid(grid_size=grid_size)

                    start_time = time.time()
                    print('Starting grid calculations...')
                    print(f'Grid size: {grid_size}x{grid_size}')
                    print(f'Number of grid points: {grid_size**2}')
                    counter = 0
                    
                    for i in range(grid_size):
                        for j in range(grid_size):

                            if (i, j) in self.data.keys():
                                print(f'Skipping already calculated point ({i}, {j})')
                                continue


                            x = coords[0][i, j]
                            y = coords[1][i, j]


                            print('Counter:', counter)
                            counter += 1

                            print(f'Calculating for position: ({x:.2f}, {y:.2f})')
                            # coord_data = self.find_optimal_config(x, y)
                            
                            trials = 0
                            failure_count = 0
                            consensus = 0
                            E_min = None


                            innter_start_time = time.time()

                            while consensus < consensus_accept:
                                trials += 1
                                print(f'Trial {trials} for position ({x:.2f}, {y:.2f})')
                                print(f'Consensus: {consensus}/{consensus_accept}')

                                coord_data = self.minimize_water_height(x, y)

                                E = coord_data['E'][0]
                                r_H1 = np.linalg.norm(coord_data['H1'])
                                r_H2 = np.linalg.norm(coord_data['H2'])

                                if E_min is None:
                                    E_min = E
                                    consensus = 1
                                elif abs(E - E_min) < E_error:
                                    consensus += 1
                                    print(f'Energy difference {abs(E - E_min):.4f} eV is within error threshold {E_error:.4f} eV. Consensus count: {consensus}')
                                elif E > E_min + E_error:
                                    failure_count+=1
                                elif E < E_min - E_error:
                                    if r_H1 > 1.2 or r_H2 > 1.2:
                                        print('Water dissociated. Trying again...')
                                        failure_count += 1
                                    else:
                                        E_min = E
                                        consensus = 1
                                        print(f'New minimum energy found: {E_min:.4f} eV at position ({x:.2f}, {y:.2f})')

                                if failure_count >= max_fails:
                                    raise ValueError(f'Failed to reach consensus after {failure_count} attempts for position ({x:.2f}, {y:.2f}).')


                            inner_end_time = time.time()
                            inner_elapsed_time = inner_end_time - innter_start_time
                            print(f'Completed trials for position ({x:.2f}, {y:.2f}) in {inner_elapsed_time:.2f} seconds.')
                            print(f'Trials: {trials}, Consensus: {consensus}, Failure count: {failure_count}')
                            
                            self.data[(i, j)] = coord_data

                            print(f'Storing data for position ({x:.2f}, {y:.2f}): {coord_data}')
                            saving_status = 'In Progress' if i+j < 2*grid_size -2 else 'Completed'
                            
                            # Saving data to file
                            self.save_data({(i,j):coord_data},status=saving_status)  # Save data after each point
                            if saving_status == 'Completed':
                                self.grid_calc_status = 'Completed'                        

                            


                    calc_success = True
                    print('Grid calculations completed.')

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f'Total elapsed time for grid calculations: {elapsed_time:.2f} seconds')
                else: 
                    print(f'Unexpected status: {status}. Expected "Completed" or "In Progress". Exiting.')
                    calc_success = False
                    
        
            finally:
                sys.stdout = original_stdout  # Reset standard output to console

            return calc_success


    


    def minimize_water_height(self,
                              x,y,
                              optimiser=BFGS,
                              max_steps=3000,
                              ):

        counter = 0
        convergence = False
        calc = MACECalculator(model_path=self.model_path,device='cuda')

        opt_traj_dir_name = 'optimisation_trajectories'
        if not os.path.exists(opt_traj_dir_name):
                os.makedirs(opt_traj_dir_name)

        if self.name is None:
            name = 'interface'
        else:
            name = self.name

        opt_traj_name ='./'+ f"{opt_traj_dir_name}/{name}_opt.traj"


        
        while convergence is False and counter < 3:

            O_pos = np.array([x,y,5 + self.z_interface]) #initial O position
            
            calc = MACECalculator(model_path=self.model_path,device='cuda')
            interface = copy.deepcopy(self.substrate)
            interface.set_calculator(calc)

            H2O_r = 0.957 #OH bond length
            H2O_angle = 37.75/180 * np.pi #angle between x-axis and O-H displacemnet 
            H2O_disp = [H2O_r*np.array([-np.cos(H2O_angle),np.sin(H2O_angle),0]),
                        H2O_r*np.array([np.cos(H2O_angle),np.sin(H2O_angle),0]),
                        np.array([0,0,0])]
            water = ase.Atoms('H2O',positions = H2O_disp)
            euler_angles = np.random.rand(3) * 180 * [2,1,2]
            water.euler_rotate(euler_angles[0],euler_angles[1],euler_angles[2]) 
            new_pos = water.get_positions() + [ O_pos , O_pos , O_pos ]
            water.set_positions(new_pos)


            interface.extend(water)

            #minimize water orientation
            substrate_atom_indices = np.arange(0,self.num_substrate_atoms,1) 
            # We fix substrate and water O positions

            water_O_index = len(substrate_atom_indices) + 2

            fixed_substrate = FixAtoms(indices=substrate_atom_indices)

            fixed_line = FixedLine(water_O_index, [0, 0, 1])

            interface.set_constraint([fixed_substrate, fixed_line])


            dyn = optimiser(interface,trajectory=opt_traj_name,append_trajectory=True)


            # opt = MinimaHopping(atoms=system)
            # opt(totalsteps=10)


            start = time.time()
            convergence = dyn.run(fmax=self.f_max,steps=max_steps)
            end = time.time()
            if convergence is False:
                counter += 1
                print('Counter:', counter)


        if convergence is False:
            print('Completely failed to converge after multiple attempts.')
            raise ValueError(f'Optimisation did not converge for position ({O_pos[0]:.2f}, {O_pos[1]:.2f}) after {counter} attempts.')
        


        opt_conf = ase.io.read(opt_traj_name,index = '-1')

        opt_time = end-start
        print(f'Optimisation time: {opt_time:.2f} seconds')
        print(f'Converged: {convergence}')


        O_pos = opt_conf.get_positions()[self.num_substrate_atoms+2]
        H1 = opt_conf.get_positions()[self.num_substrate_atoms] - O_pos
        H2 = opt_conf.get_positions()[self.num_substrate_atoms+1] - O_pos
        E = opt_conf.get_potential_energy() - self.E_substrate - self.E_water
        z_O_min = O_pos[2] - self.z_interface

        print(f'E_substrate: {self.E_substrate:.2f} eV')
        print(f'E_water: {self.E_water:.2f} eV')
        print(f'Final energy: {E:.2f} eV')
        print(f'Final O position: {O_pos}')
        print(f'Final H1 vector: {H1}')
        print(f'Final H2 vector: {H2}')
        print('')

        
        data = {'xy': [], 'z': [], 'E': [], 'H1': [], 'H2': []}    


        data['xy'].extend([x,y])
        data['z'].append(z_O_min)
        data['E'].append(E)
        data['H1'].append(H1)
        data['H2'].append(H2)

        return data




    def load_existing_data(self):
        """
        Returns the data collected during the grid calculations.
        The data is in the format: {(x,y): {'z': [], 'E': [], 'H1': [], 'H2': []}}
        """
        # format: {(x,y): {'z': [], 'E': [], 'H1': [], 'H2': []}}

        filename = self.save_filepath
        if not os.path.exists(filename):
            raise FileNotFoundError(f'Tried to load existing data, but {filename} does not exist!')

        with open(filename, 'r') as f:
            loaded_data = json.load(f)

        raw_data = loaded_data['Data']
        status = loaded_data['Status']
        
        restored_data = {}
        for key_str, value in raw_data.items():
            key = tuple(map(float, key_str.strip("()").split(",")))
            restored_data[key] = {}
            print(f'Loading data for key: {key}')
            for k, v in value.items():
                if isinstance(v, list) and isinstance(v[0], list):  # Likely array data
                    restored_data[key][k] = [np.array(arr) for arr in v]
                else:
                    restored_data[key][k] = v

        for key, value in restored_data.items():
            if key in self.data.keys():
                raise ValueError(f'Key {key} already exists in the data. You\'re trying to overload it.')
            self.data[key] = value 


        return status


    def save_data(self,new_data,status: Literal['Completed', 'In Progress']):

        """
        Adds or updates entries in a JSON file containing a dictionary.

        Parameters:
        - new_data (dict): in the format: {(x,y): {'z': [], 'E': [], 'H1': [], 'H2': []}}
        - filename (str): Path to the JSON file.
        """

        filename = self.save_filepath

        # Load existing data if file exists, otherwise start fresh
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                raw_data = json.load(f)
        
            existing_data = raw_data['Data']

        else:
            existing_data = {}
 
        # Merge new_data into existing_data
        for key, value in new_data.items():
            if str(key) in existing_data:
                raise KeyError(f"Erro whilst saving data. Key '{key}' already exists in the file. Use a different key or modify the existing entry.")
            else:
                # Insert new key
                for key, value in new_data.items():
                    key_str = str(key)  # Convert tuple to string
                    existing_data[key_str] = {}
                    for k, v in value.items():
                        if isinstance(v, list) and isinstance(v[0], np.ndarray):
                            existing_data[key_str][k] = [arr.tolist() for arr in v]
                        else:
                            existing_data[key_str][k] = v

        data_to_save = {'Data': existing_data,
                        'Status': status}

        # Write back to file
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)





def save_calculator_data(data, filename,status: Literal['Completed', 'In Progress']):
    """Convert and save calculator data to JSON."""
    # format: {(x,y): {'z': [float], 'E': [float], 'H1': [ndarray], 'H2': [ndarray]}}
    json_safe_data = {}
    for key, value in data.items():
        key_str = str(key)  # Convert tuple to string
        json_safe_data[key_str] = {}
        for k, v in value.items():
            if isinstance(v, list) and isinstance(v[0], np.ndarray):
                json_safe_data[key_str][k] = [arr.tolist() for arr in v]
            else:
                json_safe_data[key_str][k] = v

    save_data = {
        'Data': json_safe_data,
        'Status': status  # Save the status as well
                }
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)


def load_calculator_data(filename):
    """Load calculator data from JSON and restore original types."""
    with open(filename, 'r') as f:
        loaded_data = json.load(f)

    raw_data= loaded_data['Data']  # Extract the 'Data' part
    status = loaded_data['Status']  # Extract the 'Status' part
    restored_data = {}
    for key_str, value in raw_data.items():
        key = tuple(map(float, key_str.strip("()").split(",")))  # Convert string back to tuple of floats
        restored_data[key] = {}
        for k, v in value.items():
            if isinstance(v, list) and isinstance(v[0], list):  # Likely array data
                restored_data[key][k] = [np.array(arr) for arr in v]
            else:
                restored_data[key][k] = v

    return restored_data, status



def get_mesh_vals_from_data(data,grid_size=None):


    if grid_size is None:
        grid_size = int(np.sqrt(len(data)))

    x_mesh = np.zeros((grid_size, grid_size))
    y_mesh = np.zeros((grid_size, grid_size))
    E_mesh = np.zeros((grid_size, grid_size))


    for i in range(grid_size):
        for j in range(grid_size):
            x_mesh[i, j] = data[(i, j)]['xy'][0]
            y_mesh[i, j] = data[(i, j)]['xy'][1]
            E_mesh[i, j] = np.min(data[(i, j)]['E'][0])



    return x_mesh, y_mesh, E_mesh



