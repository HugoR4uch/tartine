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
from ase.optimize.minimahopping import MinimaHopping


from . import interface_analysis_tools
from . import water_analyser

from mace.calculators import MACECalculator
from mace.calculators.foundations_models import mace_mp



class monomer_PES_calculator:
    def __init__(self,
                name,
                substrate,
                model_path,
                E_water = None,
                E_substrate=None,
                savefile_path=None,
                f_max=0.005,
                ):
        

        self.model_path= model_path
        self.name = name
        self.f_max = f_max
        self.substrate = copy.deepcopy(substrate)
        self.substrate.cell[2] = np.array([0,0,100])
        self.calc = MACECalculator(model_path=model_path,device='cuda') 
        self.num_substrate_atoms = len(substrate)

        self.z_interface = np.max(substrate.get_positions()[:,2])
        self.save_filepath = savefile_path

        self.data = {} # format: {(x,y): {'z': [], 'E': [], 'H1': [], 'H2': []}}


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
        primitive, origin_shift = interface_analysis_tools.get_substrate_primitive_unit_cell(self.substrate,symprec=1e-2)
        v1 = primitive[0]
        v2 = primitive[1]

        print('Generating grid for water optimization...')
        print(f'Primitive unit cell vectors:\nv1: {v1}\nv2: {v2}')
        print(f'Origin shift: {origin_shift}')
        print(f'Grid size: {grid_size}x{grid_size}')


        coords = np.zeros((2,grid_size+1,grid_size+1))
        for i in range(grid_size+1):
            for j in range(grid_size+1):
                u1 = v1 * i / (grid_size+1) + origin_shift[0]
                u2 = v2 * j / (grid_size+1) + origin_shift[1]
                u = u1 + u2
                coords[0][i, j] = u[0]
                coords[1][i, j] = u[1]

        return coords
    


    def do_grid_calcs(self, grid_size=10, consensus_accept=3, E_error=0.005, max_fails=4):
        """
        Performs water orientation optimization for each point in the grid defined by the primitive unit cell of the substrate.
        The results are stored in self.data.
        """
        coords = self.get_calculator_grid(grid_size=grid_size)

        start_time = time.time()
        print('Starting grid calculations...')
        print(f'Grid size: {grid_size}x{grid_size}')
        print(f'Number of grid points: {grid_size**2}')

        for i in range(grid_size):
            for j in range(grid_size):
                x = coords[0][i, j]
                y = coords[1][i, j]

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

                    if E_min is None: 
                        E_min = E
                    elif abs(E - E_min) < E_error:
                        consensus += 1
                        print(f'Energy difference {abs(E - E_min):.4f} eV is within error threshold {E_error:.4f} eV. Consensus count: {consensus}')
                    elif E > E_min + E_error:
                        failure_count+=1
                        if failure_count >= max_fails:
                            raise ValueError(f'Failed to reach consensus after {failure_count} attempts for position ({x:.2f}, {y:.2f}).')
                        else:
                            continue
                    elif E < E_min - E_error:
                        failure_count = 0
                        E_min = E
                        consensus = 0
                        print(f'New minimum energy found: {E_min:.4f} eV at position ({x:.2f}, {y:.2f})')



                inner_end_time = time.time()
                inner_elapsed_time = inner_end_time - innter_start_time
                print(f'Completed trials for position ({x:.2f}, {y:.2f}) in {inner_elapsed_time:.2f} seconds.')
                print(f'Trials: {trials}, Consensus: {consensus}, Failure count: {failure_count}')
                
                self.data[(i, j)] = coord_data


                



        print('Grid calculations completed.')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Total elapsed time for grid calculations: {elapsed_time:.2f} seconds')


    


    def minimize_water_height(self,
                              x,y,
                              optimiser=FIRE,
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



    def minimize_water_orientation(self,
                                   O_pos,
                                    OH_vecs=None,
                                   optimiser=FIRE,
                                   max_steps=1000,
                                   ):

        counter = 0
        convergence = False


        calc = MACECalculator(model_path=self.model_path,device='cuda') 

        while convergence is False and counter <3:

            
            calc = MACECalculator(model_path=self.model_path,device='cuda')
            interface = copy.deepcopy(self.substrate)
            interface.set_calculator(calc)

            if OH_vecs is not None:
                H1 = O_pos + OH_vecs[0]
                H2 = O_pos + OH_vecs[1]

                H2O_disp=[H1, H2, O_pos]
                water = ase.Atoms('H2O',positions = H2O_disp)

            else:
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
            indices_to_fix = np.append(substrate_atom_indices,len(substrate_atom_indices)+2)
            # We fix substrate and water O positions
            

            water_O_index = len(substrate_atom_indices) + 2
            water_H1_index = len(substrate_atom_indices)
            water_H2_index = len(substrate_atom_indices) + 1

            c = FixAtoms(indices=indices_to_fix)
            # bond = FixInternals( bonds = [
            #     [0.957,[water_O_index,water_H1_index]],
            #     [0.957,[water_O_index,water_H2_index]]
            #     ],
            #     angles_deg = [
            #         [104.5,[water_H1_index,water_O_index,water_H2_index]],
            #          ])

            # interface.set_constraint([c,bond])

            interface.set_constraint(c)

            opt_traj_dir_name = 'optimisation_trajectories'
            if not os.path.exists(opt_traj_dir_name):
                os.makedirs(opt_traj_dir_name)

            if self.name is None:
                name = 'interface'
            else:
                name = self.name

            opt_traj_name ='./'+ f"{opt_traj_dir_name}/{name}-{O_pos[0]:.1f}-{O_pos[1]:.1f}_new_opt.traj"


            dyn = optimiser(interface,trajectory=opt_traj_name,append_trajectory=True)


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
        if not convergence:
            print('Warning: Optimisation did not converge!')
        

        O_pos = opt_conf.get_positions()[self.num_substrate_atoms+2]
        H1 = opt_conf.get_positions()[self.num_substrate_atoms] - O_pos
        H2 = opt_conf.get_positions()[self.num_substrate_atoms+1] - O_pos
        E = opt_conf.get_potential_energy() - self.E_substrate - self.E_water
        print(f'E_substrate: {self.E_substrate:.2f} eV')
        print(f'E_water: {self.E_water:.2f} eV')
        print(f'Final energy: {E:.2f} eV')
        print(f'Final O position: {O_pos}')


        return H1, H2, E





    def save_data(self, savefile_path):

        #check if xy already exists
        #saves z data
        #saves E(z)
        #save H1 and H2 vectors
        pass





def save_calculator_data(data, filename):
    """Convert and save calculator data to JSON."""
    json_safe_data = {}
    for key, value in data.items():
        key_str = str(key)  # Convert tuple to string
        json_safe_data[key_str] = {}
        for k, v in value.items():
            if isinstance(v, list) and isinstance(v[0], np.ndarray):
                json_safe_data[key_str][k] = [arr.tolist() for arr in v]
            else:
                json_safe_data[key_str][k] = v
    with open(filename, 'w') as f:
        json.dump(json_safe_data, f, indent=2)


def load_calculator_data(filename):
    """Load calculator data from JSON and restore original types."""
    with open(filename, 'r') as f:
        raw_data = json.load(f)

    restored_data = {}
    for key_str, value in raw_data.items():
        key = tuple(map(float, key_str.strip("()").split(",")))  # Convert string back to tuple of floats
        restored_data[key] = {}
        for k, v in value.items():
            if isinstance(v, list) and isinstance(v[0], list):  # Likely array data
                restored_data[key][k] = [np.array(arr) for arr in v]
            else:
                restored_data[key][k] = v
    return restored_data



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