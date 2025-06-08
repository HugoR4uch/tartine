import numpy as np 
import ase.io
import ase
import copy 
from ase.data import vdw_radii
import os
import matplotlib.pyplot as plt

import ase.io.aims

from . import fhi_aims_input_file_tools
from . import fhi_aims_output_analysis

from ase import units
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase.io import read, write
import time



class BindingCurveAnalyser:

    def __init__(self,system_name=None,water_energy=None):
        # self.z_approach_vals = [] # distance between monomer and adsorption site in A
        # self.binding_energy_vals = [] # in eV
        # self.convergence_mask = None
        # self.system_name=system_name
        # self.plot_convergence = plot_convergence
        # self.energy_factor = energy_factor # Energy factor to convert to eV
        self.system_name=system_name
        self.binding_curves = {}    
        self.water_energy = water_energy
        self.binding_frames = []

    def add_binding_curve(self,curve_name,z_appraoch_vals,binding_energy_vals,docking_atom=None,convergence_mask=None):

        self.binding_curves[curve_name] = {'z_approach_vals':z_appraoch_vals,
                                            'binding_energy_vals':binding_energy_vals,
                                            'convergence_mask':convergence_mask,
                                            'docking_atom':docking_atom}

        # if convergence_mask is None:
        #     self.plot_convergence = False
        # else:
        #     self.plot_convergence = True
        #     self.convergence_mask = convergence_mask

        # self.z_approach_vals = z_appraoch_vals
        # self.binding_energy_vals = binding_energy_vals
        

    def plot_binding_curves(self,curves_to_plot=None,plot_dir_path='./'):

        if curves_to_plot is None:
            curves_to_plot = self.binding_curves.keys()

        for curve_name in curves_to_plot:
            
            energy_vals = self.binding_curves[curve_name]['binding_energy_vals']
            z_vals = self.binding_curves[curve_name]['z_approach_vals']
            convergence_mask = self.binding_curves[curve_name]['convergence_mask']
            if  convergence_mask is None:
                plot_convergence = False
            else:
                plot_convergence = True
            docking_atom = self.binding_curves['docking_atom'] 


            if plot_convergence:
                plt.scatter(z_vals[self.convergence_mask], energy_vals[self.convergence_mask], marker='.',color='black',label=curve_name)
                plt.scatter(z_vals[~self.convergence_mask], energy_vals[~self.convergence_mask], marker='x',color='black',label='Not converged')
            else:
                plt.scatter(self.z_approach_vals, energy_vals, marker='.')
                
           

        if docking_atom is None:
            plt.title(self.system_name + ' Binding Curve')
        else:
            plt.title(self.system_name +f' (H2O -> {self.adsorp_element})' +' Binding Curve')

        plt.xlabel('Displacement (A)')
        plt.ylabel('Binding Energy (eV)')
        plt.legend()
        plt.grid() 
        plt.savefig(os.path.join(plot_dir_path,self.system_name+'binding_curve.png'))
        plt.close()


    def find_docking_atom_index(self,binding_curve_frame,water_O_index=-1):
        #Feed an Atoms object for the binding curve. Finds which atom H2O docks to.
        positions = binding_curve_frame.positions
        substrate_positions = positions[:-3]
        substrate_indices = np.arange(len(substrate_positions))
        substrate_atoms = binding_curve_frame[:-3]
        substrate_indices = len(substrate_atoms)
        O_substrate_distances = binding_curve_frame.get_all_distances(mic=True,vector=True)[water_O_index]
        xy_displacements= O_substrate_distances[:,0:2][:-3]
        z_displacements = abs(O_substrate_distances[:,2])[:-3]
        # print(z_displacements)
        xy_distances = np.linalg.norm(xy_displacements,axis=1)
        # print(xy_distances)
        # print(O_substrate_distances)
        # [water_O_index][substrate_positions]
        # print()
        min_xy_distance = np.min(xy_distances[:-3])
        # print(np.where(xy_distances==min_xy_distance))

        possible_docking_atom_indices = np.where(xy_distances==min_xy_distance)[0]
        min_z= np.min(z_displacements[possible_docking_atom_indices])
        smallest_z_possible_docking = np.where(z_displacements[possible_docking_atom_indices]==min_z)
        docking_index = possible_docking_atom_indices[smallest_z_possible_docking[0][0]]


        # print(docking_atom_index)

        return docking_index


    # def find_adsorbate_atom(O_water_index=-1): 
    #     #NOT IMPLEMENTED
    #     pass


    # def plot_MACE_and_ref_curves(self,ref_curve_dir_path):
    #     #NOT IMPLEMENTED
    #     pass


class FHI_AIMS_BindingCurveAnalyser(BindingCurveAnalyser):
    
    
    def __init__(self,system_name=None,water_energy=None):
        super().__init__(system_name,water_energy)

    def load_binding_curve_data(self,
                                binding_curve_dir,
                                system_name=None):
        

        def get_isolated_substrate_dir(calc_dirs):
            possible_substrate_dirs = [calc for calc in calc_dirs if 'isolated' in calc and system_name in calc]
            if len(possible_substrate_dirs)>1:
                raise ValueError('Multiple isolated substrate directories found. Specify isolated substrate directory.')
            if len(possible_substrate_dirs)==0:
                raise ValueError('No isolated substrate directory found.')

            isolated_substrate_dir_path = os.path.join(binding_curve_dir,possible_substrate_dirs[0])
            return isolated_substrate_dir_path
        

        def get_isolated_water_dir(calc_dirs,):
            possible_isolated_water_dirs = [calc for calc in calc_dirs if 'water' in calc] 
            if len(possible_isolated_water_dirs)<1:
                raise ValueError('Multiple isolated water directories found. Specify isolated water directory.')
            if len(possible_isolated_water_dirs)==0:
                raise ValueError('No isolated water directory found.')
            else:
                isolated_water_dir_path = os.path.join(binding_curve_dir,possible_isolated_water_dirs[0])

            return isolated_water_dir_path


        if system_name is None:
            system_name = binding_curve_dir.split('binding_curve')[-2].split('/')[-1]
        
        print('System Name: ',system_name)

        z_approach_vals = []
        total_energy_vals = []
        convergence_mask = []
        substrate_energy = None
        docking_atom = None

        calc_dirs = os.listdir(binding_curve_dir)
        binding_calc_dirs = [calc_dir for calc_dir in calc_dirs if 'binding' in calc_dir]
        
        for i,calc_dir in enumerate(binding_calc_dirs):
            calc_path = os.path.join(binding_curve_dir,calc_dir)


            #THE summarise_aims_output FUNCTION IS BROKEN!!!
            # summary=fhi_aims_output_analysis.summarise_aims_output(calc_path)
            # energy = summary[calc_path]['Corrected Energy']
            # convergence = summary[calc_path]['Convergence']
            
            
            fd = os.path.join(calc_path,'aims.out')
            try: 
                frame = ase.io.aims.read_aims_output(fd)
                energy = frame.get_potential_energy()
                convergence=True
                total_energy_vals.append(energy)

                docking_atom_index = self.find_docking_atom_index(frame)
                docking_atom = frame[docking_atom_index].symbol
                z_approach = frame.positions[-1][2] - frame.positions[docking_atom_index][2]

                z_approach_vals.append(z_approach)
                
            except:
                print('Calculation not converged: ',calc_path)
                convergence=False
            
            
            print('Convergence: ',convergence)
            convergence_mask.append(convergence)

        if self.water_energy is None:
            isolated_water_dir_path = get_isolated_water_dir(calc_dirs)
            
            water_atoms = ase.io.aims.read_aims_output(os.path.join(isolated_water_dir_path,'aims.out'))

            # summary=fhi_aims_output_analysis.summarise_aims_output(isolated_water_dir_path)
            self.water_energy = water_atoms.get_potential_energy()

        isolated_substrate_dir_path = get_isolated_substrate_dir(calc_dirs) 

        isolated_substrate = ase.io.aims.read_aims_output(os.path.join(isolated_substrate_dir_path,'aims.out'))
        substrate_energy = isolated_substrate.get_potential_energy()

        # summary=fhi_aims_output_analysis.summarise_aims_output(isolated_substrate_dir_path)
        # substrate_energy = summary[isolated_substrate_dir_path]['Corrected Energy']

        binding_energy_vals = np.array(total_energy_vals) - substrate_energy - self.water_energy


        self.add_binding_curve(system_name,
                                z_approach_vals,
                                binding_energy_vals,
                                docking_atom=docking_atom,
                                convergence_mask=convergence_mask)

    def save_binding_traj(self,traj_file_path,binding_curve_dir):

        system_name = binding_curve_dir.split('binding_curve')[-2].split('/')[-1]
        
        print('Getting traj from: ',system_name)

        calc_dirs = os.listdir(binding_curve_dir)
        binding_calc_dirs = [calc_dir for calc_dir in calc_dirs if 'binding' in calc_dir]
        
        frames = []
        z_approach_vals = []
        for i,calc_dir in enumerate(binding_calc_dirs):
            calc_path = os.path.join(binding_curve_dir,calc_dir)
           
            frame = ase.io.read(os.path.join(calc_path,'geometry.in'),format='aims')
            frames.append(frame)

           
            docking_atom_index = self.find_docking_atom_index(frame)
            z_approach = frame.positions[-1][2] - frame.positions[docking_atom_index][2]
            z_approach_vals.append(z_approach)


        sort = np.argsort(z_approach_vals)

        sorted_trajectory = [frames[i] for i in sort]

        ase.io.write(traj_file_path,sorted_trajectory,format='xyz')


class MACE_BindingCurveAnalyser(BindingCurveAnalyser):
    def __init__(self):
        super().__init__(self)

    def load_binding_curve_data(self,binding_curve_dir,config_filename=None,system_name=None):
        z_approach_vals = []
        total_energy_vals = []
        # docking_atom = None
        substrate_energy = None
        water_energy = None
        docking_atom = None

        
        def ad_hoc_reader_of_xyz_files(xyz_filepath):
            # Previously, for whatever reason, these files are poorly formatted. Have to manual read
            xyz_info = open(xyz_filepath,'r').readlines() # THIS CAUSES ERRORS!
            for data in xyz_info[1].split():
                if 'energy' in data:
                    energy = float(data.split('=')[-1])
            config = ase.io.read(xyz_filepath,format='xyz')
            xyz_info.close()
            return energy,config

        def get_isolated_substrate_dir(calc_dirs,system_name):
            possible_substrate_dirs = [calc for calc in calc_dirs if 'isolated' in calc and system_name in calc]
            if len(possible_substrate_dirs)>1:
                raise ValueError('Multiple isolated substrate directories found. Specify isolated substrate directory.')
            if len(possible_substrate_dirs)==0:
                raise ValueError('No isolated substrate directory found.')

            isolated_substrate_dir_path = os.path.join(binding_curve_dir,possible_substrate_dirs[0])
            return isolated_substrate_dir_path
        

        def get_isolated_water_dir(calc_dirs):
            possible_isolated_water_dirs = [calc for calc in calc_dirs if 'water' in calc] 
            if len(possible_isolated_water_dirs)<1:
                raise ValueError('Multiple isolated water directories found. Specify isolated water directory.')
            if len(possible_isolated_water_dirs)==0:
                raise ValueError('No isolated water directory found.')
            else:
                isolated_water_dir_path = os.path.join(binding_curve_dir,possible_isolated_water_dirs[0])

            return isolated_water_dir_path



        calc_dirs = os.listdir(binding_curve_dir)
        binding_calc_dirs = [calc_dir for calc_dir in calc_dirs if 'binding' in calc_dir]
        

        for i,calc_dir in enumerate(binding_calc_dirs):
            calc_path = os.path.join(binding_curve_dir,calc_dir)
            

            if system_name is None:
                system_name = binding_curve_dir.split('binding_curve')[-2].split('/')[-1]
            
            if config_filename is None:
                calc_files = os.listdir(calc_path)
                possible_config_files = [file for file in calc_files if '.xyz' in file]
                
                if len(possible_config_files)!=1:
                    raise ValueError('Multiple config files found in calc dir. Specify config file name.')

            config_path = os.path.join(calc_path,possible_config_files[0])
            
            energy,config = ad_hoc_reader_of_xyz_files(config_path)


            docking_atom_index = self.find_docking_atom_index(config)
            docking_atom = config[docking_atom_index].symbol
            z_approach = config.positions[-1][2] - config.positions[docking_atom_index][2]
            
            z_approach_vals.append(z_approach)
            total_energy_vals.append(energy)


    
        if self.water_energy is None:
            isolated_water_dir_path = get_isolated_water_dir(calc_dirs)

            for file in os.listdir(isolated_water_dir_path):
                if '.xyz' in file:
                    isolated_water_file = file
            
            water_xyz_path = os.path.join(isolated_water_dir_path,isolated_water_file)
            water_energy,conf = ad_hoc_reader_of_xyz_files(water_xyz_path)




        isolated_substrate_dir_path = get_isolated_substrate_dir(calc_dirs,system_name) 

        for file in os.listdir(isolated_substrate_dir_path):
            if '.xyz' in file:
                isolated_substrate_file = file

        isolated_substrate_path = os.path.join(isolated_substrate_dir_path,isolated_substrate_file)

        config = ase.io.read(isolated_substrate_path,format='extxyz')

        substrate_energy = config.get_potential_energy()


        binding_energy_vals = np.array(total_energy_vals) - substrate_energy - water_energy


        self.add_binding_curve(system_name,
                                z_approach_vals,
                                binding_energy_vals,
                                docking_atom=docking_atom,
                                convergence_mask=None)


class BindingCurveMaker:

    """
    Class for finding binding curves for water on substrates.
    """

    def __init__(self,
                 calc_dir,
                 substrate_file,
                 substrate_name=None,
                 adsorp_element=None,
                 z_approach_vals=None,
                 ):
        self.calc_dir = calc_dir
        self.substrate_name=substrate_name

        self.substrate = ase.io.read(substrate_file,format='proteindatabank')
        self.substrate.set_pbc([True,True,True])
        substrate_cell = np.array(self.substrate.cell)
        substrate_cell[2][2] =+ 50
        self.substrate.set_cell(substrate_cell)
        angle = 14.5 /180 * np.pi
        self.water = ase.Atoms('H2O', positions=np.array([[0,-0.95,0],[0.95*np.cos(angle),0.95*np.sin(angle),0],[0,0,0]]))
        self.water.set_cell([50,50,50])



        if z_approach_vals is None:
            z_approach = np.linspace(1,6,14)
            #Adding point where water is 10 A above the substrate:
            z_approach = np.append(z_approach,10)
            self.z_approach_vals = z_approach
        else:
            z_approach = z_approach_vals
            if max(z_approach_vals) < 10:
                z_approach = np.append(z_approach,10)
            self.z_approach_vals = z_approach
        
        
        self.vacuum_width = 15.0
        self.binding_configs = []
        self.n_atoms = len(self.substrate)

        if adsorp_element is None:
            self.adsorp_element = None
        else:
            self.adsorp_element= adsorp_element
 

    def make_binding_curve_atoms(self):
        
        substrate_z_vals = self.substrate.positions[:,2]

        # Shifting to 5 A above bottom of cell
        substrate_z_vals = self.substrate.positions[:,2]
        substrate_bottom_z_val = np.min(substrate_z_vals)
        self.substrate.positions[:,2] += 5 - substrate_bottom_z_val
        substrate_z_vals = copy.deepcopy(self.substrate.positions[:,2])

        #adsorp_element_indices are indices of ALL adsorption element atoms
        if self.adsorp_element is not None:
            adsorp_element_indices = [atom.index for atom in self.substrate if atom.symbol == self.adsorp_element]
        else:
            adsorp_element_indices = [atom.index for atom in self.substrate]
        
        max_adsorp_element_z_val = np.max(substrate_z_vals[adsorp_element_indices])
        substrate_bottom_z_val = np.min(substrate_z_vals)
        all_candidate_substrate_z_vals_adsorption_indices= np.where([substrate_z_vals==max_adsorp_element_z_val])[1]
        
        candidate_substrate_z_vals_adsorption_indices = [index for index in all_candidate_substrate_z_vals_adsorption_indices if index in adsorp_element_indices]

        adsorp_index = np.random.choice(candidate_substrate_z_vals_adsorption_indices)
        if self.adsorp_element is None:
            self.adsorp_element = self.substrate[adsorp_index].symbol
        x = self.substrate.positions[adsorp_index][0]
        y = self.substrate.positions[adsorp_index][1]

        # Creating binding curve configs
        substrate_with_monomer = copy.deepcopy(self.substrate)
        for i,z in enumerate(self.z_approach_vals):
            # Add water
            adsorbate_water = copy.deepcopy(self.water)
            adsorbate_water.positions[:,2] = z + max_adsorp_element_z_val
            adsorbate_water.positions[:,0] += x
            adsorbate_water.positions[:,1] += y
            substrate_with_monomer.extend(adsorbate_water)

            # Adjust cell dimensions
            substrate_with_monomer.cell[2][2] = (z+ max_adsorp_element_z_val)+(self.vacuum_width - 5)# -5 as already shifted up 
            


            self.binding_configs.append(copy.deepcopy(substrate_with_monomer))

            
            # Remove water - will add at new position in next iteration
            del substrate_with_monomer[-3:]

        return self.binding_configs


    def get_binding_curve_data(self):
        return self.binding_configs


    def set_binding_curve_data(self,new_binding_configs):
        self.binding_configs = new_binding_configs
        

    def get_water_monomer(self):
        return self.water
    
    def get_substrate(self):
        return self.substrate
    
    def set_water_monomer(self,new_water):
        self.water = new_water

    def set_substrate(self,new_substrate):
        self.substrate = new_substrate




class  AIMSBindingCurveMaker(BindingCurveMaker):
    
    def __init__(self,
                 calc_dir,
                 substrate_file,
                 substrate_name,
                 adsorp_element=None,
                 z_approach_vals=None,
                 time_hrs = 1.5,
                 qos='standard',
                 n_tasks=128,
                 num_nodes=1,
                 budget_allocation='e05-surfin-mic',
                 scf_params=None):
            
        
        super().__init__(calc_dir,
                         substrate_file,
                         substrate_name,
                         adsorp_element,
                         z_approach_vals)


        self.time_hrs = time_hrs
        self.qos = qos
        self.n_tasks = n_tasks
        self.num_nodes = num_nodes
        self.budget_allocation = budget_allocation
        self.basis_sets_dir='/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/fhi-aims/fhi-aims.240920_2/species_defaults/defaults_2020/light'
        self.default_control_file_path='/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/config_files/control_default.in'
        self.default_slurm_file_path='/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/config_files/run_fhi_aims_DFT_ARCHER2.slurm'


        if scf_params is None:
            self.mixer= 'pulay'
            self.charge_mixing = '0.05' 
            self.occupation_type= 'fermi'
            self.smearing = 0.2
            self.preconditioner = 2
            self.max_scf_cycles = 200
        else:
            self.mixer = scf_params['mixer']
            self.charge_mixing = scf_params['charge_mixing']
            self.occupation_type = scf_params['occupation_type']
            self.smearing = scf_params['smearing']
            self.preconditioner = scf_params['preconditioner']
            self.max_scf_cycles = scf_params['max_scf_cycles']




    def make_calc_dirs(self,make_isolated_substrate_dir=False,make_isolated_water_dir=False,save_configs=False):
        
        if len(self.binding_configs)==0:
            super().make_binding_curve_atoms()

        for i,config in enumerate(self.binding_configs):

            #Creating binding curve directories
            dir_path = os.path.join(self.calc_dir, f'binding_{self.substrate_name}_{i}')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            
            #Adding aims input files and slurm file
            fhi_aims_input_file_tools.make_FHI_AIMS_calc_dir(config,
                                                            dir_path,
                                                            time_hrs=self.time_hrs,
                                                            basis_sets_dir=self.basis_sets_dir,
                                                            default_control_file_path=self.default_control_file_path,
                                                            default_slurm_file_path=self.default_slurm_file_path,
                                                            project_name=f'binding_{self.substrate_name}_{i}',
                                                            qos=self.qos,n_tasks=self.n_tasks,
                                                            num_nodes=self.num_nodes,
                                                            budget_allocation=self.budget_allocation,
                                                            mixer= self.mixer,
                                                            charge_mixing = self.charge_mixing, 
                                                            occupation_type= self.occupation_type,
                                                            smearing =self.smearing,
                                                            preconditioner = self.preconditioner,
                                                            max_scf_cycles = self.max_scf_cycles,)

            
            if save_configs:
                ase.io.write(os.path.join(dir_path,'binding_config.xyz'), config, format='extxyz')


        if make_isolated_substrate_dir:

            #Creating isolated substrate calc dir
            dir_path = os.path.join(self.calc_dir, f'isolated_{self.substrate_name}')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            #Adding aims input files and slurm file
            fhi_aims_input_file_tools.make_FHI_AIMS_calc_dir(self.substrate,
                                                            dir_path,
                                                            time_hrs=self.time_hrs,
                                                            basis_sets_dir=self.basis_sets_dir,
                                                            default_control_file_path=self.default_control_file_path,
                                                            default_slurm_file_path=self.default_slurm_file_path,
                                                            project_name=f'isolated_{self.substrate_name}',
                                                            qos=self.qos,
                                                            n_tasks=self.n_tasks,
                                                            num_nodes=self.num_nodes,
                                                            budget_allocation=self.budget_allocation,
                                                            mixer= self.mixer,
                                                            charge_mixing = self.charge_mixing, 
                                                            occupation_type= self.occupation_type,
                                                            smearing =self.smearing,
                                                            preconditioner = self.preconditioner,
                                                            max_scf_cycles = self.max_scf_cycles,)


            if save_configs:
                ase.io.write(os.path.join(dir_path,self.substrate_name+'substrate.xyz'), self.substrate, format='extxyz')

        if make_isolated_water_dir:

            #Creating isolated water calc dir
            dir_path = os.path.join(self.calc_dir, f'isolated_water')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            #Adding aims input files and slurm file
            fhi_aims_input_file_tools.make_FHI_AIMS_calc_dir(self.water, dir_path, time_hrs=self.time_hrs,
                                    basis_sets_dir=self.basis_sets_dir,
                                    default_control_file_path=self.default_control_file_path,
                                    default_slurm_file_path=self.default_slurm_file_path,
                                    project_name=f'isolated_water',qos=self.qos,n_tasks=self.n_tasks,num_nodes=self.num_nodes,budget_allocation=self.budget_allocation,
                                    mixer= self.mixer,
                                    charge_mixing = self.charge_mixing, 
                                    occupation_type= self.occupation_type,
                                    smearing =self.smearing,
                                    preconditioner = self.preconditioner,
                                    max_scf_cycles = self.max_scf_cycles,)

            if save_configs:
                ase.io.write(os.path.join(dir_path,self.substrate_name+'water.xyz'), self.water, format='extxyz')


class MACEBindingCurveMaker(BindingCurveMaker):
    def __init__(self,
                 calc_dir,
                 substrate_file,
                 substrate_name=None,
                 adsorp_element=None,
                 z_approach_vals=None,
                 model='/home/hr492/michaelides-share/hr492/Projects/tartine_project/mace_models/mace_agnesi_medium.model'):
        
        super().__init__(calc_dir,
                         substrate_file,
                         substrate_name,
                         adsorp_element,
                         z_approach_vals)

        self.model = model
        self.interface_energies = []
        self.water_energy= None
        self.substrate_energy = None
        self.binding_energies = []


    def find_binding_config_energies(self):

        if self.binding_configs == []:
            raise ValueError('No binding configurations loaded.')
        # self.binding_configs = self.make_binding_curve_atoms()


        from mace.calculators.foundations_models import mace_mp
        calculator = mace_mp(model=self.model,  dispersion=True, device='cuda')

        for i,config in enumerate(self.binding_configs):
            system_calculator = copy.deepcopy(calculator) # otherwise calculator stores energy of water
            config.set_calculator(system_calculator)
            energy = config.get_potential_energy()
            self.interface_energies.append(energy)
        

        system_calculator = copy.deepcopy(calculator)
        self.substrate.set_calculator(system_calculator)
        self.substrate_energy = self.substrate.get_potential_energy()
        print('substrate_energy',self.substrate_energy)

        system_calculator = copy.deepcopy(calculator)
        self.water.set_calculator(system_calculator)
        self.water_energy = self.water.get_potential_energy()
        print('water_energy',self.water_energy)


        
        binding_energies = np.array(self.interface_energies) - self.substrate_energy - self.water_energy
        print('binding_energies',binding_energies)
        self.binding_energies = binding_energies
        return binding_energies
    

    def plot_binding_curve(self,plot_dir_path):
        # binding_energies = self.find_binding_config_energies()
        print(self.binding_energies)
        analyser = BindingCurveAnalyser(self.substrate_name,plot_convergence=False,adsorp_element=self.adsorp_element)
        analyser.load_binding_curve_data(self.z_approach_vals,self.binding_energies)
        analyser.plot_binding_curve(plot_dir_path=plot_dir_path)

        pass

    
