import ase
import ase.io
import numpy as np
import copy
from ase.constraints import FixAtoms
from mace.calculators.foundations_models import mace_mp
from ase.optimize import BFGS
import time
import os
from tartines.utils import load_density_profile_data
from tartines.interfaces_building import water_adder
from tartines.interface_analysis import interface_analysis_tools
from tartines.interface_analysis import water_analyser
from mace.calculators import MACECalculator


# Tools for loading density profiles


def bulk_dissociated_interface_multi_builder(
                            model_path,
                            interfaces_dir,
                            substrates_dir,
                            density_profile_data_dir,
                            dissociation_fraction, # Fraction of water to dissociate
                            species_to_add = 'OH', # Can also be 'H3O'
                            optimise_interfacial_water=True,
                            force_correct_number_of_ions_added = True,
                            interfaces_whitelist=None,
                            interfaces_blacklist=None,
                            dissociated_interfaces_save_dir = None,
                            **kwargs
                            ):
    
    interfaces_filenames = os.listdir(interfaces_dir)
    
    for filename in interfaces_filenames:


        interface_name_data = filename.split('_')
        interface_name  = interface_name_data[1]+'_'+interface_name_data[2] 
        replica_index = interface_name_data[3]

        if interfaces_whitelist is not None:
            if interface_name not in interfaces_whitelist:
                print('Skipping',interface_name,'as it is not in the whitelist')
                continue

        if interfaces_blacklist is not None:
            if interface_name in interfaces_blacklist:
                print('Skipping',interface_name,'as it is in the blacklist')
                continue


    
        print('Loading interface for:',interface_name)       

        interface_file_path = os.path.join(interfaces_dir,filename)
        interface = ase.io.read(interface_file_path)

        print('Loading substrate for:',interface_name)

        substrates_file_path = os.path.join(substrates_dir,interface_name+'.pdb') 
        if not os.path.exists(substrates_file_path):
             raise FileNotFoundError(f"Substrate file not found: {substrates_file_path}")
        
        substrate = ase.io.read(substrates_file_path)

        print('Building',filename,' dissociated interface - replica: ',replica_index)

        interface_dissociation_fraction = kwargs["dissociation_frac_dict"].get(interface_name,dissociation_fraction)
        print("Using dissociation fraction:",interface_dissociation_fraction)

        print('Species to add:',species_to_add)

        dissociated_interface_builder = DissociationBuilder(
            model_path=model_path,
            interface=interface,
            substrate=substrate,
            interface_name=interface_name,
            species_to_add=species_to_add,
            optimise_interfacial_water = optimise_interfacial_water,
            force_correct_number_of_species=force_correct_number_of_ions_added,
        )

        dissociated_interface_builder.load_interface_simulation_data(
            density_profile_data_dir,
            interface_dissociation_fraction,
            )

        start_time = time.time()
        dissociated_interface_builder.build_dissociated_interface()
        end_time = time.time()

        print('Time taken:',end_time-start_time)

        dissociated_interface = dissociated_interface_builder.interface

        #Saving interface
        if dissociated_interfaces_save_dir is None:
            dissociated_interfaces_save_dir = 'dissociated_interfaces'
        if not os.path.exists(dissociated_interfaces_save_dir):
            os.makedirs(dissociated_interfaces_save_dir)

        ase.io.write(dissociated_interfaces_save_dir+'/dissociated-interface_'+interface_name+'.xyz',dissociated_interface,format='extxyz')
        
    return True
    

class DissociationBuilder:
    
    
    def __init__(self,
                 model_path,
                 interface,
                 substrate,
                 interface_name,
                 species_to_add='OH',
                 dissociation_z_min=None,
                 dissociation_z_max=None,
                 num_waters_dissociate=None,
                 add_counterions=False,
                 optimise_interfacial_water=True,
                 only_relax_added_atoms=True,
                 force_correct_number_of_species = True,
                 **kwargs
                 ):
        

        # Input params
        self.model_path = model_path
        self.interface = interface
        self.substrate = substrate
        self.name = interface_name
        self.species_to_add = species_to_add
        print('Species to add in dissociation:',self.species_to_add)
        self.dissociation_z_min = dissociation_z_min
        self.dissociation_z_max = dissociation_z_max
        self.num_waters_dissociate = num_waters_dissociate
        self.add_counterions = add_counterions
        self.only_relax_added_atoms = only_relax_added_atoms
        self.optimise_interfacial_water = optimise_interfacial_water
        self.force_correct_number_of_species = force_correct_number_of_species

        if species_to_add not in {'OH','H3O'}:
            raise ValueError('species_to_add is',species_to_add,' but should be either, "OH" or "H3O".')
 
        # Loaded data params
        self.density_profile_data_dir = None
        self.dissociation_fraction = None
        
        # Internal data
        self.lattice_vectors = None
        self.expected_waters_in_contact = None
        self.added_atom_indices =  None





    def load_interface_simulation_data(
            self,
            density_profile_data_dir,
            dissociation_fraction,
            ):

        self.density_profile_data_dir = density_profile_data_dir
        self.dissociation_fraction = dissociation_fraction

        density_profile_data_path = os.path.join(self.density_profile_data_dir,self.name+'density_profile.csv')
        
        print('Loading interface simulation density profiles data from: ',density_profile_data_path)
        
        if not os.path.exists(density_profile_data_path):
            raise FileNotFoundError(f"Density profile data file not found: {density_profile_data_path}")

        meta,z, O_density,_ = load_density_profile_data(density_profile_data_path) 


        self.dissociation_z_min = meta['contact_layer_start']
        self.dissociation_z_max = meta['contact_layer_end']
        self.lattice_vectors = np.array([meta['v_1'],meta['v_2'],meta['v_3']]) 

        mask = (z>=self.dissociation_z_min)&(z<=self.dissociation_z_max)
        conversion = 0.03345


        water_cont_layer_density = ( 
            np.trapz(O_density[mask],z[mask])  
            * conversion
        )

        print('Expected contact layer water number density per nm^2:',water_cont_layer_density*100 )

        interface_area = np.linalg.norm(np.cross( self.interface.cell[0] , self.interface.cell[1])) 

        print('Area of interface to dissociate:',interface_area)


        expected_waters_in_contact = water_cont_layer_density * interface_area

        print('Expected number of water molecules in contact layer:',expected_waters_in_contact)

        self.num_waters_dissociate = int( expected_waters_in_contact * self.dissociation_fraction)

        print('Loaded data from MD sims: ',self.dissociation_fraction,r'% dissociating, this gives'
              ,self.num_waters_dissociate,'dissocated waters') 
        



    def build_dissociated_interface(self,**kwargs):


        print('Building dissociated interface with the following parameters:')
        print(f"Number of waters to dissociate: {self.num_waters_dissociate}")
        print(f"Dissociation z min: {self.dissociation_z_min}")
        print(f"Dissociation z max: {self.dissociation_z_max}")
        print(f"Species to add: {self.species_to_add}")
        print(f"Optimise interfacial water: {self.optimise_interfacial_water}")
        print(f"Only relax added atoms: {self.only_relax_added_atoms}")
        print(f"Force correct number of species: {self.force_correct_number_of_species}")
        print(f'Add counterions: {self.add_counterions}')

        dissociation_layer_water_atom_indices = interface_analysis_tools.interfacial_water_criterion(self.interface,self.substrate,self.dissociation_z_min,self.dissociation_z_max)
        water_atom_indices = np.arange(len(self.substrate), len(self.interface))
        non_dissociated_water_atom_indices = np.setdiff1d(water_atom_indices, dissociation_layer_water_atom_indices)
        substrate_indices = np.arange(0,len(self.substrate))
        analyser = water_analyser.Analyser(self.interface,substrate_indices)
        voronoi_dict = analyser.get_voronoi_dict()
        
        dissociation_layer_water_O_indices = [index for index in  dissociation_layer_water_atom_indices if self.interface[index].symbol=='O']
        non_dissociated_water_O_indices = [index for index in  non_dissociated_water_atom_indices if self.interface[index].symbol=='O']

        successful_build = False
        build_attempts = 0

        while not successful_build: 
            print()
            print('Build attempt:',build_attempts+1)
            build_attempts+=1
            dissociated_interface = copy.deepcopy(self.interface)


            dissociation_contact_water_O_indices = np.random.choice(dissociation_layer_water_O_indices,self.num_waters_dissociate,replace=False)
            dissociation_contact_O_positions = {O_atom_index:self.interface[O_atom_index].position for O_atom_index in dissociation_contact_water_O_indices}

            if self.add_counterions:
                counter_dissociation_water_O_indices = np.random.choice(non_dissociated_water_O_indices,self.num_waters_dissociate,replace=False)
                counter_dissociation_contact_O_positions = {O_atom_index:self.interface[O_atom_index].position for O_atom_index in counter_dissociation_water_O_indices}

            atoms_to_delete = []
            
            if self.add_counterions:
                all_water_O_indices_to_delete = np.append(counter_dissociation_water_O_indices, dissociation_contact_water_O_indices) 
            else:
                all_water_O_indices_to_delete = dissociation_contact_water_O_indices

            print('O indices of waters to delete:',all_water_O_indices_to_delete)

            for O_atom_index in all_water_O_indices_to_delete:
                print(O_atom_index,type(O_atom_index))
                O_atom_index = int(O_atom_index)
                atoms_to_delete.extend([O_atom_index])
                atoms_to_delete.extend(voronoi_dict[O_atom_index])

            print('Replacing',len(atoms_to_delete),'water molecules and replacing with ions.')

            
            mask = np.ones(len(self.interface), dtype=bool)
            mask[atoms_to_delete] = False
            dissociated_interface = dissociated_interface[mask]


            adder_tool = water_adder.WaterAdder(dissociated_interface)

            indices_of_added_ion_species = []
            indices_of_added_counter_ion_species = []

            for O_position in dissociation_contact_O_positions.values():
                if self.species_to_add == 'OH':
                    adder_tool.add_OH(O_position)
                    indices_of_added_ion_species.extend([len(adder_tool.structure)-1,
                                                            len(adder_tool.structure) -2])
                                                            
                if self.species_to_add == 'H3O':
                    adder_tool.add_H3O(O_position)

                    indices_of_added_ion_species.extend([len(adder_tool.structure)-1,
                                                                    len(adder_tool.structure) -2,
                                                                    len(adder_tool.structure) -3,
                                                                    len(adder_tool.structure) -4])

            if self.add_counterions:
                for O_position in counter_dissociation_contact_O_positions.values():
                    if self.species_to_add == 'OH':
                        adder_tool.add_H3O(O_position)
                        indices_of_added_counter_ion_species.extend([len(adder_tool.structure)-1,
                                                                len(adder_tool.structure) -2,
                                                                len(adder_tool.structure) -3,
                                                                len(adder_tool.structure) -4])
                    if self.species_to_add == 'H3O':
                        adder_tool.add_OH(O_position)

                        indices_of_added_counter_ion_species.extend([len(adder_tool.structure)-1,
                                                                len(adder_tool.structure) -2],
                                                                )


            print('Species added successfully.')
            print('Added contact layer atoms:', indices_of_added_ion_species)
            print('Added bulk atoms:', indices_of_added_counter_ion_species)

            self.added_atom_indices = indices_of_added_ion_species + indices_of_added_counter_ion_species

            dissociated_interface = adder_tool.structure


            successful_build = True 

            if self.optimise_interfacial_water == True:

                successful_build = False
                print('Relaxing geometry of dissociated interface')
                print('Relaxing only added species:',self.only_relax_added_atoms)


                structure_to_relax = dissociated_interface

                if self.only_relax_added_atoms is True:
                    indices_to_relax = self.added_atom_indices
                else:
                    indices_to_relax = None

                

                print('Indices relaxed during optimisation:',indices_to_relax)

                relaxed_interface = self.relax_dissociated_water_interface_geom(structure_to_relax,indices_to_relax)
                
                if not self.force_correct_number_of_species:
                    successful_build = True
                else:
                    print('Ensuring that the dissociation was done correctly')
                    dissociated_analyser_tool = water_analyser.Analyser(relaxed_interface,substrate_indices)
                    dissociated_voronoi = dissociated_analyser_tool.get_voronoi_dict()
                    #Need to get new voronoi, as H atoms may have hopped to new species
                    
                    if self.species_to_add == 'OH': 
                        coordination_for_checking =1
                    if self.species_to_add == 'H3O':
                        coordination_for_checking = 3

                    
                    optimised_ion_O_indices = [
                                            O_index 
                                            for O_index in dissociated_voronoi.keys() 
                                            if len(dissociated_voronoi[O_index])
                                            ==coordination_for_checking
                                            ]
                    
                    print('Optimised O atom indices of ',
                          self.species_to_add,
                          ' ions added to system:',
                          optimised_ion_O_indices
                          )
                    
                    dissociation_layer_ion_O_indices = [O_index
                                                   for O_index in optimised_ion_O_indices
                                                   ]
                    
                        
                    number_of_dissociation_layer_ions = len(dissociation_layer_ion_O_indices)

                    print('Optimised O atom indices of ',
                          self.species_to_add,
                          ' ions in the dissociation layer:',
                          dissociation_layer_ion_O_indices
                          )


                    print(number_of_dissociation_layer_ions,' ',
                          self.species_to_add, ' ions in dissociation layer, when ',  
                          self.num_waters_dissociate,
                          ' requested',
                          )

                    if number_of_dissociation_layer_ions == self.num_waters_dissociate:
                        successful_build = True
                    
                    if number_of_dissociation_layer_ions != self.num_waters_dissociate:
                        successful_build = False

                    print('Optimisation success:',successful_build)

                dissociated_interface = relaxed_interface

        self.interface = dissociated_interface
        
            



    def relax_dissociated_water_interface_geom(self,
                                               input_structure,
                                               indices_to_relax=None,
                                               optimiser=BFGS,
                                               max_steps=1000): 
        if indices_to_relax is None:
            indices_to_relax = np.arange(len(self.substrate),len(input_structure))

        indices_to_constrain = np.setdiff1d(np.arange(0,len(input_structure)),indices_to_relax)

        print('indices actually constrained during optimisation:',indices_to_constrain)

        num_attempts = 0
        convergence = False
        while not convergence: 
            print(f'Optimisation attempt for {self.name}:',num_attempts)

            num_attempts+=1
            structure_to_relax = copy.deepcopy(input_structure)
            structure_to_relax.calc = MACECalculator(model_path=self.model_path,device='cuda')
            c = FixAtoms(indices=indices_to_constrain)
            structure_to_relax.set_constraint(c)

            opt_traj_dir_name = 'ions_optimisation_trajectories'
            if not os.path.exists(opt_traj_dir_name):
                os.makedirs(opt_traj_dir_name)

            if self.name is None:
                name = 'interface'
            else:
                name = self.name

            opt_traj_name = './' + opt_traj_dir_name + '/' +str(name) + '_new_opt.traj'

            dyn = optimiser(structure_to_relax,trajectory=opt_traj_name)

            start = time.time()
            convergence = dyn.run(fmax=0.05,steps=max_steps)
            end = time.time()

            opt_traj = ase.io.read(opt_traj_name,index = ':')
            num_opt_steps = len(opt_traj)
            opt_time = end-start
            self.time_per_model_eval = opt_time / num_opt_steps

        structure_to_relax.set_constraint(None)
        structure_to_relax.calc = None

        return structure_to_relax