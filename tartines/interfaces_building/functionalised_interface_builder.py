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
from tartines.interfaces_building import dissociated_interface_builder
from tartines.interfaces_building import interface_builder
from mace.calculators import MACECalculator




def water_split_interface_multi_builder(
                            substrates_dir,
                            functionalisation_fraction, # Fraction of water to dissociate
                            model_path,
                            elements_to_reduce_dict:dict,
                            elements_to_oxidise_dict:dict,
                            substrate_H_distance_dict:dict,
                            substrate_OH_distance_dict:dict,
                            substrates_whitelist=None,
                            substrates_blacklist=None,
                            functionalised_interfaces_save_dir = None,
                            **kwargs
                            ):
    
    substrates_filenames = os.listdir(substrates_dir)

    # Default interface building parameters
    water_thickness = kwargs.get('water_thickness',15.0)
    water_density = kwargs.get('water_density',1.0)
    intersubstrate_gap = kwargs.get('intersubstrate_gap',50)
    water_substrate_gap = kwargs.get('water_substrate_gap',2)
    surface_tolerance_dict = kwargs.get('surface_tolerance_dict',None)


    for filename in substrates_filenames:

        substrate_name_data = filename.split('.')[0].split('_')
        substrate_name  = substrate_name_data[0]+'_'+substrate_name_data[1] 

        if substrates_whitelist is not None:
            if substrate_name not in substrates_whitelist:
                print('Skipping',substrate_name,'as it is not in the whitelist')
                continue

        if substrates_blacklist is not None:
            if substrate_name in substrates_blacklist:
                print('Skipping',substrate_name,'as it is in the blacklist')
                continue

        print('Loading substrate for:',substrate_name)

        substrates_file_path = os.path.join(substrates_dir,substrate_name+'.pdb') 
        if not os.path.exists(substrates_file_path):
             raise FileNotFoundError(f"Substrate file not found: {substrates_file_path}")
        
        substrate = ase.io.read(substrates_file_path,format='proteindatabank')

        # Picking substrate atoms to add H to:
        if surface_tolerance_dict is not None:
            surface_tolerance = surface_tolerance_dict.get(substrate_name,0.1)
        else:
            surface_tolerance = 0.1

        elements_to_reduce = elements_to_reduce_dict.get(substrate_name,None)
        elements_to_oxidise = elements_to_oxidise_dict.get(substrate_name,None)

        if elements_to_reduce is None or elements_to_oxidise is None:
            raise ValueError(f"Elements to reduce or oxidise not specified for substrate: {substrate_name}")


        water_split_frac_dict = kwargs.get("dissociation_frac_dict",None)
        if water_split_frac_dict is None:
            dissociation_fraction= functionalisation_fraction
            print('Dissocitaion fraction used is:', functionalisation_fraction)
        else:
            dissociation_fraction = water_split_frac_dict.get(substrate_name,functionalisation_fraction)
            print('Dissociation fraction dict found, using dissocation frac:',dissociation_fraction)
        


        reduced_atoms, oxidised_atoms = find_ion_docking_sites(
            substrate,
            dissociation_fraction,
            elements_to_reduce,
            elements_to_oxidise,
            surface_tolerance=surface_tolerance,
            substrate_name=substrate_name,
            avoid_neighboring_sites = True,
            top_layer_indices=kwargs.get('top_layer_indices_dict',{}).get(substrate_name,None)
        )

        dissociated_substrate = substrate_dissociator(
            substrate,
            substrate_H_distance=substrate_H_distance_dict.get(substrate_name),
            substrate_OH_distance=substrate_OH_distance_dict.get(substrate_name),
            atoms_to_reduce = reduced_atoms,
            atoms_to_oxidise = oxidised_atoms,
            model_path = model_path,
            name=substrate_name
        )



        # ase.io.write('dissociated_substrate_'+substrate_name+'.xyz',dissociated_substrate,format='extxyz')

        interface_builder_tool = interface_builder.InterfaceBuilder(model_path,
                                                 dissociated_substrate,
                                                 water_thickness,
                                                 water_density,
                                                 intersubstrate_gap,
                                                 name=substrate_name)
        
        interface_builder_tool.build_bulk_interface(water_substrate_gap=water_substrate_gap,
                                                optimise_interfacial_water=True,
                                                )
        final_interface = interface_builder_tool.interface

        #Saving interface
        if functionalised_interfaces_save_dir is None:
            functionalised_interfaces_save_dir = 'dissociated_interfaces'
        if not os.path.exists(functionalised_interfaces_save_dir):
            os.makedirs(functionalised_interfaces_save_dir)

        ase.io.write(functionalised_interfaces_save_dir+'/interface_'+substrate_name+'.xyz',final_interface,format='extxyz')
        
    return True





def reduced_interface_multi_builder(
                            substrates_dir,
                            functionalisation_fraction, # Fraction of water to dissociate
                            model_path,
                            elements_to_reduce_dict:dict=None,
                            substrates_whitelist=None,
                            substrates_blacklist=None,
                            functionalised_interfaces_save_dir = None,
                            **kwargs
                            ):
    
    substrates_filenames = os.listdir(substrates_dir)

    # Default interface building parameters
    water_thickness = kwargs.get('water_thickness',15.0)
    water_density = kwargs.get('water_density',1.0)
    intersubstrate_gap = kwargs.get('intersubstrate_gap',50)
    water_substrate_gap = kwargs.get('water_substrate_gap',2)
    
    for filename in substrates_filenames:

        substrate_name_data = filename.split('.')[0].split('_')
        substrate_name  = substrate_name_data[0]+'_'+substrate_name_data[1] 

        if substrates_whitelist is not None:
            if substrate_name not in substrates_whitelist:
                print('Skipping',substrate_name,'as it is not in the whitelist')
                continue

        if substrates_blacklist is not None:
            if substrate_name in substrates_blacklist:
                print('Skipping',substrate_name,'as it is in the blacklist')
                continue

        print('Loading substrate for:',substrate_name)

        substrates_file_path = os.path.join(substrates_dir,substrate_name+'.pdb') 
        if not os.path.exists(substrates_file_path):
             raise FileNotFoundError(f"Substrate file not found: {substrates_file_path}")
        
        substrate = ase.io.read(substrates_file_path,format='proteindatabank')

        # Picking substrate atoms to add H to:
        top_layer_atom_indices = interface_analysis_tools.find_top_layer_indices(substrate)

        print('Top layer atom indices:',top_layer_atom_indices)
        
        elements_to_reduce = elements_to_reduce_dict.get(substrate_name,None)

        if elements_to_reduce is None:
            candidate_atoms_to_reduce = top_layer_atom_indices
        else:
            candidate_atoms_to_reduce = [index for index in top_layer_atom_indices 
                                        if substrate[index].symbol in elements_to_reduce]

        atoms_to_reduce = np.random.choice(candidate_atoms_to_reduce,int(len(candidate_atoms_to_reduce)*functionalisation_fraction),replace=False)

        # Building reduced interface   
        print('Building',filename,' dissociated interface')

        start_time = time.time()


        reduced_substrate = substrate_reducer(
            substrate,
            atoms_to_reduce,
            model_path,name=substrate_name)




        interface_builder_tool = interface_builder.InterfaceBuilder(model_path,
                                                 reduced_substrate,
                                                 water_thickness,
                                                 water_density,
                                                 intersubstrate_gap,
                                                 name=substrate_name)
        
        interface_builder_start_time = time.time()
        interface_builder_tool.build_bulk_interface(water_substrate_gap=water_substrate_gap,
                                                optimise_interfacial_water=True,
                                                )
        interface_builder_end_time = time.time()

        un_pH_balanced_water_interface = interface_builder_tool.interface

        print('Time taken to build water interface:',interface_builder_end_time-interface_builder_start_time)
        print('Adding ions to bulk to balance pH')


        dissociated_interface_builder_tool = dissociated_interface_builder.DissociationBuilder(
            model_path,
            un_pH_balanced_water_interface,
            reduced_substrate,
            substrate_name,
            species_to_add='OH',
            dissociation_z_min=5,
            dissociation_z_max=30, # Just add above 5A
            num_waters_dissociate=len(atoms_to_reduce),
            add_counterions=False,
            optimise_interfacial_water=True,
            only_relax_added_atoms=True,
            force_correct_number_of_species=False,
            )
        
        dissociation_start_time = time.time()
        dissociated_interface_builder_tool.build_dissociated_interface()
        dissociation_end_time = time.time()

        print('Time taken:',dissociation_end_time-dissociation_start_time)

        end_time = time.time()

        print('Time taken:',end_time-start_time)

        pH_balanced_interface = dissociated_interface_builder_tool.interface

        #Saving interface
        if functionalised_interfaces_save_dir is None:
            functionalised_interfaces_save_dir = 'functionalised_interfaces'
        if not os.path.exists(functionalised_interfaces_save_dir):
            os.makedirs(functionalised_interfaces_save_dir)

        ase.io.write(functionalised_interfaces_save_dir+'/functionalised-interface_'+substrate_name+'.xyz',pH_balanced_interface,format='extxyz')
        
    return True




def find_ion_docking_sites(substrate,
                           functionalisation_fraction,
                           elements_to_reduce,
                           elements_to_oxidise,
                           surface_tolerance=0.1,
                           avoid_neighboring_sites=True,
                           **kwargs):


        top_layer_atom_indices = kwargs.get('top_layer_indices',None)
        if top_layer_atom_indices is None:
            top_layer_atom_indices = interface_analysis_tools.find_top_layer_indices(substrate,tolerance=surface_tolerance)

        print('Top layer atom indices:',top_layer_atom_indices)

        candidate_atoms_to_reduce = set([index for index in top_layer_atom_indices 
                                    if substrate[index].symbol in elements_to_reduce])
        
        candidate_atoms_to_oxidise = set([index for index in top_layer_atom_indices 
                                    if substrate[index].symbol in elements_to_oxidise])

        
        # ase.io.write('substrate_'+kwargs.get('substrate_name','material')+'.xyz',substrate,format='extxyz')

        print('Candidate atoms to reduce:',candidate_atoms_to_reduce)
        print('Candidate atoms to oxidise:',candidate_atoms_to_oxidise)
        print('Finding suitable atoms to reduce and oxidise...')
        
        num_waters_to_split = int(round(len(candidate_atoms_to_reduce)*functionalisation_fraction))

        print('Number of substrate atoms eligible for reduction:',len(candidate_atoms_to_reduce))
        print('Functionalisation fraction:',functionalisation_fraction)
        print('Number of waters to split:',num_waters_to_split)



        oxidised_atoms = set([])
        reduced_atoms = set([])

        trying_to_find_suitable_atoms = True
        attempts = 0

        trial_atoms_to_reduce = copy.deepcopy(candidate_atoms_to_reduce)
        trial_atoms_to_oxidise = copy.deepcopy(candidate_atoms_to_oxidise)

        while trying_to_find_suitable_atoms:
            print()
        

            if len(trial_atoms_to_reduce) == 0:
                attempts += 1
                oxidised_atoms = set([])
                reduced_atoms = set([])
                trial_atoms_to_reduce = copy.deepcopy(candidate_atoms_to_reduce)
                trial_atoms_to_oxidise = copy.deepcopy(candidate_atoms_to_oxidise)

            if attempts > 10:
                raise ValueError('Could not find suitable atoms to reduce and oxidise after 10 attempts. Please check the substrate structure and the specified elements to reduce and oxidise.')

            trial_reduce_atom = np.random.choice(list(trial_atoms_to_reduce))
            print('Trying to reduce atom index:',trial_reduce_atom)
            distances=substrate.get_distances(trial_reduce_atom, list(trial_atoms_to_oxidise),mic=True)
            min_distance = distances[np.argmin(distances)]
            closest_oxidise_atoms = [index for index,distance in zip(list(trial_atoms_to_oxidise),distances) if np.abs(distance-min_distance) < 0.3]
            
            print('User has chosen to avoid neighboring sites:',avoid_neighboring_sites)
            if avoid_neighboring_sites:
                possible_oxidation_sites = trial_atoms_to_oxidise - set(closest_oxidise_atoms)
            else:
                possible_oxidation_sites = set(closest_oxidise_atoms)

            suitable_choices=[index for index in possible_oxidation_sites if ( (index not in oxidised_atoms)
                                                                               and (index not in reduced_atoms)
                                                                               and (index != trial_reduce_atom)
                                                                            )]

            
            print('Suitable oxidise atom choices:',suitable_choices)
            if len(suitable_choices) == 0:
                print('No suitable oxidise atom found for reduce atom index:',trial_reduce_atom)
                trial_atoms_to_reduce -= {trial_reduce_atom}
            else:
                chosen_oxidise_atom = np.random.choice(suitable_choices)
                oxidised_atoms.add(chosen_oxidise_atom)
                reduced_atoms.add(trial_reduce_atom)

                trial_atoms_to_oxidise -= {trial_reduce_atom}
                trial_atoms_to_oxidise -= {chosen_oxidise_atom}
                trial_atoms_to_reduce -= {chosen_oxidise_atom}
                trial_atoms_to_reduce -= {trial_reduce_atom}

                print('Chosen oxidise atom index:',chosen_oxidise_atom)

            if len(oxidised_atoms) == num_waters_to_split:
                trying_to_find_suitable_atoms = False

            
        print('Atoms to reduce (add H):',reduced_atoms)
        print('Atoms to oxidise (add OH):',oxidised_atoms)

        return list(reduced_atoms),list(oxidised_atoms)




def substrate_dissociator(substrate,
                          substrate_H_distance,
                          substrate_OH_distance,
                          atoms_to_reduce,
                          atoms_to_oxidise,
                          model_path,
                          ion_tilt=0,
                          add_water_above_OH=False, # Helps prevent recombination
                          optimise_H_atoms=False,
                          name='dissociated_substrate'):

    substrate_to_reduce = copy.deepcopy(substrate)

    # Adding H atoms to substrate
    H_atoms_to_add = ase.Atoms()
    OH_atoms_to_add = ase.Atoms()

    print('Adding H atoms to substrate above indices:', atoms_to_reduce)
    for atom_index in atoms_to_reduce:
        pos_add = substrate_to_reduce[atom_index].position
        print('Adding H above position:', pos_add, 'for atom_index', atom_index)

        h_pos = pos_add + np.array([substrate_H_distance * np.sin(np.radians(ion_tilt)),
                                    0.0,
                                    substrate_H_distance * np.cos(np.radians(ion_tilt))])
        print('H position:', h_pos)
        H_atom = ase.Atom('H', position=h_pos)
        H_atoms_to_add.append(H_atom)

    print('Adding OH groups to substrate above indices:', atoms_to_oxidise)
    for atom_index in atoms_to_oxidise:
        pos_add = substrate_to_reduce[atom_index].position
        print('Adding OH above position:', pos_add, 'for atom_index', atom_index)
        o_pos = pos_add + np.array([0.0, 0.0, substrate_OH_distance])
        h_pos = pos_add + np.array([0.98 * np.sin(-np.radians(ion_tilt)),
                                    0.0,
                                    substrate_OH_distance + 0.98* np.cos(-np.radians(ion_tilt))])
        print('O position:', o_pos)
        print('H (of OH) position:', h_pos)

        O_atom = ase.Atom('O', position=o_pos)
        H_atom = ase.Atom('H', position=h_pos)

        OH_atoms_to_add.append(O_atom)
        OH_atoms_to_add.append(H_atom)

        if add_water_above_OH:
            water_O_pos = pos_add + np.array([0.0, 0.0, substrate_OH_distance + 2.8])
            water_H1_pos = water_O_pos + np.array([0.98 * np.sin(np.radians(104.5/2)),
                                                   0.0,
                                                   0.98 * np.cos(np.radians(104.5/2))])
            
            water_H2_pos = water_O_pos + np.array([-0.98 * np.sin(np.radians(104.5/2)),
                                                   0.0,
                                                   0.98 * np.cos(np.radians(104.5/2))])
            
            print('Adding water above OH to prevent recombination')

            water_O_atom = ase.Atom('O', position=water_O_pos)
            water_H1_atom = ase.Atom('H', position=water_H1_pos)
            water_H2_atom = ase.Atom('H', position=water_H2_pos)

            OH_atoms_to_add.append(water_O_atom)
            OH_atoms_to_add.append(water_H1_atom)
            OH_atoms_to_add.append(water_H2_atom)




    new_substrate_positions = np.append(
        substrate_to_reduce.get_positions(),
        H_atoms_to_add.get_positions(),
        axis=0
    )
    new_substrate_positions = np.append(
        new_substrate_positions,
        OH_atoms_to_add.get_positions(),
        axis=0
    )

    new_substrate_symbols = np.append(
        substrate_to_reduce.get_chemical_symbols(),
        H_atoms_to_add.get_chemical_symbols(),
        axis=0
    )
    new_substrate_symbols = np.append(
        new_substrate_symbols,
        OH_atoms_to_add.get_chemical_symbols(),
        axis=0
    )

    new_substrate = ase.Atoms(new_substrate_symbols, new_substrate_positions)

    new_cell = substrate_to_reduce.get_cell().copy()
    new_cell[2, 2] += 50.0 
    new_substrate.set_cell(new_cell)
    new_substrate.set_pbc(substrate_to_reduce.get_pbc())

    n_sub = len(substrate_to_reduce)
    n_H = len(H_atoms_to_add)
    n_OH = len(OH_atoms_to_add)

    added_H_indices = list(range(n_sub, n_sub + n_H))

    added_OH_indices = list(range(n_sub + n_H, n_sub + n_H + n_OH))

    added_O_indices = []
    added_OH_H_indices = []
    for i, _ in enumerate(atoms_to_oxidise):
        O_idx = n_sub + n_H + 2 * i
        H_idx = n_sub + n_H + 2 * i + 1
        added_O_indices.append(O_idx)
        added_OH_H_indices.append(H_idx)

    print("Substrate atom indices: 0 ..", n_sub - 1)
    print("Added H indices (isolated H):", added_H_indices)
    print("Added OH indices (O and H within OH groups):", added_OH_indices)
    print("  OH O indices:", added_O_indices)
    print("  OH H indices:", added_OH_H_indices)

    print('Setting up MACE calculator')
    calculator = MACECalculator(model_path=model_path, device='cuda')
    new_substrate.set_calculator(calculator)

  
    ase.io.write(name + '_before_optimisation.xyz', new_substrate, format='extxyz')

    print('Stage 1: Optimising OH atoms (H atoms constrained)')

    fix_indices_stage1 = list(range(n_sub)) + added_H_indices
    constraint_stage1 = FixAtoms(indices=fix_indices_stage1)
    new_substrate.set_constraint(constraint_stage1)
    opt1 = BFGS(new_substrate, append_trajectory=True, trajectory=name + '_ion_opt.traj')
    opt1.run(fmax=0.02)

   
    print('Stage 2: Optimising H atoms (OH atoms constrained)')

    if optimise_H_atoms:
        print('Optimising H atoms added to substrate')
        fix_indices_stage2 = list(range(n_sub)) + added_OH_indices
        constraint_stage2 = FixAtoms(indices=fix_indices_stage2)
        new_substrate.set_constraint(constraint_stage2)

        opt2 = BFGS(new_substrate, append_trajectory=True, trajectory=name + '_ion_opt.traj')
        opt2.run(fmax=0.02)
    else:
        print('Skipping optimisation of added H atoms as per user request')

    return new_substrate


# def substrate_dissociator(substrate,
#                           substrate_H_distance,
#                           substrate_OH_distance,
#                           atoms_to_reduce,
#                           atoms_to_oxidise,
#                           model_path,
#                           name='dissociated_substrate'):
    
#     substrate_to_reduce = copy.deepcopy(substrate)

#     # Adding H atoms to substrate

#     H_atoms_to_add = ase.Atoms()
#     OH_atoms_to_add = ase.Atoms()

#     print('Adding H atoms to substrate above indices:',atoms_to_reduce)
#     for atom_index in atoms_to_reduce:
#         pos_add = substrate_to_reduce[atom_index].position
#         print('Adding H above position:',pos_add, 'for atom_index', atom_index)
#         print('H position:',pos_add+np.array([0,0,substrate_H_distance]))
#         H_atom = ase.Atom('H',position=pos_add+np.array([0,0,substrate_H_distance]))
#         H_atoms_to_add.append(H_atom)
    
#     print('Adding OH groups to substrate above indices:',atoms_to_oxidise)
#     for atom_index in atoms_to_oxidise:
#         pos_add = substrate_to_reduce[atom_index].position
#         print('Adding OH above position:',pos_add, 'for atom_index', atom_index)
#         print('O position:',pos_add+np.array([0,0,substrate_OH_distance]))
#         O_atom = ase.Atom('O',position=pos_add+np.array([0,0,substrate_OH_distance]))
#         H_atom = ase.Atom('H',position=pos_add+np.array([0,0,substrate_OH_distance+0.98]))
#         OH_atoms_to_add.append(O_atom)
#         OH_atoms_to_add.append(H_atom)
    
#     new_substrate_positions = np.append(substrate_to_reduce.get_positions(),H_atoms_to_add.get_positions(),axis=0)
#     new_substrate_positions = np.append(new_substrate_positions,OH_atoms_to_add.get_positions(),axis=0)
#     new_substrate_symbols = np.append(substrate_to_reduce.get_chemical_symbols(),H_atoms_to_add.get_chemical_symbols(),axis=0)
#     new_substrate_symbols = np.append(new_substrate_symbols,OH_atoms_to_add.get_chemical_symbols(),axis=0)
#     new_substrate = ase.Atoms(new_substrate_symbols,new_substrate_positions)
    
#     new_cell = substrate_to_reduce.get_cell().copy()
#     new_cell[2,2] += 50.0  # Increase the z-d
    
#     new_substrate.set_cell(new_cell)
#     new_substrate.set_pbc(substrate_to_reduce.get_pbc())

#     # Optimising substrate with added H atoms
#     print('Optimising dissociated OH ions with MACE')
#     calculator = MACECalculator(model_path=model_path,
#                                 device='cuda')
    
#     new_substrate.set_calculator(calculator)

#     constraint = FixAtoms(indices=[atom.index for atom in substrate_to_reduce if atom.index < len(substrate)])
#     new_substrate.set_constraint(constraint)

#     optimiser = BFGS(new_substrate, trajectory=name+'_ions_opt.traj')
#     optimiser.run(fmax=0.02)

#     constraint = FixAtoms(indices=[atom.index for atom in substrate_to_reduce if atom.index < len(substrate)])
#     new_substrate.set_constraint(constraint)

#     optimiser = BFGS(new_substrate, trajectory=name+'_ions_opt.traj')
#     optimiser.run(fmax=0.02)

#     return new_substrate





def substrate_reducer(substrate,atoms_to_reduce,model_path,name='reduced_substrate'):

    substrate_to_reduce = copy.deepcopy(substrate)

    # Adding H atoms to substrate

    H_atoms_to_add = ase.Atoms()

    print('Adding H atoms to substrate above indices:',atoms_to_reduce)
    for atom_index in atoms_to_reduce:
        pos_add = substrate_to_reduce[atom_index].position
        print('Adding H above position:',pos_add, 'for atom_index', atom_index)
        print('H position:',pos_add+np.array([0,0,1]))
        H_atom = ase.Atom('H',position=pos_add+np.array([0,0,1]))
        H_atoms_to_add.append(H_atom)
    
    new_substrate_positions = np.append(substrate_to_reduce.get_positions(),H_atoms_to_add.get_positions(),axis=0)
    new_substrate_symbols = np.append(substrate_to_reduce.get_chemical_symbols(),H_atoms_to_add.get_chemical_symbols(),axis=0)
    new_substrate = ase.Atoms(new_substrate_symbols,new_substrate_positions)
    
    new_cell = substrate_to_reduce.get_cell().copy()
    new_cell[2,2] += 50.0  # Increase the z-d
    
    new_substrate.set_cell(new_cell)
    new_substrate.set_pbc(substrate_to_reduce.get_pbc())
    
    print(new_substrate.positions)

    # Optimising substrate with added H atoms
    print('Optimising reduced substrate geometry with MACE')
    if model_path is not None:
        calculator = MACECalculator(model_path=model_path,
                                    device='cuda')
        
        new_substrate.set_calculator(calculator)

        constraint = FixAtoms(indices=[atom.index for atom in substrate_to_reduce if atom.index < len(substrate)])
        new_substrate.set_constraint(constraint)

        optimiser = BFGS(new_substrate, trajectory=name+'.traj')
        optimiser.run(fmax=0.02)

    return new_substrate