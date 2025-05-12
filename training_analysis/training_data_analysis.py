import tartines.reference_calc_tools.aims_tools.fhi_aims_output_analysis as oa
import os 
import matplotlib.pyplot as plt
import numpy as np
import ase

class TrainingDataAnalyser: 

    def __init__(self,test_set_frac=0.05):
        
        self.test_set_fraction = test_set_frac

        # Statistics
        self.num_frames_loaded = 0
        self.num_unconverged_frames = 0
        self.num_frames_excluded = 0
        self.num_training_frames = 0
        self.num_test_set_frames = 0
        self.num_isolated_atoms = 0
        
        # Training Data structure {System: {frame_name: frame's ase.Atoms object}}
        self.isolated_atoms = {'IsolatedAtoms':{}}
        self.training_set = {}
        self.test_set = {}

    

    def load_training_frames(self,
                             training_frames_dir,
                             add_to_test_set=False,
                             isolated_atoms=False,
                             frames_to_exclude=None,
                             system_name=None,):
        
        """
        Takes ABSOLUTE path to directory with DFT calc dirs.

        Loads ase.Atoms objects from calc dirs. Adds to correct data sets.

        If loading isolated atoms, each calc directory name must be the atomic symbol, e.g. 'Si'
        """


        # These stats will be printed at the end
        num_frames_loaded = 0
        num_unconverged_frames = 0 
        num_frames_excluded = 0
        num_training_frames = 0
        num_test_set_frames = 0
        
        if isolated_atoms:
            print('Loading Isolated Atom Frames')

        # (Optional) Extracting system name from path
        if system_name is None:
            system_name=training_frames_dir.split('/')[-1]
            if system_name =='':
                # e.g. if dir ends with '/'
                system_name = system_name.split('/')[-2]

        loaded_configs = {}

        # Loads Atoms objects from calc dirs

        print('awdadw',os.listdir(training_frames_dir))

        for frame_name in os.listdir(training_frames_dir):
            print('loading calc for:',frame_name)
            calc_dir_path = os.path.join(training_frames_dir,frame_name)
            if os.path.isdir(calc_dir_path):
                
                num_frames_loaded+=1

                # Excluding frames, if needed
                if frames_to_exclude is not None:
                    if frame_name in frames_to_exclude:
                        print('Excluding:', frame_name)
                        num_frames_excluded+=1
                        continue

                # Loading Atoms object
                aims_output_path = os.path.join(calc_dir_path,'aims.out')
                try:
                    atoms = ase.io.aims.read_aims_output(aims_output_path,non_convergence_ok=False)
                    print(frame_name,': Converged!')
                except:
                    print(frame_name, ': Not Converged!')
                    num_unconverged_frames+=1
                    continue

                

                # Naming Atoms objects
                if isolated_atoms:
                    # The config_type value specified by MACE docs for isolated atoms:
                    atoms.info['config_type'] = 'IsolatedAtom'
                    if frame_name in self.isolated_atoms['IsolatedAtoms'].keys():
                        print('Element:',frame_name, ' already in isolated atoms!')
                        continue
                    else:
                        self.isolated_atoms['IsolatedAtoms'][frame_name] = atoms
                else:
                    # Naming configs makes post-processing easier
                    atoms.info['config_name'] = frame_name

                    # Adding config if not isolated
                    loaded_configs[frame_name] = atoms
                

        if not isolated_atoms:

            if add_to_test_set:
                num_remaining_frames = num_frames_loaded - num_frames_excluded - num_unconverged_frames
                num_test_set_frames = int(num_remaining_frames * self.test_set_fraction)
                
                if num_test_set_frames == 0:
                    num_test_set_frames = 1
                
                add_to_test_set = np.random.choice(list(loaded_configs.keys()), num_test_set_frames, replace=False)
                self.training_set[system_name] = { key: loaded_configs[key] for key in loaded_configs.keys() if key not in add_to_test_set} 
                self.test_set[system_name] = { key: loaded_configs[key] for key in add_to_test_set}
            
                num_training_frames = num_remaining_frames - num_test_set_frames

            else:
                self.training_set[system_name] = loaded_configs
                num_training_frames = num_frames_loaded - num_frames_excluded - num_unconverged_frames
                

        # Updating stats
        self.num_frames_loaded += num_frames_loaded
        self.num_unconverged_frames += num_unconverged_frames
        self.num_frames_excluded += num_frames_excluded
        self.num_training_frames += num_training_frames
        self.num_test_set_frames += num_test_set_frames


        # Printing stats
        print(f'Loaded {num_frames_loaded} frames for system {system_name}')
        print(f'Excluded {num_frames_excluded} frames')
        print(f'Unconverged frames: {num_unconverged_frames}')
        print(f'Frames used for training: {num_training_frames}')
        print(f'Frames used for testing: {num_test_set_frames}')


        

    def make_training_files(self, train_files_path='./',train_filename='train.xyz',test_filename='test.xyz'):
        
        if not self.training_set:
            raise ValueError('No training data loaded. Try using the load_training_frames method.')
        if not self.test_set:
            print('WARNING: No test data loaded. Try using the load_training_frames method.')
        if not self.isolated_atoms:
            print('WARNING: No isolated atoms loaded. Try using the load_training_frames method.')


        if not os.path.exists(train_files_path):
            os.makedirs(train_files_path)




        print(self.test_set)
        print(self.training_set)


        # Compiling training set

        training_atoms = []
        test_atoms = []

        unique_elements_in_training_set = set()

        for system_name in self.training_set.keys():
            for config_name in self.training_set[system_name].keys():
                atoms = self.training_set[system_name][config_name]
                training_atoms.append(atoms)
                symbols = atoms.get_chemical_symbols()
                unique_elements_in_training_set.update(set(symbols))
        
        for system_name in self.test_set.keys():
            for config_name in self.test_set[system_name].keys():
                atoms = self.test_set[system_name][config_name]
                test_atoms.append(atoms)
        
        for system_name in self.isolated_atoms['IsolatedAtoms'].keys():
            atoms = self.isolated_atoms['IsolatedAtoms'][system_name]
            training_atoms.append(atoms)
            

        isolated_atoms_elements = list( self.isolated_atoms['IsolatedAtoms'].keys() )
        isolated_atoms_elements = set( isolated_atoms_elements) 

        # Writing training set
        ase.io.write(os.path.join(train_files_path,train_filename),training_atoms,format='extxyz')
        training_configs = len(training_atoms)
        print(f'Wrote {training_configs} training frames (and isolated atoms) to {train_files_path+train_filename}')
        
        # Writing test set
        if len(test_atoms) > 0:
            ase.io.write(os.path.join(train_files_path,test_filename),test_atoms,format='extxyz')
            test_configs = len(test_atoms)
            print(f'Wrote {test_configs} test frames to {train_files_path+test_filename}')

        # Stats
        print(f'Loaded {self.num_frames_loaded} frames')
        print(f'Excluded {self.num_frames_excluded} frames')
        print(f'Unconverged frames: {self.num_unconverged_frames}')
        print(f'Frames used for training: {self.num_training_frames}')
        print(f'Frames used for testing: {self.num_test_set_frames}')

        # Isolated atoms
        print('Unique elements in training set:',unique_elements_in_training_set)
        print('Isolated atoms:',isolated_atoms_elements)
        if isolated_atoms_elements!=unique_elements_in_training_set:
            print('WARNING: Isolated atoms not in training set!')








####################################################################################################
# Below is crap which is unlikely to be used ever again.
####################################################################################################



# def get_calc_dir_info(self,calc_dir_path,calc_dir_name=None):
    
#     # STUPID FUNCTION! SHOULD JUST RETURN THE INFERRED NAME!!!

#     """
#     Takes the name of a calc dir (dir where DFT calc was done).
#     (Can manually pass name of the system, will be deduced otherwise)
#     Returns a dictionary with the system name and absolute path the calc dir. 
#     """

#     calc_data = {}

#     if calc_dir_name is None:
#         dir_name =calc_dir_path.split('/')[-1] 
#         if dir_name =='':
#             # In case path ends with '/'
#             dir_name = dir_name.split('/')[-2]
#         calc_data['name'] = dir_name

#     calc_data['path'] = calc_dir_path

    
#     return calc_data






# def make_train_file(self,frames_to_exclude = None, train_file_path='./',train_filename='train.xyz'):
    
#     number_of_systems = 0
#     number_of_excluded_frames = 0
#     number_of_loaded_frames= 0
#     number_of_unconverged_frames = 0


#     training_atoms=[]


#     # Check if all atoms in training/test sets are in isolated atoms
#     # Check if all isolated atoms are in training/test sets
#     # Warns if not 


#     if not self.training_set:
#         raise ValueError('No training data loaded. Try using the load_training_frames method.')
    
#     for system_name in self.training_set.keys():
#         number_of_systems+=1
#         print(system_name)

        
#         for calc_name in self.training_set[system_name].keys():
            
#             # Excluding frames, if needed
#             if frames_to_exclude is not None:
#                 if calc_name in frames_to_exclude:
#                     print('Exculding', calc_name)
#                     number_of_excluded_frames+=1
#                     continue
            

#             # Loading atoms object
#             print('Loading:',calc_name)
#             number_of_loaded_frames+=1
#             calc_dir_path = self.training_set[system_name][calc_name]['path']
#             aims_output_path = os.path.join(calc_dir_path,'aims.out')
#             try:
#                 atoms = ase.io.aims.read_aims_output(aims_output_path,non_convergence_ok=False)
#             except:
#                 print('Not Converged!',aims_output_path)
#                 number_of_unconverged_frames+=1
#                 continue

#             # Checking if isolated atom
#             if 'isolated_atom' in self.training_set[system_name][calc_name]:
#                 if self.training_set[system_name][calc_name]['isolated_atom'] is True:
#                     atoms.info['config_type'] = 'IsolatedAtom'


#             # Adding frame and trajectory index to atoms object
#             atoms.info['frame_name'] = system_name


    
#     ase.io.write(os.path.join(train_file_path,train_filename),training_atoms,format='extxyz')


#     print(f'Loaded {number_of_loaded_frames} frames')
#     print(f'Unconverged frames: {number_of_unconverged_frames}')
#     print(f'Frames used for training: {number_of_loaded_frames-number_of_unconverged_frames}')




# def plot_convergence_statistics(self,title_name=None,plot_path='./',plot_filename='convergence_statistics.png',systems_to_plot=None):
    
#     if not self.training_set:
#         raise ValueError('No training data loaded. Try using the load_training_frames method.')
    

#     all_systems_convergence_info = {}
#     for system_name in self.training_set.keys():
#         if systems_to_plot is not None:
#             if system_name not in systems_to_plot:
#                 continue
#         system_convergence_results = []
#         for calc_name in self.training_set[system_name].keys():
#             calc_dir_path = self.training_set[system_name][calc_name]['path']
#             summary = oa.summarise_aims_output(calc_dir=calc_dir_path,out_file_name='aims.out', write_summary=False)
#             #NOTE: I want to change the output analyser so that I dont need to do this:
#             summary = summary[calc_dir_path]
#             system_convergence_results.append(summary['Convergence'])

#         all_systems_convergence_info[system_name] = system_convergence_results

#     system_names = list(all_systems_convergence_info.keys())
#     print('System names:',system_names)

#     num_successes = np.array([np.sum(convergence_list) for convergence_list in all_systems_convergence_info.values()])
#     num_trials = np.array([len(convergence_list) for convergence_list in all_systems_convergence_info.values()])
#     success_rates = num_successes / num_trials * 100
#     num_fails = num_trials - num_successes
#     print('num_successes',num_successes)
#     print('num_trials',num_trials)
#     print('success_rates',success_rates)

#     # Set up bar positions
#     x = np.arange(len(system_names))

#     # Create bar chart with adjusted figure size
#     fig, ax1 = plt.subplots(figsize=(len(system_names) * 2, 6))

#     # Bar chart for number of fails
#     bars1 = ax1.bar(x - 0.2, num_fails, 0.4, label='Number of Fails', color='orange')
#     ax1.set_ylabel('Number of DFT calculations', color='black')

#     # Create a second y-axis for the number of trials
#     ax2 = ax1.twinx()
#     bars2 = ax2.bar(x + 0.2, num_successes, 0.4, label='Number of Successes', color='lightblue')
#     # ax2.set_ylabel('Number of Unconverged DFT Calculations', color='orange')
#     ax2.set_yticks([])

#     # Add labels and title
#     if title_name is not None:
#         # ax1.set_xlabel('Systems in Group: ' + title_name)
#         ax1.set_title('DFT Convergence for Group: ' + title_name)
#     else:
#         ax1.set_xlabel('Systems')
#         ax1.set_title('Convergence Statistics for Different Systems')

#     ax1.set_xticks(x)
#     ax1.set_xticklabels(system_names)

#     # Add legends
#     # ax1.legend(loc='upper left')
#     # ax2.legend(loc='upper right')

#     # Add text annotations
#     for i in range(len(system_names)):
#         ax1.text(x[i] - 0.2,  2, f'Fails:\n{num_fails[i]}', ha='center', color='black')
#         ax2.text(x[i] + 0.2,  2, f'Passes:\n{num_successes[i]}', ha='center', color='black')

#     plt.savefig(plot_path+plot_filename)





        


# def get_energies_from_isolated_atom_calcs(self,isolated_atoms_dir_path):
#     """
#     Loads isolated atom calc dirs. 
#     Temporarily added to training set.
#     Energies extracted.
#     Isolated atoms removed from training set.
#     """


#     atomic_energies = {}

#     self.load_training_frames(isolated_atoms_dir_path,system_name='isolated_atoms',isolated_atoms=True)
    
#     raise ValueError('This is not working yet!!!!')


#     for element in self.training_set['isolated_atoms'].keys():
        
#         calc_dir_path = self.training_set['isolated_atoms'][element]['path']
#         summary = oa.summarise_aims_output(calc_dir=calc_dir_path,out_file_name='aims.out', write_summary=False)
        
#         #NOTE: I want to change the output analyser so that I dont need to do this:
#         #NOTE: THIS IS CRAZY AND BROKEN -> THIS IS THE UNCORRECTED ENERGY; I TAKE THE ELECTRONIC FREE ENERGY FOR REFERENCE CALCS!
#         summary = summary[calc_dir_path]
#         print(summary['Convergence'])
#         print(element)
#         z = ase.data.atomic_numbers[element]
#         atomic_energies[f"{z}"] = summary['Energy']

#     del self.training_set['isolated_atoms']

#     return atomic_energies

    

#     self.training_set[system_name][calc_data['name']]