import tartines.reference_calc_tools.aims_tools.fhi_aims_output_analysis as oa
import os 
import matplotlib.pyplot as plt
import numpy as np
import ase

class TrainingDataAnalyser: 

    def __init__(self,test_set_frac=0.05):
        
        """
        frames_to_exclude has the form: {'system_dir': [blacklist_calc_dir_path, ...]}
        E.g. frames_to_exclude = {'../ref_calcs/Pt_111_binding': ['../ref_calcs/Pt_111_binding/Pt_111_binding_14', ...], ...}
        """


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

    
    def frame_multi_loader(self,
                           group_name,
                           frames_path,
                           selection_fraction=1.0,
                           selection_type='interval',
                           add_to_test_set=False,
                           isolated_atoms=False,
                           frames_to_exclude={}
                           ):
        
        """
        Loads frames from a directory with multiple systems.
        Takes ABSOLUTE path to directory with DFT calc dirs.
        """
        
        group_data = {}

        for sys_dir in os.listdir(frames_path):
            sys_dir_path = os.path.join(frames_path, sys_dir)
            if os.path.isdir(sys_dir_path):
                
                print('loading:',sys_dir, 'from', sys_dir_path)

                data = self.load_training_frames(sys_dir_path,
                                            system_name=None,
                                            selection_fraction=selection_fraction,
                                            selection_type=selection_type,
                                            add_to_test_set=add_to_test_set,
                                            isolated_atoms=isolated_atoms,
                                            frames_to_exclude = frames_to_exclude,
                                            group_name=group_name)

                group_data[sys_dir] = data

        
        total_frames_loaded = sum([data['num_frames_loaded'] for data in group_data.values()])
        total_unconverged_frames = sum([data['num_unconverged_frames'] for data in group_data.values()])
        total_frames_excluded = sum([data['num_frames_excluded'] for data in group_data.values()])
        total_training_frames = sum([data['num_training_frames'] for data in group_data.values()])
        total_test_set_frames = sum([data['num_test_set_frames'] for data in group_data.values()])


        print(f'Loaded {total_frames_loaded} frames for group {group_name}')
        print(f'Excluded {total_frames_excluded} frames for group {group_name}')
        print(f'Unconverged frames: {total_unconverged_frames} for group {group_name}')
        print(f'Frames used for training: {total_training_frames} for group {group_name}')
        print(f'Frames used for testing: {total_test_set_frames} for group {group_name}')
        

        return group_data




    def load_training_frames(self,
                             training_frames_dir,
                             selection_fraction=1.0,
                             selection_type='interval',
                             add_to_test_set=False,
                             frames_to_exclude={},
                             isolated_atoms=False,
                             system_name=None,
                             group_name=None
                             ):
        
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

        if group_name is None:
            group_name = 'misc'

        loaded_configs = {}

        # Loads Atoms objects from calc dirs


        ref_calc_dirs = os.listdir(training_frames_dir)

        if selection_fraction < 1.0:
            
            num_selected_calcs = int(len(ref_calc_dirs) * selection_fraction)
            if num_selected_calcs == 0:
                raise Exception('WARNING: No frames selected! Selection fraction is too low!')

            

            if selection_type == 'interval':
                # Selecting every nth frame
                selected_ref_calc_dirs = ref_calc_dirs[::int(1/selection_fraction)]
            elif selection_type == 'random':
                # Randomly selecting a fraction of frames
                selected_ref_calc_dirs = np.random.choice(ref_calc_dirs, num_selected_calcs, replace=False)
        
        else:
            # Selecting all frames
            selected_ref_calc_dirs = ref_calc_dirs

            
        for frame_name in selected_ref_calc_dirs:

            print('loading calc for:',frame_name)
            calc_dir_path = os.path.join(training_frames_dir,frame_name)
            if os.path.isdir(calc_dir_path):
                
                num_frames_loaded+=1

                # Excluding frames, if needed
                if system_name in frames_to_exclude.keys():
                    if frame_name in frames_to_exclude[system_name]:
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
                        atoms.info['config_name'] = frame_name
                        atoms.info['config_group'] = 'IsolatedAtoms'
                        self.isolated_atoms['IsolatedAtoms'][frame_name] = atoms
                else:
                    # Naming configs makes post-processing easier
                    atoms.info['config_name'] = frame_name
                    atoms.info['config_group'] = group_name

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


        return {
            'num_frames_loaded': num_frames_loaded,
            'num_unconverged_frames': num_unconverged_frames,
            'num_frames_excluded': num_frames_excluded,
            'num_training_frames': num_training_frames,
            'num_test_set_frames': num_test_set_frames,
            }


        

    def make_training_files(self, train_files_path='./',train_filename='train.xyz',test_filename='test.xyz'):
        
        if not self.training_set:
            raise ValueError('No training data loaded. Try using the load_training_frames method.')
        if not self.test_set:
            print('WARNING: No test data loaded. Try using the load_training_frames method.')
        if not self.isolated_atoms:
            print('WARNING: No isolated atoms loaded. Try using the load_training_frames method.')


        if not os.path.exists(train_files_path):
            os.makedirs(train_files_path)



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


