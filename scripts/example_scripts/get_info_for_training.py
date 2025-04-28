import sys
sys.path.append('/home/hr492/michaelides-share/hr492/Projects/tartine_project/software')
import tartines.training_analysis.training_data_analysis as ta
import os

isolated_atom_calc_dirs_path = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_training/gen_2_ref_calcs/isolated_atom_ref_calcs'
binding_curve_ref_frames = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_training/gen_2_ref_calcs/binding_curves'
training_frames_path = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_training/gen_2_ref_calcs/traj_frames'


# This bit makes sure that inappropriate frames are excluded ()


binding_curves_too_close = ['r-TiO2_110binding_curve',
                            'WS2_binding_curve',
                            'MoS2_binding_curve',
                            'MoSe2_binding_curve',
                            'WSe2_binding_curve',]

frames_to_exclude_by_system={}

# Traj frames
for sys_dir in os.listdir(training_frames_path):
    sys_dir_path = os.path.join(training_frames_path, sys_dir)
    if os.path.isdir(sys_dir_path):
        print('------------------')
        print(sys_dir)
        print('------------------')

        if sys_dir not in frames_to_exclude_by_system:
            frames_to_exclude_by_system[sys_dir] = []


# Binding curves
for sys_dir in os.listdir(binding_curve_ref_frames):
    sys_dir_path = os.path.join(binding_curve_ref_frames, sys_dir)
    if os.path.isdir(sys_dir_path):
        print('------------------')
        print(sys_dir)
        print('------------------')

        if sys_dir not in frames_to_exclude_by_system:
            frames_to_exclude_by_system[sys_dir] = []

        for calc_dir in os.listdir(sys_dir_path):
            calc_dir_path = os.path.join(sys_dir_path, calc_dir)
            if os.path.isdir(calc_dir_path):
                print(calc_dir)
                

                if sys_dir in binding_curves_too_close:
                    for i in [0,1,2,3,4,5,6]:
                        if calc_dir.endswith('_'+str(i)):
                            print('too close')
                            frames_to_exclude_by_system[sys_dir].append(calc_dir)

                for i in [11,12,13,14]:
                    if calc_dir.endswith('_'+str(i)):
                        print('too far')

                        frames_to_exclude_by_system[sys_dir].append(calc_dir)

                if 'isolated' in calc_dir:
                    print('isolated')
                    frames_to_exclude_by_system[sys_dir].append(calc_dir)
        print()



# This bit loads the actual ref calcs


analyser = ta.TrainingDataAnalyser(test_set_frac=0.05)

analyser.load_training_frames(isolated_atom_calc_dirs_path,
                              system_name='isolated_atoms',
                              isolated_atoms=True)


# Binding Frames not added to test set
for sys_dir in os.listdir(binding_curve_ref_frames):
    sys_dir_path = os.path.join(binding_curve_ref_frames, sys_dir)
    if os.path.isdir(sys_dir_path):
        
        print('loading:',sys_dir)
        analyser.load_training_frames(sys_dir_path,
                                    system_name=None,
                                    frames_to_exclude=frames_to_exclude_by_system[sys_dir],
                                    add_to_test_set=False,
                                    isolated_atoms=False,)
                                

for sys_dir in os.listdir(training_frames_path):
    sys_dir_path = os.path.join(training_frames_path, sys_dir)
    if os.path.isdir(sys_dir_path):
        print('loading:',sys_dir)
        analyser.load_training_frames(sys_dir_path,
                                    system_name=None,
                                    frames_to_exclude=frames_to_exclude_by_system[sys_dir],
                                    add_to_test_set=True,
                                    isolated_atoms=False,
                                    )
                            


# This bit creates the files

analyser.make_training_files(train_files_path='./training',train_filename='train_II.xyz',test_filename='test_II.xyz')


# # analyser.training_frames_multi_loader(test_set_path)
# analyser.training_frames_multi_loader(training_frames_path)
# analyser.training_frames_multi_loader(binding_curve_ref_frames)
# analyser.load_training_frames(isolated_atom_calc_dirs_path,system_name='isolated_atoms',isolated_atoms=True)


# frames_to_exclude=[
#     # "BN_",
#     # "graphene_",
#     # "WS2_",
#     # "WSe2_",
#     # "MoS2_",
#     # "MoSe2_",
#     # "SiO2-H_0001",
#     # "kaolinite-Al_",
#     #"kaolinite-Si_", #Yair said no
#     # "NaF_",
#     #"NaCl", #Yair said no
#     # "KF_",
#     # "KCl_",
#     # "KI_",
#     # "AgCl_",
#     # "AgI_",
#     # "a-TiO2_101",
#     # "Au_100",
#     # "Au_111",
#     # "Au_110",
#     # "Cu_111",
#     # "Cu_110",
#     # "Cu_100",
#     "Pt_100",
#     # "Pt_110",
#     # "Pt_111",
#     # "Mg_0001",
#     #'Ru_0001', #Yair said no
#     # "Al_111",
#     # "Ti_0001",
#     # "Pd_111"
#     # "MgO_001",
#     # "KBr_",
#     # "SiO2_0001",
#     # "CaF2_111",
#     # "NaCl_",
#     # "Pd_110",
#     # "Pd_100",
#     "Pt_100binding_curve",
#     "Pt_110binding_curve",
#     "Pt_111binding_curve"
# ]

# # training_dir_path = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_e_training'

# # analyser.make_train_file(systems_to_exclude=systems_to_exclude,train_file_path='./',train_filename='train.xyz')
# # analyser.load_training_frames(training_frames_path)


# analyser.make_train_file(frames_to_exclude=frames_to_exclude,train_file_path='./',train_filename='train_II.xyz')
# print('Training file created.')

