import sys
sys.path.append('/home/hr492/michaelides-share/hr492/Projects/tartine_project/software')
import tartines.training_analysis.training_data_analysis as ta
import json

isolated_atom_calc_dirs_path = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_b_training/data/isolated_atom_ref_calcs'
# isolated_atom_calc_dirs_path = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_training_data/isolated_atom_ref_calcs_no_spin'


# Getting atomic energies

analyser = ta.TrainingDataAnalyser()
atomic_energies_dict = analyser.get_energies_from_isolated_atom_calcs(isolated_atom_calc_dirs_path)
print('Atomic energies: ')
print(atomic_energies_dict)


# Getting training file

binding_curve_ref_frames = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_b_training/binding_curve_ref_frames'
# redeemed_sys_training_frames_path = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_training_data/redeemed_systems'
training_frames_path = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_a_training/training_frames'
# test_set_path = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_a_training/test_set'


# analyser.training_frames_multi_loader(test_set_path)
analyser.training_frames_multi_loader(training_frames_path)
analyser.training_frames_multi_loader(binding_curve_ref_frames)
analyser.load_training_frames(isolated_atom_calc_dirs_path,system_name='isolated_atoms',isolated_atoms=True)


systems_to_exclude=[
    # "BN_",
    # "graphene_",
    # "WS2_",
    # "WSe2_",
    # "MoS2_",
    # "MoSe2_",
    # "SiO2-H_0001",
    # "kaolinite-Al_",
    #"kaolinite-Si_", #Yair said no
    # "NaF_",
    #"NaCl", #Yair said no
    # "KF_",
    # "KCl_",
    # "KI_",
    # "AgCl_",
    # "AgI_",
    # "a-TiO2_101",
    # "Au_100",
    # "Au_111",
    # "Au_110",
    # "Cu_111",
    # "Cu_110",
    # "Cu_100",
    "Pt_100",
    # "Pt_110",
    # "Pt_111",
    # "Mg_0001",
    #'Ru_0001', #Yair said no
    # "Al_111",
    # "Ti_0001",
    # "Pd_111"
    # "MgO_001",
    # "KBr_",
    # "SiO2_0001",
    # "CaF2_111",
    # "NaCl_",
    # "Pd_110",
    # "Pd_100",
    "Pt_100binding_curve",
    "Pt_110binding_curve",
    "Pt_111binding_curve"
]

# training_dir_path = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_e_training'

# analyser.make_train_file(systems_to_exclude=systems_to_exclude,train_file_path='./',train_filename='train.xyz')
# analyser.load_training_frames(training_frames_path)
analyser.make_train_file(systems_to_exclude=systems_to_exclude,train_file_path='./',train_filename='train_g.xyz')
print('Training file created.')

