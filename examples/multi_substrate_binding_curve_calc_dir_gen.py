import sys
sys.path.append('/home/hr492/michaelides-share/hr492/Projects/tartine_project/software')
import software.tartines.reference_calc_tools.cp2k_binding_curves as cp2k_binding_curves
import os


dir_name = 'oxide_binding_calc_dirs'
substrate_dir = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/oxide_interfaces/substrates/'
#substrate_dir = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/metal_interfaces_screening/metal_substrates'
#dir_name = 'metal_binding_calc_dirs'
#substrate_dir = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/halide_salt_and_2D_interfaces/salts_2D_substrates'
#dir_name = 'halide_2D_binding_calc_dirs'

adsorp_elements = {
    'a-TiO2_101': 'Ti',
    'Al2O3_0001': 'Al',
    'Al2O3-H_0001': 'H',
    'CaF2_111': 'Ca',
    'Fe2O3_0001': 'Fe',
    'Fe2O3-H_0001': 'Fe',
    'kaolinite-Al_': 'H',
    'kaolinite-Si_': 'Si',
    'MgO_001': 'Mg',
    'Mg_0001': 'Mg',
    #'Pd_100': 'Pd',
    'Pt-non-orth_111': 'Pt',
    'r-TiO2_110': 'Ti',
    'SiO2_0001': 'Si',
    'SiO2-H_0001': 'H',
    'AgCl_': 'Ag',
    'AgI_': 'Ag',
    'BN_': 'B',
    'graphene_': 'C',
    'KF_': 'K',
    'KCl_': 'K',
    'KBr_': 'K',
    'KI_': 'K',
    'MoS2_': 'Mo',
    'MoSe2_': 'Mo',
    'NaCl_': 'Na',
    'NaBr_': 'Na',
    'NaF_': 'Na',
    'NaI_': 'Na',
    'WS2_': 'W',
    'WSe2_': 'W'
}   

smearing_dir = {'Al_111':True,
                'Au_100':True,
                'Au_110':True,
                'Au_111':True,
                'Pd_100':True,
                'Pd_110':True,
                'Pd_111':True,
                'Pt_100':True,
                'Pt_110':True,
                'Pt_111':True,
                'Cu_100':True,
                'Cu_110':True,
                'Cu_111':True,
                'Ru_0001':True,
                'Ti_0001':True,
                'Mg_0001':True,
                'Pt-non-orth_111':True,
                'graphene_':True,
                }


dir_path = './'+dir_name
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

cp2k_binding_curves.make_binding_curve_calc_dirs(substrate_dir,
                                            cluster='archer2',
                                            calc_dir_path = dir_path,
                                            time_hrs = 0.33,
                                            adsorp_elements_dict=adsorp_elements,
                                            wave_cutoff=1200,
                                            config_file_path = '../../config_files',
                                            default_cp2k_inp_file_path='/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/single_point.inp',
                                            smearing_dict=smearing_dir)
                                            



