
import sys
sys.path.append('/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools')
import binding_curves
import os

substrate_path = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/oxide_interfaces/substrates/kaolinite-Al_.pdb'
adsorp_element = 'H'


dir_path = './test_binding_calc_dirs'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

binding_curves.cp2k_binding_curve_calc_dir(substrate_path,adsorp_element,
                            dir_path,
                            '/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines/reference_calc_tools/single_point.inp',
                            project_name='test',
                            wave_cutoff=1000)

