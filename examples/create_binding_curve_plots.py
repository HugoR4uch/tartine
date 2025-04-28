import sys

sys.path.append('/home/hr492/michaelides-share/hr492/Projects/tartine_project/software/tartines') # path to module dirs

import software.tartines.reference_calc_tools.cp2k_binding_curves as cp2k_binding_curves


cp2k_binding_curves.plot_binding_curves('/scratch/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/water_binding_curves/graphene_binding_curve',
                                   plots_dir_path='.')

