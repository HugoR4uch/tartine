import sys
sys.path.append('/home/hr492/michaelides-share/hr492/Projects/tartine_project/software')
import tartines.training_analysis.training_data_analysis as ta


all_systems = [
    "BN_",
    "graphene_",
    "WS2_",
    "WSe2_",
    "MoS2_",
    "MoSe2_",
    "SiO2-H_0001",
    "kaolinite-Al_",
    "kaolinite-Si_", #Yair said no
    "NaF_",
    "NaCl", #Yair said no
    "KF_",
    "KCl_",
    "KI_",
    "AgCl_",
    "AgI_",
    "a-TiO2_101",
    "Au_100",
    "Au_111",
    "Au_110",
    "Cu_111",
    "Cu_110",
    "Cu_100",
    "Pt_100",
    "Pt_110",
    "Pt_111",
    "Mg_0001",
    'Ru_0001', #Yair said no
    "Al_111",
    "Ti_0001",
    "Pd_111"
    "MgO_001",
    "KBr_",
    "SiO2_0001",
    "CaF2_111",
    "NaCl_",
    "Pd_110",
    "Pd_100"
]

training_frames_path = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_training_data/training_frames/'
redeemed_systems_path = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_training_data/redeemed_systems/'
analyser = ta.TrainingDataAnalyser()
analyser.training_frames_multi_loader(training_frames_path)
analyser.training_frames_multi_loader(redeemed_systems_path)

# Redeemed Systems
redeemed_systems = [
    "MgO_001",
    "KBr_",
    "SiO2_0001",
    "CaF2_111",
    "NaCl_",
    "Pd_110",
    "Pd_100"
]

analyser.plot_convergence_statistics(title_name='Redeemed Systems',
                                     plot_filename='Redeemed_systems_convergence_statistics.png',
                                     systems_to_plot=redeemed_systems)

# Oxides
oxides= [
    "SiO2-H_0001",
    "kaolinite-Al_",
    "kaolinite-Si_", #Yair said no
    "a-TiO2_101",
    "MgO_001",
    "SiO2_0001",
]

analyser.plot_convergence_statistics(title_name='Oxides',
                                     plot_filename='oxide_convergence_statistics.png',
                                     systems_to_plot=oxides)

# Ionic
ionic= [
    "NaF_",
    "NaCl", #Yair said no
    "KF_",
    "KCl_",
    "KI_",
    "AgCl_",
    "AgI_",
    "KBr_",
    "NaCl_",
]


analyser.plot_convergence_statistics(title_name='Ionic',
                                     plot_filename='ionic_convergence_statistics.png',
                                     systems_to_plot=ionic)
# Metals
metals= [
    "Au_100",
    "Au_111",
    "Au_110",
    "Cu_111",
    "Cu_110",
    "Cu_100",
    "Pt_100",
    "Pt_110",
    "Pt_111",
    "Mg_0001",
    'Ru_0001', #Yair said no
    "Al_111",
    "Ti_0001",
    "Pd_111"
    "MgO_001",
    "Pd_110",
    "Pd_100"
]

analyser.plot_convergence_statistics(title_name='Metals',
                                     plot_filename='Metals_convergence_statistics.png',
                                     systems_to_plot=metals)


# 2D Materials
D2_mats =  [
    "BN_",
    "graphene_",
    "WS2_",
    "WSe2_",
    "MoS2_",
    "MoSe2_",
]

analyser.plot_convergence_statistics(title_name='2D Materials',
                                     plot_filename='2D_materials_convergence_statistics.png',
                                     systems_to_plot=D2_mats)