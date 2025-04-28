import sys
import os
import ase
import ase.io.aims
import copy
import ase.io
import numpy as np
import matplotlib.pyplot as plt
from mace.calculators import MACECalculator
sys.path.append('/home/hr492/michaelides-share/hr492/Projects/tartine_project/software')
from tartines.reference_calc_tools.aims_tools import fhi_aims_binding_curves as bc


isolated_water_energy = -2082.47323127425
water_O_index = -1

binding_curves_dir = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_training/gen_2_ref_calcs/binding_curves'

binding_curve_plots_dir = './binding_curve_plots'

models = {'agnesi':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/data/mace_models/mace_agnesi_medium.model',
        #   'finetune':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_f_training/data/naive_training/tartine_0_stagetwo_compiled.model',
        #   'scratch':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_f_training/data/scratch_training/scratch_tartine_0_stagetwo_compiled.model',
            'gen_II':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_training/training_gen_2/tartine_II_stagetwo_compiled.model',}


systems_to_ignore = ['kaolinite-Si_','kaolinite-Al_']

model_cutoff_distance = 6.0
make_short_range_energy_corrections = True # Shifts (MACE and DFT) binding energies to zero at point nearest to r_cutoff



def load_binding_configs_from_geometry(binding_curve_dir):
    # Loads binding configs and energies from DFT ref calcs
    
    binding_frames = []
    frame_indices = []

    dirs = os.listdir(binding_curve_dir)
    for dir in dirs:
        path = os.path.join(binding_curve_dir,dir)
        if not os.path.isdir(path):
            continue
        if 'binding' not in dir:
            continue

        index = dir.split('_')[-1]
        frame_indices.append(int(index))


        geometry_path = os.path.join(path,'geometry.in')
        
        
        
        print('Reading:',geometry_path)
        atoms =  ase.io.read(geometry_path,format='aims')
        

        binding_frames.append(atoms)


    return frame_indices, binding_frames


def load_binding_configs_from_aims(binding_curve_dir):
    # Loads binding configs and energies from DFT ref calcs
    
    binding_frames = []
    frame_indices = []
    convergence_mask = []

    dirs = os.listdir(binding_curve_dir)
    for dir in dirs:
        path = os.path.join(binding_curve_dir,dir)
        if not os.path.isdir(path):
            continue
        if 'binding' not in dir:
            continue

        index = dir.split('_')[-1]
        frame_indices.append(int(index))


        aims_output_path = os.path.join(path,'aims.out')
        
        convergence = True
        
        try:
            print('Reading:',aims_output_path)
            atoms =  ase.io.aims.read_aims_output(aims_output_path)
        except:
            print('No convergence for:',aims_output_path)
            atoms =  ase.io.aims.read_aims_output(aims_output_path,non_convergence_ok=True)
            convergence = False

        convergence_mask.append(convergence)

        binding_frames.append(atoms)


    return frame_indices, binding_frames, convergence_mask


def load_substrate_config(binding_curve_dir,file_name):
    #filename can be 'geometry.in' or 'aims.out'
    substrate_file_path = None
    dirs = os.listdir(binding_curve_dir)
    for dir in dirs:
        if 'isolated' in dir and 'water' not in dir:
            if os.path.isdir(os.path.join(binding_curve_dir,dir)):
                substrate_file_path = os.path.join(binding_curve_dir,dir)


    if file_name == 'geometry.in':
        geometry_file_path = os.path.join(substrate_file_path,'geometry.in')
        atoms= ase.io.read(geometry_file_path,format='aims')

    
    elif file_name == 'aims.out':
        aims_file_path = os.path.join(substrate_file_path,'aims.out')
        try:
            atoms =  ase.io.aims.read_aims_output(aims_file_path)
        except:
            print('Error reading aims output file:',aims_file_path)
            return False
    
    return atoms

            


def find_docking_atom_index(binding_curve_frame,water_O_index=-1):
        # Feed an Atoms object for the binding curve. Finds which atom H2O docks to.

        # How it works:
        # 1. Get all distances from water O atom to all other atoms in the system
        # 2. Get all atoms with the smallest xy displacement from water O atom
        # 3. Get the substrate atom with the smallest (magnitude) z displacement from water O atom
        # 4. Return the index of that atom

        O_substrate_distances = binding_curve_frame.get_all_distances(mic=True,vector=True)[water_O_index]
        xy_displacements= O_substrate_distances[:,0:2][:-3]
        z_displacements = abs(O_substrate_distances[:,2])[:-3]

        xy_distances = np.linalg.norm(xy_displacements,axis=1)
        min_xy_distance = np.min(xy_distances[:-3])
        possible_docking_atom_indices = np.where(xy_distances==min_xy_distance)[0]

        min_z= np.min(z_displacements[possible_docking_atom_indices])
        smallest_z_possible_docking = np.where(z_displacements[possible_docking_atom_indices]==min_z)
        docking_index = possible_docking_atom_indices[smallest_z_possible_docking[0][0]]

        return docking_index






def get_DFT_binding_curve_data(binding_curve_dir,
                               water_O_index=water_O_index,
                               isolated_water_energy = isolated_water_energy,
                               ):

    
    frame_indices, binding_configs, convergence_mask = load_binding_configs_from_aims(binding_curve_dir)


    # Loading substrate 
    substrate_file_path = None
    dirs = os.listdir(binding_curve_dir)
    for dir in dirs:
        if 'isolated' in dir and 'water' not in dir:
            if os.path.isdir(os.path.join(binding_curve_dir,dir)):
                substrate_file_path = os.path.join(binding_curve_dir,dir)
    if substrate_file_path is None:
        raise ValueError('No isolated substrate directory found in binding curve dir')
    aims_file_path = os.path.join(substrate_file_path,'aims.out')
    try:
        substrate =  ase.io.aims.read_aims_output(aims_file_path)
    except:
        raise('Error reading isolated substrate aims output file')
    substrate_energy = substrate.get_potential_energy()




    z_approach_vals = []
    binding_config_energies = []
    docking_atom = None
    for config in binding_configs:
        
        docking_index = find_docking_atom_index(config,water_O_index=water_O_index)
        docking_atom = config[docking_index].symbol

        z_approach = config.get_all_distances(mic=True,vector=True)[water_O_index][docking_index][2] 
        z_approach = np.abs(z_approach)
        config_energy = config.get_potential_energy()

        binding_config_energies.append(config_energy)
        z_approach_vals.append(z_approach)


    binding_energies = np.array(binding_config_energies)[convergence_mask] - substrate_energy - isolated_water_energy
    z_approach_vals = np.array(z_approach_vals)[convergence_mask]
    frame_indices = np.array(frame_indices)[convergence_mask]

    sort = np.argsort(z_approach_vals)
    z_approach_vals = np.array(z_approach_vals)[sort]
    binding_energies = np.array(binding_energies)[sort]
    convergence_mask = np.array(convergence_mask)[sort]
    frame_indices = np.array(frame_indices)[sort]

    return z_approach_vals,binding_energies,frame_indices,convergence_mask,docking_atom




def get_MACE_binding_curve_data(binding_curve_dir,
                                model_path,
                                water_O_index=water_O_index,
                                ):
    


    frame_indices, binding_configs = load_binding_configs_from_geometry(binding_curve_dir)


    substrate_file_path = None
    dirs = os.listdir(binding_curve_dir)
    for dir in dirs:
        if 'isolated' in dir and 'water' not in dir:
            if os.path.isdir(os.path.join(binding_curve_dir,dir)):
                substrate_file_path = os.path.join(binding_curve_dir,dir)
    if substrate_file_path is None:
        raise ValueError('No isolated substrate directory found in binding curve dir')

    substrate_geometry_path = os.path.join(substrate_file_path,'geometry.in')
    substrate = ase.io.read(substrate_geometry_path, format='aims')





    angle = 14.5 /180 * np.pi
    water = ase.Atoms('H2O', positions=np.array([[0,-0.95,0],[0.95*np.cos(angle),0.95*np.sin(angle),0],[0,0,0]]))
    water.set_cell([50,50,50])

    
    # Finding energies for ase.Atoms objects

    # calculator = MACECalculator(model_path=model_path,device='cuda')
    print(substrate)
    substrate.calc = MACECalculator(model_path=model_path,device='cuda')
    substrate_energy = substrate.get_potential_energy()
    water.calc = MACECalculator(model_path=model_path,device='cuda')
    isolated_water_energy = water.get_potential_energy()

    z_approach_vals = []
    binding_config_energies = []

    docking_atom = None

    for config in binding_configs:
        
        docking_index = find_docking_atom_index(config,water_O_index=water_O_index)
        docking_atom = config[docking_index].symbol

        z_approach = config.get_all_distances(mic=True,vector=True)[water_O_index][docking_index][2]
        z_approach = np.abs(z_approach) 

        config.calc = MACECalculator(model_path=model_path,device='cuda')
        config_energy = config.get_potential_energy()

        binding_config_energies.append(config_energy)
        z_approach_vals.append(z_approach)


    binding_energies = np.array(binding_config_energies) - substrate_energy - isolated_water_energy
    z_approach_vals = np.array(z_approach_vals)
    frame_indices = np.array(frame_indices)

    sort = np.argsort(z_approach_vals)
    z_approach_vals = np.array(z_approach_vals)[sort]
    binding_energies = np.array(binding_energies)[sort]
    frame_indices = np.array(frame_indices)[sort]


    

    return z_approach_vals,binding_energies,frame_indices, docking_atom




####################################################################################################
# Doing the plotting
####################################################################################################

binding_curves_dirs = os.listdir(binding_curves_dir)
# plot_dir = './binding_curve_plots'
if not os.path.isdir(binding_curve_plots_dir):
    os.mkdir(binding_curve_plots_dir)

# binding_trajs_dir = './binding_trajectories'
# binding_curves_dirs = ['MoS2_binding_curve']


for binding_curve_dir in binding_curves_dirs:
    
    sysname = binding_curve_dir.split('binding_curve')[-2].split('/')[-1]

    if sysname in systems_to_ignore:
        continue

    # if sysname not in systems_in_training_set:
    #     continue
        #pass

    fig_path =  os.path.join(binding_curve_plots_dir,f"{sysname}.png")

    if os.path.isfile(fig_path):
        continue

    binding_curve_path = os.path.join(binding_curves_dir,binding_curve_dir)

    print(sysname)

    z_approach_vals,binding_energies,frame_indices,convergence_mask,docking_atom = get_DFT_binding_curve_data(binding_curve_path)

    
    print('Binding energies:',binding_energies)
    print('Z approach vals:',z_approach_vals)
    print()

    if make_short_range_energy_corrections:
        index = np.abs(z_approach_vals - model_cutoff_distance).argmin()
        zeroing_frame_index = frame_indices[index]
        zeroing_frame_DFT_E = binding_energies[index]

    
    mace_energies = {}
    mace_z_vals = {}
    for model_name,model in models.items():

        MACE_z_vals, MACE_energies, frame_indices, docking_atom = get_MACE_binding_curve_data(binding_curve_path,model)


        if make_short_range_energy_corrections:
            zeroing_frame_MACE_E = MACE_energies[np.where(frame_indices==zeroing_frame_index)[0][0]]
            MACE_energies = MACE_energies - zeroing_frame_MACE_E + zeroing_frame_DFT_E

        mace_energies[model_name] = MACE_energies
        mace_z_vals[model_name] = MACE_z_vals



    # if make_short_range_energy_corrections:
    #     binding_energies = binding_energies - zeroing_frame_DFT_E


    plt.plot(z_approach_vals,binding_energies)
    plt.scatter(z_approach_vals[~convergence_mask],binding_energies[~convergence_mask],marker='x')
    plt.scatter(z_approach_vals[convergence_mask],binding_energies[convergence_mask],marker='.',color='black',label='revPBE-D3')
    for model_name in mace_energies.keys():
        plt.plot(mace_z_vals[model_name],mace_energies[model_name],label=model_name)
    

    if make_short_range_energy_corrections:
        title = sysname +f' (H2O -> {docking_atom})' +' Binding Curve (with short-range energy correction)'
        ylabel = 'Binding Energy (eV) (with short-range energy correction)'
    else:
        title = sysname +f' (H2O -> {docking_atom})' +' Binding Curve'
        ylabel = 'Binding Energy (eV)'
    
    
    plt.title(title)
    plt.xlabel('Displacement (A)')
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    
    plt.savefig(fig_path,dpi=400)
    plt.close()





# def get_MACE_binding_binding_curve_data(binding_curve_dir,model):

#     substrate_config_path  = get_substrate_file(binding_curve_dir)

#     # Computing MACE model binding curve

#     substrate = ase.io.read(substrate_config_path,format='aims')

#     # Checking if system atomic elements are in model training set 
#     binding_curve_elements = get_unique_elements([substrate])

#     substrate.set_pbc([True,True,True])
#     substrate_cell = np.array(substrate.cell)
#     substrate_cell[2][2] =+ 50
#     substrate.set_cell(substrate_cell)
#     angle = 14.5 /180 * np.pi
#     water = ase.Atoms('H2O', positions=np.array([[0,-0.95,0],[0.95*np.cos(angle),0.95*np.sin(angle),0],[0,0,0]]))
#     water.set_cell([50,50,50])

    

#     calculator = MACECalculator(model_path=model,device='cuda')

#     system_energies = []

#     for system in [water,substrate]:
#         sys_calculator = copy.deepcopy(calculator)
#         system.set_calculator(sys_calculator)
#         print(system)
#         energy = system.get_potential_energy()
#         system_energies.append(energy)

#     MACE_water_energy = system_energies[0]
#     substrate_energy = system_energies[1]

#     frame_indices,binding_frames = load_binding_frames(binding_curve_dir)
    
#     sort = np.argsort(frame_indices)

#     z_approach_vals = []
#     interface_energies = []
#     for frame in binding_frames:
#         dock_index = find_docking_atom_index(frame)
#         z_approach_val = frame.positions[-1][2]- frame.positions[dock_index][2]
#         z_approach_vals.append(z_approach_val)
#         system_calculator = copy.deepcopy(calculator)
#         frame.set_calculator(system_calculator)
#         energy = frame.get_potential_energy()
#         interface_energies.append(energy)


#     # print(interface_energies)
#     # print(water_energy)
#     # print(substrate_energy)
#     frame_indices = np.array(frame_indices)[sort]
#     # binding_frames = [binding_frames[i] for i in sort]
#     interface_energies = np.array(interface_energies)[sort]
#     z_approach_vals = np.array(z_approach_vals)[sort]

#     MACE_energies = interface_energies - substrate_energy - MACE_water_energy

#     return z_approach_vals,MACE_energies




# def plot_binding_curves(binding_curve_path,models=None,plot_dir = './new_binding_curve_plots'):
        
#         if not os.path.isdir(plot_dir):
#             os.mkdir(plot_dir)


#         z_approach_vals,binding_energy_vals,convergence_mask,docking_atom,system_name = get_DFT_binding_curve_data(binding_curve_path)
        
#         mace_energies = {}
#         mace_z_vals = {}
#         for model_name,model in models.items():

#             MACE_z_vals, MACE_energies = get_MACE_binding_energies(binding_curve_path,model)
#             if MACE_energies is not None:
#                 mace_energies[model_name] = MACE_energies
#                 mace_z_vals[model_name] = MACE_z_vals


#         plt.plot(z_approach_vals,binding_energy_vals)
#         plt.scatter(z_approach_vals[~convergence_mask],binding_energy_vals[~convergence_mask],marker='x')
#         plt.scatter(z_approach_vals[convergence_mask],binding_energy_vals[convergence_mask],marker='.',color='black',label='revPBE-D3')
#         for model_name in mace_energies.keys():
#             plt.plot(z_approach_vals,mace_energies[model_name],label=model_name)
        
        
        
#         plt.title(system_name +f' (H2O -> {docking_atom})' +' Binding Curve')
#         plt.xlabel('Displacement (A)')
#         plt.ylabel('Binding Energy (eV)')
#         plt.grid()
#         plt.legend()
        
#         plt.savefig(os.path.join(plot_dir,f"{system_name}.png"),dpi=400)
#         plt.close()


    # binding_maker = bc.MACEBindingCurveMaker(calc_dir = calc_dir,
    #                                          substrate_file=substrate_config_path,
    #                                          model=model)

    # frame_indices,binding_frames = load_binding_frames(calc_dir)
    # MACE_energies = compute_mace_binding_curve_info(frame_indices,binding_frames,binding_maker,valid_elements=training_set_elements)



# if __name__ == '__main__':

    
#     # training = ase.io.read('/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/testing_gen_1_models/train.xyz',index =':')
#     # training_set_elements = get_unique_elements(training)
#     # print(training_set_elements)


#     # calc_dir  = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/water_binding_curves/binding_curves/BN_binding_curve'
#     # calc_dir  = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/water_binding_curves/binding_curves/graphene_binding_curve'
#     # model = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/mace_models/gen_1_models/naive_models/tartine_0_stagetwo_compiled.model'
#     # model = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/mace_models/gen_1_models/scratch_models/scratch_tartine_0_stagetwo_compiled.model'
#     # model = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/mace_models/mace_agnesi_medium.model'
#     # model = None



#     models = {'agnesi':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/data/mace_models/mace_agnesi_medium.model',
#             #   'finetune':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_f_training/data/naive_training/tartine_0_stagetwo_compiled.model',
#             #   'scratch':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_f_training/data/scratch_training/scratch_tartine_0_stagetwo_compiled.model',
#               'gen_II':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_training/training_gen_2/tartine_II_stagetwo_compiled.model',}


#     systems_in_training_set = [
#     "AgCl_",
#     "a-TiO2_101",
#     "Au_100",
#     "Au_110",
#     "Au_111",
#     "BN_",
#     "CaF2_111",
#     "Cu_100",
#     "Cu_110",
#     "Cu_111",
#     "graphene_",
#     "kaolinite-Al_",
#     "KBr_",
#     "KCl_",
#     "KF_",
#     "Mg_0001",
#     "MgO_001",
#     "MoS2_",
#     "MoSe2_",
#     "NaCl_",
#     "NaF_",
#     "Pd_111",
#     "Pt_100",
#     "Pt_110",
#     "Pt_111",
#     "SiO2-H_0001",
#     "WS2_",
#     "WSe2_"
# ]


#     binding_curves_dir = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_training/gen_2_ref_calcs/binding_curves'
#     binding_curves_dirs = os.listdir(binding_curves_dir)
#     plot_dir = './binding_curve_plots'
#     if not os.path.isdir(plot_dir):
#         os.mkdir(plot_dir)

#     # binding_trajs_dir = './binding_trajectories'
#     # binding_curves_dirs = ['MoS2_binding_curve']


#     for binding_curve_dir in binding_curves_dirs:
        
#         sysname = binding_curve_dir.split('binding_curve')[-2].split('/')[-1]

#         # if sysname not in systems_in_training_set:
#         #     continue
#             #pass

#         fig_path =  os.path.join(plot_dir,f"{sysname}.png")

#         if os.path.isfile(fig_path):
#             continue

#         binding_curve_path = os.path.join(binding_curves_dir,binding_curve_dir)

#         z_approach_vals,binding_energy_vals,convergence_mask,docking_atom,system_name = get_DFT_binding_curve_data(binding_curve_path)
        
#         mace_energies = {}
#         mace_z_vals = {}
#         for model_name,model in models.items():

#             MACE_z_vals, MACE_energies = get_MACE_binding_energies(binding_curve_path,model)
#             if MACE_energies is not None:
#                 mace_energies[model_name] = MACE_energies
#                 mace_z_vals[model_name] = MACE_z_vals



#         plt.plot(z_approach_vals,binding_energy_vals)
#         plt.scatter(z_approach_vals[~convergence_mask],binding_energy_vals[~convergence_mask],marker='x')
#         plt.scatter(z_approach_vals[convergence_mask],binding_energy_vals[convergence_mask],marker='.',color='black',label='revPBE-D3')
#         for model_name in mace_energies.keys():
#             plt.plot(mace_z_vals[model_name],mace_energies[model_name],label=model_name)
        
        
        
#         plt.title(system_name +f' (H2O -> {docking_atom})' +' Binding Curve')
#         plt.xlabel('Displacement (A)')
#         plt.ylabel('Binding Energy (eV)')
#         plt.grid()
#         plt.legend()
        
#         plt.savefig(fig_path,dpi=400)
#         plt.close()




# # from mace.calculators.foundations_models import mace_mp
# # model='/home/hr492/michaelides-share/hr492/Projects/tartine_project/mace_models/gen_1_models/naive_models/tartine_0_stagetwo_compiled.model'
# # calculator = mace_mp(model='/home/hr492/michaelides-share/hr492/Projects/tartine_project/mace_models/gen_1_models/naive_models/tartine_0_stagetwo_compiled.model', 
# #                     dispersion=True,
# #                     device='cuda')
# # system_calculator = copy.deepcopy(calculator)
# # energy = binding_frames[0].get_potential_energy()

# # config = ase.Atoms('I2',[[0,0,0],[0,0,2.5]])
# # config.calc = calculator
# # config.get_all_distances()
# # E = config.get_potential_energy()
# # print(E)


# # for dir in binding_curves_dirs:
# #     dir_path = os.path.join(binding_curves_dir, dir)
# #     print(dir_path)
# #     system_name = dir.split('binding_curve')[-2].split('/')[-1]
# #     analyser = bc.BindingCurveAnalyser(system_name,water_energy=water_energy)
#     fhi_aims_analyser = bc.FHI_AIMS_BindingCurveAnalyser(water_energy=water_energy)
#     scratch_mace_analyser = bc.MACE_BindingCurveAnalyser(model='/home/hr492/michaelides-share/hr492/Projects/tartine_project/mace_models/gen_1_models/scratch_models/scratch_tartine_0_stagetwo_compiled.model')
#     finetune_mace_analyser = bc.MACE_BindingCurveAnalyser(model='/home/hr492/michaelides-share/hr492/Projects/tartine_project/mace_models/gen_1_models/naive_models/tartine_0_stagetwo_compiled.model')
#     mace_analyser = bc.MACE_BindingCurveAnalyser()
#     fhi_aims_analyser.load_binding_curve_data(dir_path)
#     mace_analyser.load_binding_curve_data(dir_path)

    
    
#     fhi_aims_analyser.save_binding_traj(binding_trajs_dir+system_name+'_traj.xyz',dir_path)
    
    
#     DFT_z_vals = fhi_aims_analyser.binding_curves[system_name]['z_approach_vals']
#     DFT_energy_vals = fhi_aims_analyser.binding_curves[system_name]['binding_energy_vals']
#     MACE_z_vals = mace_analyser.binding_curves[system_name]['z_approach_vals']
#     MACE_energy_vals = mace_analyser.binding_curves[system_name]['binding_energy_vals']
#     convergence_mask = fhi_aims_analyser.binding_curves[system_name]['convergence_mask']
#     docking_atom = fhi_aims_analyser.binding_curves[system_name]['docking_atom']

#     convergence_mask=np.array(convergence_mask)
#     DFT_z_vals = np.array(DFT_z_vals)
#     sort = np.argsort(DFT_z_vals)
#     DFT_z_vals = DFT_z_vals[sort]
#     DFT_z_mask = [z>=2 for z in DFT_z_vals]
#     DFT_z_vals = DFT_z_vals[DFT_z_mask]
#     DFT_energy_vals = np.array(DFT_energy_vals)[sort]
#     DFT_energy_vals = DFT_energy_vals[DFT_z_mask]
#     convergence_mask = convergence_mask[sort]
#     convergence_mask=convergence_mask[DFT_z_mask]

#     MACE_z_vals = np.array(MACE_z_vals)
#     sort = np.argsort(MACE_z_vals)
#     MACE_z_vals = MACE_z_vals[sort]
#     MACE_energy_vals = np.array(MACE_energy_vals)[sort]
#     MACE_z_mask = [z>=2 for z in MACE_z_vals]
#     MACE_z_vals = MACE_z_vals[MACE_z_mask]
#     MACE_energy_vals = MACE_energy_vals[MACE_z_mask]


#     if np.sum(convergence_mask)==0:
#         continue
    
#     plt.plot(DFT_z_vals, DFT_energy_vals, marker='.',color='black',label='revPBE-D3')
#     plt.scatter(DFT_z_vals[~convergence_mask], DFT_energy_vals[~convergence_mask], marker='x',color='black')
#     plt.plot(MACE_z_vals, MACE_energy_vals, marker='.',label='MACE-mp0b',color='blue')
        
    

#     plt.title(system_name +f' (H2O -> {docking_atom})' +' Binding Curve')

#     plt.xlabel('Displacement (A)')
#     plt.ylabel('Binding Energy (eV)')
#     plt.legend()
#     plt.grid() 
#     plt.savefig(os.path.join('./binding_curve_plots',system_name+'binding_curve.png'))
#     plt.close()


#     # analyser.add_binding_curve('FHI_AIMS',
#     #                            z_appraoch_vals,
#     #                            binding_energy_vals,
#     #                            docking_atom=None,
#     #                            convergence_mask=None)

#     # print(analyser.binding_curves)
    
#     # bc.plot_binding_curve(dir)
