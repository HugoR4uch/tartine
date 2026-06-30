from ase.visualize.plot import plot_atoms
import ase.io
import ase
import matplotlib.pyplot as plt
import sys
from tartines.interface_analysis import interface_analysis_tools
from tartines.interface_analysis import analysis_plotting_tools
from tartines.interface_analysis import angular_plotting_tools
from tartines.interface_analysis import H_bond_plotting_tools
from tartines.interface_analysis import water_analyser
from tartines.utils import load_density_profile_data
import os
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
import csv


plt.rcParams.update({
    'font.size': 20,              # general font size
    'axes.titlesize': 25,         # plot title
    'axes.labelsize': 20,         # x/y axis labels
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman']
})


simulation_runs = {
    # "Acid_1_75": "/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/dissociated_interface_sims/acid_1/75_dissoc_sims",
    # "Acid_1_50": "/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/dissociated_interface_sims/acid_1/50_dissoc_sims",
    # "Acid_1_25": "/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/dissociated_interface_sims/acid_1/25_dissoc_sims",
    # "Acid_2_25": "/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/dissociated_interface_sims/acid_2/25_dissoc_sims",
    # "Acid_2_50": "/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/dissociated_interface_sims/acid_2/50_dissoc_sims",
    # "Acid_2_75": "/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/dissociated_interface_sims/acid_2/75_dissoc_sims",
    # "GenVIII_25": "/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/dissociated_interface_sims/complete/25_dissoc_sims",
    # "GenVIII_50": "/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/dissociated_interface_sims/complete/50_dissoc_sims",
    # "GenVIII_75": "/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/dissociated_interface_sims/complete/75_dissoc_sims",
    # 'GenVII_25':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_7_no_damping/dissociated_interfaces_sims/gen_7_0_dissoc_sims/25_dissoc_sims',
    # 'GenVII_50':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_7_no_damping/dissociated_interfaces_sims/gen_7_0_dissoc_sims/50_dissoc_sims',
    # 'GenVII_75':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_7_no_damping/dissociated_interfaces_sims/gen_7_0_dissoc_sims/75_dissoc_sims',
    # 'GenVIII':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/gen_8_sims/trajectories_complete',
    # 'Water_1':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/gen_8_sims/trajectories_water_1',
    # 'Water_2':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/gen_8_sims/trajectories_water_2',
    # 'Xavi_model':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/misc/testing_Xavis_graphene_and_BN_model/simulations_with_his_models',
    # 'Graphene_model':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/misc/testing_Xavis_graphene_and_BN_model/simulations/graphene_model',
    # 'BN_model':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/misc/testing_Xavis_graphene_and_BN_model/simulations/BN_model',
    # 'BN_graphene_model':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/misc/testing_Xavis_graphene_and_BN_model/simulations/combined_MPA',
    # 'GenVII':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_7_no_damping/simulations_for_training_configs/trajectories',
    # 'Gen_VIII_more_epochs_1':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/gen_8_sims/more_epochs_1_trajs',
    # 'Gen_VIII_more_epochs_2':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/gen_8_sims/more_epochs_2_trajs',
    # 'Gen_VIII_larger_model':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/gen_8_sims/bigger_model_trajs'
    'Gen IX':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_9_work/simulations/trajectories'
}

current_model_generation = 'Gen IX'

# Which analyses to run
plot_thermodynamics = False
plot_contact_layer_z_density = False
save_data = True
plot_angular_distributions = False
plot_euler_heatmaps = False
plot_species_resolved_euler_density = True
plot_species_sum_density = True
skip_plotted_systems = False
plot_multi_gen_density = False
plot_H_bonds_vs_z = False
plot_contact_layer_xy_free_energy = False
plot_substrate_xy_motion_transform = False
make_snapshot = False
plot_dissociation_statistics = False
plot_default_z_density_profile = False
plot_multi_sim_densities=False

# Analysis parameters
T_target = 330
equilib_end_frame = 4000
min_prominence_of_z_density_peaks=0.25
xy_free_energy_sampling_interval = 10
xy_free_energy_n_xy_bins = 80
xy_free_energy_n_z_bins = 600
xy_free_energy_reference_z_bounds = (8.0, 10.0)
account_for_substrate_xy_motion = True
substrate_xy_motion_transform_stride = xy_free_energy_sampling_interval
xy_top_layer_tolerance = 0.5
euler_sampling_interval = 10
euler_n_bins = 100
species_resolved_density_z_min = 0.0
species_resolved_density_z_max = 10.0
species_resolved_density_bins = 200
species_resolved_density_sampling_interval = 10
allow_overlapping_euler_species = False
# euler_species_z_bounds_mode = "interface"
euler_species_z_bounds_mode = "interface_to_next_trough"
# euler_species_z_bounds_mode = "interface_to_manual_z_max"
euler_species_manual_z_max = 8.0

euler_species_partition_file = os.path.join(
    os.path.dirname(interface_analysis_tools.__file__),
    "euler_species_partitions.json",
)



material_to_euler_species_partition = {
    "NaCl_": "salt_species_partitioning",
    "NaF_": "salt_species_partitioning",
    "NaBr_": "salt_species_partitioning",
    "NaI_": "salt_species_partitioning",
    "KCl_": "salt_species_partitioning",
    "KF_": "salt_species_partitioning",
    "KBr_": "salt_species_partitioning",
    "KI_": "salt_species_partitioning",
    # "MgO_001": "salt_species_partitioning",
    "Pt_111": "metals_species_partitioning",
    "Ag_111": "metals_species_partitioning",
    "Au_111": "metals_species_partitioning",
    "Pd_111": "metals_species_partitioning",
    "Cu_111": "metals_species_partitioning",
    "Ru_0001": "metals_species_partitioning",
}




# Previous Model Directories:
older_gen_sim_dirs ={
    'Gen VII': '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_7_no_damping/simulations_for_training_configs/trajectories',
    # 'Gen VIII - Water 1': '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/gen_8_sims/trajectories_water_1',
    'Gen VIII':'/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_8_work/gen_8_sims/trajectories_complete'
    }

AIMD_density_profiles = {
    'Pt_111': [
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Pt-111_298_RPBE+D3_2024_Dominguez-Flores.csv',
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Pt-111_330_optB88-vdW_2025_Gading.csv',
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Pt-111_330_PBE+D3_2017_Le.csv',
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Pt-111_300_PBE+d3_2024_Li.csv',
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Pt-111_300_PBE+d3_2021_Mikkelsen.csv',
        # '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Pt-111_300_RPBE+d3_2020_Heenen.csv',
        # NEW (shifted): additional adjusted dataset
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_7_work/AIMD_data/shifted_AIMD_data/Pt-111_300_RPBE+d3(shifted)_2020_Heenen.csv',
    ],
    'Pt_100': [
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Pt-100_330_optB88-vdW_2025_Gading.csv',
    ],
    'Ag_111': [
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Ag-111_298_RPBE+D3_2024_Dominguez-Flores.csv',
    ],
    'Au_111': [
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Au-111_298_RPBE+D3_2024_Dominguez-Flores.csv',
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Au-111_330_optB88-vdW_2025_Gading.csv',
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Au-111_330_PBE+D3_2017_Le.csv',
        # UPDATED -> shifted
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_7_work/AIMD_data/shifted_AIMD_data/Au-111_300_RPBE+d3(shifted)_2020_Heenen.csv',
    ],
    'Au_100': [
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Au-100_330_optB88-vdW_2025_Gading.csv',
    ],
    'Cu_100': [
        # UPDATED -> shifted
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_7_work/AIMD_data/shifted_AIMD_data/Cu-100_300_RPBE+d3(shifted)_2016_Natarajan.csv',
    ],
    'Cu_110': [
        # UPDATED -> shifted
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_7_work/AIMD_data/shifted_AIMD_data/Cu-110_300_RPBE+d3(shifted)_2016_Natarajan.csv',
    ],
    'Cu_111': [
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Cu-111_300_PBE+d3_2023_Li.csv',
        # UPDATED -> shifted
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_7_work/AIMD_data/shifted_AIMD_data/Cu-111_300_RPBE+d3(shifted)_2016_Natarajan.csv',
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_7_work/AIMD_data/shifted_AIMD_data/Cu-111_300_RPBE+d3(shifted)_2020_Heenen.csv',
    ],
    'graphene_': [
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Graphene_330_optB88-vdW_2025_Gading.csv',
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Graphene_300_revPBE+D3__ICE-Group.csv',
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/graphene_300_optB88-vdW_2014_Tocci.csv',
    ],
    'BN_': [
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/hBN_300_revPBE+D3__ICE-Group.csv',
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/hBN_300_optB88-vdW_2014_Tocci.csv',
    ],
    'MoS2_': [
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/MoS2_330_optB88-vdW_2025_Gading.csv',
    ],
    'Pd_111': [
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Pd-111_298_RPBE+D3_2024_Dominguez-Flores.csv',
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/Pd-111_330_PBE+D3_2017_Le.csv',
    ],
    'r-TiO2_110': [
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/r-TiO2-110_300_optB88-vdW_2021_Schran.csv',
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/r-TiO2-110_330_SCAN_2023_Wen.csv',
    ],
    'a-TiO2_101': [
        # UPDATED -> shifted (also fixes earlier interface typo)
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_7_work/AIMD_data/shifted_AIMD_data/a-TiO2-101_330_SCAN(shifted)_2020_Calegari-Andrade.csv',
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/a-TiO2-101_330_SCAN_2023_Li.csv',
    ],
    'Al2O3_0001': [
        # UPDATED -> shifted
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_7_work/AIMD_data/shifted_AIMD_data/Al2O3-0001_300_revPBD-d3(shifted)_2023_Zhang.csv',
    ],
    'Al2O3_0001_H': [
        # UPDATED -> shifted
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_7_work/AIMD_data/shifted_AIMD_data/Al2O3-0001-H_300_revPBE-d3(shifted)_2023_Zhang.csv',
        # UPDATED -> shifted
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_7_work/AIMD_data/shifted_AIMD_data/Al2O3-0001-H_330_PBE-d3(shifted)_2024_Du.csv',
    ],
    'MgO_001': [
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/MgO-001_330_PBE0-d3_2021_Ding.csv',
    ],
    'SiO2_H_0001': [
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/SiO2-0001-H_330_BLYP3_2012_Sulpizi.csv',
    ],
    'WSe2_': [
        '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_2_work/gen_2_simulations/simulation_analysis/AIMD_density_profiles/WSe2_350_rev-vdW-DF2_2023_Siheng.csv',
    ],
}


code_to_name = {
    'a-TiO2_101': r'a-TiO$_2$ (101)',
    'Au_111': 'Au (111)',
    'BN_': 'h-BN',
    'graphene_': 'Graphene',
    'MgO_001': 'MgO (001)',
    'MoS2_': r'MoS$_2$',
    'MoSe2_': r'MoSe$_2$',
    'NaCl_': 'NaCl',
    'Pt_100': 'Pt (100)',
    'Pt_110': 'Pt (110)',
    'Pt_111': 'Pt (111)',
    'r-TiO2_110': r'r-TiO$_2$ (110)',
    'WS2_': r'WS$_2$',
}

predefined_contact_layer_z = {}


systems_to_ignore = [
                    # 'Ti_0001',
                    #  'AgCl_',
                    #  'Cu_110',
                    #  'Au_110',
                    #  'KF_',
                    #  'KI_',
                    #  'BN_',
                    # 'Mg_0001'
                     ]


analyse_only_these_systems = {
    "NaCl_",    
    "NaF_",
    "NaBr_",
    "NaI_",
    "KCl_",
    "KF_",
    "KBr_",
    "KI_",
    "Pt_111",
    "Ag_111",
    "Au_111",
    "Pd_111",
    "Cu_111",
    "Ru_0001",
    }




def run_all_analyses_for_run(
    simulation_runs_dir: str,
    run_key: str,
    current_model_generation: str = "Current Gen",
):
    # ---------------------------------------------------------------------
    # Per-run output directories (keyed by run_key)
    # ---------------------------------------------------------------------
    out_root = os.path.join("./analysis_outputs", run_key)
    os.makedirs(out_root, exist_ok=True)

    density_profile_data_dir = os.path.join(out_root, "density_profiles_data")
    density_vs_z_figures_dir = os.path.join(out_root, "density_vs_z_figures")
    snapshot_dir = os.path.join(out_root, "snapshots")
    fragments_plots_dir = os.path.join(out_root, "fragments_plots")
    angular_data_dir = os.path.join(out_root, "angular_data")
    angular_plots_dir = os.path.join(out_root, "angular_plots")
    xy_free_energy_data_dir = os.path.join(out_root, "xy_free_energy_data")
    xy_free_energy_plots_dir = os.path.join(out_root, "xy_free_energy_plots")
    species_resolved_density_data_dir = os.path.join(out_root, "species_resolved_density_data")
    species_resolved_density_plots_dir = os.path.join(out_root, "species_resolved_density_plots")

    contact_layer_density_vs_z_figures_dir = os.path.join(out_root, "contact_layer_density_vs_z_figures")
    density_vs_aimd_figures_dir = os.path.join(out_root, "density_vs_aimd_figures")
    multi_gen_density_figures_dir = os.path.join(out_root, "multi_gen_density_figures")
    contact_layer_angle_vs_z_plots_dir = os.path.join(out_root, "angle_vs_z_plots")  # keeping your name
    all_sims_density_vs_z_figures_dir =  os.path.join(out_root, "multi_sim_vs_z_plots")

    hbond_data_dir = os.path.join(out_root, "hbond_data")
    hbond_plots_dir = os.path.join(out_root, "hbond_plots")
    # (If you actually have separate dirs for angular plots etc, add them here)

    # Create dirs lazily is fine, but pre-creating avoids repeated checks

    for d in [
        hbond_data_dir,
        hbond_plots_dir,
        density_profile_data_dir,
        density_vs_z_figures_dir,
        snapshot_dir,
        fragments_plots_dir,
        contact_layer_density_vs_z_figures_dir,
        density_vs_aimd_figures_dir,
        multi_gen_density_figures_dir,
        contact_layer_angle_vs_z_plots_dir,
        all_sims_density_vs_z_figures_dir,
        angular_data_dir,
        angular_plots_dir,
        xy_free_energy_data_dir,
        xy_free_energy_plots_dir,
        species_resolved_density_data_dir,
        species_resolved_density_plots_dir,
    ]:
        print(f"Ensuring directory exists: {d}")
        os.makedirs(d, exist_ok=True)


    print("\n" + "=" * 100)
    print(f"Starting Analysis for run_key = {run_key}")
    print(f"simulation_runs_dir = {simulation_runs_dir}")
    print(f"output_root        = {out_root}")
    print("=" * 100 + "\n")

    euler_species_partitions = {}
    if plot_species_resolved_euler_density:
        euler_species_partitions = interface_analysis_tools.load_euler_species_partitions(
            euler_species_partition_file
        )
        print(f"Loaded Euler species partitions from: {euler_species_partition_file}")

    grouped_simulation_details = analysis_plotting_tools.get_grouped_simulation_details(simulation_runs_dir)

    # ---------------------------------------------------------------------
    # Already plotted systems (PER RUN)
    # ---------------------------------------------------------------------
    already_plotted_systems = set()
    if skip_plotted_systems and os.path.isdir(density_vs_z_figures_dir):
        for file in os.listdir(density_vs_z_figures_dir):
            if file.endswith(".png") and "_density_vs_z" in file:
                system_name = file.split("_density_vs_z")[0]
                already_plotted_systems.add(system_name)

    # ---------------------------------------------------------------------
    # Loop systems
    # ---------------------------------------------------------------------

    for name, dirs in tqdm(grouped_simulation_details.items(), desc=f"Processing systems ({run_key})"):
        name = f"{name}"
        plot_name = code_to_name[name] if name in code_to_name else name

        print("FOUND SYSTEM:", repr(name))

        if analyse_only_these_systems and name not in analyse_only_these_systems:
            print("Skipping:", repr(name))
            continue

        if name in systems_to_ignore:
            continue
        if skip_plotted_systems and name in already_plotted_systems:
            continue

        substrates, trajectories, logfile_paths, traj_indices = analysis_plotting_tools.extract_simulation_data(
            simulation_runs_dir,
            name,
            dirs,
            equilib_end_frame,
        )

        # -----------------------------------------------------------------
        # Thermodynamics
        # -----------------------------------------------------------------
        if plot_thermodynamics:
            analysis_plotting_tools.make_thermodynamics_plot(
                name,
                logfile_paths,
                T_target,
                equilib_end_frame=equilib_end_frame,
                end_frame=-1,
                figures_dir=os.path.join(out_root, "thermodynamics_figures"),
            )


        # -----------------------------------------------------------------
        # Z density profiles
        # -----------------------------------------------------------------
        O_data = interface_analysis_tools.get_z_density_profile(
            trajectories,
            substrates[0],
            z_min=-1,
            z_max=30,
            plot_all_profiles=False,
        )
        H_data = interface_analysis_tools.get_z_density_profile(
            trajectories,
            substrates[0],
            z_min=-1,
            z_max=30,
            plot_all_profiles=False,
            species="H",
        )

        H_bin_centers, H_average_density, H_errors = H_data
        O_bin_centers, O_average_density, O_errors = O_data

        if plot_default_z_density_profile:
            analysis_plotting_tools.plot_default_density_vs_z(
                name,
                O_bin_centers,
                O_average_density,
                O_errors,
                H_bin_centers,
                H_average_density,
                H_errors,
                density_vs_z_figures_dir=density_vs_z_figures_dir,
            )

        # AIMD comparison
        if name in AIMD_density_profiles:
            analysis_plotting_tools.plot_density_vs_AIMD(
                O_bin_centers,
                O_average_density,
                name,
                current_model_generation,
                AIMD_density_profiles,
                density_vs_aimd_figures_dir=density_vs_aimd_figures_dir,
                plot_name=plot_name,
                z_plot_max=10,
            )

        # multi-gen
        if plot_multi_gen_density:
            analysis_plotting_tools.multi_gen_density_plot(
                O_bin_centers,
                O_average_density,
                name,
                name_of_this_gen=current_model_generation,
                older_gen_sim_dirs=older_gen_sim_dirs,
                multi_gen_density_figures_dir=multi_gen_density_figures_dir,
                equilib_end_frame=equilib_end_frame,
                plot_name=plot_name,
                z_plot_max=10,
                plot_legend=False,
            )

        #Plotting all profiles
        if plot_multi_sim_densities:
            print(f'Plotting multi sim densities for {name}...')
            multi_sim_O_data = interface_analysis_tools.get_z_density_profile(trajectories,
                                                                    substrates[0],
                                                                    z_min=-1,
                                                                    z_max=30,
                                                                    plot_all_profiles=True,
                                                                    )
            
            for mass_density in multi_sim_O_data[1]:
                plt.plot(multi_sim_O_data[0],mass_density)
            
            plt.xlabel('Distance from surface (Å)')
            plt.ylabel(r'Water Density ($gcm^{-3}$)')
            plt.title(f'{name} Density for all trajectories')
            plt.grid(alpha=0.2)
            plot_path = os.path.join(all_sims_density_vs_z_figures_dir,name+'multi_sim.png')
            plt.savefig(plot_path)
            plt.close()

        # -----------------------------------------------------------------
        # Contact-layer peak/trough logic (robust to all-zero densities)
        # -----------------------------------------------------------------
        contact_layer_cutoff = 5.5
        prominence = min_prominence_of_z_density_peaks

        O_avg = np.asarray(O_average_density, dtype=float)
        H_avg = np.asarray(H_average_density, dtype=float)

        O_peaks, _ = find_peaks(O_avg, distance=2, prominence=prominence)
        O_troughs, _ = find_peaks(-O_avg, distance=2, prominence=prominence)
        H_peaks, _ = find_peaks(H_avg, distance=2, prominence=prominence)
        H_troughs, _ = find_peaks(-H_avg, distance=2, prominence=prominence)

        peak_O_z_vals = np.asarray(O_bin_centers)[O_peaks] if len(O_peaks) else np.array([])
        trough_O_z_vals = np.asarray(O_bin_centers)[O_troughs] if len(O_troughs) else np.array([])
        peak_H_z_vals = np.asarray(H_bin_centers)[H_peaks] if len(H_peaks) else np.array([])
        trough_H_z_vals = np.asarray(H_bin_centers)[H_troughs] if len(H_troughs) else np.array([])

        # default starts: first nonzero density bin, else fallback
        nzO = np.where(O_avg > 0)[0]
        z_min_contact_O = float(np.asarray(O_bin_centers)[nzO[0]]) if len(nzO) else float(np.min(O_bin_centers))

        # If peak/trough detection fails, optionally use predefined overrides
        O_contact_layer_start = z_min_contact_O
        if len(O_peaks) == 0 or len(O_troughs) == 0:
            if name in predefined_contact_layer_z:
                O_contact_layer_start = predefined_contact_layer_z[name][0]
                O_contact_layer_end = predefined_contact_layer_z[name][1]
            else:
                # fallback: start at first nonzero, end at cutoff
                O_contact_layer_end = contact_layer_cutoff
        else:
            # normal: end at last trough < cutoff, else cutoff
            contact_layer_troughs = [z for z in trough_O_z_vals if z < contact_layer_cutoff]
            O_contact_layer_end = float(np.max(contact_layer_troughs)) if len(contact_layer_troughs) else float(contact_layer_cutoff)

        # -----------------------------------------------------------------
        # Species-resolved Euler rho(z)
        # -----------------------------------------------------------------
        if plot_species_resolved_euler_density:
            partition_name = material_to_euler_species_partition.get(name)

            if partition_name is None:
                print(f"No Euler species partition configured for {name}; skipping")
            elif partition_name not in euler_species_partitions:
                print(
                    f"Euler species partition {partition_name!r} for {name} "
                    "was not found in the partition file; skipping"
                )
            else:
                base_species_definitions = euler_species_partitions[partition_name]

                if euler_species_z_bounds_mode == "interface":
                    species_z_bounds = [
                        float(O_contact_layer_start),
                        float(O_contact_layer_end),
                    ]
                elif euler_species_z_bounds_mode == "interface_to_next_trough":
                    species_z_bounds = [
                        float(O_contact_layer_start),
                        float(
                            interface_analysis_tools.get_first_trough_after(
                                trough_O_z_vals,
                                O_contact_layer_end,
                                fallback=O_contact_layer_end,
                            )
                        ),
                    ]
                elif euler_species_z_bounds_mode == "interface_to_manual_z_max":
                    species_z_bounds = [
                        float(O_contact_layer_start),
                        float(euler_species_manual_z_max),
                    ]
                else:
                    raise ValueError(
                        "Unknown euler_species_z_bounds_mode: "
                        f"{euler_species_z_bounds_mode}"
                    )

                print(
                    f"Euler species config for {name}: partition={partition_name}, "
                    f"z_bounds_mode={euler_species_z_bounds_mode}, "
                    f"z_bounds=[{species_z_bounds[0]:.3f}, {species_z_bounds[1]:.3f}], "
                    f"sampling_interval={species_resolved_density_sampling_interval}, "
                    f"manual_z_max={euler_species_manual_z_max}"
                )

                species_definitions = interface_analysis_tools.set_euler_species_z_bounds(
                    base_species_definitions,
                    species_z_bounds,
                )

                z_min_tag = f"{species_z_bounds[0]:.3f}".replace("-", "m").replace(".", "p")
                z_max_tag = f"{species_z_bounds[1]:.3f}".replace("-", "m").replace(".", "p")
                species_density_file = os.path.join(
                    species_resolved_density_data_dir,
                    (
                        f"{name}_{partition_name}_{euler_species_z_bounds_mode}"
                        f"_z_{z_min_tag}_{z_max_tag}"
                        f"_stride_{species_resolved_density_sampling_interval}"
                        "_species_resolved_density.csv"
                    ),
                )

                if save_data and os.path.exists(species_density_file):
                    print(f"Loading species-resolved Euler density data for {name}")
                    species_profiles = interface_analysis_tools.load_species_resolved_density_profiles(
                        species_density_file
                    )
                else:
                    print(f"Generating species-resolved Euler density data for {name}")
                    species_profiles = interface_analysis_tools.get_species_resolved_euler_z_density_profiles(
                        trajectories,
                        substrates[0],
                        species_definitions,
                        z_min=species_resolved_density_z_min,
                        z_max=species_resolved_density_z_max,
                        bins=species_resolved_density_bins,
                        sampling_interval=species_resolved_density_sampling_interval,
                        include_species_sum=plot_species_sum_density,
                        allow_overlapping_species=allow_overlapping_euler_species,
                    )

                    if save_data:
                        interface_analysis_tools.save_species_resolved_density_profiles(
                            species_density_file,
                            species_profiles,
                            metadata={
                                "system": name,
                                "partition_name": partition_name,
                                "species_definitions": species_definitions,
                                "species_z_bounds": species_z_bounds,
                                "z_bounds_mode": euler_species_z_bounds_mode,
                                "manual_z_max": euler_species_manual_z_max,
                                "density_z_min": species_resolved_density_z_min,
                                "density_z_max": species_resolved_density_z_max,
                                "density_bins": species_resolved_density_bins,
                                "sampling_interval": species_resolved_density_sampling_interval,
                                "allow_overlapping_species": allow_overlapping_euler_species,
                            },
                        )

                analysis_plotting_tools.plot_species_resolved_density_profiles(
                    name,
                    species_profiles,
                    figures_dir=species_resolved_density_plots_dir,
                    plot_name=plot_name,
                    z_plot_max=species_resolved_density_z_max,
                    plot_species_sum=plot_species_sum_density,
                    O_contact_layer_start=O_contact_layer_start,
                    O_contact_layer_end=O_contact_layer_end,
                    species_z_bounds=species_z_bounds,
                    partition_name=partition_name,
                )

        # -----------------------------------------------------------------
        # Angular distributions
        # -----------------------------------------------------------------
        if plot_angular_distributions:

            z_xlim_contact = (O_contact_layer_start, O_contact_layer_end)

            # -----------------------------
            # Load or generate OH data
            # -----------------------------
            try:
                OH_z_bins, OH_cos_bins, OH_counts = angular_plotting_tools.load_costheta_z_histogram(
                    angular_data_dir,
                    f"{name}_OH",
                )
                print(f"Loaded OH angular data for {name}")

            except Exception:
                print(f"Generating OH angular data for {name}")

                OH_z_bins, OH_cos_bins, OH_counts = angular_plotting_tools.get_binned_interfacial_angular_data(
                    substrates[0],
                    trajectories,
                    mode="OH",
                    z_min=0.0,
                    z_max=30.0,
                    n_z_bins=600,
                    n_cos_bins=100,
                    sampling_interval=10,
                    return_by_traj=True,
                )

                angular_plotting_tools.save_costheta_z_histogram(
                    angular_data_dir,
                    f"{name}_OH",
                    OH_z_bins,
                    OH_cos_bins,
                    OH_counts,
                )

            # -----------------------------
            # Load or generate dipole data
            # -----------------------------
            try:
                dip_z_bins, dip_cos_bins, dip_counts = angular_plotting_tools.load_costheta_z_histogram(
                    angular_data_dir,
                    f"{name}_dipole",
                )
                print(f"Loaded dipole angular data for {name}")

            except Exception:
                print(f"Generating dipole angular data for {name}")

                dip_z_bins, dip_cos_bins, dip_counts = angular_plotting_tools.get_binned_interfacial_angular_data(
                    substrates[0],
                    trajectories,
                    mode="dipole",
                    z_min=0.0,
                    z_max=30.0,
                    n_z_bins=600,
                    n_cos_bins=100,
                    sampling_interval=10,
                    return_by_traj=True,
                )

                angular_plotting_tools.save_costheta_z_histogram(
                    angular_data_dir,
                    f"{name}_dipole",
                    dip_z_bins,
                    dip_cos_bins,
                    dip_counts,
                )

            # -----------------------------
            # 1. P(cos theta_dip), by traj
            # -----------------------------
            angular_plotting_tools.plot_interfacial_angular_distribution(
                dip_z_bins,
                dip_cos_bins,
                dip_counts,
                z_xlim=z_xlim_contact,
                plot_by_traj=True,
                filename=os.path.join(
                    angular_plots_dir,
                    f"{name}_dipole_contact_layer_Pcostheta_by_traj.png",
                ),
                plot_mean=True,
                title=f"{plot_name}: dipole orientation",
                xlabel=r"$\cos(\theta_{\mathrm{dipole}})$",
                ylabel=r"$P(\cos\theta_{\mathrm{dipole}})$",
                
            )

            # -----------------------------
            # 2. P(cos theta_OH), by traj
            # -----------------------------
            angular_plotting_tools.plot_interfacial_angular_distribution(
                OH_z_bins,
                OH_cos_bins,
                OH_counts,
                z_xlim=z_xlim_contact,
                plot_by_traj=True,
                filename=os.path.join(
                    angular_plots_dir,
                    f"{name}_OH_contact_layer_Pcostheta_by_traj.png",
                ),
                plot_mean=True,
                title=f"{plot_name}: OH orientation",
                
            )

            # -----------------------------
            # 3. P(z, cos theta_OH), summed
            # -----------------------------
            angular_plotting_tools.plot_costheta_z_histogram(
                OH_z_bins,
                OH_cos_bins,
                OH_counts,
                filename=os.path.join(
                    angular_plots_dir,
                    f"{name}_OH_contact_layer_costheta_vs_z.png",
                ),
                z_xlim=z_xlim_contact,
                title=f"{plot_name}: OH orientation vs z",
                density=True,
            )

            # -----------------------------
            # 4. P(z, cos theta_dip), summed
            # -----------------------------
            angular_plotting_tools.plot_costheta_z_histogram(
                dip_z_bins,
                dip_cos_bins,
                dip_counts,
                filename=os.path.join(
                    angular_plots_dir,
                    f"{name}_dipole_contact_layer_costheta_vs_z.png",
                ),
                z_xlim=z_xlim_contact,
                title=f"{plot_name}: dipole orientation vs z",
                density=True,
            )

        # -----------------------------------------------------------------
        # Contact-layer Euler pitch/roll heat map
        # -----------------------------------------------------------------
        if plot_euler_heatmaps:

            euler_system_name = f"{name}_contact_layer_pitch_roll"
            euler_plot = os.path.join(
                angular_plots_dir,
                f"{euler_system_name}_heatmap.png",
            )

            try:
                if not save_data:
                    raise FileNotFoundError

                euler_data = angular_plotting_tools.load_euler_angle_data(
                    angular_data_dir,
                    euler_system_name,
                )
                print(f"Loaded Euler pitch/roll data for {name}")

            except Exception:
                print(f"Generating Euler pitch/roll data for {name}")

                euler_data = angular_plotting_tools.get_interfacial_euler_angle_data(
                    substrates[0],
                    trajectories,
                    z_min=O_contact_layer_start,
                    z_max=O_contact_layer_end,
                    sampling_interval=euler_sampling_interval,
                    trajectory_indices=traj_indices,
                )

                if save_data:
                    angular_plotting_tools.save_euler_angle_data(
                        angular_data_dir,
                        euler_system_name,
                        euler_data,
                    )

            angular_plotting_tools.plot_euler_pitch_roll_heatmap(
                euler_data,
                filename=euler_plot,
                title=f"{plot_name}: contact-layer pitch vs roll",
                bins=euler_n_bins,
                density=False,
            )

        # -----------------------------------------------------------------
        # Contact-layer z density plot
        # -----------------------------------------------------------------
        if plot_contact_layer_z_density:
            analysis_plotting_tools.plot_contact_layer_z_density(
                name,
                O_bin_centers,
                O_average_density,
                O_errors,
                H_bin_centers,
                H_average_density,
                H_errors,
                peak_O_z_vals,
                O_contact_layer_start,
                O_contact_layer_end,
                density_vs_z_figures_dir=contact_layer_density_vs_z_figures_dir,
            )

        # -----------------------------------------------------------------
        # Contact-layer F(x, y)
        # -----------------------------------------------------------------
        if plot_substrate_xy_motion_transform:
            substrate_xy_motion_suffix = (
                "_substrate_xy_motion_corrected"
                if account_for_substrate_xy_motion
                else "_substrate_xy_uncorrected"
            )
            substrate_xy_motion_plot = os.path.join(
                xy_free_energy_plots_dir,
                f"{name}{substrate_xy_motion_suffix}.png",
            )

            analysis_plotting_tools.plot_top_layer_xy_motion_transform(
                trajectories,
                substrates[0],
                filename=substrate_xy_motion_plot,
                stride=substrate_xy_motion_transform_stride,
                num_layers=None,
                tolerance=xy_top_layer_tolerance,
                account_for_substrate_xy_motion=account_for_substrate_xy_motion,
                title=f"{plot_name}: substrate top-layer xy transform",
            )

            print(f"Saved substrate xy motion diagnostic plot: {substrate_xy_motion_plot}")

        if plot_contact_layer_xy_free_energy:
            xy_motion_suffix = (
                f"_primitive_uv_substrate_xy_corrected_top_tol_{xy_top_layer_tolerance:g}"
                if account_for_substrate_xy_motion
                else f"_primitive_uv_top_tol_{xy_top_layer_tolerance:g}"
            )
            xy_free_energy_file = os.path.join(
                xy_free_energy_data_dir,
                f"{name}_contact_layer_xy_free_energy_counts{xy_motion_suffix}.npz",
            )
            xy_free_energy_plot = os.path.join(
                xy_free_energy_plots_dir,
                f"{name}_contact_layer_F_uv{xy_motion_suffix}.png",
            )

            z_bounds_contact = (O_contact_layer_start, O_contact_layer_end)

            if save_data and os.path.exists(xy_free_energy_file):
                print(f"Loading contact-layer F(u, v) data for {name}")
                xy_free_energy_data = interface_analysis_tools.load_xy_free_energy_histogram(
                    xy_free_energy_file
                )
            else:
                print(f"Generating contact-layer F(u, v) data for {name}")
                xy_free_energy_data = interface_analysis_tools.get_xy_free_energy_profile(
                    trajectories=trajectories,
                    substrate=substrates[0],
                    z_bounds=z_bounds_contact,
                    reference_z_bounds=xy_free_energy_reference_z_bounds,
                    temperature=T_target,
                    n_xy_bins=xy_free_energy_n_xy_bins,
                    n_z_bins=xy_free_energy_n_z_bins,
                    z_min=0.0,
                    z_max=30.0,
                    sampling_interval=xy_free_energy_sampling_interval,
                    tolerance=xy_top_layer_tolerance,
                    num_layers=None,
                    species="O",
                    return_by_traj=True,
                    account_for_substrate_xy_motion=account_for_substrate_xy_motion,
                )

                if save_data:
                    interface_analysis_tools.save_xy_free_energy_histogram(
                        xy_free_energy_file,
                        xy_free_energy_data["u_bins"],
                        xy_free_energy_data["v_bins"],
                        xy_free_energy_data["z_bins"],
                        xy_free_energy_data["counts"],
                        z_bounds=xy_free_energy_data["z_bounds"],
                        reference_z_bounds=xy_free_energy_data["reference_z_bounds"],
                        temperature=xy_free_energy_data["temperature"],
                        primitive_cell=xy_free_energy_data.get("primitive_cell"),
                        origin_shift=xy_free_energy_data.get("origin_shift"),
                        sampling_interval=xy_free_energy_data.get("sampling_interval"),
                        top_layer_tolerance=xy_free_energy_data.get(
                            "top_layer_tolerance",
                            xy_top_layer_tolerance,
                        ),
                        account_for_substrate_xy_motion=xy_free_energy_data.get(
                            "account_for_substrate_xy_motion",
                            account_for_substrate_xy_motion,
                        ),
                    )

            analysis_plotting_tools.plot_contact_layer_xy_free_energy(
                xy_free_energy_data["u_bins"],
                xy_free_energy_data["v_bins"],
                xy_free_energy_data["free_energy_uv"],
                filename=xy_free_energy_plot,
                plot_name=plot_name,
                z_bounds=xy_free_energy_data["z_bounds"],
                reference_z_bounds=xy_free_energy_data["reference_z_bounds"],
                primitive_cell=xy_free_energy_data.get("primitive_cell"),
                origin_xy=(
                    xy_free_energy_data["origin_shift"][:2]
                    if "origin_shift" in xy_free_energy_data
                    else None
                ),
            )

            print(f"Saved contact-layer F(u, v) plot: {xy_free_energy_plot}")

        
        # -----------------------------------------------------------------
        # H-bonds vs z & stratification
        # -----------------------------------------------------------------
        if plot_H_bonds_vs_z:


            layer_occupancies_plot = os.path.join(
                hbond_plots_dir,
                f"{name}_layer_occupancies.png",
            )

            hbond_json = os.path.join(
                hbond_data_dir,
                f"{name}_hbond_donor_acceptor_data.json",
            )

            hbond_vs_z_plot = os.path.join(
                hbond_plots_dir,
                f"{name}_hbond_vs_z.png",
            )

            hbond_vs_z_data_file = os.path.join(
                hbond_data_dir,
                f"{name}_hbond_vs_z_data.json",
            )

            hbond_stratification_plot = os.path.join(
                hbond_plots_dir,
                f"{name}_hbond_stratification.png",
            )

            layer_occupancies_file = os.path.join(
                hbond_data_dir,
                f"{name}_layer_occupancies.npz",
            )

            layer_hbond_data_file = os.path.join(
                hbond_data_dir,
                f"{name}_layer_hbond_all_regions_data.json",
            )

            hbond_sampling_interval = 100
            substrate_indices = np.arange(len(substrates[0]))

            # -----------------------------
            # 0/1. Load or compute H-bond donor/acceptor data
            # -----------------------------
            if save_data and os.path.exists(hbond_json):
                print(f"Loading H-bond donor/acceptor data for {name}")
                hbond_donor_acceptor_data = H_bond_plotting_tools.load_hbond_donor_acceptor_data(
                    hbond_json
                )
            else:
                print(f"Generating H-bond donor/acceptor data for {name}")

                hbond_donor_acceptor_data = H_bond_plotting_tools.get_hbond_donor_acceptor_data(
                    trajectories=trajectories,
                    substrate_indices=substrate_indices,
                    sampling_interval=hbond_sampling_interval,
                    angle_cut=30,
                    include_substrate=True,
                    num_substrate_layers_for_reference=None,
                )

                if save_data:
                    H_bond_plotting_tools.save_hbond_donor_acceptor_data(
                        hbond_donor_acceptor_data,
                        hbond_json,
                    )

            # -----------------------------
            # 2. H-bonds vs z, water O only
            # -----------------------------
            analyser = water_analyser.Analyser(
                trajectories[0][0],
                substrate_indices=substrate_indices,
            )
            water_O_indices = analyser.aqua_O_indices

            hbond_z_data = H_bond_plotting_tools.get_hbond_z_data(
                hbond_donor_acceptor_data,
                water_O_indices=water_O_indices,
            )

            hbond_vs_z_plot_data = H_bond_plotting_tools.plot_hbond_vs_z(
                hbond_z_data,
                z_bin_width=0.25,
                z_min=0.0,
                z_max=O_contact_layer_end + 8.0,
                filename=hbond_vs_z_plot,
            )

            if save_data:
                H_bond_plotting_tools.save_hbond_vs_z_plot_data_json(
                    hbond_vs_z_data_file,
                    hbond_vs_z_plot_data,
                )

            # -----------------------------
            # 3. Stratification analysis
            # -----------------------------
            z_bounds = [
                [O_contact_layer_start, O_contact_layer_end],
            ]

            matrices, metadata = H_bond_plotting_tools.hbond_donor_acceptor_data_to_layer_matrices(
                hbond_donor_acceptor_data,
                z_bounds=z_bounds,
                z_cut=3.5,
                return_metadata=True,
            )

            # -----------------------------
            # Load or compute layer occupancies, for all trajectories
            # -----------------------------
            if save_data and os.path.exists(layer_occupancies_file):
                print(f"Loading layer occupancies for {name}")
                layer_populations, loaded_z_bounds = interface_analysis_tools.load_layer_occupancies(
                    layer_occupancies_file
                )
            else:
                print(f"Generating layer occupancies for {name}")

                layer_populations = interface_analysis_tools.get_layer_occupancies_from_traj(
                    trajectories,
                    z_bounds=z_bounds,
                    substrate_indices=substrate_indices,
                    sampling_interval=hbond_sampling_interval,
                )

                if save_data:
                    interface_analysis_tools.save_layer_occupancies(
                        layer_occupancies_file,
                        layer_populations,
                        z_bounds,
                    )

            interface_analysis_tools.plot_layer_occupancies(
                layer_populations,
                z_bounds=z_bounds,
                filename=layer_occupancies_plot,
                title=f"{plot_name}: layer occupancies",
                alpha_traj=0.25,
            )

            layer_hbond_data = H_bond_plotting_tools.compute_layer_hbond_all_regions_data(
                layer_populations,
                matrices,
                region_labels=metadata["region_labels"],
            )

            if save_data:
                H_bond_plotting_tools.save_layer_hbond_all_regions_data(
                    layer_hbond_data_file,
                    layer_hbond_data,
                )

            H_bond_plotting_tools.plot_layer_hbond_all_regions_data(
                layer_hbond_data,
                title=f"{plot_name}: layer-resolved H-bond stratification",
                save_path=hbond_stratification_plot,
            )

            print(f"Saved H-bond vs z plot: {hbond_vs_z_plot}")
            print(f"Saved H-bond stratification plot: {hbond_stratification_plot}")




        # -----------------------------------------------------------------
        # Dissociation statistics
        # -----------------------------------------------------------------
        if plot_dissociation_statistics:
            analysis_plotting_tools.plot_dissociation_statistics(
                name,
                trajectories=trajectories,
                substrate=substrates[0],
                z_min=-10,
                z_max=100,
                sampling_interval=20,
                dt=5,
                fragments_plots_dir=fragments_plots_dir,
                save_results=True,
            )

        # -----------------------------------------------------------------
        # Save density profile data
        # -----------------------------------------------------------------
        if save_data:
            unit_cell = np.array(trajectories[0][0].get_cell())
            out = np.column_stack((O_bin_centers, O_average_density, H_average_density))
            filename = os.path.join(density_profile_data_dir, name + "density_profile.csv")

            metadata_info = [
                "contact_layer_start:" + str(O_contact_layer_start),
                "contact_layer_end:" + str(O_contact_layer_end),
                "peaks:" + str(peak_O_z_vals),
                "troughs:" + str(trough_O_z_vals),
                "H_peaks:" +str(peak_H_z_vals),
                "H_troughs:" + str(trough_H_z_vals),
                "v_1:" + str(unit_cell[0]),
                "v_2:" + str(unit_cell[1]),
                "v_3:" + str(unit_cell[2]),
            ]
            metadata = ";".join(metadata_info)

            with open(filename, "w") as f:
                f.write(f"{metadata}\n")
                f.write("z, O_density, H_density\n")
                np.savetxt(f, out, delimiter=",", fmt="%g")

        # -----------------------------------------------------------------
        # Snapshots
        # -----------------------------------------------------------------
        if make_snapshot:
            analysis_plotting_tools.make_interface_snapshot(
                name, trajectories[0][-1], substrates[0], snapshot_dir=snapshot_dir
            )

            num_substrate = len(substrates[0])
            substrate_indices = np.arange(num_substrate)
            z_interface = trajectories[0][-1].get_positions()[substrate_indices][:, 2].max()

            atoms = trajectories[0][-1]
            pos = atoms.get_positions()
            in_contact_layer = pos[:, 2] < (z_interface + O_contact_layer_end)

            contact_layer_only_atoms = atoms[in_contact_layer]
            contact_layer_only_atoms.wrap()

            analysis_plotting_tools.make_interface_snapshot(
                name + "contact_water_only",
                trajectories[0][-1],
                contact_layer_only_atoms,
                snapshot_dir=snapshot_dir,
            )







for run_key, sim_dir in simulation_runs.items():
    run_all_analyses_for_run(sim_dir, run_key)
