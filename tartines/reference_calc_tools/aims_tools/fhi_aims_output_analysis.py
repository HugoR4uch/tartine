#!/home/hr492/software/miniconda3/envs/mace_env/bin/python

import ase 
import ase.io
import numpy as np
import os
import ase.io.aims
import argparse

def extract_simulation_status(output_file_path):

    # Convergence? 

    # Extracting forces and energy from 'Energy and forces in a compact form:'

    # Electronic free energy

    # Electronic ground state (uncorrected)

    # Number of SCF cycles

    # Cell

    # Num Atoms

    # Total wall Time

    return 


def get_labelled_structure(aims_output_path):

    atoms = ase.io.aims.read_aims_output(aims_output_path,non_convergence_ok=False)

    return atoms



def summarise_aims_output(calc_dir, out_file_name='aims.out', write_mode='a',write_summary=True):
    summary = {}
    calc_dir_list = [d for d in os.listdir(calc_dir) if os.path.isdir(os.path.join(calc_dir, d))]
    if len(calc_dir_list)==0:
        calc_dir_list = [calc_dir]
    else:
        print('Multiple directories found, summarising all of them')

    for calc_index , dir_name in enumerate(calc_dir_list):
        
        #If summarising the results of only one calc
        if len(calc_dir_list)==0:
            aims_output_path = os.path.join(calc_dir, out_file_name)
        else:
            aims_output_path = os.path.join(calc_dir, dir_name, out_file_name)
        
        
        if os.path.isfile(aims_output_path):    
            calc_summary = {}


            print(aims_output_path)
            atoms = ase.io.aims.read_aims_output(aims_output_path,non_convergence_ok=True)
            cell = atoms.cell
            calc_summary['Cell_z'] = np.array(cell)[2][2]
            calc_summary['max_Z'] = np.max(atoms.positions[:,2])

            with open(aims_output_path, 'r') as file:
                calc_summary['Convergence'] = False
                scf_iter_times = []
                scf_iter_energ_change = []
                scf_iter_charge_change = []
                scf_iter_energ = []
                for line in file:
                    if '| Number of atoms' in line:
                        num_atoms = int(line.split(':')[1].strip())
                        calc_summary['Number of atoms'] = num_atoms
                    
                    if 'Have a nice day' in line:
                        calc_summary['Convergence'] = True

                    if '| Total energy uncorrected' in line:
                        energy = float(line.split(':')[1].strip().split()[0])
                        calc_summary['Energy'] = energy
                    
                    if '| Electronic free energy' in line:
                        free_energy = float(line.split(':')[1].split()[0].strip())
                        calc_summary['Free Energy'] = free_energy
                    
                    if '| Total energy corrected' in line:
                        corrected_energy_str = line.split(':')[1].strip().split()[0]
                        corrected_energy = float(corrected_energy_str)
                        calc_summary['Corrected Energy'] = corrected_energy

                    if '| Time for this iteration' in line:
                        iteration_wall_time = line.split(':')[1].strip().split('s')[1]
                        iteration_wall_time= float(iteration_wall_time)
                        scf_iter_times.append(iteration_wall_time)
                    
                    if '| Change of total energy' in line:
                        change_energy = float(line.split(':')[1].strip().split()[0])
                        scf_iter_energ_change.append(change_energy)

                    if '| Change of charge density' in line:
                        change_charge_density = float(line.split(':')[1].strip().split()[0])
                        scf_iter_charge_change.append(change_charge_density)

                    # if '| Total energy (T->0) per atom' in line:
                    #     energy_ev = line.split(':')[1].strip().split('eV')[0].strip()
                    #     scf_iter_energ.append(float(energy_ev))

                    
                    if 'Total energy                  :' in line:
                        energy_ev = line.split('Ha')[1].strip().split('eV')[0]
                        scf_iter_energ.append(float(energy_ev))


                mean_scf_iter_time = np.mean(scf_iter_times)
                total_scf_time = np.sum(scf_iter_times)
                calc_summary['SCF_iterations']=np.arange(1,len(scf_iter_times)+1)
                calc_summary['SCF_energy_changes'] = np.array(scf_iter_energ_change)
                calc_summary['SCF_charge_changes'] = np.array(scf_iter_charge_change)
                calc_summary['SCF_energies'] = np.array(scf_iter_energ)
                calc_summary['Total SCF Time'] = total_scf_time

                # if calc_summary['Convergence']:
                #     calc_summary['SCF_energies'] = np.array(scf_iter_energies)[:-1] # as if converged last value will be one with dispersion
                # else:
                #     calc_summary['SCF_energies'] = np.array(scf_iter_energies)


                summary[dir_name] = calc_summary
                
                
                if write_summary is not False:
                    if calc_index == 0:
                        calc_write_mode = write_mode
                    else:
                        calc_write_mode = 'a'
                    with open('./analysis.out', calc_write_mode) as output_file:
                        if write_summary == 'minimal':
                            output_file.write(f"Directory: {dir_name}\n")
                            output_file.write(f"Convergence: {calc_summary['Convergence']}\n")
                            output_file.write(f"SCF Cycles: {len(calc_summary['SCF_energy_changes'])}\n")
                            output_file.write(f"Total SCF Time: {str(total_scf_time)}\n")
                        if write_summary is True:
                            output_file.write(f"Number of atoms: {calc_summary['Number of atoms']}\n")
                            output_file.write(f"Mean SCF Iteration Time: {str(mean_scf_iter_time)}\n")
                            output_file.write(f"Max Z: {calc_summary['max_Z']}\n")
                            output_file.write(f"Cell Z: {calc_summary['Cell_z']}\n")
                            if calc_summary['Convergence']:
                                output_file.write(f"Energy: {calc_summary['Energy']}\n")
                                output_file.write(f"Free Energy: {calc_summary['Free Energy']}\n")
                                output_file.write(f"Corrected Energy: {calc_summary['Corrected Energy']}\n")
                            output_file.write("\n")

    return summary


if __name__ == "__main__":
    pass
    # parser = argparse.ArgumentParser(description="Process some inputs.")

    # parser.add_argument("--aims_output_filename", type=str, default='aims.out', help="Specify the filename")
    # parser.add_argument("--write_mode", type=str, default='a', help="Specify the write_mode mode. Can be 'a' for append or 'w' to over-write")
    # parser.add_argument("--calc_dir", type=str, required=True, help="Specify the directory containing calculation directories")

    # args = parser.parse_args()

    # summary = summarise_aims_output(args.calc_dir, args.aims_output_filename, args.write_mode)


    # import matplotlib.pyplot as plt

    # interface_classes = ['Oxides','ionic','metals','2Dmat']

    # ##########################
    # make_analysis_logs = False
    # ##########################

    # for interface_class in interface_classes:

    #     summary_yair_Oxides = summarise_aims_output('/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_training_data/fhi_aims_testing/second_round_of_my_runs/'+interface_class,out_file_name='aims1.out', write_mode='a', write_summary=make_analysis_logs)
    #     summary_me_Oxides = summarise_aims_output('/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/gen_1_training_data/fhi_aims_testing/second_round_of_my_runs/'+interface_class,out_file_name='aims.out', write_summary=True)

    #     system_names = summary_yair_Oxides.keys()

    #     plot_dir_name = './scf_plots'
    #     if not os.path.exists(plot_dir_name):
    #         os.mkdir(plot_dir_name)

    #     for system_name in system_names:
    #         yair_summary = summary_yair_Oxides[system_name]
    #         me_summary = summary_me_Oxides[system_name]

    #         me_convergence = me_summary['Convergence'] 
    #         yair_convergence = yair_summary['Convergence'] 
    #         scf_energy_me = abs(me_summary['SCF_energy_changes'] )
    #         scf_energy_yair = abs(yair_summary['SCF_energy_changes'] )
    #         scf_iter_me = np.arange(len(scf_energy_me)) 
    #         scf_iter_yair = np.arange(len(scf_energy_yair)) 


    #         scf_charge_me = me_summary['SCF_charge_changes']
    #         scf_charge_yair = yair_summary['SCF_charge_changes']

    #         fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    #         # Energy change plot
    #         axs[0].plot(scf_iter_me, scf_energy_me, label='Me. Converged:'+str(me_convergence))
    #         axs[0].plot(scf_iter_yair, scf_energy_yair, label='Yair. Converged:'+str(yair_convergence), linestyle='--')
    #         axs[0].set_xlabel('SCF Iteration')
    #         axs[0].set_ylabel('SCF Energy Change Magnitude (eV)')
    #         axs[0].legend()
    #         axs[0].set_yscale('log')
    #         axs[0].grid()
    #         axs[0].set_title('Energy Change')

    #         # Charge change plot
    #         axs[1].plot(scf_iter_me, scf_charge_me, label='Me. Converged:'+str(me_convergence))
    #         axs[1].plot(scf_iter_yair, scf_charge_yair, label='Yair. Converged:'+str(yair_convergence), linestyle='--')
    #         axs[1].set_xlabel('SCF Iteration')
    #         axs[1].set_ylabel('SCF Charge Change')
    #         axs[1].legend()
    #         axs[1].set_yscale('log')
    #         axs[1].grid()
    #         axs[1].set_title('Charge Change')

    #         fig.suptitle(system_name)
    #         plt.tight_layout()
    #         plt.savefig(f'{plot_dir_name}/{system_name}.png')
    #         plt.close()
