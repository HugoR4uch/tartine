from ase.visualize.plot import plot_atoms
import ase.io
import ase
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/hr492/michaelides-share/hr492/Projects/tartine_project/software')
from tartines.interface_analysis import interface_analysis_tools
from tartines.interface_analysis import water_analyser
import os
import time
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
import pandas as pd
import json



def convert_numpy_types_for_json(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'item'):  # Handle numpy scalars that have .item() method
        return obj.item()
    elif hasattr(obj, 'tolist'):  # Handle any numpy-like objects with tolist method
        return obj.tolist()
    else:
        return obj



def plot_angular_distributions(
                               name,
                               substrate,
                               trajectories,
                               O_z_min,
                               O_z_max,
                               contact_layer_angle_vs_z_plots_dir='./',
                               contact_layer_angular_plots_dir='./',
                               sampling_interval=20,
                               z_sampling_min = 0,
                               z_sampling_max = 20,
                               z_sampling_increment = 0.05,
                               theta_sampling_increment_degrees = 1,
                               num_cos_bins=50,
                               save_data = True,
                               load_data = True,
                               plot_mean_theta_vs_z = True,
                               trajectory_indices = None,
                               plot_cos_theta=False,
                               rho_z_alpha=0.7,
                               ):

        # Set default trajectory indices if not provided
        if trajectory_indices is None:
            trajectory_indices = [i+1 for i in range(len(trajectories))]
        elif len(trajectory_indices) != len(trajectories):
            raise ValueError(f"Length of trajectory_indices ({len(trajectory_indices)}) must match number of trajectories ({len(trajectories)})")

        # Create system-specific subdirectories
        system_plots_dir = os.path.join(contact_layer_angle_vs_z_plots_dir, f"{name}_angular_plots")
        system_data_dir = os.path.join(contact_layer_angular_plots_dir, f"{name}_angular_data")
        
        if not os.path.exists(system_plots_dir):
            os.makedirs(system_plots_dir)

        if not os.path.exists(system_data_dir):
            os.makedirs(system_data_dir)

        # Define binning
        z_bins = np.arange(z_sampling_min, z_sampling_max + z_sampling_increment, z_sampling_increment)
        theta_bins = np.arange(0, 180 + theta_sampling_increment_degrees, theta_sampling_increment_degrees)
        
        # Initialize angular_data structure
        angular_data = {}
        
        # Data file paths
        data_file_path = f"{system_data_dir}/{name}_angular_data.json"
        
        # Load existing data if requested and available
        if load_data and os.path.exists(data_file_path):
            with open(data_file_path, 'r') as f:
                angular_data = json.load(f)
            print(f"Loaded existing angular data from {data_file_path}")
        else:
            # Collect angular data for each trajectory and frame
            for traj_idx, traj in enumerate(trajectories):
                angular_data[traj_idx] = {}
                for frame_idx, frame in enumerate(tqdm(traj[::sampling_interval], desc=f"Processing traj {trajectory_indices[traj_idx]} frames for Angular analysis {name}")):
                    data = interface_analysis_tools.get_interfacial_z_vs_dipole_angles(frame, substrate)
                    _, z, angles = data
                    
                    angular_data[traj_idx][frame_idx * sampling_interval] = {
                        'angles': angles.tolist() if hasattr(angles, 'tolist') else list(angles),
                        'z_vals': z.tolist() if hasattr(z, 'tolist') else list(z)
                    }
            
            # Save data if requested
            if save_data:
                with open(data_file_path, 'w') as f:
                    json.dump(angular_data, f, indent=2)
                print(f"Saved angular data to {data_file_path}")
        
        # Create binned distributions for each trajectory
        num_trajectories = len(trajectories)
        traj_distributions_full = []
        traj_distributions_contact = []
        
        # Initialize cos(theta) distributions if needed
        if plot_cos_theta:
            # Create evenly spaced cos(theta) bin edges from -1 to +1
            cos_theta_bins = np.linspace(-1, 1, num_cos_bins + 1)
            cos_theta_centers = (cos_theta_bins[:-1] + cos_theta_bins[1:]) / 2
            
            traj_distributions_full_costheta = []
            traj_distributions_contact_costheta = []
        
        for traj_idx in range(num_trajectories):
            # Collect all z and angle data for this trajectory
            all_z_vals = []
            all_angles = []

            # Handle both integer and string keys (depending on whether data was loaded from JSON or not)
            traj_key = str(traj_idx) if str(traj_idx) in angular_data else traj_idx
            
            for frame_data in angular_data[traj_key].values():
                all_z_vals.extend(frame_data['z_vals'])
                all_angles.extend(frame_data['angles'])
            
            all_z_vals = np.array(all_z_vals)
            all_angles = np.array(all_angles)
            
            # Create 2D histogram for full z range (use density=True for probability density)
            hist_full, z_edges, theta_edges = np.histogram2d(all_z_vals, all_angles, bins=[z_bins, theta_bins], density=True)
            traj_distributions_full.append(hist_full)
            
            # Create 2D histogram for contact layer only (use density=True for probability density)
            contact_mask = (all_z_vals >= O_z_min) & (all_z_vals <= O_z_max)
            hist_contact, _, _ = np.histogram2d(all_z_vals[contact_mask], all_angles[contact_mask], bins=[z_bins, theta_bins], density=True)
            traj_distributions_contact.append(hist_contact)
            
            # Create cos(theta) histograms if requested
            if plot_cos_theta:
                # Convert angles to cos(theta) values
                all_cos_theta_vals = np.cos(np.radians(all_angles))
                
                # Create 2D histogram for cos(theta) - full range
                costheta_hist_full, _, cos_theta_edges = np.histogram2d(all_z_vals, all_cos_theta_vals, 
                                                                      bins=[z_bins, cos_theta_bins], density=True)
                traj_distributions_full_costheta.append(costheta_hist_full)
                
                # Create 2D histogram for cos(theta) - contact layer
                costheta_hist_contact, _, _ = np.histogram2d(all_z_vals[contact_mask], all_cos_theta_vals[contact_mask], 
                                                           bins=[z_bins, cos_theta_bins], density=True)
                traj_distributions_contact_costheta.append(costheta_hist_contact)
            
            # Save distribution data for this trajectory if requested
            if save_data:
                # Save full range distribution
                np.savetxt(f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_full_distribution.csv", 
                          hist_full, delimiter=',')
                
                # Save contact layer distribution
                np.savetxt(f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_contact_distribution.csv", 
                          hist_contact, delimiter=',')
                
                # Save cos(theta) distributions if requested
                if plot_cos_theta:
                    np.savetxt(f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_full_costheta_distribution.csv", 
                              costheta_hist_full, delimiter=',')
                    np.savetxt(f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_contact_costheta_distribution.csv", 
                              costheta_hist_contact, delimiter=',')
                
                # Save bin edges
                np.savetxt(f"{system_data_dir}/{name}_z_bin_edges.csv", 
                          z_edges, delimiter=',')
                np.savetxt(f"{system_data_dir}/{name}_theta_bin_edges.csv", 
                          theta_edges, delimiter=',')
                
                # Save cos(theta) bin edges if requested
                if plot_cos_theta:
                    np.savetxt(f"{system_data_dir}/{name}_costheta_bin_edges.csv", 
                              cos_theta_edges, delimiter=',')
        
        # Load cos(theta) distributions if requested and available
        if plot_cos_theta:
            print('Loading or creating cos(theta) distributions with evenly spaced cos(theta) bins')
            
            # Try to load existing cos(theta) distributions
            costheta_data_exists = True
            for traj_idx in range(num_trajectories):
                full_costheta_path = f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_full_costheta_distribution.csv"
                contact_costheta_path = f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_contact_costheta_distribution.csv"
                costheta_edges_path = f"{system_data_dir}/{name}_costheta_bin_edges.csv"
                
                if not (os.path.exists(full_costheta_path) and os.path.exists(contact_costheta_path) and os.path.exists(costheta_edges_path)):
                    costheta_data_exists = False
                    break
            
            if load_data and costheta_data_exists:
                print("Loading existing cos(theta) distribution data...")
                # Load cos(theta) bin edges
                cos_theta_edges = np.loadtxt(f"{system_data_dir}/{name}_costheta_bin_edges.csv", delimiter=',')
                cos_theta_centers = (cos_theta_edges[:-1] + cos_theta_edges[1:]) / 2
                
                # Load cos(theta) distributions for each trajectory
                for traj_idx in range(num_trajectories):
                    full_costheta_hist = np.loadtxt(f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_full_costheta_distribution.csv", delimiter=',')
                    contact_costheta_hist = np.loadtxt(f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_contact_costheta_distribution.csv", delimiter=',')
                    
                    traj_distributions_full_costheta.append(full_costheta_hist)
                    traj_distributions_contact_costheta.append(contact_costheta_hist)
                
                print(f"Loaded cos(theta) distributions for {num_trajectories} trajectories")
            else:
                print("cos(theta) distribution data not found or load_data=False, but distributions should have been created above.")
        
        # Create mesh for plotting
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
        Z, Theta = np.meshgrid(z_centers, theta_centers, indexing='ij')
        
        # Ensure cos(theta) centers are defined when needed
        if plot_cos_theta and 'cos_theta_centers' not in locals():
            # If cos(theta) data wasn't loaded, define centers from the bins created above
            if 'cos_theta_edges' in locals():
                cos_theta_centers = (cos_theta_edges[:-1] + cos_theta_edges[1:]) / 2
            else:
                # Fallback: create cos(theta) bins if they don't exist
                num_cos_bins = len(theta_bins) - 1
                cos_theta_bins = np.linspace(-1, 1, num_cos_bins + 1)
                cos_theta_centers = (cos_theta_bins[:-1] + cos_theta_bins[1:]) / 2


      
        
        # Plot 1: 2 x Num_trajectories subplot of z,theta histograms
        fig, axes = plt.subplots(2, num_trajectories, figsize=(5*num_trajectories, 8))

        if num_trajectories == 1:
            axes = axes.reshape(-1, 1)
        
        # Determine color scale limits for consistent colorbar across all subplots
        if plot_cos_theta:
            all_distributions_full = traj_distributions_full_costheta
            all_distributions_contact = traj_distributions_contact_costheta
            # Use cos_theta_edges if available, otherwise create from centers
            if 'cos_theta_edges' in locals():
                y_edges = cos_theta_edges
            else:
                # Reconstruct edges from centers if needed
                y_edges = np.linspace(-1, 1, len(cos_theta_centers) + 1)
            y_label = r'$\cos(\theta)$'
            colorbar_label = 'Probability Density'
        else:
            all_distributions_full = traj_distributions_full
            all_distributions_contact = traj_distributions_contact
            y_edges = theta_edges
            y_label = 'Angle [°]'
            colorbar_label = 'Probability Density'
        
        # Find global min and max for consistent color scale
        vmin_full = min([np.min(dist) for dist in all_distributions_full])
        vmax_full = max([np.max(dist) for dist in all_distributions_full])
        vmin_contact = min([np.min(dist) for dist in all_distributions_contact])
        vmax_contact = max([np.max(dist) for dist in all_distributions_contact])
        
        for traj_idx in range(num_trajectories):
            plot_traj_distributions_full = all_distributions_full[traj_idx]
            plot_traj_distributions_contact = all_distributions_contact[traj_idx]

            # Top row: full z range
            im1 = axes[0, traj_idx].pcolormesh(z_edges, y_edges, plot_traj_distributions_full.T, 
                                             cmap='Blues', vmin=vmin_full, vmax=vmax_full)
            axes[0, traj_idx].set_title(f'Traj {trajectory_indices[traj_idx]}')
            axes[0, traj_idx].set_xlabel('z [Å]')
            axes[0, traj_idx].set_ylabel(y_label)
            
            # Bottom row: contact layer only (use same data but limit x-axis view)
            im2 = axes[1, traj_idx].pcolormesh(z_edges, y_edges, plot_traj_distributions_full.T, 
                                             cmap='Blues', vmin=vmin_full, vmax=vmax_full)
            axes[1, traj_idx].set_xlabel('z [Å]')
            axes[1, traj_idx].set_ylabel(y_label)
            axes[1, traj_idx].set_xlim(O_z_min, O_z_max)
        
        # Add row titles
        fig.text(0.02, 0.75, 'Distribution for Entire Slab', rotation=90, 
                fontsize=14, ha='center', va='center', weight='bold')
        fig.text(0.02, 0.25, 'Distribution for the Contact Layer', rotation=90, 
                fontsize=14, ha='center', va='center', weight='bold')
        
        # Add single colorbar for the entire figure
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im1, cax=cbar_ax, label=colorbar_label)

        fig.suptitle('Angle vs z distributions', fontsize=16, weight='bold')
        plt.tight_layout()


        if plot_cos_theta:
            plt.savefig(system_plots_dir + '/' + name + '_costheta_vs_z_by_traj.png', dpi=300)
        else:
            plt.savefig(system_plots_dir + '/' + name + '_angle_vs_z_by_traj.png', dpi=300)
        plt.close()
        
        # Plot 2: Combined data from all trajectories - Full range
        if plot_cos_theta:
            combined_hist_full = np.sum(traj_distributions_full_costheta, axis=0)
            # Use cos_theta_edges if available, otherwise create from centers
            if 'cos_theta_edges' in locals():
                y_edges_combined = cos_theta_edges
            else:
                y_edges_combined = np.linspace(-1, 1, len(cos_theta_centers) + 1)
            y_label_combined = r'$\cos(\theta)$'
            title_combined = f'{name} Water Dipole cos(θ) Distributions'
            filename_combined = system_plots_dir + '/' + name + '_costheta_vs_z_combined_full.png'
        else:
            combined_hist_full = np.sum(traj_distributions_full, axis=0)
            y_edges_combined = theta_edges
            y_label_combined = 'Water Dipole Angle [°]'
            title_combined = f'{name} Water Dipole Angle Distributions'
            filename_combined = system_plots_dir + '/' + name + '_angle_vs_z_combined_full.png'
            
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(z_edges, y_edges_combined, combined_hist_full.T, cmap='Blues')
        plt.colorbar(label='Probability Density')
        plt.xlabel('z [Å]')
        plt.ylabel(y_label_combined)
        plt.title(title_combined)
        plt.tight_layout()
        plt.savefig(filename_combined, dpi=300)
        plt.close()
        
        # Plot 3: Combined data from all trajectories - Contact layer
        if plot_cos_theta:
            combined_hist_contact = np.sum(traj_distributions_contact_costheta, axis=0)
            # Use cos_theta_edges if available, otherwise create from centers
            if 'cos_theta_edges' in locals():
                y_edges_contact = cos_theta_edges
            else:
                y_edges_contact = np.linspace(-1, 1, len(cos_theta_centers) + 1)
            y_label_contact = 'cos(θ)'
            title_contact = f'{name} Contact Layer Water Dipole cos(θ) Distribution'
            filename_contact = system_plots_dir + '/' + name + '_costheta_vs_z_combined_contact.png'
        else:
            combined_hist_contact = np.sum(traj_distributions_contact, axis=0)
            y_edges_contact = theta_edges
            y_label_contact = 'Water Dipole Angle [°]'
            title_contact = f'{name} Water Dipole Angle Distribution (Contact Layer)'
            filename_contact = system_plots_dir + '/' + name + '_angle_vs_z_combined_contact.png'
            
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(z_edges, y_edges_contact, combined_hist_contact.T, cmap='Blues')
        plt.colorbar(label='Probability Density')
        plt.xlabel('z [Å]')
        plt.ylabel(y_label_contact)
        plt.title(title_contact)
        plt.xlim(O_z_min, O_z_max)
        plt.tight_layout()
        plt.savefig(filename_contact, dpi=300)
        plt.close()
        
        # Plot 4: Mean theta vs z plots if requested
        if plot_mean_theta_vs_z:
            # Calculate water density rho(z) for background plotting
            print("Calculating water density rho(z) for background plotting...")
            
            # Collect all z values from all trajectories and frames for density calculation
            all_z_vals_for_density = []
            for traj_idx in range(num_trajectories):
                # Handle both integer and string keys
                traj_key = str(traj_idx) if str(traj_idx) in angular_data else traj_idx
                
                for frame_data in angular_data[traj_key].values():
                    all_z_vals_for_density.extend(frame_data['z_vals'])
            
            all_z_vals_for_density = np.array(all_z_vals_for_density)
            
            # Create density histogram using the same z bins
            density_counts, _ = np.histogram(all_z_vals_for_density, bins=z_bins, density=True)
            density_z_centers = z_centers
            
            # Calculate mean theta for each z bin for each trajectory
            fig, axes = plt.subplots(1, num_trajectories, figsize=(5*num_trajectories, 5))
            if num_trajectories == 1:
                axes = [axes]
            
            all_traj_mean_theta = []
            
            for traj_idx in range(num_trajectories):
                # Calculate weighted mean for each z bin
                mean_values_z = []
                for z_idx in range(len(z_centers)):
                    if plot_cos_theta:
                        # Use cos(theta) distribution directly for proper <cos(theta)> calculation
                        costheta_weights = traj_distributions_full_costheta[traj_idx][z_idx, :]
                        costheta_values = cos_theta_centers
                        
                        if np.sum(costheta_weights) > 0:
                            mean_value = np.average(costheta_values, weights=costheta_weights)
                        else:
                            mean_value = np.nan
                    else:
                        # Use regular theta distribution and theta centers
                        weights = traj_distributions_full[traj_idx][z_idx, :]
                        values = theta_centers
                        
                        if np.sum(weights) > 0:
                            mean_value = np.average(values, weights=weights)
                        else:
                            mean_value = np.nan
                    mean_values_z.append(mean_value)
                
                all_traj_mean_theta.append(mean_values_z)
                
                # Determine the y-axis range for the mean values (excluding NaN)
                valid_mean_values = [v for v in mean_values_z if not np.isnan(v)]
                if valid_mean_values:
                    max_mean_value = max(valid_mean_values)
                    min_mean_value = min(valid_mean_values)
                    y_range = max_mean_value - min_mean_value
                    
                    # Scale density to fit 10% above the max mean value
                    if np.max(density_counts) > 0:
                        density_scaled = (density_counts / np.max(density_counts)) * (max_mean_value * 1.1)
                    else:
                        density_scaled = density_counts
                    
                    # Plot background density as filled area (subtle grey)
                    axes[traj_idx].fill_between(density_z_centers, 0, density_scaled, 
                                               alpha=rho_z_alpha, color='lightgrey', zorder=0)
                
                # Plot mean theta/cos(theta) curve
                axes[traj_idx].plot(z_centers, mean_values_z, 'o-', markersize=3, zorder=2)
                axes[traj_idx].set_xlabel('z [Å]')
                if plot_cos_theta:
                    axes[traj_idx].set_ylabel('Mean cos(θ)')
                    axes[traj_idx].set_title(f'Traj {trajectory_indices[traj_idx]}: Mean cos(θ) vs Z')
                else:
                    axes[traj_idx].set_ylabel('Mean Theta [°]')
                    axes[traj_idx].set_title(f'Traj {trajectory_indices[traj_idx]}: Mean Theta vs Z')
                axes[traj_idx].grid(True, alpha=0.3)
                axes[traj_idx].set_xlim(z_sampling_min, z_sampling_max)
                
                # Create a twin axis for density label (no ticks)
                ax_density = axes[traj_idx].twinx()
                ax_density.set_ylabel('ρ(z)', color='grey', alpha=0.9)
                ax_density.set_yticks([])  # Remove ticks
                ax_density.spines['right'].set_color('grey')
                ax_density.spines['right'].set_alpha(0.7)
            
            plt.tight_layout()
            if plot_cos_theta:
                plt.savefig(system_plots_dir + '/' + name + '_mean_costheta_vs_z_by_traj.png', dpi=300)
            else:
                plt.savefig(system_plots_dir + '/' + name + '_mean_theta_vs_z_by_traj.png', dpi=300)
            plt.close()
            
            # Plot 5: Combined mean values vs z from all trajectories
            combined_mean_values = []
            for z_idx in range(len(z_centers)):
                if plot_cos_theta:
                    # Use combined cos(theta) distribution for proper <cos(theta)> calculation
                    combined_costheta_weights = combined_hist_full[z_idx, :]  # This is already the cos(theta) combined hist
                    if np.sum(combined_costheta_weights) > 0:
                        mean_value = np.average(cos_theta_centers, weights=combined_costheta_weights)
                    else:
                        mean_value = np.nan
                else:
                    # Use regular theta approach
                    combined_weights = combined_hist_full[z_idx, :]
                    if np.sum(combined_weights) > 0:
                        mean_value = np.average(theta_centers, weights=combined_weights)
                    else:
                        mean_value = np.nan
                combined_mean_values.append(mean_value)
            
            plt.figure(figsize=(8, 6))
            
            # Determine the y-axis range for the combined mean values (excluding NaN)
            valid_combined_mean_values = [v for v in combined_mean_values if not np.isnan(v)]
            if valid_combined_mean_values:
                max_combined_mean_value = max(valid_combined_mean_values)
                min_combined_mean_value = min(valid_combined_mean_values)
                
                # Scale density to fit 10% above the max mean value
                if np.max(density_counts) > 0:
                    density_scaled_combined = (density_counts / np.max(density_counts)) * (max_combined_mean_value * 1.1)
                else:
                    density_scaled_combined = density_counts
                
                # Plot background density as filled area (subtle grey)
                plt.fill_between(density_z_centers, 0, density_scaled_combined, 
                               alpha=rho_z_alpha, color='lightgrey', zorder=0)
            
            # Plot combined mean theta/cos(theta) curve
            plt.plot(z_centers, combined_mean_values, 'o-', markersize=4, linewidth=2, zorder=2)
            plt.xlabel('z [Å]')
            if plot_cos_theta:
                plt.ylabel('Mean cos(θ)')
                plt.title(f'{name} Mean Water Dipole θ vs z')
            else:
                plt.ylabel('Mean Theta [°]')
                plt.title(f'{name} Mean Water Dipole Angle vs $z$')
            plt.grid(True, alpha=0.3)
            plt.xlim(z_sampling_min, z_sampling_max)
            
            # Create a twin axis for density label (no ticks)
            ax_density_combined = plt.gca().twinx()
            ax_density_combined.set_ylabel(f'ρ(z)', color='grey', alpha=0.9)
            ax_density_combined.set_yticks([])  # Remove ticks
            ax_density_combined.spines['right'].set_color('grey')
            ax_density_combined.spines['right'].set_alpha(0.7)
            
            plt.tight_layout()
            if plot_cos_theta:
                plt.savefig(system_plots_dir + '/' + name + '_mean_costheta_vs_z_combined.png', dpi=300)
            else:
                plt.savefig(system_plots_dir + '/' + name + '_mean_theta_vs_z_combined.png', dpi=300)
            plt.close()
        
        # Plot 6: Contact layer theta histograms P(theta) for O_z_min to O_z_max range
        contact_layer_theta_data = []
        
        # Check if contact layer histogram data already exists
        contact_hist_data_path = f"{system_data_dir}/{name}_contact_layer_theta_histograms.json"
        
        if load_data and os.path.exists(contact_hist_data_path):
            with open(contact_hist_data_path, 'r') as f:
                contact_hist_data = json.load(f)
            print(f"Loaded existing contact layer theta histogram data from {contact_hist_data_path}")
        else:
            # Calculate contact layer theta histograms for each trajectory
            contact_hist_data = {}
            
            for traj_idx in range(num_trajectories):
                # Handle both integer and string keys
                traj_key = str(traj_idx) if str(traj_idx) in angular_data else traj_idx
                
                # Collect angles for contact layer only
                contact_angles = []
                for frame_data in angular_data[traj_key].values():
                    z_vals = np.array(frame_data['z_vals'])
                    angles = np.array(frame_data['angles'])
                    
                    # Filter for contact layer
                    contact_mask = (z_vals >= O_z_min) & (z_vals <= O_z_max)
                    contact_angles.extend(angles[contact_mask].tolist())
                
                # Create histogram
                hist_counts, hist_edges = np.histogram(contact_angles, bins=theta_bins, density=True)
                hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
                
                contact_hist_data[str(traj_idx)] = {
                    'hist_counts': hist_counts.tolist(),
                    'hist_centers': hist_centers.tolist(),
                    'hist_edges': hist_edges.tolist()
                }
            
            # Save contact layer histogram data
            if save_data:
                with open(contact_hist_data_path, 'w') as f:
                    json.dump(contact_hist_data, f, indent=2)
                print(f"Saved contact layer theta histogram data to {contact_hist_data_path}")
                
                # Also save individual CSV files for each trajectory
                for traj_idx in range(num_trajectories):
                    traj_data = contact_hist_data[str(traj_idx)]
                    csv_data = np.column_stack([traj_data['hist_centers'], traj_data['hist_counts']])
                    np.savetxt(f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_contact_theta_histogram.csv", 
                              csv_data, delimiter=',', header='theta_center,probability_density', comments='')
        
        # Plot individual trajectory histograms
        if plot_cos_theta:
            # Plot cos(theta) histograms for individual trajectories
            fig, axes = plt.subplots(1, num_trajectories, figsize=(5*num_trajectories, 5))
            if num_trajectories == 1:
                axes = [axes]
            
            for traj_idx in range(num_trajectories):
                traj_data = contact_hist_data[str(traj_idx)]
                hist_centers = np.array(traj_data['hist_centers'])
                hist_counts = np.array(traj_data['hist_counts'])
                
                # Create expanded theta data points weighted by histogram counts
                expanded_thetas = []
                for j in range(len(hist_centers)):
                    count = int(hist_counts[j] * 1000)  # Scale up for discrete sampling
                    expanded_thetas.extend([hist_centers[j]] * count)
                
                if len(expanded_thetas) > 0:
                    expanded_thetas = np.array(expanded_thetas)
                    expanded_cos_thetas = np.cos(np.radians(expanded_thetas))
                    
                    # Use evenly spaced cos(theta) bins for proper histogram
                    evenly_spaced_cos_bins = np.linspace(-1, 1, num_cos_bins + 1)
                    cos_hist_counts, cos_hist_edges = np.histogram(expanded_cos_thetas, bins=evenly_spaced_cos_bins, density=True)
                    cos_hist_centers = (cos_hist_edges[:-1] + cos_hist_edges[1:]) / 2
                    
                    axes[traj_idx].plot(cos_hist_centers, cos_hist_counts, 'o-', linewidth=2, markersize=4, color='lightcoral')
                
                axes[traj_idx].set_xlabel('cos(θ)')
                axes[traj_idx].set_ylabel('Probability Density')
                axes[traj_idx].set_title(f'Traj {trajectory_indices[traj_idx]}: P(cos(θ)) in Contact Layer\n({O_z_min:.1f}-{O_z_max:.1f} Å)')
                axes[traj_idx].grid(True, alpha=0.3)
                axes[traj_idx].set_xlim(-1, 1)
            
            plt.tight_layout()
            plt.savefig(system_plots_dir + '/' + name + '_contact_costheta_hist_by_traj.png', dpi=300)
        else:
            # Plot theta histograms for individual trajectories
            fig, axes = plt.subplots(1, num_trajectories, figsize=(5*num_trajectories, 5))
            if num_trajectories == 1:
                axes = [axes]
            
            for traj_idx in range(num_trajectories):
                traj_data = contact_hist_data[str(traj_idx)]
                hist_centers = np.array(traj_data['hist_centers'])
                hist_counts = np.array(traj_data['hist_counts'])
                
                axes[traj_idx].plot(hist_centers, hist_counts, 'o-', linewidth=2, markersize=4)
                axes[traj_idx].set_xlabel('Water Dipole Angle [°]')
                axes[traj_idx].set_ylabel('Probability Density')
                axes[traj_idx].set_title(f'Traj {trajectory_indices[traj_idx]}: P(θ) in Contact Layer\n({O_z_min:.1f}-{O_z_max:.1f} Å)')
                axes[traj_idx].grid(True, alpha=0.3)
                axes[traj_idx].set_xlim(0, 180)
            
            plt.tight_layout()
            plt.savefig(system_plots_dir + '/' + name + '_contact_theta_hist_by_traj.png', dpi=300)
        plt.close()
        
        # Plot combined histogram from all trajectories
        all_contact_angles = []
        for traj_idx in range(num_trajectories):
            # Handle both integer and string keys
            traj_key = str(traj_idx) if str(traj_idx) in angular_data else traj_idx
            
            for frame_data in angular_data[traj_key].values():
                z_vals = np.array(frame_data['z_vals'])
                angles = np.array(frame_data['angles'])
                
                # Filter for contact layer
                contact_mask = (z_vals >= O_z_min) & (z_vals <= O_z_max)
                all_contact_angles.extend(angles[contact_mask].tolist())
        
        # Create combined histogram
        combined_hist_counts, combined_hist_edges = np.histogram(all_contact_angles, bins=theta_bins, density=True)
        combined_hist_centers = (combined_hist_edges[:-1] + combined_hist_edges[1:]) / 2
        
        # Save combined histogram data
        if save_data:
            combined_csv_data = np.column_stack([combined_hist_centers, combined_hist_counts])
            np.savetxt(f"{system_data_dir}/{name}_combined_contact_theta_histogram.csv", 
                      combined_csv_data, delimiter=',', header='theta_center,probability_density', comments='')
        
        if plot_cos_theta:
            # Create expanded theta data points weighted by histogram counts for proper cos(theta) transformation
            expanded_combined_thetas = []
            for j in range(len(combined_hist_centers)):
                count = int(combined_hist_counts[j] * 1000)  # Scale up for discrete sampling
                expanded_combined_thetas.extend([combined_hist_centers[j]] * count)
            
            if len(expanded_combined_thetas) > 0:
                expanded_combined_thetas = np.array(expanded_combined_thetas)
                expanded_combined_cos_thetas = np.cos(np.radians(expanded_combined_thetas))
                
                # Use evenly spaced cos(theta) bins for proper histogram
                evenly_spaced_cos_bins = np.linspace(-1, 1, num_cos_bins + 1)
                combined_cos_hist_counts, combined_cos_hist_edges = np.histogram(expanded_combined_cos_thetas, bins=evenly_spaced_cos_bins, density=True)
                combined_cos_hist_centers = (combined_cos_hist_edges[:-1] + combined_cos_hist_edges[1:]) / 2
                
                # Create cos(theta) histogram plot
                plt.figure(figsize=(8, 6))
                plt.plot(combined_cos_hist_centers, combined_cos_hist_counts, 'o-', linewidth=2, markersize=4, color='lightcoral')
                plt.xlabel('cos(θ)')
                plt.ylabel('Probability Density')
                plt.title(f'{name}: Dipole Angle Distribution in Contact Layer: \n({O_z_min:.1f}-{O_z_max:.1f} Å)')
                plt.grid(True, alpha=0.3)
                plt.xlim(-1, 1)
                plt.tight_layout()
                plt.savefig(system_plots_dir + '/' + name + '_contact_costheta_hist_combined.png', dpi=300)
        else:
            # Create theta histogram plot
            plt.figure(figsize=(8, 6))
            plt.plot(combined_hist_centers, combined_hist_counts, 'o-', linewidth=2, markersize=4, color='skyblue')
            plt.xlabel('Water Dipole Angle [°]')
            plt.ylabel('Probability Density')
            plt.title(f'{name}: Dipole Angle Distribution in Contact Layer: \n({O_z_min:.1f}-{O_z_max:.1f} Å)')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 180)
            plt.tight_layout()
            plt.savefig(system_plots_dir + '/' + name + '_contact_theta_hist_combined.png', dpi=300)
        plt.close()




def plot_layer_angular_distributions(
                                     name,
                                     layer_bounds,
                                     contact_layer_angular_plots_dir='./',
                                     plot_cos_theta=False,
                                     theta_sampling_increment_degrees=1,
                                     ):
    """
    Plot angular distributions P(theta) or P(cos(theta)) for each layer using existing data.
    
    This function reads data created by plot_angular_distributions and creates layer-specific
    angular distribution plots by combining data from all z bins within each layer's bounds.
    
    Parameters:
    -----------
    name : str
        Name identifier for the system (must match the data created by plot_angular_distributions)
    layer_bounds : dict
        Dictionary mapping layer indices to [z_min, z_max] tuples
    contact_layer_angular_plots_dir : str, optional
        Base directory where angular data is stored (default: './')
    trajectory_indices : list, optional
        Trajectory indices (used for informational purposes only)
    plot_cos_theta : bool, optional
        Whether to plot cos(theta) instead of theta (default: False)
    theta_sampling_increment_degrees : int, optional
        Theta sampling increment in degrees (default: 1)
    """
    
    # Create system-specific subdirectories
    system_plots_dir = os.path.join(contact_layer_angular_plots_dir, f"{name}_angular_plots")
    system_data_dir = os.path.join(contact_layer_angular_plots_dir, f"{name}_angular_data")
    
    if not os.path.exists(system_plots_dir):
        os.makedirs(system_plots_dir)
    
    # Check if required data files exist
    data_file_path = f"{system_data_dir}/{name}_angular_data.json"
    z_edges_path = f"{system_data_dir}/{name}_z_bin_edges.csv"
    theta_edges_path = f"{system_data_dir}/{name}_theta_bin_edges.csv"
    costheta_edges_path = f"{system_data_dir}/{name}_costheta_bin_edges.csv"
    
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Angular data file not found: {data_file_path}. Please run plot_angular_distributions first.")
    if not os.path.exists(z_edges_path):
        raise FileNotFoundError(f"Z bin edges file not found: {z_edges_path}. Please run plot_angular_distributions first.")
    if not os.path.exists(theta_edges_path):
        raise FileNotFoundError(f"Theta bin edges file not found: {theta_edges_path}. Please run plot_angular_distributions first.")
    
    # Check for cos(theta) data if needed - make it optional
    costheta_edges_available = plot_cos_theta and os.path.exists(costheta_edges_path)
    
    # Load existing angular data
    with open(data_file_path, 'r') as f:
        angular_data = json.load(f)
    print(f"Loaded existing angular data from {data_file_path}")
    
    # Load bin edges
    z_edges = np.loadtxt(z_edges_path, delimiter=',')
    theta_edges = np.loadtxt(theta_edges_path, delimiter=',')
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    
    # Load cos(theta) data if available, otherwise create evenly spaced bins
    if plot_cos_theta:
        if costheta_edges_available:
            costheta_edges = np.loadtxt(costheta_edges_path, delimiter=',')
            costheta_centers = (costheta_edges[:-1] + costheta_edges[1:]) / 2
            print("Using existing cos(theta) bin edges")
        else:
            # Create evenly spaced cos(theta) bins from -1 to +1 (same number as theta bins)
            num_cos_bins = len(theta_bins) - 1
            costheta_edges = np.linspace(-1, 1, num_cos_bins + 1)
            costheta_centers = (costheta_edges[:-1] + costheta_edges[1:]) / 2
            print("Creating evenly spaced cos(theta) bins")
    
    # Define theta bins for histograms
    theta_bins = np.arange(0, 180 + theta_sampling_increment_degrees, theta_sampling_increment_degrees)
    
    # Get all angular data combined from all trajectories
    all_z_vals = []
    all_angles = []
    
    # Determine number of trajectories from the data
    traj_keys = [key for key in angular_data.keys() if str(key).isdigit() or isinstance(key, int)]
    num_trajectories = len(traj_keys)
    
    # Load cos(theta) distribution data if needed - simplified approach
    # Since the exact cos(theta) distribution files may not exist or have naming issues,
    # we'll compute cos(theta) distributions from the available theta data
    if plot_cos_theta:
        print("Computing cos(theta) distributions from theta data...")
    
    for traj_key in traj_keys:
        for frame_data in angular_data[traj_key].values():
            all_z_vals.extend(frame_data['z_vals'])
            all_angles.extend(frame_data['angles'])
    
    all_z_vals = np.array(all_z_vals)
    all_angles = np.array(all_angles)
    
    print(f"Loaded {len(all_z_vals)} data points from {num_trajectories} trajectories")
    
    # Filter out bulk layers (assume bulk layers have higher indices)
    interface_layers = {}
    for layer_idx, bounds in layer_bounds.items():
        # Skip layers that might be bulk (you can modify this criterion as needed)
        if layer_idx < max(layer_bounds.keys()):  # Exclude the highest layer index (assumed to be bulk)
            interface_layers[layer_idx] = bounds
    
    # Create subplot for each interface layer
    num_interface_layers = len(interface_layers)
    fig, axes = plt.subplots(1, num_interface_layers, figsize=(8*num_interface_layers, 8))
    if num_interface_layers == 1:
        axes = [axes]
    
    layer_colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon', 'lightsteelblue']
    
    for i, (layer_idx, (z_min, z_max)) in enumerate(sorted(interface_layers.items())):
        # Filter angles for this layer
        layer_mask = (all_z_vals >= z_min) & (all_z_vals <= z_max)
        layer_angles = all_angles[layer_mask]
        
        print(f"Layer {layer_idx} ({z_min:.1f}-{z_max:.1f} Å): {len(layer_angles)} data points")
        
        if len(layer_angles) > 0:
            if plot_cos_theta:
                # Transform theta angles to cos(theta) for this layer
                layer_cos_angles = np.cos(np.radians(layer_angles))
                
                # Create cos(theta) histogram for this specific layer using the appropriate bins
                cos_hist_counts, _ = np.histogram(layer_cos_angles, bins=costheta_edges, density=True)
                cos_hist_centers = costheta_centers
                
                # Plot cos(theta) histogram with filled area for better visibility
                color = layer_colors[i % len(layer_colors)]
                axes[i].fill_between(cos_hist_centers, 0, cos_hist_counts, 
                                   alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
                axes[i].plot(cos_hist_centers, cos_hist_counts, 'o-', 
                           color='darkblue', markersize=3, linewidth=1.5)
                
                axes[i].set_xlabel('cos(θ)')
                axes[i].set_ylabel('Probability Density')
                axes[i].set_title(f'Layer {layer_idx}: P(cos(θ))\n({z_min:.1f}-{z_max:.1f} Å)')
                axes[i].set_xlim(-1, 1)
                
                # Add statistics text
                mean_cos_theta = np.mean(layer_cos_angles)
                std_cos_theta = np.std(layer_cos_angles)
                axes[i].text(0.05, 0.95, r'$\langle$cos(θ)$\rangle$ ='+f' {mean_cos_theta:.3f}\nσ = {std_cos_theta:.3f}', 
                            transform=axes[i].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            else:
                # Create theta histogram
                theta_hist_counts, theta_hist_edges = np.histogram(layer_angles, bins=theta_bins, density=True)
                theta_hist_centers = (theta_hist_edges[:-1] + theta_hist_edges[1:]) / 2
                
                # Plot theta histogram with filled area for better visibility
                color = layer_colors[i % len(layer_colors)]
                axes[i].fill_between(theta_hist_centers, 0, theta_hist_counts, 
                                   alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
                axes[i].plot(theta_hist_centers, theta_hist_counts, 'o-', 
                           color='darkblue', markersize=3, linewidth=1.5)
                
                axes[i].set_xlabel('Water Dipole Angle [°]')
                axes[i].set_ylabel('Probability Density')
                axes[i].set_title(f'Contact Layer {layer_idx}: P(θ)\n({z_min:.1f}-{z_max:.1f} Å)')
                axes[i].set_xlim(0, 180)
                
                # Add statistics text
                mean_theta = np.mean(layer_angles)
                std_theta = np.std(layer_angles)
                axes[i].text(0.05, 0.95, r'$\langle$θ$\rangle$ ='+f' {mean_theta:.1f}°\nσ = {std_theta:.1f}°', 
                            transform=axes[i].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        else:
            # No data for this layer
            axes[i].text(0.5, 0.5, 'No data\navailable', 
                        transform=axes[i].transAxes, ha='center', va='center',
                        fontsize=14, alpha=0.5)
            axes[i].set_xlabel('cos(θ)' if plot_cos_theta else 'Water Dipole Angle [°]')
            axes[i].set_ylabel('Probability Density')
            axes[i].set_title(f'Layer {layer_idx}: P({"cos(θ)" if plot_cos_theta else "θ"})\n({z_min:.1f}-{z_max:.1f} Å)')
        
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    if plot_cos_theta:
        filename = f"{system_plots_dir}/{name}_layer_costheta_distributions.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved layer cos(θ) distributions to {filename}")
    else:
        filename = f"{system_plots_dir}/{name}_layer_theta_distributions.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved layer θ distributions to {filename}")
    
    plt.close()
    
    # Print summary statistics
    print("\nLayer Angular Distribution Summary:")
    print("=" * 50)
    for layer_idx, (z_min, z_max) in sorted(interface_layers.items()):
        layer_mask = (all_z_vals >= z_min) & (all_z_vals <= z_max)
        layer_angles = all_angles[layer_mask]
        
        if len(layer_angles) > 0:
            if plot_cos_theta:
                layer_cos_angles = np.cos(np.radians(layer_angles))
                mean_val = np.mean(layer_cos_angles)
                std_val = np.std(layer_cos_angles)
                print(f"Layer {layer_idx} ({z_min:.1f}-{z_max:.1f} Å): {len(layer_angles)} points")
                print(r"  $\langle$ cos(θ) $\rangle$ = "+f"{mean_val:.3f} ± {std_val:.3f}")
            else:
                mean_val = np.mean(layer_angles)
                std_val = np.std(layer_angles)
                print(f"Layer {layer_idx} ({z_min:.1f}-{z_max:.1f} Å): {len(layer_angles)} points")
                print(r"  $\langle$ θ $\rangle$ = "+f"{mean_val:.1f}° ± {std_val:.1f}°")
        else:
            print(f"Layer {layer_idx} ({z_min:.1f}-{z_max:.1f} Å): No data")
        print()


def plot_layer_OH_angular_distributions(
                                        name,
                                        layer_bounds,
                                        contact_layer_angular_plots_dir='./',
                                        plot_cos_theta=False,
                                        theta_sampling_increment_degrees=1,
                                        ):
    """
    Plot OH angular distributions P(theta) or P(cos(theta)) for each layer using existing OH data.
    
    This function reads data created by plot_OH_angular_distributions and creates layer-specific
    OH angular distribution plots by combining data from all z bins within each layer's bounds.
    Each water molecule contributes two data points (one for each OH bond).
    
    Parameters:
    -----------
    name : str
        Name identifier for the system (must match the data created by plot_OH_angular_distributions)
    layer_bounds : dict
        Dictionary mapping layer indices to [z_min, z_max] tuples
    contact_layer_angular_plots_dir : str, optional
        Base directory where OH angular data is stored (default: './')
    plot_cos_theta : bool, optional
        Whether to plot cos(theta) instead of theta (default: False)
    theta_sampling_increment_degrees : int, optional
        Theta sampling increment in degrees (default: 1)
    """
    
    # Create system-specific subdirectories for OH data
    system_plots_dir = os.path.join(contact_layer_angular_plots_dir, f"{name}_OH_angular_plots")
    system_data_dir = os.path.join(contact_layer_angular_plots_dir, f"{name}_OH_angular_data")
    
    if not os.path.exists(system_plots_dir):
        os.makedirs(system_plots_dir)
    
    # Check if required OH data files exist
    data_file_path = f"{system_data_dir}/{name}_OH_angular_data.json"
    z_edges_path = f"{system_data_dir}/{name}_OH_z_bin_edges.csv"
    theta_edges_path = f"{system_data_dir}/{name}_OH_theta_bin_edges.csv"
    costheta_edges_path = f"{system_data_dir}/{name}_OH_costheta_bin_edges.csv"
    
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"OH angular data file not found: {data_file_path}. Please run plot_OH_angular_distributions first.")
    if not os.path.exists(z_edges_path):
        raise FileNotFoundError(f"OH z bin edges file not found: {z_edges_path}. Please run plot_OH_angular_distributions first.")
    if not os.path.exists(theta_edges_path):
        raise FileNotFoundError(f"OH theta bin edges file not found: {theta_edges_path}. Please run plot_OH_angular_distributions first.")
    
    # Check for cos(theta) data if needed - make it optional
    costheta_edges_available = plot_cos_theta and os.path.exists(costheta_edges_path)
    
    # Load existing OH angular data
    with open(data_file_path, 'r') as f:
        angular_data = json.load(f)
    print(f"Loaded existing OH angular data from {data_file_path}")
    
    # Load bin edges
    z_edges = np.loadtxt(z_edges_path, delimiter=',')
    theta_edges = np.loadtxt(theta_edges_path, delimiter=',')
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    
    # Load cos(theta) data if available, otherwise create evenly spaced bins
    if plot_cos_theta:
        if costheta_edges_available:
            costheta_edges = np.loadtxt(costheta_edges_path, delimiter=',')
            costheta_centers = (costheta_edges[:-1] + costheta_edges[1:]) / 2
            print("Using existing cos(theta) bin edges")
        else:
            # Create evenly spaced cos(theta) bins from -1 to +1 (same number as theta bins)
            num_cos_bins = len(theta_edges) - 1
            costheta_edges = np.linspace(-1, 1, num_cos_bins + 1)
            costheta_centers = (costheta_edges[:-1] + costheta_edges[1:]) / 2
            print("Creating evenly spaced cos(theta) bins")
    
    # Define theta bins for histograms
    theta_bins = np.arange(0, 180 + theta_sampling_increment_degrees, theta_sampling_increment_degrees)
    
    # Get all OH angular data combined from all trajectories
    all_z_vals = []
    all_angles = []
    
    # Determine number of trajectories from the data
    traj_keys = [key for key in angular_data.keys() if str(key).isdigit() or isinstance(key, int)]
    num_trajectories = len(traj_keys)
    
    # Load cos(theta) distribution data if needed - simplified approach
    # Since the exact cos(theta) distribution files may not exist or have naming issues,
    # we'll compute cos(theta) distributions from the available theta data
    if plot_cos_theta:
        print("Computing cos(theta) distributions from OH theta data...")
    
    for traj_key in traj_keys:
        for frame_data in angular_data[traj_key].values():
            all_z_vals.extend(frame_data['z_vals'])
            all_angles.extend(frame_data['angles'])
    
    all_z_vals = np.array(all_z_vals)
    all_angles = np.array(all_angles)
    
    print(f"Loaded {len(all_z_vals)} OH bond data points from {num_trajectories} trajectories")
    print(f"Note: Each water molecule contributes 2 OH bond angles")
    
    # Filter out bulk layers (assume bulk layers have higher indices)
    interface_layers = {}
    for layer_idx, bounds in layer_bounds.items():
        # Skip layers that might be bulk (you can modify this criterion as needed)
        if layer_idx < max(layer_bounds.keys()):  # Exclude the highest layer index (assumed to be bulk)
            interface_layers[layer_idx] = bounds
    
    # Create subplot for each interface layer
    num_interface_layers = len(interface_layers)
    fig, axes = plt.subplots(1, num_interface_layers, figsize=(8*num_interface_layers, 8))
    if num_interface_layers == 1:
        axes = [axes]
    
    layer_colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon', 'lightsteelblue']
    
    for i, (layer_idx, (z_min, z_max)) in enumerate(sorted(interface_layers.items())):
        # Filter OH angles for this layer
        layer_mask = (all_z_vals >= z_min) & (all_z_vals <= z_max)
        layer_angles = all_angles[layer_mask]
        
        print(f"Layer {layer_idx} ({z_min:.1f}-{z_max:.1f} Å): {len(layer_angles)} OH bond data points")
        
        if len(layer_angles) > 0:
            if plot_cos_theta:
                # Transform theta angles to cos(theta) for this layer
                layer_cos_angles = np.cos(np.radians(layer_angles))
                
                # Create cos(theta) histogram for this specific layer using the appropriate bins
                cos_hist_counts, _ = np.histogram(layer_cos_angles, bins=costheta_edges, density=True)
                cos_hist_centers = costheta_centers
                
                # Plot cos(theta) histogram with filled area for better visibility
                color = layer_colors[i % len(layer_colors)]
                axes[i].fill_between(cos_hist_centers, 0, cos_hist_counts, 
                                   alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
                axes[i].plot(cos_hist_centers, cos_hist_counts, 'o-', 
                           color='darkred', markersize=3, linewidth=1.5)
                
                axes[i].set_xlabel('cos(θ)')
                axes[i].set_ylabel('Probability Density')
                axes[i].set_title(f'Layer {layer_idx}: OH P(cos(θ))\n({z_min:.1f}-{z_max:.1f} Å)')
                axes[i].set_xlim(-1, 1)
                
                # Add statistics text
                mean_cos_theta = np.mean(layer_cos_angles)
                std_cos_theta = np.std(layer_cos_angles)
                axes[i].text(0.05, 0.95, r'$\langle$cos(θ)$\rangle$ ='+f' {mean_cos_theta:.3f}\nσ = {std_cos_theta:.3f}', 
                            transform=axes[i].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            else:
                # Create theta histogram
                theta_hist_counts, theta_hist_edges = np.histogram(layer_angles, bins=theta_bins, density=True)
                theta_hist_centers = (theta_hist_edges[:-1] + theta_hist_edges[1:]) / 2
                
                # Plot theta histogram with filled area for better visibility
                color = layer_colors[i % len(layer_colors)]
                axes[i].fill_between(theta_hist_centers, 0, theta_hist_counts, 
                                   alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
                axes[i].plot(theta_hist_centers, theta_hist_counts, 'o-', 
                           color='darkred', markersize=3, linewidth=1.5)
                
                axes[i].set_xlabel('Water OH Angle [°]')
                axes[i].set_ylabel('Probability Density')
                axes[i].set_title(f'Layer {layer_idx}: OH P(θ)\n({z_min:.1f}-{z_max:.1f} Å)')
                axes[i].set_xlim(0, 180)
                
                # Add statistics text
                mean_theta = np.mean(layer_angles)
                std_theta = np.std(layer_angles)
                axes[i].text(0.05, 0.95, r'$\langle$θ$\rangle$ ='+f' {mean_theta:.1f}°\nσ = {std_theta:.1f}°', 
                            transform=axes[i].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        else:
            # No data for this layer
            axes[i].text(0.5, 0.5, 'No data\navailable', 
                        transform=axes[i].transAxes, ha='center', va='center',
                        fontsize=14, alpha=0.5)
            axes[i].set_xlabel('cos(θ)' if plot_cos_theta else 'Water OH Angle [°]')
            axes[i].set_ylabel('Probability Density')
            axes[i].set_title(f'Contact Layer {layer_idx}: OH P({"cos(θ)" if plot_cos_theta else "θ"})\n({z_min:.1f}-{z_max:.1f} Å)')
        
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    if plot_cos_theta:
        filename = f"{system_plots_dir}/{name}_layer_OH_costheta_distributions.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved layer OH cos(θ) distributions to {filename}")
    else:
        filename = f"{system_plots_dir}/{name}_layer_OH_theta_distributions.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved layer OH θ distributions to {filename}")
    
    plt.close()
    
    # Print summary statistics
    print("\nLayer OH Angular Distribution Summary:")
    print("=" * 55)
    for layer_idx, (z_min, z_max) in sorted(interface_layers.items()):
        layer_mask = (all_z_vals >= z_min) & (all_z_vals <= z_max)
        layer_angles = all_angles[layer_mask]
        
        if len(layer_angles) > 0:
            if plot_cos_theta:
                layer_cos_angles = np.cos(np.radians(layer_angles))
                mean_val = np.mean(layer_cos_angles)
                std_val = np.std(layer_cos_angles)
                print(f"Layer {layer_idx} ({z_min:.1f}-{z_max:.1f} Å): {len(layer_angles)} OH bond points")
                print(r"  $\langle$ cos(θ) $\rangle$ = "+f"{mean_val:.3f} ± {std_val:.3f}")
            else:
                mean_val = np.mean(layer_angles)
                std_val = np.std(layer_angles)
                print(f"Layer {layer_idx} ({z_min:.1f}-{z_max:.1f} Å): {len(layer_angles)} OH bond points")
                print(r"  $\langle$ θ $\rangle$ = "+f"{mean_val:.1f}° ± {std_val:.1f}°")
        else:
            print(f"Layer {layer_idx} ({z_min:.1f}-{z_max:.1f} Å): No data")
        print()


def plot_OH_angular_distributions(
                               name,
                               substrate,
                               trajectories,
                               O_z_min,
                               O_z_max,
                               contact_layer_angle_vs_z_plots_dir='./',
                               contact_layer_angular_plots_dir='./',
                               sampling_interval=20,
                               z_sampling_min = 0,
                               z_sampling_max = 20,
                               z_sampling_increment = 0.05,
                               theta_sampling_increment_degrees = 1,
                               num_cos_bins=50,
                               save_data = True,
                               load_data = True,
                               plot_mean_theta_vs_z = True,
                               trajectory_indices = None,
                               plot_cos_theta=False,
                               rho_z_alpha=0.7,
                               ):
    """
    Plot OH angular distributions P(theta) or P(cos(theta)) for water molecules.
    
    This function analyzes OH bond angles instead of water dipole angles, providing
    insights into the orientation of individual OH bonds relative to the surface normal.
    Each water molecule contributes two data points (one for each OH bond).
    
    Parameters are identical to plot_angular_distributions but uses OH bond angles
    instead of dipole moment angles.
    """

    # Set default trajectory indices if not provided
    if trajectory_indices is None:
        trajectory_indices = [i+1 for i in range(len(trajectories))]
    elif len(trajectory_indices) != len(trajectories):
        raise ValueError(f"Length of trajectory_indices ({len(trajectory_indices)}) must match number of trajectories ({len(trajectories)})")

    # Create system-specific subdirectories with OH prefix
    system_plots_dir = os.path.join(contact_layer_angle_vs_z_plots_dir, f"{name}_OH_angular_plots")
    system_data_dir = os.path.join(contact_layer_angular_plots_dir, f"{name}_OH_angular_data")
    
    if not os.path.exists(system_plots_dir):
        os.makedirs(system_plots_dir)

    if not os.path.exists(system_data_dir):
        os.makedirs(system_data_dir)

    # Define binning
    z_bins = np.arange(z_sampling_min, z_sampling_max + z_sampling_increment, z_sampling_increment)
    theta_bins = np.arange(0, 180 + theta_sampling_increment_degrees, theta_sampling_increment_degrees)
    
    # Initialize angular_data structure
    angular_data = {}
    
    # Data file paths with OH prefix
    data_file_path = f"{system_data_dir}/{name}_OH_angular_data.json"
    
    # Load existing data if requested and available
    if load_data and os.path.exists(data_file_path):
        with open(data_file_path, 'r') as f:
            angular_data = json.load(f)
        print(f"Loaded existing OH angular data from {data_file_path}")
    else:
        # Collect OH angular data for each trajectory and frame
        for traj_idx, traj in enumerate(trajectories):
            angular_data[traj_idx] = {}
            for frame_idx, frame in enumerate(tqdm(traj[::sampling_interval], desc=f"Processing traj {trajectory_indices[traj_idx]} frames for OH Angular analysis {name}")):
                # Use OH angle function instead of dipole angle function
                data = interface_analysis_tools.get_interfacial_z_vs_OH_angles(frame, substrate)
                _, z, angles = data
                
                angular_data[traj_idx][frame_idx * sampling_interval] = {
                    'angles': angles.tolist() if hasattr(angles, 'tolist') else list(angles),
                    'z_vals': z.tolist() if hasattr(z, 'tolist') else list(z)
                }
        
        # Save data if requested
        if save_data:
            with open(data_file_path, 'w') as f:
                json.dump(angular_data, f, indent=2)
            print(f"Saved OH angular data to {data_file_path}")
    
    # Create binned distributions for each trajectory
    num_trajectories = len(trajectories)
    traj_distributions_full = []
    traj_distributions_contact = []
    
    # Initialize cos(theta) distributions if needed
    if plot_cos_theta:
        # Create evenly spaced cos(theta) bin edges from -1 to +1
        cos_theta_bins = np.linspace(-1, 1, num_cos_bins + 1)
        cos_theta_centers = (cos_theta_bins[:-1] + cos_theta_bins[1:]) / 2
        
        traj_distributions_full_costheta = []
        traj_distributions_contact_costheta = []
    
    for traj_idx in range(num_trajectories):
        # Collect all z and angle data for this trajectory
        all_z_vals = []
        all_angles = []

        # Handle both integer and string keys (depending on whether data was loaded from JSON or not)
        traj_key = str(traj_idx) if str(traj_idx) in angular_data else traj_idx
        
        for frame_data in angular_data[traj_key].values():
            all_z_vals.extend(frame_data['z_vals'])
            all_angles.extend(frame_data['angles'])
        
        all_z_vals = np.array(all_z_vals)
        all_angles = np.array(all_angles)
        
        # Create 2D histogram for full z range (use density=True for probability density)
        hist_full, z_edges, theta_edges = np.histogram2d(all_z_vals, all_angles, bins=[z_bins, theta_bins], density=True)
        traj_distributions_full.append(hist_full)
        
        # Create 2D histogram for contact layer only (use density=True for probability density)
        contact_mask = (all_z_vals >= O_z_min) & (all_z_vals <= O_z_max)
        hist_contact, _, _ = np.histogram2d(all_z_vals[contact_mask], all_angles[contact_mask], bins=[z_bins, theta_bins], density=True)
        traj_distributions_contact.append(hist_contact)
        
        # Create cos(theta) histograms if requested
        if plot_cos_theta:
            # Convert angles to cos(theta) values
            all_cos_theta_vals = np.cos(np.radians(all_angles))
            
            # Create 2D histogram for cos(theta) - full range
            costheta_hist_full, _, cos_theta_edges = np.histogram2d(all_z_vals, all_cos_theta_vals, 
                                                                  bins=[z_bins, cos_theta_bins], density=True)
            traj_distributions_full_costheta.append(costheta_hist_full)
            
            # Create 2D histogram for cos(theta) - contact layer
            costheta_hist_contact, _, _ = np.histogram2d(all_z_vals[contact_mask], all_cos_theta_vals[contact_mask], 
                                                       bins=[z_bins, cos_theta_bins], density=True)
            traj_distributions_contact_costheta.append(costheta_hist_contact)
        
        # Save distribution data for this trajectory if requested
        if save_data:
            # Save full range distribution
            np.savetxt(f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_OH_full_distribution.csv", 
                      hist_full, delimiter=',')
            
            # Save contact layer distribution
            np.savetxt(f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_OH_contact_distribution.csv", 
                      hist_contact, delimiter=',')
            
            # Save cos(theta) distributions if requested
            if plot_cos_theta:
                np.savetxt(f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_OH_full_costheta_distribution.csv", 
                          costheta_hist_full, delimiter=',')
                np.savetxt(f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_OH_contact_costheta_distribution.csv", 
                          costheta_hist_contact, delimiter=',')
            
            # Save bin edges
            np.savetxt(f"{system_data_dir}/{name}_OH_z_bin_edges.csv", 
                      z_edges, delimiter=',')
            np.savetxt(f"{system_data_dir}/{name}_OH_theta_bin_edges.csv", 
                      theta_edges, delimiter=',')
            
            # Save cos(theta) bin edges if requested
            if plot_cos_theta:
                np.savetxt(f"{system_data_dir}/{name}_OH_costheta_bin_edges.csv", 
                          cos_theta_edges, delimiter=',')
    
    # Load cos(theta) distributions if requested and available
    if plot_cos_theta:
        print('Loading or creating OH cos(theta) distributions with evenly spaced cos(theta) bins')
        
        # Try to load existing cos(theta) distributions
        costheta_data_exists = True
        for traj_idx in range(num_trajectories):
            full_costheta_path = f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_OH_full_costheta_distribution.csv"
            contact_costheta_path = f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_OH_contact_costheta_distribution.csv"
            costheta_edges_path = f"{system_data_dir}/{name}_OH_costheta_bin_edges.csv"
            
            if not (os.path.exists(full_costheta_path) and os.path.exists(contact_costheta_path) and os.path.exists(costheta_edges_path)):
                costheta_data_exists = False
                break
        
        if load_data and costheta_data_exists:
            print("Loading existing OH cos(theta) distribution data...")
            # Load cos(theta) bin edges
            cos_theta_edges = np.loadtxt(f"{system_data_dir}/{name}_OH_costheta_bin_edges.csv", delimiter=',')
            cos_theta_centers = (cos_theta_edges[:-1] + cos_theta_edges[1:]) / 2
            
            # Load cos(theta) distributions for each trajectory
            for traj_idx in range(num_trajectories):
                full_costheta_hist = np.loadtxt(f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_OH_full_costheta_distribution.csv", delimiter=',')
                contact_costheta_hist = np.loadtxt(f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_OH_contact_costheta_distribution.csv", delimiter=',')
                
                traj_distributions_full_costheta.append(full_costheta_hist)
                traj_distributions_contact_costheta.append(contact_costheta_hist)
            
            print(f"Loaded OH cos(theta) distributions for {num_trajectories} trajectories")
        else:
            print("OH cos(theta) distribution data not found or load_data=False, but distributions should have been created above.")
    
    # Create mesh for plotting
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    Z, Theta = np.meshgrid(z_centers, theta_centers, indexing='ij')
    
    # Ensure cos(theta) centers are defined when needed
    if plot_cos_theta and 'cos_theta_centers' not in locals():
        # If cos(theta) data wasn't loaded, define centers from the bins created above
        if 'cos_theta_edges' in locals():
            cos_theta_centers = (cos_theta_edges[:-1] + cos_theta_edges[1:]) / 2
        else:
            # Fallback: create cos(theta) bins if they don't exist
            num_cos_bins = len(theta_bins) - 1
            cos_theta_bins = np.linspace(-1, 1, num_cos_bins + 1)
            cos_theta_centers = (cos_theta_bins[:-1] + cos_theta_bins[1:]) / 2

    # Plot 1: 2 x Num_trajectories subplot of z,theta histograms
    fig, axes = plt.subplots(2, num_trajectories, figsize=(5*num_trajectories, 8))

    if num_trajectories == 1:
        axes = axes.reshape(-1, 1)
    
    # Determine color scale limits for consistent colorbar across all subplots
    if plot_cos_theta:
        all_distributions_full = traj_distributions_full_costheta
        all_distributions_contact = traj_distributions_contact_costheta
        # Use cos_theta_edges if available, otherwise create from centers
        if 'cos_theta_edges' in locals():
            y_edges = cos_theta_edges
        else:
            # Reconstruct edges from centers if needed
            y_edges = np.linspace(-1, 1, len(cos_theta_centers) + 1)
        y_label = r'$\cos(\theta)$'
        colorbar_label = 'Probability Density'
    else:
        all_distributions_full = traj_distributions_full
        all_distributions_contact = traj_distributions_contact
        y_edges = theta_edges
        y_label = 'OH Angle [°]'
        colorbar_label = 'Probability Density'
    
    # Find global min and max for consistent color scale
    vmin_full = min([np.min(dist) for dist in all_distributions_full])
    vmax_full = max([np.max(dist) for dist in all_distributions_full])
    vmin_contact = min([np.min(dist) for dist in all_distributions_contact])
    vmax_contact = max([np.max(dist) for dist in all_distributions_contact])
    
    for traj_idx in range(num_trajectories):
        plot_traj_distributions_full = all_distributions_full[traj_idx]
        plot_traj_distributions_contact = all_distributions_contact[traj_idx]

        # Top row: full z range
        im1 = axes[0, traj_idx].pcolormesh(z_edges, y_edges, plot_traj_distributions_full.T, 
                                         cmap='Blues', vmin=vmin_full, vmax=vmax_full)
        axes[0, traj_idx].set_title(f'Traj {trajectory_indices[traj_idx]}')
        axes[0, traj_idx].set_xlabel('z [Å]')
        axes[0, traj_idx].set_ylabel(y_label)
        
        # Bottom row: contact layer only (use same data but limit x-axis view)
        im2 = axes[1, traj_idx].pcolormesh(z_edges, y_edges, plot_traj_distributions_full.T, 
                                         cmap='Blues', vmin=vmin_full, vmax=vmax_full)
        axes[1, traj_idx].set_xlabel('z [Å]')
        axes[1, traj_idx].set_ylabel(y_label)
        axes[1, traj_idx].set_xlim(O_z_min, O_z_max)
    
    # Add row titles
    fig.text(0.02, 0.75, 'OH Distribution for Entire Slab', rotation=90, 
            fontsize=14, ha='center', va='center', weight='bold')
    fig.text(0.02, 0.25, 'OH Distribution for the Contact Layer', rotation=90, 
            fontsize=14, ha='center', va='center', weight='bold')
    
    # Add single colorbar for the entire figure
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label=colorbar_label)

    fig.suptitle('OH Angle vs z distributions', fontsize=16, weight='bold')
    plt.tight_layout()

    if plot_cos_theta:
        plt.savefig(system_plots_dir + '/' + name + '_OH_costheta_vs_z_by_traj.png', dpi=300)
    else:
        plt.savefig(system_plots_dir + '/' + name + '_OH_angle_vs_z_by_traj.png', dpi=300)
    plt.close()
    
    # Plot 2: Combined data from all trajectories - Full range
    if plot_cos_theta:
        combined_hist_full = np.sum(traj_distributions_full_costheta, axis=0)
        # Use cos_theta_edges if available, otherwise create from centers
        if 'cos_theta_edges' in locals():
            y_edges_combined = cos_theta_edges
        else:
            y_edges_combined = np.linspace(-1, 1, len(cos_theta_centers) + 1)
        y_label_combined = r'$\cos(\theta)$'
        title_combined = f'{name} Water OH cos(θ) Distributions'
        filename_combined = system_plots_dir + '/' + name + '_OH_costheta_vs_z_combined_full.png'
    else:
        combined_hist_full = np.sum(traj_distributions_full, axis=0)
        y_edges_combined = theta_edges
        y_label_combined = 'Water OH Angle [°]'
        title_combined = f'{name} Water OH Angle Distributions'
        filename_combined = system_plots_dir + '/' + name + '_OH_angle_vs_z_combined_full.png'
        
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(z_edges, y_edges_combined, combined_hist_full.T, cmap='Blues')
    plt.colorbar(label='Probability Density')
    plt.xlabel('z [Å]')
    plt.ylabel(y_label_combined)
    plt.title(title_combined)
    plt.tight_layout()
    plt.savefig(filename_combined, dpi=300)
    plt.close()
    
    # Plot 3: Combined data from all trajectories - Contact layer
    if plot_cos_theta:
        combined_hist_contact = np.sum(traj_distributions_contact_costheta, axis=0)
        # Use cos_theta_edges if available, otherwise create from centers
        if 'cos_theta_edges' in locals():
            y_edges_contact = cos_theta_edges
        else:
            y_edges_contact = np.linspace(-1, 1, len(cos_theta_centers) + 1)
        y_label_contact = 'cos(θ)'
        title_contact = f'{name} Contact Layer Water OH cos(θ) Distribution'
        filename_contact = system_plots_dir + '/' + name + '_OH_costheta_vs_z_combined_contact.png'
    else:
        combined_hist_contact = np.sum(traj_distributions_contact, axis=0)
        y_edges_contact = theta_edges
        y_label_contact = 'Water OH Angle [°]'
        title_contact = f'{name} Water OH Angle Distribution (Contact Layer)'
        filename_contact = system_plots_dir + '/' + name + '_OH_angle_vs_z_combined_contact.png'
        
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(z_edges, y_edges_contact, combined_hist_contact.T, cmap='Blues')
    plt.colorbar(label='Probability Density')
    plt.xlabel('z [Å]')
    plt.ylabel(y_label_contact)
    plt.title(title_contact)
    plt.xlim(O_z_min, O_z_max)
    plt.tight_layout()
    plt.savefig(filename_contact, dpi=300)
    plt.close()
    
    # Plot 4: Mean theta vs z plots if requested
    if plot_mean_theta_vs_z:
        # Calculate water density rho(z) for background plotting
        print("Calculating water density rho(z) for background plotting...")
        
        # Collect all z values from all trajectories and frames for density calculation
        all_z_vals_for_density = []
        for traj_idx in range(num_trajectories):
            # Handle both integer and string keys
            traj_key = str(traj_idx) if str(traj_idx) in angular_data else traj_idx
            
            for frame_data in angular_data[traj_key].values():
                all_z_vals_for_density.extend(frame_data['z_vals'])
        
        all_z_vals_for_density = np.array(all_z_vals_for_density)
        
        # Create density histogram using the same z bins
        density_counts, _ = np.histogram(all_z_vals_for_density, bins=z_bins, density=True)
        density_z_centers = z_centers
        
        # Calculate mean theta for each z bin for each trajectory
        fig, axes = plt.subplots(1, num_trajectories, figsize=(5*num_trajectories, 5))
        if num_trajectories == 1:
            axes = [axes]
        
        all_traj_mean_theta = []
        
        for traj_idx in range(num_trajectories):
            # Calculate weighted mean for each z bin
            mean_values_z = []
            for z_idx in range(len(z_centers)):
                if plot_cos_theta:
                    # Use cos(theta) distribution directly for proper <cos(theta)> calculation
                    costheta_weights = traj_distributions_full_costheta[traj_idx][z_idx, :]
                    costheta_values = cos_theta_centers
                    
                    if np.sum(costheta_weights) > 0:
                        mean_value = np.average(costheta_values, weights=costheta_weights)
                    else:
                        mean_value = np.nan
                else:
                    # Use regular theta distribution and theta centers
                    weights = traj_distributions_full[traj_idx][z_idx, :]
                    values = theta_centers
                    
                    if np.sum(weights) > 0:
                        mean_value = np.average(values, weights=weights)
                    else:
                        mean_value = np.nan
                mean_values_z.append(mean_value)
            
            all_traj_mean_theta.append(mean_values_z)
            
            # Determine the y-axis range for the mean values (excluding NaN)
            valid_mean_values = [v for v in mean_values_z if not np.isnan(v)]
            if valid_mean_values:
                max_mean_value = max(valid_mean_values)
                min_mean_value = min(valid_mean_values)
                y_range = max_mean_value - min_mean_value
                
                # Scale density to fit 10% above the max mean value
                if np.max(density_counts) > 0:
                    density_scaled = (density_counts / np.max(density_counts)) * (max_mean_value * 1.1)
                else:
                    density_scaled = density_counts
                
                # Plot background density as filled area (subtle grey)
                axes[traj_idx].fill_between(density_z_centers, 0, density_scaled, 
                                           alpha=rho_z_alpha, color='lightgrey', zorder=0)
            
            # Plot mean theta/cos(theta) curve
            axes[traj_idx].plot(z_centers, mean_values_z, 'o-', markersize=3, zorder=2)
            axes[traj_idx].set_xlabel('z [Å]')
            if plot_cos_theta:
                axes[traj_idx].set_ylabel('Mean cos(θ)')
                axes[traj_idx].set_title(f'Traj {trajectory_indices[traj_idx]}: Mean OH cos(θ) vs Z')
            else:
                axes[traj_idx].set_ylabel('Mean OH Theta [°]')
                axes[traj_idx].set_title(f'Traj {trajectory_indices[traj_idx]}: Mean OH Theta vs Z')
            axes[traj_idx].grid(True, alpha=0.3)
            axes[traj_idx].set_xlim(z_sampling_min, z_sampling_max)
            
            # Create a twin axis for density label (no ticks)
            ax_density = axes[traj_idx].twinx()
            ax_density.set_ylabel('ρ(z)', color='grey', alpha=0.7)
            ax_density.set_yticks([])  # Remove ticks
            ax_density.spines['right'].set_color('grey')
            ax_density.spines['right'].set_alpha(0.7)
        
        plt.tight_layout()
        if plot_cos_theta:
            plt.savefig(system_plots_dir + '/' + name + '_mean_OH_costheta_vs_z_by_traj.png', dpi=300)
        else:
            plt.savefig(system_plots_dir + '/' + name + '_mean_OH_theta_vs_z_by_traj.png', dpi=300)
        plt.close()
        
        # Plot 5: Combined mean values vs z from all trajectories
        combined_mean_values = []
        for z_idx in range(len(z_centers)):
            if plot_cos_theta:
                # Use combined cos(theta) distribution for proper <cos(theta)> calculation
                combined_costheta_weights = combined_hist_full[z_idx, :]  # This is already the cos(theta) combined hist
                if np.sum(combined_costheta_weights) > 0:
                    mean_value = np.average(cos_theta_centers, weights=combined_costheta_weights)
                else:
                    mean_value = np.nan
            else:
                # Use regular theta approach
                combined_weights = combined_hist_full[z_idx, :]
                if np.sum(combined_weights) > 0:
                    mean_value = np.average(theta_centers, weights=combined_weights)
                else:
                    mean_value = np.nan
            combined_mean_values.append(mean_value)
        
        plt.figure(figsize=(8, 6))
        
        # Determine the y-axis range for the combined mean values (excluding NaN)
        valid_combined_mean_values = [v for v in combined_mean_values if not np.isnan(v)]
        if valid_combined_mean_values:
            max_combined_mean_value = max(valid_combined_mean_values)
            min_combined_mean_value = min(valid_combined_mean_values)
            
            # Scale density to fit 10% above the max mean value
            if np.max(density_counts) > 0:
                density_scaled_combined = (density_counts / np.max(density_counts)) * (max_combined_mean_value * 1.1)
            else:
                density_scaled_combined = density_counts
            
            # Plot background density as filled area (subtle grey)
            plt.fill_between(density_z_centers, 0, density_scaled_combined, 
                           alpha=rho_z_alpha, color='lightgrey', zorder=0)
        
        # Plot combined mean theta/cos(theta) curve
        plt.plot(z_centers, combined_mean_values, 'o-', markersize=4, linewidth=2, zorder=2)
        plt.xlabel('z [Å]')
        if plot_cos_theta:
            plt.ylabel('Mean cos(θ)')
            plt.title(f'{name} Mean Water OH cos(θ) vs z')
        else:
            plt.ylabel('Mean OH Theta [°]')
            plt.title(f'{name} Mean Water OH Angle vs z')
        plt.grid(True, alpha=0.3)
        plt.xlim(z_sampling_min, z_sampling_max)
        
        # Create a twin axis for density label (no ticks)
        ax_density_combined = plt.gca().twinx()
        ax_density_combined.set_ylabel(f'ρ(z)', color='grey', alpha=0.7)
        ax_density_combined.set_yticks([])  # Remove ticks
        ax_density_combined.spines['right'].set_color('grey')
        ax_density_combined.spines['right'].set_alpha(0.7)
        
        plt.tight_layout()
        if plot_cos_theta:
            plt.savefig(system_plots_dir + '/' + name + '_mean_OH_costheta_vs_z_combined.png', dpi=300)
        else:
            plt.savefig(system_plots_dir + '/' + name + '_mean_OH_theta_vs_z_combined.png', dpi=300)
        plt.close()
    
    # Plot 6: Contact layer OH theta histograms P(theta) for O_z_min to O_z_max range
    contact_layer_theta_data = []
    
    # Check if contact layer histogram data already exists
    contact_hist_data_path = f"{system_data_dir}/{name}_OH_contact_layer_theta_histograms.json"
    
    if load_data and os.path.exists(contact_hist_data_path):
        with open(contact_hist_data_path, 'r') as f:
            contact_hist_data = json.load(f)
        print(f"Loaded existing OH contact layer theta histogram data from {contact_hist_data_path}")
    else:
        # Calculate contact layer theta histograms for each trajectory
        contact_hist_data = {}
        
        for traj_idx in range(num_trajectories):
            # Handle both integer and string keys
            traj_key = str(traj_idx) if str(traj_idx) in angular_data else traj_idx
            
            # Collect angles for contact layer only
            contact_angles = []
            for frame_data in angular_data[traj_key].values():
                z_vals = np.array(frame_data['z_vals'])
                angles = np.array(frame_data['angles'])
                
                # Filter for contact layer
                contact_mask = (z_vals >= O_z_min) & (z_vals <= O_z_max)
                contact_angles.extend(angles[contact_mask].tolist())
            
            # Create histogram
            hist_counts, hist_edges = np.histogram(contact_angles, bins=theta_bins, density=True)
            hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
            
            contact_hist_data[str(traj_idx)] = {
                'hist_counts': hist_counts.tolist(),
                'hist_centers': hist_centers.tolist(),
                'hist_edges': hist_edges.tolist()
            }
        
        # Save contact layer histogram data
        if save_data:
            with open(contact_hist_data_path, 'w') as f:
                json.dump(contact_hist_data, f, indent=2)
            print(f"Saved OH contact layer theta histogram data to {contact_hist_data_path}")
            
            # Also save individual CSV files for each trajectory
            for traj_idx in range(num_trajectories):
                traj_data = contact_hist_data[str(traj_idx)]
                csv_data = np.column_stack([traj_data['hist_centers'], traj_data['hist_counts']])
                np.savetxt(f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_OH_contact_theta_histogram.csv", 
                          csv_data, delimiter=',', header='theta_center,probability_density', comments='')
    
    # Plot individual trajectory histograms
    if plot_cos_theta:
        # Plot cos(theta) histograms for individual trajectories
        fig, axes = plt.subplots(1, num_trajectories, figsize=(5*num_trajectories, 5))
        if num_trajectories == 1:
            axes = [axes]
        
        for traj_idx in range(num_trajectories):
            traj_data = contact_hist_data[str(traj_idx)]
            hist_centers = np.array(traj_data['hist_centers'])
            hist_counts = np.array(traj_data['hist_counts'])
            
            # Create expanded theta data points weighted by histogram counts
            expanded_thetas = []
            for j in range(len(hist_centers)):
                count = int(hist_counts[j] * 1000)  # Scale up for discrete sampling
                expanded_thetas.extend([hist_centers[j]] * count)
            
            if len(expanded_thetas) > 0:
                expanded_thetas = np.array(expanded_thetas)
                expanded_cos_thetas = np.cos(np.radians(expanded_thetas))
                
                # Use evenly spaced cos(theta) bins for proper histogram
                evenly_spaced_cos_bins = np.linspace(-1, 1, num_cos_bins + 1)
                cos_hist_counts, cos_hist_edges = np.histogram(expanded_cos_thetas, bins=evenly_spaced_cos_bins, density=True)
                cos_hist_centers = (cos_hist_edges[:-1] + cos_hist_edges[1:]) / 2
                
                axes[traj_idx].plot(cos_hist_centers, cos_hist_counts, 'o-', linewidth=2, markersize=4, color='lightcoral')
            
            axes[traj_idx].set_xlabel('cos(θ)')
            axes[traj_idx].set_ylabel('Probability Density')
            axes[traj_idx].set_title(f'Traj {trajectory_indices[traj_idx]}: P(cos(θ)) OH in Contact Layer\n({O_z_min:.1f}-{O_z_max:.1f} Å)')
            axes[traj_idx].grid(True, alpha=0.3)
            axes[traj_idx].set_xlim(-1, 1)
        
        plt.tight_layout()
        plt.savefig(system_plots_dir + '/' + name + '_OH_contact_costheta_hist_by_traj.png', dpi=300)
    else:
        # Plot theta histograms for individual trajectories
        fig, axes = plt.subplots(1, num_trajectories, figsize=(5*num_trajectories, 5))
        if num_trajectories == 1:
            axes = [axes]
        
        for traj_idx in range(num_trajectories):
            traj_data = contact_hist_data[str(traj_idx)]
            hist_centers = np.array(traj_data['hist_centers'])
            hist_counts = np.array(traj_data['hist_counts'])
            
            axes[traj_idx].plot(hist_centers, hist_counts, 'o-', linewidth=2, markersize=4)
            axes[traj_idx].set_xlabel('Water OH Angle [°]')
            axes[traj_idx].set_ylabel('Probability Density')
            axes[traj_idx].set_title(f'Traj {trajectory_indices[traj_idx]}: P(θ) OH in Contact Layer\n({O_z_min:.1f}-{O_z_max:.1f} Å)')
            axes[traj_idx].grid(True, alpha=0.3)
            axes[traj_idx].set_xlim(0, 180)
        
        plt.tight_layout()
        plt.savefig(system_plots_dir + '/' + name + '_OH_contact_theta_hist_by_traj.png', dpi=300)
    plt.close()
    
    # Plot combined histogram from all trajectories
    all_contact_angles = []
    for traj_idx in range(num_trajectories):
        # Handle both integer and string keys
        traj_key = str(traj_idx) if str(traj_idx) in angular_data else traj_idx
        
        for frame_data in angular_data[traj_key].values():
            z_vals = np.array(frame_data['z_vals'])
            angles = np.array(frame_data['angles'])
            
            # Filter for contact layer
            contact_mask = (z_vals >= O_z_min) & (z_vals <= O_z_max)
            all_contact_angles.extend(angles[contact_mask].tolist())
    
    # Create combined histogram
    combined_hist_counts, combined_hist_edges = np.histogram(all_contact_angles, bins=theta_bins, density=True)
    combined_hist_centers = (combined_hist_edges[:-1] + combined_hist_edges[1:]) / 2
    
    # Save combined histogram data
    if save_data:
        combined_csv_data = np.column_stack([combined_hist_centers, combined_hist_counts])
        np.savetxt(f"{system_data_dir}/{name}_combined_OH_contact_theta_histogram.csv", 
                  combined_csv_data, delimiter=',', header='theta_center,probability_density', comments='')
    
    if plot_cos_theta:
        # Create expanded theta data points weighted by histogram counts for proper cos(theta) transformation
        expanded_combined_thetas = []
        for j in range(len(combined_hist_centers)):
            count = int(combined_hist_counts[j] * 1000)  # Scale up for discrete sampling
            expanded_combined_thetas.extend([combined_hist_centers[j]] * count)
        
        if len(expanded_combined_thetas) > 0:
            expanded_combined_thetas = np.array(expanded_combined_thetas)
            expanded_combined_cos_thetas = np.cos(np.radians(expanded_combined_thetas))
            
            # Use evenly spaced cos(theta) bins for proper histogram
            evenly_spaced_cos_bins = np.linspace(-1, 1, num_cos_bins + 1)
            combined_cos_hist_counts, combined_cos_hist_edges = np.histogram(expanded_combined_cos_thetas, bins=evenly_spaced_cos_bins, density=True)
            combined_cos_hist_centers = (combined_cos_hist_edges[:-1] + combined_cos_hist_edges[1:]) / 2
            
            # Create cos(theta) histogram plot
            plt.figure(figsize=(8, 6))
            plt.plot(combined_cos_hist_centers, combined_cos_hist_counts, 'o-', linewidth=2, markersize=4, color='lightcoral')
            plt.xlabel('cos(θ)')
            plt.ylabel('Probability Density')
            plt.title(f'{name}: OH Angle Distribution in Contact Layer: \n({O_z_min:.1f}-{O_z_max:.1f} Å)')
            plt.grid(True, alpha=0.3)
            plt.xlim(-1, 1)
            plt.tight_layout()
            plt.savefig(system_plots_dir + '/' + name + '_OH_contact_costheta_hist_combined.png', dpi=300)
    else:
        # Create theta histogram plot
        plt.figure(figsize=(8, 6))
        plt.plot(combined_hist_centers, combined_hist_counts, 'o-', linewidth=2, markersize=4, color='skyblue')
        plt.xlabel('Water OH Angle [°]')
        plt.ylabel('Probability Density')
        plt.title(f'{name}: OH Angle Distribution in Contact Layer: \n({O_z_min:.1f}-{O_z_max:.1f} Å)')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 180)
        plt.tight_layout()
        plt.savefig(system_plots_dir + '/' + name + '_OH_contact_theta_hist_combined.png', dpi=300)
    plt.close()

    print(f"OH angular distribution analysis complete for {name}")
    print(f"Note: Each water molecule contributes 2 data points (one for each OH bond)")
    print(f"Total data points: {len(all_z_vals_for_density)} OH bond angle measurements") 



def plot_OH_vs_dipole_angle_distribution(name,
                                       substrate,
                                       trajectories,
                                       O_z_min,
                                       O_z_max,
                                       contact_layer_angle_plots_dir='./',
                                       sampling_interval=20,
                                       save_data=True,
                                       load_data=True,
                                       trajectory_indices=None,
                                       num_costheta_bins=50,
                                       ):
    """
    Analyze OH bond cos(theta) vs dipole cos(theta) for water molecules.
    
    For each frame, this function uses get_interfacial_z_vs_OH_angles to get OH angles
    and get_dipole_vs_interface_angles to get dipole angles, converts them to cos(theta),
    then creates a dictionary with structure {O atom index: [dipole_costheta, OH_costheta]}.
    
    Parameters:
    -----------
    name : str
        System name for file naming
    substrate : ase.Atoms
        Substrate structure
    trajectories : list
        List of trajectory objects
    O_z_min : float
        Minimum z-coordinate for contact layer
    O_z_max : float  
        Maximum z-coordinate for contact layer
    contact_layer_angle_plots_dir : str
        Directory for output files
    sampling_interval : int
        Sampling interval for frames
    save_data : bool
        Whether to save data to files
    load_data : bool
        Whether to load existing data
    trajectory_indices : list, optional
        Indices for trajectory labeling
    num_costheta_bins : int, optional
        Number of bins for cos(theta) histograms (default: 50)
        
    Returns:
    --------
    dict : Dictionary with structure {O_atom_index: [dipole_costheta, OH_costheta]}
    """
    
    # Set default trajectory indices if not provided
    if trajectory_indices is None:
        trajectory_indices = [i+1 for i in range(len(trajectories))]
    elif len(trajectory_indices) != len(trajectories):
        raise ValueError(f"Length of trajectory_indices ({len(trajectory_indices)}) must match number of trajectories ({len(trajectories)})")

    # Create system-specific subdirectories
    system_data_dir = os.path.join(contact_layer_angle_plots_dir, f"{name}_OH_vs_dipole_data")
    
    if not os.path.exists(system_data_dir):
        os.makedirs(system_data_dir)

    # Initialize data structure
    combined_angle_data = {}
    
    # Data file paths
    data_file_path = f"{system_data_dir}/{name}_OH_vs_dipole_angle_data.json"
    
    # Load existing data if requested and available
    if load_data and os.path.exists(data_file_path):
        with open(data_file_path, 'r') as f:
            combined_angle_data = json.load(f)
        print(f"Loaded existing OH vs dipole angle data from {data_file_path}")
    else:
        # Collect both OH and dipole angle data for each trajectory and frame
        for traj_idx, traj in enumerate(trajectories):
            combined_angle_data[traj_idx] = {}
            
            for frame_idx, frame in enumerate(tqdm(traj[::sampling_interval], 
                                                 desc=f"Processing traj {trajectory_indices[traj_idx]} frames for OH vs Dipole analysis {name}")):
                
                # Get OH angle data
                oh_indices, oh_z_vals, oh_angles = interface_analysis_tools.get_interfacial_z_vs_OH_angles(frame, substrate)
                
                # Get dipole angle data  
                dipole_angles_dict = interface_analysis_tools.get_dipole_vs_interface_angles(frame, substrate)
                
                # Create frame-specific dictionary combining both angle types
                frame_combined_data = {}
                
                # Process OH data - note that get_interfacial_z_vs_OH_angles returns 2 entries per water molecule (2 OH bonds)
                # We need to match these with the single dipole angle per water molecule
                for i, (oh_idx, oh_z, oh_angle) in enumerate(zip(oh_indices, oh_z_vals, oh_angles)):
                    if oh_idx in dipole_angles_dict:
                        dipole_angle = dipole_angles_dict[oh_idx]
                        
                        # Convert angles to cos(theta) values
                        dipole_costheta = np.cos(np.radians(dipole_angle))
                        oh_costheta = np.cos(np.radians(oh_angle))
                        
                        # Create unique key for each OH bond (since each water has 2 OH bonds)
                        # We'll use a combination of O index and OH bond number
                        oh_bond_count = frame_combined_data.get(f"{oh_idx}_count", 0)
                        unique_key = f"{oh_idx}_OH_{oh_bond_count}"
                        
                        frame_combined_data[unique_key] = {
                            'O_index': oh_idx,
                            'z': oh_z,
                            'dipole_costheta': dipole_costheta,
                            'OH_costheta': oh_costheta,
                            'OH_bond_number': oh_bond_count
                        }
                        
                        frame_combined_data[f"{oh_idx}_count"] = oh_bond_count + 1
                
                # Remove count keys and store the frame data
                frame_data = {k: v for k, v in frame_combined_data.items() if not k.endswith('_count')}
                combined_angle_data[traj_idx][frame_idx * sampling_interval] = frame_data
        
        # Save data if requested
        if save_data:
            # Convert numpy types for JSON serialization
            json_data = convert_numpy_types_for_json(combined_angle_data)
            with open(data_file_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Saved OH vs dipole cos(theta) data to {data_file_path}")
    
    # Create simplified dictionary with structure {O atom index: [dipole_costheta, OH_costheta]}
    # This aggregates all data points across all trajectories and frames
    simplified_dict = {}
    
    # Also collect data for contact layer filtering and histogram plotting
    all_dipole_costheta = []
    all_oh_costheta = []
    contact_dipole_costheta = []
    contact_oh_costheta = []
    
    for traj_idx, traj_data in combined_angle_data.items():
        # Handle both integer and string keys (from JSON loading)
        traj_key = str(traj_idx) if str(traj_idx) in combined_angle_data else traj_idx
        
        for frame_idx, frame_data in traj_data.items():
            for unique_key, angle_data in frame_data.items():
                o_index = angle_data['O_index']
                dipole_costheta = angle_data['dipole_costheta']
                oh_costheta = angle_data['OH_costheta']
                z_val = angle_data['z']
                
                # Store each OH bond separately with a unique identifier
                simplified_key = f"{o_index}_{unique_key}"  # This includes trajectory and frame info
                simplified_dict[simplified_key] = [dipole_costheta, oh_costheta]
                
                # Collect all cos(theta) data
                all_dipole_costheta.append(dipole_costheta)
                all_oh_costheta.append(oh_costheta)
                
                # Filter for contact layer (O_z_min <= z <= O_z_max)
                if O_z_min <= z_val <= O_z_max:
                    contact_dipole_costheta.append(dipole_costheta)
                    contact_oh_costheta.append(oh_costheta)
    
    # Convert to numpy arrays for histogram processing
    all_dipole_costheta = np.array(all_dipole_costheta)
    all_oh_costheta = np.array(all_oh_costheta)
    contact_dipole_costheta = np.array(contact_dipole_costheta)
    contact_oh_costheta = np.array(contact_oh_costheta)
    
    print(f"OH vs dipole cos(theta) analysis complete for {name}")
    print(f"Total data points: {len(simplified_dict)} OH bond measurements")
    print(f"Contact layer data points: {len(contact_dipole_costheta)} OH bond measurements")
    print(f"Note: Each water molecule contributes 2 data points (one for each OH bond)")
    
    # Create 2D histogram for contact layer data
    if len(contact_dipole_costheta) > 0:
        # Define histogram file paths
        histogram_file = f"{system_data_dir}/{name}_OH_vs_dipole_2D_costheta_histogram.csv"
        dipole_edges_file = f"{system_data_dir}/{name}_dipole_costheta_bin_edges.csv"
        oh_edges_file = f"{system_data_dir}/{name}_oh_costheta_bin_edges.csv"
        
        # Check if histogram data already exists and should be loaded
        if load_data and os.path.exists(histogram_file) and os.path.exists(dipole_edges_file) and os.path.exists(oh_edges_file):
            print(f"Loading existing 2D cos(theta) histogram data from {histogram_file}")
            
            # Load bin edges
            dipole_edges = np.loadtxt(dipole_edges_file, delimiter=',')
            oh_edges = np.loadtxt(oh_edges_file, delimiter=',')
            
            # Load histogram data - skip first row (header) and first column (OH centers)
            histogram_data = np.loadtxt(histogram_file, delimiter=',', skiprows=1)
            oh_centers = histogram_data[:, 0]  # First column is OH centers
            hist_2d = histogram_data[:, 1:].T  # Rest is the histogram data, transposed back
            
            # Calculate dipole centers from edges
            dipole_centers = (dipole_edges[:-1] + dipole_edges[1:]) / 2
            
        else:
            print(f"Creating new 2D cos(theta) histogram from {len(contact_dipole_costheta)} contact layer data points")
            
            # Define cos(theta) histogram bins - evenly spaced from -1 to +1
            dipole_bins = np.linspace(-1, 1, num_costheta_bins + 1)  # cos(theta) bins for dipole
            oh_bins = np.linspace(-1, 1, num_costheta_bins + 1)     # cos(theta) bins for OH
            
            # Create 2D histogram
            hist_2d, dipole_edges, oh_edges = np.histogram2d(contact_dipole_costheta, contact_oh_costheta, 
                                                             bins=[dipole_bins, oh_bins], density=True)
            
            # Calculate bin centers for plotting
            dipole_centers = (dipole_edges[:-1] + dipole_edges[1:]) / 2
            oh_centers = (oh_edges[:-1] + oh_edges[1:]) / 2
            
            # Save histogram data to CSV
            if save_data:
                # Save the histogram data with row/column headers
                header_line = 'OH_costheta_centers,' + ','.join([f'{dc:.3f}' for dc in dipole_centers])
                np.savetxt(histogram_file, 
                          np.column_stack([oh_centers, hist_2d.T]), 
                          delimiter=',', 
                          header=header_line, 
                          comments='',
                          fmt='%.6f')
                
                # Also save bin edges
                np.savetxt(dipole_edges_file, dipole_edges, delimiter=',')
                np.savetxt(oh_edges_file, oh_edges, delimiter=',')
                
                print(f"Saved 2D cos(theta) histogram data to {histogram_file}")
        
        # Create plots directory
        plots_dir = os.path.join(contact_layer_angle_plots_dir, f"{name}_OH_vs_dipole_costheta_plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Plot 2D histogram
        plt.figure(figsize=(10, 8))
        
        # Use pcolormesh for the 2D histogram
        X, Y = np.meshgrid(dipole_edges, oh_edges)
        im = plt.pcolormesh(X, Y, hist_2d.T, cmap='Blues')
        
        plt.colorbar(im, label='Probability Density')
        plt.xlabel('Water Dipole cos(θ)')
        plt.ylabel('Water OH cos(θ)')
        plt.title(f'{name}: OH vs Dipole cos(θ) Distribution\nContact Layer ({O_z_min:.1f}-{O_z_max:.1f} Å)')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Add reference lines
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{name}_OH_vs_dipole_2D_costheta_histogram.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a scatter plot for comparison
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot with alpha for overlapping points
        plt.scatter(contact_dipole_costheta, contact_oh_costheta, alpha=0.1, s=1, c='blue')
        
        plt.xlabel('Water Dipole cos(θ)')
        plt.ylabel('Water OH cos(θ)')
        plt.title(f'{name}: OH vs Dipole cos(θ) Scatter Plot\nContact Layer ({O_z_min:.1f}-{O_z_max:.1f} Å)')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.grid(True, alpha=0.3)
        
        # Add diagonal line for reference
        plt.plot([-1, 1], [-1, 1], 'r--', alpha=0.5, label='OH cos(θ) = Dipole cos(θ)')
        # Add reference lines
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{name}_OH_vs_dipole_costheta_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved 2D cos(theta) histogram plot to {plots_dir}/{name}_OH_vs_dipole_2D_costheta_histogram.png")
        print(f"Saved cos(theta) scatter plot to {plots_dir}/{name}_OH_vs_dipole_costheta_scatter.png")
        
    else:
        print("No data points found in the contact layer for histogram plotting")
    
    return simplified_dict





def generate_euler_angle_data(name,
                             substrate,
                             trajectories,
                             O_z_min,
                             O_z_max,
                             contact_layer_angle_plots_dir='./',
                             sampling_interval=20,
                             save_data=True,
                             load_data=True,
                             trajectory_indices=None,
                             ):
    """
    Generate raw Euler angle data (cos(phi), cos(theta), z) for water molecules.
    
    Parameters:
    -----------
    name : str
        System name for file naming
    substrate : ase.Atoms
        Substrate structure
    trajectories : list
        List of trajectory objects
    O_z_min : float
        Minimum z-coordinate for contact layer
    O_z_max : float  
        Maximum z-coordinate for contact layer
    contact_layer_angle_plots_dir : str
        Directory for output files
    sampling_interval : int
        Sampling interval for frames
    save_data : bool
        Whether to save data to files
    load_data : bool
        Whether to load existing data
    trajectory_indices : list, optional
        Indices for trajectory labeling
        
    Returns:
    --------
    dict : Dictionary with structure {traj_idx: {'cos_phi': [...], 'cos_theta': [...], 'z_vals': [...]}}
    """
    
    # Set default trajectory indices if not provided
    if trajectory_indices is None:
        trajectory_indices = [i+1 for i in range(len(trajectories))]
    elif len(trajectory_indices) != len(trajectories):
        raise ValueError(f"Length of trajectory_indices ({len(trajectory_indices)}) must match number of trajectories ({len(trajectories)})")

    # Create system-specific subdirectories
    system_data_dir = os.path.join(contact_layer_angle_plots_dir, f"{name}_euler_angle_data")
    
    if not os.path.exists(system_data_dir):
        os.makedirs(system_data_dir)

    def load_euler_data():
        """Load existing Euler angle data if available."""
        euler_data = {}
        all_files_exist = True
        
        for traj_idx in range(len(trajectories)):
            data_file_path = f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_euler_raw_data.csv"
            
            if load_data and os.path.exists(data_file_path):
                try:
                    df = pd.read_csv(data_file_path)
                    euler_data[traj_idx] = {
                        'cos_phi': df['cos_phi'].values,
                        'cos_theta': df['cos_theta'].values,
                        'z_vals': df['z_vals'].values
                    }
                    print(f"Loaded existing Euler angle data for trajectory {trajectory_indices[traj_idx]} from {data_file_path}")
                except Exception as e:
                    print(f"Error loading Euler angle data for trajectory {trajectory_indices[traj_idx]}: {e}")
                    all_files_exist = False
                    break
            else:
                all_files_exist = False
                break
        
        return euler_data if all_files_exist else None

    def generate_euler_data():
        """Generate new Euler angle data for all trajectories."""
        print(f"Generating new Euler angle data for {name}")
        euler_data = {}
        
        for traj_idx, traj in enumerate(trajectories):
            print(f"Processing trajectory {trajectory_indices[traj_idx]}/{len(trajectories)}")
            
            # Calculate interface indices trajectory
            interface_indices_traj = interface_analysis_tools.find_water_species_indices_traj(
                traj[::sampling_interval],
                substrate,
                interface_analysis_tools.interfacial_water_criterion,
                z_min=O_z_min,
                z_max=O_z_max,
            )
            
            # Get Euler angle data
            raw_euler_data = interface_analysis_tools.get_euler_angle_data(
                [traj[::sampling_interval]],
                substrate,
                sampling_indices_trajectory=interface_indices_traj,
            )
            
            # Extract phi, theta, z data
            phi_vals, theta_vals, z_vals = raw_euler_data[0], raw_euler_data[1], raw_euler_data[2]
            
            # Convert to cosine values
            cos_phi_vals = np.cos(phi_vals)
            cos_theta_vals = np.cos(theta_vals)
            
            # Store data
            euler_data[traj_idx] = {
                'cos_phi': cos_phi_vals,
                'cos_theta': cos_theta_vals,
                'z_vals': z_vals
            }
            
            print(f"Generated {len(cos_phi_vals)} Euler angle data points for trajectory {trajectory_indices[traj_idx]}")
        
        return euler_data

    def save_euler_data(euler_data):
        """Save Euler angle data to CSV files."""
        if save_data:
            for traj_idx, data in euler_data.items():
                data_file_path = f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_euler_raw_data.csv"
                
                df = pd.DataFrame({
                    'cos_phi': data['cos_phi'],
                    'cos_theta': data['cos_theta'],
                    'z_vals': data['z_vals']
                })
                df.to_csv(data_file_path, index=False)
                print(f"Saved Euler angle raw data for trajectory {trajectory_indices[traj_idx]} to {data_file_path}")

    # Main execution flow
    # 1. Try to load existing data
    euler_data = load_euler_data()
    
    # 2. Generate new data if loading failed
    if euler_data is None:
        euler_data = generate_euler_data()
        # 3. Save the generated data
        save_euler_data(euler_data)
    
    return euler_data


def plot_euler_angle_distributions(name,
                                  substrate,
                                  trajectories,
                                  O_z_min,
                                  O_z_max,
                                  contact_layer_angle_plots_dir='./',
                                  sampling_interval=20,
                                  save_data=True,
                                  load_data=True,
                                  trajectory_indices=None,
                                  num_costheta_bins=50,
                                  ):
    """
    Plot Euler angle distributions using raw data.
    
    Parameters:
    -----------
    name : str
        System name for file naming
    substrate : ase.Atoms
        Substrate structure
    trajectories : list
        List of trajectory objects
    O_z_min : float
        Minimum z-coordinate for contact layer
    O_z_max : float  
        Maximum z-coordinate for contact layer
    contact_layer_angle_plots_dir : str
        Directory for output files
    sampling_interval : int
        Sampling interval for frames
    save_data : bool
        Whether to save binned data to files
    load_data : bool
        Whether to load existing binned data
    trajectory_indices : list, optional
        Indices for trajectory labeling
    num_costheta_bins : int, optional
        Number of bins for cos(theta) histograms (default: 50)
    """
    
    # Set default trajectory indices if not provided
    if trajectory_indices is None:
        trajectory_indices = [i+1 for i in range(len(trajectories))]
    elif len(trajectory_indices) != len(trajectories):
        raise ValueError(f"Length of trajectory_indices ({len(trajectory_indices)}) must match number of trajectories ({len(trajectories)})")

    # Create system-specific subdirectories
    system_data_dir = os.path.join(contact_layer_angle_plots_dir, f"{name}_euler_angle_data")
    system_plots_dir = os.path.join(contact_layer_angle_plots_dir, f"{name}_euler_angle_plots")
    
    if not os.path.exists(system_plots_dir):
        os.makedirs(system_plots_dir)

    def load_binned_data():
        """Load existing binned Euler angle data if available."""
        binned_data = {}
        
        # Check for individual trajectory binned data
        for traj_idx in range(len(trajectories)):
            phi_file = f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_cos_phi_histogram.csv"
            theta_file = f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_cos_theta_histogram.csv"
            hist2d_file = f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_2d_histogram.npz"
            
            if load_data and os.path.exists(phi_file) and os.path.exists(theta_file) and os.path.exists(hist2d_file):
                try:
                    phi_data = np.loadtxt(phi_file, delimiter=',', skiprows=1)
                    theta_data = np.loadtxt(theta_file, delimiter=',', skiprows=1)
                    hist2d_data = np.load(hist2d_file)
                    
                    binned_data[traj_idx] = {
                        'cos_phi_centers': phi_data[:, 0],
                        'cos_phi_counts': phi_data[:, 1],
                        'cos_theta_centers': theta_data[:, 0],
                        'cos_theta_counts': theta_data[:, 1],
                        'hist2d': hist2d_data['hist2d'],
                        'cos_theta_edges': hist2d_data['cos_theta_edges'],
                        'cos_phi_edges': hist2d_data['cos_phi_edges']
                    }
                except Exception as e:
                    print(f"Error loading binned data for trajectory {trajectory_indices[traj_idx]}: {e}")
                    return None
            else:
                return None
        
        # Check for combined data if multiple trajectories
        if len(trajectories) > 1:
            combined_phi_file = f"{system_data_dir}/{name}_combined_cos_phi_histogram.csv"
            combined_theta_file = f"{system_data_dir}/{name}_combined_cos_theta_histogram.csv"
            combined_hist2d_file = f"{system_data_dir}/{name}_combined_2d_histogram.npz"
            
            if load_data and os.path.exists(combined_phi_file) and os.path.exists(combined_theta_file) and os.path.exists(combined_hist2d_file):
                try:
                    combined_phi_data = np.loadtxt(combined_phi_file, delimiter=',', skiprows=1)
                    combined_theta_data = np.loadtxt(combined_theta_file, delimiter=',', skiprows=1)
                    combined_hist2d_data = np.load(combined_hist2d_file)
                    
                    binned_data['combined'] = {
                        'cos_phi_centers': combined_phi_data[:, 0],
                        'cos_phi_counts': combined_phi_data[:, 1],
                        'cos_theta_centers': combined_theta_data[:, 0],
                        'cos_theta_counts': combined_theta_data[:, 1],
                        'hist2d': combined_hist2d_data['hist2d'],
                        'cos_theta_edges': combined_hist2d_data['cos_theta_edges'],
                        'cos_phi_edges': combined_hist2d_data['cos_phi_edges']
                    }
                except Exception as e:
                    print(f"Error loading combined binned data: {e}")
                    return None
            else:
                return None
        
        print(f"Loaded existing binned Euler angle data for {name}")
        return binned_data

    def generate_binned_data(raw_euler_data):
        """Generate binned histogram data from raw Euler angle data."""
        print(f"Generating binned Euler angle data for {name}")
        binned_data = {}
        
        # Define bins for cos(phi) from 0 to 1 and cos(theta) from -1 to 1
        cos_phi_bins = np.linspace(0, 1, num_costheta_bins + 1)
        cos_theta_bins = np.linspace(-1, 1, num_costheta_bins + 1)
        cos_phi_centers = (cos_phi_bins[:-1] + cos_phi_bins[1:]) / 2
        cos_theta_centers = (cos_theta_bins[:-1] + cos_theta_bins[1:]) / 2
        
        # Process individual trajectories
        for traj_idx, data in raw_euler_data.items():
            cos_phi_counts, _ = np.histogram(data['cos_phi'], bins=cos_phi_bins, density=True)
            cos_theta_counts, _ = np.histogram(data['cos_theta'], bins=cos_theta_bins, density=True)
            
            # Create 2D histogram: cos(theta) vs cos(phi)
            hist2d, cos_theta_edges, cos_phi_edges = np.histogram2d(
                data['cos_theta'], data['cos_phi'], 
                bins=[cos_theta_bins, cos_phi_bins], 
                density=True
            )
            
            binned_data[traj_idx] = {
                'cos_phi_centers': cos_phi_centers,
                'cos_phi_counts': cos_phi_counts,
                'cos_theta_centers': cos_theta_centers,
                'cos_theta_counts': cos_theta_counts,
                'hist2d': hist2d,
                'cos_theta_edges': cos_theta_edges,
                'cos_phi_edges': cos_phi_edges
            }
        
        # Generate combined data if multiple trajectories
        if len(trajectories) > 1:
            combined_cos_phi = np.concatenate([data['cos_phi'] for data in raw_euler_data.values()])
            combined_cos_theta = np.concatenate([data['cos_theta'] for data in raw_euler_data.values()])
            
            combined_cos_phi_counts, _ = np.histogram(combined_cos_phi, bins=cos_phi_bins, density=True)
            combined_cos_theta_counts, _ = np.histogram(combined_cos_theta, bins=cos_theta_bins, density=True)
            
            # Create combined 2D histogram
            combined_hist2d, combined_cos_theta_edges, combined_cos_phi_edges = np.histogram2d(
                combined_cos_theta, combined_cos_phi, 
                bins=[cos_theta_bins, cos_phi_bins], 
                density=True
            )
            
            binned_data['combined'] = {
                'cos_phi_centers': cos_phi_centers,
                'cos_phi_counts': combined_cos_phi_counts,
                'cos_theta_centers': cos_theta_centers,
                'cos_theta_counts': combined_cos_theta_counts,
                'hist2d': combined_hist2d,
                'cos_theta_edges': combined_cos_theta_edges,
                'cos_phi_edges': combined_cos_phi_edges
            }
        
        return binned_data

    def save_binned_data(binned_data):
        """Save binned histogram data to CSV and NPZ files."""
        if save_data:
            # Save individual trajectory data
            for traj_idx in range(len(trajectories)):
                if traj_idx in binned_data:
                    data = binned_data[traj_idx]
                    
                    # Save cos(phi) histogram
                    phi_file = f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_cos_phi_histogram.csv"
                    phi_csv_data = np.column_stack([data['cos_phi_centers'], data['cos_phi_counts']])
                    np.savetxt(phi_file, phi_csv_data, delimiter=',', 
                              header='cos_phi_center,probability_density', comments='')
                    
                    # Save cos(theta) histogram
                    theta_file = f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_cos_theta_histogram.csv"
                    theta_csv_data = np.column_stack([data['cos_theta_centers'], data['cos_theta_counts']])
                    np.savetxt(theta_file, theta_csv_data, delimiter=',', 
                              header='cos_theta_center,probability_density', comments='')
                    
                    # Save 2D histogram
                    hist2d_file = f"{system_data_dir}/{name}_traj_{trajectory_indices[traj_idx]}_2d_histogram.npz"
                    np.savez(hist2d_file, 
                            hist2d=data['hist2d'],
                            cos_theta_edges=data['cos_theta_edges'],
                            cos_phi_edges=data['cos_phi_edges'])
                    
                    print(f"Saved binned histogram data for trajectory {trajectory_indices[traj_idx]}")
            
            # Save combined data if it exists
            if 'combined' in binned_data:
                combined_data = binned_data['combined']
                
                # Save combined cos(phi) histogram
                combined_phi_file = f"{system_data_dir}/{name}_combined_cos_phi_histogram.csv"
                combined_phi_csv = np.column_stack([combined_data['cos_phi_centers'], combined_data['cos_phi_counts']])
                np.savetxt(combined_phi_file, combined_phi_csv, delimiter=',', 
                          header='cos_phi_center,probability_density', comments='')
                
                # Save combined cos(theta) histogram
                combined_theta_file = f"{system_data_dir}/{name}_combined_cos_theta_histogram.csv"
                combined_theta_csv = np.column_stack([combined_data['cos_theta_centers'], combined_data['cos_theta_counts']])
                np.savetxt(combined_theta_file, combined_theta_csv, delimiter=',', 
                          header='cos_theta_center,probability_density', comments='')
                
                # Save combined 2D histogram
                combined_hist2d_file = f"{system_data_dir}/{name}_combined_2d_histogram.npz"
                np.savez(combined_hist2d_file,
                        hist2d=combined_data['hist2d'],
                        cos_theta_edges=combined_data['cos_theta_edges'],
                        cos_phi_edges=combined_data['cos_phi_edges'])
                
                print(f"Saved combined binned histogram data")

    # Main execution flow
    # 1. Try to load plotting (binned) data first
    binned_data = load_binned_data()
    
    # 2. If plotting data doesn't exist, get raw data and generate plotting data
    if binned_data is None:
        # Get raw Euler angle data (this will load or generate as needed)
        raw_euler_data = generate_euler_angle_data(
            name, substrate, trajectories, O_z_min, O_z_max,
            contact_layer_angle_plots_dir, sampling_interval, save_data, load_data, trajectory_indices
        )
        
        # Generate plotting (binned) data from raw data
        binned_data = generate_binned_data(raw_euler_data)
        
        # Save the plotting (binned) data
        save_binned_data(binned_data)
    
    # 5. Create plots
    num_trajectories = len(trajectories)
    
    if num_trajectories > 1:
        # Create subplots for individual trajectories (1D histograms)
        fig, axes = plt.subplots(2, num_trajectories, figsize=(6*num_trajectories, 10))
        
        for traj_idx in range(num_trajectories):
            if traj_idx in binned_data:
                data = binned_data[traj_idx]
                
                # Plot cos(phi) histogram
                axes[0, traj_idx].plot(data['cos_phi_centers'], data['cos_phi_counts'], 'o-', 
                                      linewidth=2, markersize=4, color='skyblue')
                axes[0, traj_idx].set_xlabel('cos(φ)')
                axes[0, traj_idx].set_ylabel('Probability Density')
                axes[0, traj_idx].set_title(f'Traj {trajectory_indices[traj_idx]}: cos(φ) Distribution')
                axes[0, traj_idx].grid(True, alpha=0.3)
                axes[0, traj_idx].set_xlim(0, 1)
                
                # Plot cos(theta) histogram
                axes[1, traj_idx].plot(data['cos_theta_centers'], data['cos_theta_counts'], 'o-', 
                                      linewidth=2, markersize=4, color='lightcoral')
                axes[1, traj_idx].set_xlabel('cos(θ)')
                axes[1, traj_idx].set_ylabel('Probability Density')
                axes[1, traj_idx].set_title(f'Traj {trajectory_indices[traj_idx]}: cos(θ) Distribution')
                axes[1, traj_idx].grid(True, alpha=0.3)
                axes[1, traj_idx].set_xlim(-1, 1)
        
        plt.tight_layout()
        plt.savefig(f"{system_plots_dir}/{name}_euler_distributions_by_traj.png", dpi=300)
        plt.close()
        
        # Create 2D histograms for individual trajectories
        fig, axes = plt.subplots(1, num_trajectories, figsize=(6*num_trajectories, 5))
        if num_trajectories == 1:
            axes = [axes]
        
        for traj_idx in range(num_trajectories):
            if traj_idx in binned_data:
                data = binned_data[traj_idx]
                
                # Plot 2D histogram: cos(theta) vs cos(phi)
                im = axes[traj_idx].pcolormesh(data['cos_theta_edges'], data['cos_phi_edges'], 
                                              data['hist2d'].T, cmap='Blues')
                axes[traj_idx].set_xlabel('cos(θ)')
                axes[traj_idx].set_ylabel('cos(φ)')
                axes[traj_idx].set_title(f'Traj {trajectory_indices[traj_idx]}: 2D Distribution')
                axes[traj_idx].set_xlim(-1, 1)
                axes[traj_idx].set_ylim(0, 1)
                plt.colorbar(im, ax=axes[traj_idx], label='Probability Density')
        
        plt.tight_layout()
        plt.savefig(f"{system_plots_dir}/{name}_euler_2d_distributions_by_traj.png", dpi=300)
        plt.close()
        
        # Create combined plots
        if 'combined' in binned_data:
            combined_data = binned_data['combined']
            
            # Combined 1D histograms
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Combined cos(phi) histogram
            ax1.plot(combined_data['cos_phi_centers'], combined_data['cos_phi_counts'], 'o-', 
                    linewidth=2, markersize=4, color='skyblue')
            ax1.set_xlabel('cos(φ)')
            ax1.set_ylabel('Probability Density')
            ax1.set_title(f'{name}: Combined cos(φ) Distribution\nContact Layer ({O_z_min:.1f}-{O_z_max:.1f} Å)')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 1)
            
            # Combined cos(theta) histogram
            ax2.plot(combined_data['cos_theta_centers'], combined_data['cos_theta_counts'], 'o-', 
                    linewidth=2, markersize=4, color='lightcoral')
            ax2.set_xlabel('cos(θ)')
            ax2.set_ylabel('Probability Density')
            ax2.set_title(f'{name}: Combined cos(θ) Distribution\nContact Layer ({O_z_min:.1f}-{O_z_max:.1f} Å)')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(-1, 1)
            
            plt.tight_layout()
            plt.savefig(f"{system_plots_dir}/{name}_euler_distributions_combined.png", dpi=300)
            plt.close()
            
            # Combined 2D histogram
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.pcolormesh(combined_data['cos_theta_edges'], combined_data['cos_phi_edges'], 
                              combined_data['hist2d'].T, cmap='Blues')
            ax.set_xlabel('cos(θ)')
            ax.set_ylabel('cos(φ)')
            ax.set_title(f'{name}: Combined 2D Distribution\nContact Layer ({O_z_min:.1f}-{O_z_max:.1f} Å)')
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1)
            plt.colorbar(im, label='Probability Density')
            plt.tight_layout()
            plt.savefig(f"{system_plots_dir}/{name}_euler_2d_distribution_combined.png", dpi=300)
            plt.close()
    
    else:
        # Single trajectory case
        if 0 in binned_data:
            data = binned_data[0]
            
            # 1D histograms
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # cos(phi) histogram
            ax1.plot(data['cos_phi_centers'], data['cos_phi_counts'], 'o-', 
                    linewidth=2, markersize=4, color='skyblue')
            ax1.set_xlabel('cos(φ)')
            ax1.set_ylabel('Probability Density')
            ax1.set_title(f'{name}: cos(φ) Distribution\nContact Layer ({O_z_min:.1f}-{O_z_max:.1f} Å)')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 1)
            
            # cos(theta) histogram
            ax2.plot(data['cos_theta_centers'], data['cos_theta_counts'], 'o-', 
                    linewidth=2, markersize=4, color='lightcoral')
            ax2.set_xlabel('cos(θ)')
            ax2.set_ylabel('Probability Density')
            ax2.set_title(f'{name}: cos(θ) Distribution\nContact Layer ({O_z_min:.1f}-{O_z_max:.1f} Å)')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(-1, 1)
            
            plt.tight_layout()
            plt.savefig(f"{system_plots_dir}/{name}_euler_distributions.png", dpi=300)
            plt.close()
            
            # 2D histogram
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.pcolormesh(data['cos_theta_edges'], data['cos_phi_edges'], 
                              data['hist2d'].T, cmap='Blues')
            ax.set_xlabel('cos(θ)')
            ax.set_ylabel('cos(φ)')
            ax.set_title(f'{name}: 2D Distribution\nContact Layer ({O_z_min:.1f}-{O_z_max:.1f} Å)')
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1)
            plt.colorbar(im, label='Probability Density')
            plt.tight_layout()
            plt.savefig(f"{system_plots_dir}/{name}_euler_2d_distribution.png", dpi=300)
            plt.close()
    
    print(f"Euler angle distribution analysis complete for {name}")
    # Calculate total data points from binned data (approximate from histogram counts)
    if 'combined' in binned_data:
        # Use combined data if available
        total_points = int(np.sum(binned_data['combined']['cos_phi_counts']) * 
                          (binned_data['combined']['cos_phi_centers'][1] - binned_data['combined']['cos_phi_centers'][0]))
        print(f"Approximate total data points: {total_points} water orientation measurements")
    else:
        # Sum from individual trajectories
        total_points = 0
        for traj_idx in range(len(trajectories)):
            if traj_idx in binned_data:
                traj_points = int(np.sum(binned_data[traj_idx]['cos_phi_counts']) * 
                                 (binned_data[traj_idx]['cos_phi_centers'][1] - binned_data[traj_idx]['cos_phi_centers'][0]))
                total_points += traj_points
        print(f"Approximate total data points: {total_points} water orientation measurements")












