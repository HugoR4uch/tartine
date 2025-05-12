import ase
import ase.io
import matplotlib.pyplot as plt
import numpy as np
import copy
from ase.constraints import FixAtoms
from mace.calculators.foundations_models import mace_mp
from ase.optimize import BFGS
import ase.data
import time
import os
from . import water_analyser
from mace.calculators import MACECalculator
from ase.visualize.plot import plot_atoms



def plot_atoms(atoms,rotation=('0x,0y,0z')):
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))  
    plot_atoms(atoms, ax=axes, rotation=('0x,0y,0z'))




def bulk_interface_multi_builder(substrate_dir,
                            intersubstrate_gap,
                            water_substrate_gap,
                            num_replicas=1,
                            thickness = {},
                            density = {},
                            optimise_interfacial_water=True,
                            ):
    
    input_filenames = os.listdir(substrate_dir)


    for filename in input_filenames:

        interface_name = filename.split('.')[0]

        if type(thickness) == dict:
            if interface_name in thickness:
                water_thickness = thickness[interface_name]
            else:
                water_thickness = 3 # Typical width of contact layer
        else:
            water_thickness = thickness
            

        if type(density) == dict:
            if interface_name in density:
                water_density = density[interface_name]
            else:
                water_density = 1
        else:
            water_density = density


        substrate_file_path = os.path.join(substrate_dir,filename)
        substrate = ase.io.read(substrate_file_path)

        for replica_index in range(num_replicas):

            print('Building',filename,'interface')


        

            interface_builder = InterfaceBuilder(substrate,
                                                 water_thickness,
                                                 water_density,
                                                 intersubstrate_gap,
                                                 name=interface_name)
            start_time = time.time()
            interface_builder.build_bulk_interface(water_substrate_gap=water_substrate_gap,
                                                   optimise_interfacial_water=optimise_interfacial_water,
                                                   )
            end_time = time.time()

            print('Time taken:',end_time-start_time)


            interface = interface_builder.get_interface()


            #Saving interface
            interface_dir_name = 'interfaces'
            if not os.path.exists(interface_dir_name):
                os.makedirs(interface_dir_name)

            name = filename.split('.')[0] +'_' + str(replica_index+1)
            ase.io.write(interface_dir_name+'/interface_'+name+'.xyz',interface,format='extxyz')
        
    return True



def find_top_layer_indices(substrate,num_layers):
    z_vals = substrate.positions[:,2]
    if num_layers == None:
        top_layer_z_val_threshold = np.max(z_vals) - 0.1 # anything 0.1 A below top atom
    else:
        top_layer_z_val_threshold = np.percentile(z_vals, 100*(1-1/num_layers))
    top_layer_indices = np.where(z_vals >= top_layer_z_val_threshold)[0]
    return top_layer_indices



""" Tools for checking physicality of interfacial water"""



def get_water_H_bond_stats(system,substrate_indices = None):
    # Initialization
    if substrate_indices is None:
        system_water = system
    else:
        system_size = len(system)
        water_indices = np.setdiff1d(np.array(range(system_size)) , np.array(substrate_indices))
        system_water = system[water_indices]
    analyser = water_analyser.Analyser([system_water])
    frame_index= 0 #as only one 'frame'
    #Getting H-bond statistics
    connectivity_matrx = analyser.get_H_bond_connectivity(frame_index)
    num_H_bonds_list = []
    num_water_O = len(analyser.water_O_indices)
    for i in range(num_water_O):
        num_H_bonds = np.sum(connectivity_matrx[i])
        num_H_bonds_list.append(num_H_bonds)
    mean_H_bonds_per_water = sum(num_H_bonds_list)/num_water_O
    return mean_H_bonds_per_water


def is_water_coordination_physical(system,substrate_indices = None):
    
    #Initialization

    if substrate_indices is None:
        system_water = system
    else:
        system_size = len(system)
        water_indices = np.setdiff1d(np.array(range(system_size)) , np.array(substrate_indices))
        system_water = system[water_indices]
    analyser = water_analyser.Analyser([system_water])
    frame_index= 0 #as only one 'frame'

    physical = True
    # Checks for water O-H coordination (should be 2 for water)
    num_voronoi_entries = []
    for i, water_index in enumerate(analyser.water_O_indices):
        num_voronoi_entries.append(len(analyser.voronoi_dicts[frame_index].get(water_index)))

    max_coordination , min_coordination = max(num_voronoi_entries) , min(num_voronoi_entries)

    if max_coordination!=2 or min_coordination!=2:
        physical = False

    return physical

    

class InterfaceBuilder:
    def __init__(self,
                 substrate,
                 water_thickness,
                 water_density,
                 intersubstrate_gap,
                 name=None):
        

                
        self.interface = None
        self.name = name
        self.intersubstrate_gap = intersubstrate_gap
        self.time_per_model_eval = None
        self.water_thickness = water_thickness
        self.water_density = water_density


        #Error handling
        if (substrate.cell.array == 0).all():
            raise ValueError('Cell of substrate is not defined')

        # Takes input substrate and adjusts z-dimension of cell, adds PBC and shifts interface to z=0.
        self.substrate = substrate

        self.num_substrate_atoms = len(self.substrate)
        model = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/data/mace_models/tartine_II_stagetwo_compiled.model'
        self.calculator = MACECalculator(model_path=model,device='cuda')


        #Finding substrate thickness
        if len(self.substrate)==0:
            self.substrate_thickness = 0
        else:
            all_substrate_z_vals = self.substrate.positions[:,2]
            self.substrate_thickness = np.max(all_substrate_z_vals) - np.min(all_substrate_z_vals)
            
        

        self.substrate = self.shift_substrate_to_cell_centre(substrate)

    def build_cluster_interface(self,
                                resolution_index = 2,
                                num_water = 1,
                                num_layers= None,
                                z_start = 3,
                                
                                ):
        

        raise NotImplementedError("Cluster interface building not yet implemented")

        """
        Builds interface with a cluster of molecules on top of the substrate.
        
        Parameters
        ----------
        resolution_index : float
            Resolution for possible x,y coordinates of the cluster.
            The higher the resolution, the more possible coordinates.
            The number of possible coordinates is 2**resolution_index.
            For example, resolution 1 means water can be placed on every top layer atom;
            resolution 2 means water can be placed between every two top layer atoms;
            resolution 3 means water can be placed between the atoms and the midpoints between atoms.
        num_water : int
            Number of water molecules in the cluster.
        num_layser : int
            Number of layers in the cluster. If None, the cluster is built with a single layer.

        """


        top_layer_indices = find_top_layer_indices(self.substrate,num_layers)

        top_layer_xy_coords = self.substrate.positions[top_layer_indices][:,0:2]

        sampling_xy_coords = copy.deepcopy(top_layer_xy_coords)

        for i in range(resolution_index):
            sampling_x = sampling_xy_coords[:,0]
            sampling_y = sampling_xy_coords[:,1]
            new_x = ( sampling_x[1:] + sampling_x[:1] ) / 2
            new_y = ( sampling_y[1:] + sampling_y[:1] ) / 2
            sampling_x = np.append((sampling_x,new_x),axis=0)
            sampling_y = np.append((sampling_y,new_y),axis=0)
            sampling_xy_coords = np.array([sampling_x,sampling_y]).T
        
        for coordinate in sampling_xy_coords:

            pass






    def build_bulk_interface(self,
                             water_substrate_gap = 2,
                             optimiser = BFGS,
                             optimise_interfacial_water=True,
                             optimise_water_before_adding=False,
                             only_freeze_substrate_bottom_half=False,
                             ):
    
        
        new_substrate = copy.deepcopy(self.substrate)
        
        water_slab = self.create_water_slab(water_thickness=self.water_thickness,
                                            density = self.water_density,
                                            z_expansion_factor=1.3,
                                            fill_using_optimisation=optimise_water_before_adding,
                                            )
        
        print("Water slab created.")

        # Shifting water to right height above the substrate
        interface_z =   ( self.intersubstrate_gap + self.substrate_thickness ) / 2 
        water_slab.positions[:,2] = water_slab.positions[:,2] - np.min(water_slab.positions[:,2]) + interface_z + water_substrate_gap
        print("Water slab interface shifted to centre of cell.")

        #Adds water to substrate
        new_substrate.extend(water_slab)
        print("Water slab added to substrate.")

        # Optimise geometry of water above substrate
        if optimise_interfacial_water:
            optimised_interface = self.optimise_interfacial_water(new_substrate,
                                                                  optimiser=optimiser,
                                                                  only_freeze_substrate_bottom_half=only_freeze_substrate_bottom_half
                                                                  )


        self.interface = optimised_interface
        print("Interface building complete.")

        return self.interface


    def get_interface(self):
        if self.interface is None:
            raise ValueError('Interface not yet built')
        return self.interface


    def shift_substrate_to_cell_centre(self,substrate):    
        """
        Returns substrate with z vals shifted such that the interface is at the middle of the cell. 
        """
        

        # Taking the substrate and creating a new object without all the tags etc:
        new_substrate = ase.Atoms(symbols=substrate.symbols,
                                    positions=substrate.positions,
                                    cell=substrate.cell,
                                    pbc=substrate.pbc)


        # Top of interface moved to z_interface = middle of cell
        if len(substrate)==0:
            pass
        else:
            all_substrate_z_vals = new_substrate.positions[:,2]
            new_substrate.positions[:,2] = new_substrate.positions[:,2] - np.max(all_substrate_z_vals) + (self.substrate_thickness + self.intersubstrate_gap)/2


        # Adjusting cell
        new_substrate.cell[2] = np.array([0,0,self.intersubstrate_gap+self.substrate_thickness])
        new_substrate.set_pbc( (True,True,True) )


        return new_substrate




    def create_water_slab(self,
                          water_thickness,
                          density =1,
                          z_expansion_factor=1.5,
                          fill_using_optimisation=False
                          ):
       

        #Create empty water slab
        empty_water_slab =ase.Atoms()
        water_box_vectors = copy.deepcopy(self.substrate.cell)
        water_box_vectors[2] = np.array([0,0,water_thickness * z_expansion_factor]) # Slightly thicker slab to allow for relaxation
        lat_vec_0=water_box_vectors[0]
        lat_vec_1=water_box_vectors[1]


        #Setting cell
        empty_water_slab.set_cell(water_box_vectors)
        empty_water_slab.pbc = (True,True,True)


        #number of water molecules for given density
        vol =  np.abs(np.dot( np.array([0,0,water_thickness]) , np.cross(lat_vec_0,lat_vec_1) ) ) 
        weight = density * 10**-1 * vol
        num = weight / 2.9915
        n_water = int(num)


        #Filling water slab
        from . import water_adder
        
        slab_builder = water_adder.WaterAdder(empty_water_slab,water_box_vectors)
        successfully_added = False
        attempts=0
        while(not successfully_added):
            n_added = slab_builder.fill_H2O(n_water,n_trials = 100000)
            if n_added == n_water:
                successfully_added = True
            else:
                attempts+=1

                # Tries to add more water after packing
                if fill_using_optimisation:
                    try:
                        water_slab = slab_builder.get_system()
                        optimised_water = self.optimise_water(water_slab)
                        slab_builder.set_system(optimised_water)
                    except:
                        print('Water slab optimisation failed during filling.')

                if attempts > 10:
                    raise ValueError('Could not fit water in slab')
    

        water_slab = slab_builder.get_system()
        return water_slab


    def optimise_water(self,
                       input_water_slab,
                       optimiser=BFGS,
                       max_steps=1000,
                       ):
        
        # Optimises water slab. 
        water = copy.deepcopy(input_water_slab)
        water.calc = self.calculator
        dyn = optimiser(water)
        convergence = dyn.run(fmax=0.1,steps=max_steps)
        if not convergence:
            raise ValueError('Water geometry optimisation did not converge')

        return water
    
        
    def optimise_interfacial_water(self,
                                   input_interface,
                                   optimiser=BFGS,
                                   max_steps=1000,
                                   only_freeze_substrate_bottom_half=False):
        

        interface = copy.deepcopy(input_interface)
        interface.calc = self.calculator


        #Freezing substrate (either whole or bottom half)
        num_substrate_atoms = len(self.substrate)
        num_interface_atoms = len(input_interface)
        substrate_atom_indices = np.arange(0,num_substrate_atoms,1) 
        interface_atom_indices = np.arange(0,num_interface_atoms,1)
        
        
        c = FixAtoms(indices=substrate_atom_indices)

        if only_freeze_substrate_bottom_half: 
            substrate_atom_z_coords = self.substrate.positions[:,2]
            interface_atom_z_coords = self.substrate.positions[:,2]
            median_z_coord = np.median(substrate_atom_z_coords)
            isin_substrate_bottom_half_mask = [coord <= median_z_coord for coord in interface_atom_z_coords]
            substrate_bottom_half_indices = interface_atom_indices[isin_substrate_bottom_half_mask]
            c = FixAtoms(indices=substrate_bottom_half_indices)
    

        #Freezing substrate
        interface.set_constraint(c)


        opt_traj_dir_name = 'optimisation_trajectories'
        if not os.path.exists(opt_traj_dir_name):
            os.makedirs(opt_traj_dir_name)

        if self.name is None:
            name = 'interface'
        else:
            name = self.name

        opt_traj_name = './' + opt_traj_dir_name + '/' +str(name) + '_new_opt.traj'

        dyn = optimiser(interface,trajectory=opt_traj_name)

        start = time.time()
        convergence = dyn.run(fmax=0.1,steps=max_steps)
        end = time.time()

        opt_traj = ase.io.read(opt_traj_name,index = ':')
        num_opt_steps = len(opt_traj)
        opt_time = end-start
        self.time_per_model_eval = opt_time / num_opt_steps


        if not convergence:
            raise ValueError('Interface geometry optimisation did not converge')

        return interface