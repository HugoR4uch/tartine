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



def bulk_interface_multi_builder(
                            model_path,
                            substrate_dir,
                            intersubstrate_gap,
                            water_substrate_gap,
                            num_replicas=1,
                            thickness = {},
                            density = {},
                            optimise_interfacial_water=True,
                            substrates_whitelist=None,
                            substrates_blacklist=None,
                            ):
    
    input_filenames = os.listdir(substrate_dir)


    for filename in input_filenames:


        interface_name = filename.split('.')[0]

        if substrates_whitelist is not None:
            if interface_name not in substrates_whitelist:
                print('Skipping',interface_name,'as it is not in the whitelist')
                continue
        if substrates_blacklist is not None:
            if interface_name in substrates_blacklist:
                print('Skipping',interface_name,'as it is in the blacklist')
                continue

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


        

            interface_builder = InterfaceBuilder(model_path,
                                                 substrate,
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
                 model_path,
                 substrate,
                 water_thickness,
                 water_density,
                 intersubstrate_gap,
                 name=None):
        

        self.model_path = model_path
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


        #Finding substrate thickness
        if len(self.substrate)==0:
            self.substrate_thickness = 0
        else:
            all_substrate_z_vals = self.substrate.positions[:,2]
            self.substrate_thickness = np.max(all_substrate_z_vals) - np.min(all_substrate_z_vals)
            
        

        self.substrate = self.shift_substrate_to_cell_centre(substrate)




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
                       max_steps=5000,
                       ):
        
        # Optimises water slab. 
        water = copy.deepcopy(input_water_slab)
        water.calc = MACECalculator(model_path=self.model_path,device='cuda')
        dyn = optimiser(water)
        convergence = dyn.run(fmax=0.05,steps=max_steps)
        if not convergence:
            raise ValueError('Water geometry optimisation did not converge')

        return water
    
        
    def optimise_interfacial_water(self,
                                   input_interface,
                                   optimiser=BFGS,
                                   max_steps=1000,
                                   only_freeze_substrate_bottom_half=False):
        

        interface = copy.deepcopy(input_interface)
        interface.calc = MACECalculator(model_path=self.model_path,device='cuda')


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
        convergence = dyn.run(fmax=0.05,steps=max_steps)
        end = time.time()

        opt_traj = ase.io.read(opt_traj_name,index = ':')
        num_opt_steps = len(opt_traj)
        opt_time = end-start
        self.time_per_model_eval = opt_time / num_opt_steps


        if not convergence:
            raise ValueError('Interface geometry optimisation did not converge')

        return interface