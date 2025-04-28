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



def write_to_logfile(message):
    with open('interface_builder_log.out', 'a') as file:
        new_line = message
        file.write(new_line + "\n")  # Write the line to the file


def plot_atoms(atoms,rotation=('0x,0y,0z')):
    from ase.visualize.plot import plot_atoms

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))  

    plot_atoms(atoms, ax=axes, rotation=('0x,0y,0z'))



def interface_multi_builder(substrate_dir,water_thickness,intersubstrate_gap,water_substrate_gap,num_replicas=1,optimise_interface=True,logfile = True,check_physicality=True,enforce_physicality = False,freeze_whole_substrate=False):
    
    input_filenames = os.listdir(substrate_dir)

    with open('interface_builder.out', 'w') as file:
        pass # clears old logfile

    for filename in input_filenames:
        substrate = ase.io.read('./'+substrate_dir+'/'+filename)
        

        for replica_index in range(num_replicas):

            interface_name = filename.split('.')[0]

            print('Building',filename,'interface')
            if logfile:
                write_to_logfile("")
                write_to_logfile("Loaded "+filename )
                write_to_logfile('Building Interface')

            interface_builder = InterfaceBuilder(substrate,water_thickness,intersubstrate_gap,water_substrate_gap,name=interface_name)
            start_time = time.time()
            interface_builder.build_interface(optimise_interface=optimise_interface,enforce_physicality=enforce_physicality,freeze_whole_substrate=freeze_whole_substrate)
            end_time = time.time()

            print('Time taken:',end_time-start_time)
            if logfile:
                write_to_logfile('Time taken: '+ str(end_time-start_time))

            interface = interface_builder.get_interface()


            #Saving interface
            interface_dir_name = 'interfaces'
            if not os.path.exists(interface_dir_name):
                os.makedirs(interface_dir_name)

            name = filename.split('.')[0] +'_' + str(replica_index+1)
            ase.io.write(interface_dir_name+'/interface_'+name+'.xyz',interface,format='extxyz')
        
    return True


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
    def __init__(self,substrate,water_thickness,intersubstrate_gap,water_substrate_gap,name=None):
        
        #Error handling
        if (substrate.cell.array == 0).all():
            raise ValueError('Cell of substrate is not defined')

        self.input_substrate = copy.deepcopy(substrate)
        self.substrate = copy.deepcopy(substrate)
        self.num_substrate_atoms = len(self.substrate)
        self.calculator = mace_mp(model='/home/hr492/michaelides-share/hr492/Projects/tartine_project/data/mace_models/mace_agnesi_medium.model',  dispersion=True, device='cuda')
        self.water_thickness = water_thickness
        self.intersubstrate_gap = intersubstrate_gap
        self.water_substrate_gap = water_substrate_gap
        self.interface = None
        self.name = name
        self.time_per_model_eval = None

        #Finding substrate thickness
        if len(self.substrate)==0:
            self.substrate_thickness = 0
        else:
            all_substrate_z_vals = self.substrate.positions[:,2]
            self.substrate_thickness = np.max(all_substrate_z_vals) - np.min(all_substrate_z_vals)
            


    def build_interface(self,optimiser = BFGS, enforce_physicality = False, print_H_bond_stats = False, optimise_interface=True,optimise_slab=False,freeze_whole_substrate=False):
        # Takes input substrate and adjusts z-dimension of cell, adds PBC and shifts interface to z=0.
        self.initialise_substrate()
        print("Substrate initialised.")

        """
        Details of how it builds water slab:
        - Creates cell with interfacial water slab dimensions. Adds pbc. Creates new empty atoms object and copies positions, symbols, cell etc
        - Will attempt to fill with water such that density = 1gcm^-3. 
        - It will check to see if the correct number H2O molecules are added attempt to do this 10 times. 
        - There is also the option to optimise the geometry of this bulk water.
        """
        #Buildin water slab
        if not enforce_physicality:
            water_slab = self.create_water_slab(density = 1, z_expansion_factor=1,optimise_slab=optimise_slab)
            print("Water slab created without enforcing physicality.")
        else:
            # Will attempt to build water slab until it is physically reasonable
            attempts_for_building_water_slab = 0
            slab_build_success = False

            # Attempting to build water slab
            while(not slab_build_success):
                water_slab = self.create_water_slab(density = 1, z_expansion_factor=1,optimise_slab=optimise_slab)
                slab_build_success = is_water_coordination_physical(water_slab)
                attempts_for_building_water_slab+=1
                print(f"Attempt {attempts_for_building_water_slab} to build water slab: {'Success' if slab_build_success else 'Failure'}")
            

        #Shifting to z_interface = cell_z / 2
        interface_z =   ( self.intersubstrate_gap + self.substrate_thickness ) /2

        #Shits the positions to correct height above substrate
        water_slab.positions[:,2] = water_slab.positions[:,2] - np.min(water_slab.positions[:,2]) + interface_z + self.water_substrate_gap
        print("Water slab positions shifted.")

        #Adds water to substrate
        self.substrate.extend(water_slab)
        print("Water slab added to substrate.")

        water_indices = range (-len(water_slab),0)
        #Checking H-bond stats before optimisation
        if print_H_bond_stats:
            un_optimised_interface_water = self.substrate[water_indices]
            mean_H_bonds_per_water_pre_optimise = get_water_H_bond_stats(un_optimised_interface_water)
            print(f"Mean H-bonds per water molecule before optimisation: {mean_H_bonds_per_water_pre_optimise}")
        

        #Optimising interface
        if optimise_interface:
            if not enforce_physicality:
                self.substrate = self.optimise_interface(self.substrate,optimiser=optimiser,freeze_whole_substrate=freeze_whole_substrate)
                print("Interface optimised without enforcing physicality.")
            else:
            # If physicality is enforced, will attempt to optimise interface until it is physically reasonable
                attempts_to_optimise_interface = 0 
                interface_optimisation_success = False
                while(not interface_optimisation_success):
                    self.substrate = self.optimise_interface(self.substrate,optimiser=optimiser,freeze_whole_substrate=freeze_whole_substrate)
                    optimised_interface_water = self.substrate[water_indices]
                    interface_optimisation_success = is_water_coordination_physical(optimised_interface_water)
                    attempts_to_optimise_interface+=1
                    print(f"Attempt {attempts_to_optimise_interface} to optimise interface: {'Success' if interface_optimisation_success else 'Failure'}")

        #Checking H-bond stats after optimisation
        if print_H_bond_stats:
            optimised_interface_water = self.substrate[water_indices]
            mean_H_bonds_per_water_post_optimise = get_water_H_bond_stats(optimised_interface_water)
            print(f"Mean H-bonds per water molecule after optimisation: {mean_H_bonds_per_water_post_optimise}")

        self.interface = self.substrate
        print("Interface building complete.")

        return self.interface


    def get_interface(self):
        if self.interface is None:
            raise ValueError('Interface not yet built')
        return self.interface


    def initialise_substrate(self):    
        """
        Shifts interface to z=0. Adapts cell to correct z dimensions. Adds PBCs.
        """
        

        #Taking the substrate and creating a new object without all the tags etc:
        blank_substrate = ase.Atoms(symbols=self.substrate.symbols,
                positions=self.substrate.positions,
                cell=self.substrate.cell,
                pbc=self.substrate.pbc)
        self.substrate = blank_substrate #This ensures that the atoms file is clean


        #Top of interface moved to z_interface = middle of cell
        if len(self.substrate)==0:
            pass
        else:
            all_substrate_z_vals = self.substrate.positions[:,2]
            self.substrate.positions[:,2] = self.substrate.positions[:,2] - np.max(all_substrate_z_vals) + (self.substrate_thickness + self.intersubstrate_gap)/2


        #Adjusting cell
        self.substrate.cell[2] = np.array([0,0,self.intersubstrate_gap+self.substrate_thickness])
        self.substrate.set_pbc( (True,True,True) )

        self.input_substrate =  copy.deepcopy(self.substrate) #This is the substrate before water is added


    def create_water_slab(self,density =1, z_expansion_factor=1.5,optimise_slab=False,water_tags=False):
        #This will be an atoms object of bulk water

        #Create empty water slab
        empty_water_slab =ase.Atoms()
        water_box_vectors = copy.deepcopy(self.substrate.cell)
        water_box_vectors[2] = np.array([0,0,self.water_thickness * z_expansion_factor]) # Slightly thicker slab to allow for relaxation
        lat_vec_0=water_box_vectors[0]
        lat_vec_1=water_box_vectors[1]

        #Setting cell
        empty_water_slab.set_cell(water_box_vectors)
        empty_water_slab.pbc = (True,True,True)

        #number of water molecules for density=1
        water_density=1
        vol =  np.abs(np.dot( np.array([0,0,self.water_thickness]) , np.cross(lat_vec_0,lat_vec_1) ) ) 
        weight = water_density * 10**-1 * vol
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
                if attempts > 10:
                    raise ValueError('Could not fit water in slab')
    
        water_slab = slab_builder.get_system()

        if optimise_slab:
            # Optimising the water slab sturcture
            # This is optimising it as if it were bulk water#
            #You may have weird wrapping issues
            water_slab = self.optimise_water(water_slab)


        if water_tags:
            water_slab.set_tags(1)

        return water_slab #error here before!


    def optimise_water(self,input_water_slab,optimiser=BFGS,max_steps=1000):
        # Optimises water slab. 
        water = copy.deepcopy(input_water_slab)
        water.calc = self.calculator
        dyn = optimiser(water)
        convergence = dyn.run(fmax=0.1,steps=max_steps)
        if not convergence:
            raise ValueError('Water geometry optimisation did not converge')

        return water
    
        
    def optimise_interface(self,input_interface,optimiser=BFGS,max_steps=1000,freeze_whole_substrate=False):
        interface = copy.deepcopy(input_interface)
        interface.calc = self.calculator


        #Freezing substrate (either whole or bottom half)
        num_substrate_atoms = len(self.input_substrate)
        num_interface_atoms = len(self.substrate)
        substrate_atom_indices = np.arange(0,num_substrate_atoms,1) #Note, we have assumed that substrates were added to tartine before water
        interface_atom_indices = np.arange(0,num_interface_atoms,1)
        
        if freeze_whole_substrate: #freeze whole substrate
            c = FixAtoms(indices=substrate_atom_indices)

        else: #freeze bottom half of substrate 
            substrate_atom_z_coords = self.input_substrate.positions[:,2]
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