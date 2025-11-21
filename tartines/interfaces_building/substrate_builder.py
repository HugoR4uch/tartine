import ase
import ase.io
import numpy as np
import pandas as pd
import os 
import time
import importlib


def existing_substrate_multi_builder(input_directory,l_x,l_y,num_layers):
    

    input_filenames = os.listdir(input_directory)

    for filename in input_filenames:
        substrate = ase.io.read('./'+input_directory +'/'+ filename)
        print()
        print('Loaded','./'+input_directory +'/'+ filename)
        print('Building Substrate')
        start_time = time.time()
        substrate = existing_substrate_builder(substrate,l_x,l_y,num_layers)
        end_time = time.time()
        print('Time taken:',end_time-start_time)
        name = filename.split('.')[0]
        if not os.path.exists('./substrates'):
            os.makedirs('./substrates')

        ase.io.write('./substrates/'+name+'.pdb',substrate,format='proteindatabank')
    
    return



def existing_substrate_builder(input_substrate,l_x,l_y,num_layers,output_filename=None):
    """
    Function to build a substrate from an existing pdb file by expanding it until it fits the desired dimensions.    

    Parameters:
    -----------
    input_filename (str): The name of the pdb file containing the substrate. This substrate must have a cell.
    l_x (float): The desired length of the substrate in the x-direction.
    l_y (float): The desired length of the substrate in the y-direction.
    num_layers (int): The number of layers in the substrate.
    output_filename (str): The name of the file (without extension) to which the substrate will be written. If None, the substrate will not be written to a file.

    Returns:
    --------
    substrate: (ase.Atoms) The built substrate with the desired dimensions.
    """

    
    #Finding correct dimensions for the supercell
    init_substrate_l_x= np.linalg.norm( input_substrate.get_cell()[0] ) 
    init_substrate_l_y= np.linalg.norm( input_substrate.get_cell()[1] ) 
    
    #Repeating unit cell to get desired dimensions
    n_x = round (l_x / init_substrate_l_x )
    n_y = round (l_y / init_substrate_l_y)
    substrate= input_substrate.repeat((n_x , n_y , num_layers )) 

    if output_filename is None:
        return substrate
    else:
        # Creating substrates directory (if it doesn't already exist)

        directory = 'substrates'
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        ase.io.write('substrates/'+output_filename+'.pdb', substrate, format='proteindatabank')
        #saved as pdb, as you need to specify the cell dimensions in the file.
        return substrate
    

def metal_substrate_multi_builder(input_filename='metals.csv',adjustment_index=2):
    
    #Importing all possible metal build functions
    from ase.build import fcc111
    from ase.build import fcc100
    from ase.build import fcc110
    from ase.build import hcp0001
    
    builder_func_dict = {
    'fcc111': fcc111,
    'fcc100': fcc100,
    'fcc110': fcc110,
    'hcp0001': hcp0001
    }
    
    #Reading data
    df = pd.read_csv(input_filename)
    data = df.to_numpy().transpose()

    names = data[0].astype(str)
    elements = data[1].astype(str)
    builders = data[2].astype(str)
    lattice_params_a = data[3].astype(float)
    lattice_params_b = data[4].astype(float)
    l_x_vals = data[5].astype(float)
    l_y_vals = data[6].astype(float)
    num_layer_vals = data[7].astype(int)


    # Building substrates
    for i,name in enumerate(names):
        element = elements[i]
        interface_build_func = builder_func_dict.get(builders[i])
        l_x = l_x_vals[i]
        l_y = l_y_vals[i]
        num_layers = num_layer_vals[i]
        if np.isnan(lattice_params_b[i]):
            lattice_params = lattice_params_a[i]
        else:
            lattice_params = [lattice_params_a[i],lattice_params_b[i]]

        substrate = metal_substrate_builder(element,l_x,l_y,num_layers,interface_build_func,lattice_params,name=name,adjustment_index=adjustment_index)

        cell = substrate.cell

        print('Build substrate for  '+name+' with cell: ',cell)

    return substrate



def metal_substrate_builder(element,l_x,l_y,num_slab_layers,interface_build_func,lattice_params,name=None,adjustment_index=2,printing=False):
    """
    Builds a metal substrate by creating a small initial substrate and then repeating it to achieve the desired dimensions.
    
    Parameters:
    -----------
    element (str): The element to be used in the substrate.
    l_x (float): The desired length of the substrate in the x-direction.
    l_y (float): The desired length of the substrate in the y-direction.
    num_slab_layers (int): The number of slab layers in the substrate.
    interface_build_func (function): A function that builds the initial substrate.
    lattice_params (tuple): The lattice parameters required by the interface_build_func.
    name (str): The name of the file (without extension) to which the substrate will be written. If None, the substrate will not be written to a file.
    search_space (int): The number of extra atoms it will add or subtract in order to find the precsise length
    
    Returns:
    --------
    substrate: (ase.Atoms) The built metal substrate with the desired dimensions. 

    NOTE: The function assumes that the substrate is orthogonal.
    NOTE: The z dimension of the substrate has not been set.
    NOTE: name is optional. If not provided, will not write to file. Must not include extension. Automatically saved as pdb file (so that cell dimensions are stored). 
    """

    # Ensure lattice_params is a list
    if not isinstance(lattice_params, (list, tuple)):
        lattice_params = [lattice_params]


    #Building small substrate which we will tile
    initial_substrate = interface_build_func(element, (2,2,num_slab_layers), *lattice_params,orthogonal=True, vacuum=1) #x=2,y=2 is arbitrary (as is vacuum size)- chosen to be smallish
        
    #Finding correct dimensions for the supercell
    init_substrate_l_x= np.linalg.norm( initial_substrate.get_cell()[0] ) 
    init_substrate_l_y= np.linalg.norm( initial_substrate.get_cell()[1] ) 


    #Repeating unit cell to get desired dimensions
    n_x = round (l_x / init_substrate_l_x )
    n_y = round (l_y / init_substrate_l_y)
    substrate= initial_substrate.repeat((n_x , n_y , 1 )) 

    #Adjusting length until correct size found
    for dn_x in range(-adjustment_index,adjustment_index+1):
        for dn_y in range(-adjustment_index,adjustment_index+1):
            if (2*n_y+dn_y) % 2 != 0:
                pass #for some reason n_y can't be even in ase
            else:
                trial_substrate = interface_build_func(element, (2*n_x+dn_x,2*n_y+dn_y,num_slab_layers), *lattice_params,orthogonal=True, vacuum=1) #2*nx ... becuase nx was number of repates of 2x2 cell
                trial_substrate_l_x= np.linalg.norm( trial_substrate.get_cell()[0] ) 
                trial_substrate_l_y= np.linalg.norm( trial_substrate.get_cell()[1] ) 
                substrate_l_x= np.linalg.norm( substrate.get_cell()[0] ) 
                substrate_l_y= np.linalg.norm( substrate.get_cell()[1] ) 


                trial_lx_improvement = abs(trial_substrate_l_x - l_x) <= abs(substrate_l_x - l_x)
                trial_ly_improvement = abs(trial_substrate_l_y - l_y) <= abs(substrate_l_y - l_y)
                if printing:
                    print('try with',2*n_x+dn_x,2*n_y+dn_y)
                    print('x error',substrate_l_x - l_x,trial_substrate_l_x-l_x  )
                    print('y error',substrate_l_y - l_y,trial_substrate_l_y-l_y)
                    print('x val',trial_substrate_l_x)
                    print('y val',trial_substrate_l_y)
                if trial_lx_improvement and trial_ly_improvement:
                    substrate = trial_substrate
            

    if name is None:
        return substrate
    else:
        # Creating substrates directory (if it doesn't already exist)

        directory = 'substrates'
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        ase.io.write('substrates/'+name+'.pdb', substrate, format='proteindatabank')
        #saved as pdb, as you need to specify the cell dimensions in the file.
        return substrate
    




