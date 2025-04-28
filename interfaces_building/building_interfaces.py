import ase
import ase.io
import numpy as np
import copy
import math
import os
import time
import matplotlib.pyplot as plt
import importlib 
import pandas as pd 

from ase.constraints import FixAtoms
from mace.calculators.foundations_models import mace_mp
from ase.optimize import BFGS
from ase.optimize import FIRE
import ase.data
from ase.constraints import StrainFilter



#Adding my modules
import sys
import interface_builder
import water_adder
import substrate_builder

importlib.reload(interface_builder)
importlib.reload(water_adder)
importlib.reload(substrate_builder)


#Section: Building the substrates


substrate_builder.metal_substrate_multi_builder('metals.csv', adjustment_index =3)


#Section: Visualizing the substrates

from ase.visualize.plot import plot_atoms

substrates_dir = './substrates'
input_filenames = os.listdir(substrates_dir)
#Creating directory to save snapshots
directory = "substrate_snapshots"
if not os.path.exists(directory):
    os.makedirs(directory)

for filename in input_filenames:
    substrate = ase.io.read('./'+substrates_dir+'/'+filename)
    name = filename.split('.')[-2]
    print('')
    print('analyzing', name)
    print('cell',substrate.cell ) 


    fig, axes = plt.subplots(1, 2, figsize=(8, 8))  

    plot_atoms(substrate, ax=axes[0], rotation=('-90x,0y,0z'))
    plot_atoms(substrate, ax=axes[1], rotation=('0x,0y,0z'))

    plt.tight_layout()  # Adjust spacing to prevent overlap
    plt.title(name+ ' interface')
    plt.savefig('substrate_snapshots/'+name+'.png')



#Section: Adding water to the interface

water_thickness= 15
intersubstrate_gap=50
water_substrate_gap = 2
substrate_dir = 'substrates'
optimise_interface= True #Optimize the entire interface (as opposed to just the water in the slab)
freeze_whole_substrate = True #When optimising interface, should the whole substrate be frozen?
enforce_physicality = False 
logfile = False 

interface_builder.interface_multi_builder(substrate_dir,water_thickness,intersubstrate_gap,water_substrate_gap,optimise_interface=optimise_interface,logfile = logfile,enforce_physicality = enforce_physicality,freeze_whole_substrate=freeze_whole_substrate)


#Section: Visualizing the interfaces

from ase.visualize.plot import plot_atoms

interfaces_dir = './interfaces'
input_filenames = os.listdir(interfaces_dir)
#Creating directory to save snapshots
directory = "interface_snapshots"
if not os.path.exists(directory):
    os.makedirs(directory)

for filename in input_filenames:
    name = filename.split('.')[-2]
    print(name)
    print(filename)
    interface = ase.io.read('./'+interfaces_dir+'/'+filename)
    
    print('')
    print('analyzing', name)
    print('cell',interface.cell ) 


    fig, axes = plt.subplots(1, 2, figsize=(8, 8))  

    plot_atoms(interface, ax=axes[0], rotation=('-90x,0y,0z'))
    plot_atoms(interface, ax=axes[1], rotation=('0x,0y,0z'))

    plt.tight_layout()  # Adjust spacing to prevent overlap
    plt.title(name+ ' interface')
    plt.savefig('interface_snapshots/'+name+'.png')