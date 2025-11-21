import ase.io
import numpy as np
import matplotlib.pyplot as plt
from ase.visualize import view
import matplotlib.animation as animation
from collections import defaultdict
from tqdm import tqdm


"""Module Functions"""
def plot_atoms(atoms, rotation='0x,0y,0z'):
    ig, ax = plt.subplots(1, 1, figsize=(8, 8))  
    
    # Plot the atoms with the specified rotation
    plot_atoms(atoms, ax=ax, rotation=rotation)
    
    # Display the plot
    plt.show()

def animate_trajectory(trajectory_list,color_list,animation_name,coord_indices=[0,1]):

    fig, ax = plt.subplots()
   
    scatter_artists=[] 
    num_trajectories  = len(trajectory_list)
    num_frames=len(trajectory_list[0])
 
    for frame_index in range(num_frames):
        frame_scatter_artist=[]
        for trajectory_index in range(num_trajectories):
            trajectory = trajectory_list[trajectory_index]
            coods1=trajectory[frame_index][:, coord_indices[0]]
            coords2 =trajectory[frame_index][:, coord_indices[1]]
            scatter_artist = ax.scatter(coods1, coords2, color = color_list[trajectory_index])
            frame_scatter_artist.append(scatter_artist)
        progress = frame_index / num_frames
        title = ax.text(0.5, 1.05, f't={progress:.2f}', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        scatter_artists.append(frame_scatter_artist+[title])
        

    ani = animation.ArtistAnimation(fig, scatter_artists, interval=1, blit=True)
                                
    
    ani.save(filename=animation_name+".gif", fps=60,writer="pillow")
    print('success')


def get_simulation_aggregated_positions(analyser,indices):
    positions= analyser.get_positions(indices)
    x=np.array([])
    y=np.array([])
    z=np.array([])
    for i in range(analyser.num_frames):
        x=np.append(x, positions[i][:,0])
        y=np.append(y, positions[i][:,1])
        z=np.append(z ,positions[i][:,2]) 
    return x,y,z
        


def histogram_to_plot(data, bins=10, range=None, density=False, **kwargs):
    """
    Convert np.histogram data to a plot using plt.plot.

    Parameters:
    - data: array-like, the input data to be histogrammed.
    - bins: int or sequence, optional, number of bins or bin edges.
    - range: tuple, optional, the lower and upper range of the bins.
    - density: bool, optional, if True, the result is the value of the probability density function.
    - kwargs: additional keyword arguments for plt.plot.

    Returns:
    - None
    """
    # Generate histogram data
    counts, bin_edges = np.histogram(data, bins=bins, range=range, density=density)

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot data
    plt.plot(bin_centers, counts, **kwargs)
    plt.xlabel('Bin Centers')
    plt.ylabel('Counts')
    plt.title('Histogram to Plot')
    plt.show()



def coordinate_wrap(atoms):
    return atoms.wrap()


class Analyser:
    
    """
    Class for analysing trajectories of water systems. We assume that all O and H belong to water molecules or related species (e.g. hydronium, hydroxyl).


    Attributes
    ----------

    trajectory : ase.Atoms object
        Trajectory of simulation.
    num_atoms : int 
        Number of atoms in simulation.    
    num_frames : int
        Number of frames in simulation.
    cell : np.array
        Cell of simulation.
    O_indices : np.array 
        Indices of O atoms in each frame of simulation. 
    H_indices : np.array
        Indices of H atoms in simulation.
    water_O_indices : np.array
        Indices of water O atoms in simulation.
    voronoi_dicts : list
        List of dictionaries of O_indices with H_indices in their voronoi region.
    free_proton_indices : np.array
        Indices of H atoms which are furthest away from O in hydronium.
    hydronium_O_indices : np.array
        Indices of hydronium O atoms.
    framework_hydroxyl_O_indices : np.array
        Indices of framework hydroxyl O atoms.
    coordinate_transform_function : function
        Function to transform coordinates.


    Methods
    ---------

    find_voronoi_dict(frame_index,distance_matrix=None)
        Returns dictionary of O_indices with H_indices in their voronoi region.
    find_hydronium_O_indices(frame_index,frame_voronoi_dict,distance_matrix=None)
        Returns indices of hydronium O atoms.
    find_free_proton_indices(frame_index,frame_voronoi_dict,frame_hydronium_O_indices,distance_matrix=None)
        Returns indices of H atoms which are furthest away from O in hydronium.
    find_neighbours(atom_index, r_cutoff,frame_index=0, distance_matrix=None)
        Returns indices of atoms within a certain distance of a given atom.
    H_bond_geometry_check(frame_index,O_D_index,H_index,O_A_index,r_OO_c = 3.5, r_OH_c = 2.4, theta_c = 30)
        Takes indices of O_donor , H , O_acceptor and tells you whether the 3 have the geometry of a H bond.
    is_H_bonded(frame_index,O_index_1,O_index_2,distance_matrix=None, r_OO_c = 3.5 , r_OH_c = 2.4, theta_c = 30)
        Returns whether two O atoms are H bonded.
    get_H_bond_connectivity(frame_index,distance_matrix=None,r_OO_c = 3.5 , r_OH_c = 2.4, theta_c = 30)
        Returns connectivity matrix of H-bonded O atoms.
    get_H_bond_clusters(frame_index,distance_matrix=None,H_bond_connectivity=None,r_OO_c = 3.5 , r_OH_c = 2.4, theta_c = 30)
        Returns clusters of H-bonded O atoms.
    get_positions(atom_indices,frame_indices=None)
        Returns positions of atoms in simulation.
    
    """




    def __init__(self,input_trajectory,stride=1,coordinate_transform_function=None):
        
        """Initialisation"""
        self.coordinate_transform_function=coordinate_transform_function #Acted on each frame to transform coords
        self.trajectory = input_trajectory.copy()[::stride]
        self.num_atoms = len( self.trajectory[0] ) 
        self.num_frames = len(self.trajectory)
        self.cell=self.trajectory[0].cell

        """Transforming coordinates"""

        if self.coordinate_transform_function is not None:
            for frame in self.trajectory:
                frame.wrap()
                new_positions=[]
                for atom in frame:
                    position = atom.position
                    new_positions.append(coordinate_transform_function(position,self.cell))
                frame.positions=new_positions
        
        """Finding indices of important species""" 
        self.O_indices = np.arange(0,self.num_atoms,1)[[atom.symbol == 'O' for atom in self.trajectory[0]]]
        self.H_indices = np.arange(0,self.num_atoms,1)[[atom.symbol == 'H' for atom in self.trajectory[0]]]

        self.water_O_indices = self.O_indices #Actually means water, hydroxyl and hydronium O atoms

        
        #Finds H atoms closest to each O atom (given by voronoi dicts)
        #Uses this to find hydroxyl, and hydronium species
        self.voronoi_dicts=  []
        self.free_proton_indices = [] #Means H which is furthest away from O in hydronium 
        self.hydronium_O_indices = []
        #self.hydroxyl_O_indices = []    #need to implement


    
        

        for frame_index in tqdm(range(len(self.trajectory)), desc="Processing Frames"):
            distance_matrix= self.trajectory[frame_index].get_all_distances(mic=True)
            
            #Finding voronoi dict & species indices
            frame_voronoi_dict = self.find_voronoi_dict(frame_index,distance_matrix)
            frame_hydronium_O_indices = self.find_hydronium_O_indices(frame_index,frame_voronoi_dict,distance_matrix)
            frame_free_proton_indices= self.find_free_proton_indices(frame_index,frame_voronoi_dict,frame_hydronium_O_indices,distance_matrix)

            #Adding species indices to lists    
            self.voronoi_dicts.append(frame_voronoi_dict)
            self.free_proton_indices.append(frame_free_proton_indices)
            self.hydronium_O_indices.append(frame_hydronium_O_indices)


        #Turning lists into numpy arrays
        self.free_proton_indices = np.array(self.free_proton_indices)
        self.hydronium_O_indices = np.array(self.hydronium_O_indices,dtype=object)


    def find_voronoi_dict(self,frame_index,distance_matrix=None):
        #Returns dictionary of O_indices with H_indices in their voronoi region
        if distance_matrix is None:
            distance_matrix= self.trajectory[frame_index].get_all_distances(mic=True)
        
        voronoi_dictionary = {i: [] for i in self.O_indices}
        
        for H_index in self.H_indices:                       
            distances=distance_matrix[H_index][self.O_indices]
            
            smallest = np.min(distances)
            tolerance=1e-7
            closest_O_index = np.where( abs (distance_matrix[H_index] - smallest )< tolerance )[0][0]
            voronoi_dictionary[closest_O_index].append(int(H_index))

        return voronoi_dictionary


    def find_hydronium_O_indices(self,frame_index,voronoi_dict,distance_matrix=None):
        hydronium_mask = [ len(voronoi_dict[i]) > 2 for i in self.water_O_indices ]
        hydronium_O_indices = self.water_O_indices[hydronium_mask]
        return hydronium_O_indices

    
    def find_free_proton_indices(self,frame_index,voronoi_dict,hydronium_O_indices,distance_matrix=None):
        if distance_matrix is None:
            distance_matrix= self.trajectory[frame_index].get_all_distances(mic=True)


        #Attach furtherst H in voronoi region of each O
        hydronium_protons = np.array([])
        for i in hydronium_O_indices:
            H_indices = voronoi_dict[i]
            distances = [distance_matrix[i][j] for j in H_indices]
            largest_distance = max(distances)
            tolerance=1e-7
            furthest_H_index = np.where( abs (distance_matrix[i] - largest_distance )< tolerance )[0][0]
            hydronium_protons =np.append(hydronium_protons,furthest_H_index)
        
        return hydronium_protons


    def find_neighbours(self, atom_index, r_cutoff,frame_index=0, distance_matrix=None):

        if distance_matrix is None:
            distance_matrix = self.trajectory[frame_index].get_all_distances(mic=True)
        
        neighbours = np.intersect1d(
            np.where(distance_matrix[atom_index] < r_cutoff),
            np.where(distance_matrix[atom_index] != 0)
        )
        return neighbours

            

    
    """ Methods for H-bond network analyis """


    def H_bond_geometry_check(self,frame_index,O_D_index,H_index,O_A_index,r_OO_c = 3.5, r_OH_c = 2.4, theta_c = 30):
        """Takes indices of O_donor , H , O_acceptor and tells you whether the 3 have the geometry of a H bond """

        v_1 = self.trajectory[frame_index].get_distances(O_D_index,H_index,mic=True,vector=True) [0]
        v_2 = self.trajectory[frame_index].get_distances(O_D_index,O_A_index,mic=True,vector=True) [0]
        v_3 = self.trajectory[frame_index].get_distances(H_index,O_A_index,mic=True,vector=True) [0]
        
        r_OH = np.linalg.norm(v_3)
        r_OO = np.linalg.norm(v_2)
        cos_theta = np.dot(v_2,v_1) / (np.linalg.norm(v_1) * np.linalg.norm(v_2))
        cos_theta_c = np.cos(30 * np.pi / 180)
    
        return r_OH < r_OH_c and cos_theta > cos_theta_c and r_OO<r_OO_c

    
    
    def is_H_bonded(self,frame_index,O_index_1,O_index_2,distance_matrix=None, r_OO_c = 3.5 , r_OH_c = 2.4, theta_c = 30):
        #Note, O_1 is donor and O_2 is acceptor
        if distance_matrix is None:
            distance_matrix = self.trajectory[frame_index].get_all_distances(mic=True)

        
        def find_local_H_indices(O_index):
            distances = distance_matrix[O_index]
            H_distances = distances[self.H_indices]
            local_H_distances = H_distances[[ 1e-7 < distance and distance < r_OH_c for distance in H_distances]]
            local_H_indices = np.where(np.isin(distances,local_H_distances))[0]
            return local_H_indices

        #Finding protons in between O atoms
        local_O_H_indices_1 = find_local_H_indices(O_index_1)
        local_O_H_indices_2 = find_local_H_indices(O_index_2)
        
        if len(local_O_H_indices_1) ==0 or len(local_O_H_indices_2) ==0:
            return False
        common_H_indices = np.intersect1d(local_O_H_indices_1,local_O_H_indices_2)
        if len(common_H_indices)==0:
            return False

        #Finding 'bonding proton' which has shortest OHO path length
        path_lengths= []

        for H_index in common_H_indices:
            common_H_pos=self.trajectory[frame_index][H_index].position
            path_length = distance_matrix[O_index_1][H_index] + distance_matrix[H_index][O_index_2]
            path_lengths.append(path_length)
   
        smallest_path_length = min(path_lengths)
        candidate_H_index = common_H_indices[ np.where( abs(path_lengths - smallest_path_length)<1e-7 )[0][0] ]       
        
        is_H_bond = self.H_bond_geometry_check(frame_index,O_index_1,candidate_H_index,O_index_2,r_OO_c,r_OH_c,theta_c)

        
        return is_H_bond
                    

    def get_H_bond_connectivity(self,frame_index,distance_matrix=None,r_OO_c = 3.5 , r_OH_c = 2.4, theta_c = 30):
        
        if distance_matrix is None:
            distance_matrix = self.trajectory[frame_index].get_all_distances(mic=True)
                            
        num_Os=len(self.O_indices)
        connectivity_matrix=np.full((num_Os, num_Os), False, dtype=bool)
        for i in range(num_Os):
            for j in range(i):
                O_index_i=self.O_indices[i]
                O_index_j=self.O_indices[j]
                H_bond=self.is_H_bonded(frame_index,O_index_i,O_index_j,distance_matrix,r_OO_c , r_OH_c , theta_c )
                if H_bond:
                    connectivity_matrix[i][j] = True
                    connectivity_matrix[j][i] = connectivity_matrix[i][j]
                    
        return connectivity_matrix
    

    def get_H_bond_clusters(self,frame_index,distance_matrix=None,H_bond_connectivity=None,r_OO_c = 3.5 , r_OH_c = 2.4, theta_c = 30):

        clusters = []
        unassigned = self.O_indices
        
        def assign_to_cluster(index):
            """Returns whether index is part of a cluster"""
    
            
            nonlocal new_cluster
            nonlocal unassigned

            #removing index from unassigned
            location_of_index=np.where(unassigned==index)[0]
            unassigned = np.delete(unassigned,location_of_index)
            
            selection_index = np.where(self.O_indices == index)[0][0] 
            neighbours_mask = H_bond_connectivity[selection_index]
            
  
            if sum(neighbours_mask) == 0:
   
                return False

            
            neighbours= self.O_indices[neighbours_mask]
   
            new_cluster=np.append(new_cluster,index)
         
            #Removing index from unassigned lis
            
            #selection index is index in list of O indices (as opposed to list of all atoms)
            unassigned_neighbours = np.intersect1d(neighbours,unassigned)
            for unassigned_neighbour_index in unassigned_neighbours:
                assign_to_cluster(unassigned_neighbour_index)
            return True

    
        if distance_matrix is None:
            distance_matrix = self.trajectory[frame_index].get_all_distances(mic=True)
        
        if H_bond_connectivity is None:
            H_bond_connectivity = self.get_H_bond_connectivity(frame_index,distance_matrix,r_OO_c , r_OH_c , theta_c  )
        

       
        while len(unassigned)>0:
     
                
            O_index = unassigned[0]
            
            new_cluster =np.array([])
            belongs_to_cluster = assign_to_cluster(O_index)
        
            
            if belongs_to_cluster: 
                clusters.append(new_cluster)
            else:
                continue
            
        return clusters

    
    
    """Getters"""  

    def get_positions(self,atom_indices,frame_indices=None):
        
        
        if frame_indices is None:
            frame_indices=np.arange(0,self.num_frames,1) #if no frame indices specified, assume whole trajectory
            

        
        positions=[]
        for frame_index in frame_indices:
            if np.issubdtype(atom_indices.dtype, np.integer):
                #Indended for cases where indices are constant throughout sim
                #Be careful, if inhomogeneous, indices will be 1-dim with N-dim elements
                frame_atom_indices=atom_indices
            else:
                frame_atom_indices=atom_indices[frame_index].astype(int)
            for atom_index in frame_atom_indices:
                frame_positions = np.array([ self.trajectory[frame_index][atom_index].position for atom_index in frame_atom_indices] )
            positions.append(frame_positions)

        return np.array(positions,dtype=object)
         