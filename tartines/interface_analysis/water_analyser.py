import ase.io
import numpy as np
import matplotlib.pyplot as plt
from ase.visualize import view
import matplotlib.animation as animation
from collections import defaultdict
from tqdm import tqdm

from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components

class Analyser:
    
    def __init__(self,
                 frame,
                 substrate_indices=None,
                 r_OO_c = 3.5,
                 r_OH_c = 2.4,
                 theta_c = 120):
        

        self.frame = frame
        self.num_atoms = len(frame)
        self.substrate_indices = substrate_indices

        if substrate_indices is not None:
            for atom in frame:
                if atom.index in substrate_indices:
                    atom.tag = 0
                else:
                    atom.tag = 1

        self.num_water_molecules = len([atom for atom in frame if atom.symbol == 'O' and atom.tag == 1])
        if self.num_water_molecules == 0:
            raise ValueError("No water molecules found in frame. Make sure tags are correct or add substrate.")


        # 1 if atom belongs to water molecule, 0 otherwise
        self.O_indices = {index for index in np.arange(0,self.num_atoms) if frame[index].symbol == 'O'} 
        self.H_indices = {index for index in np.arange(0,self.num_atoms) if frame[index].symbol == 'H'}
        self.substrate_O_indices = {index for index in np.arange(0,self.num_atoms) if frame[index].symbol == 'O' and frame[index].tag == 0}
        self.substrate_H_indices = {index for index in np.arange(0,self.num_atoms) if frame[index].symbol == 'H' and frame[index].tag == 0}
        #Aqua denotes HO, H2O, H3O, etc. 
        self.aqua_O_indices = self.O_indices - self.substrate_O_indices
        self.aqua_H_indices = self.H_indices - self.substrate_H_indices



        self.distance_matrix= self.frame.get_all_distances(mic=True)
        
        self.voronoi_dict= None
        self.undirected_H_bond_connectivity = None
        self.directed_H_bond_connectivity = None

        self.r_OO_c = r_OO_c
        self.r_OH_c = r_OH_c
        self.theta_c = theta_c


    def get_voronoi_dict(self,O_indices=None,H_indices=None,include_substrate=False):
        #Returns dictionary of O_indices with H_indices in their voronoi region
        
        if include_substrate is None and include_substrate:
            raise ValueError("Cannot include substrate atoms if substrate_indices was not provided at initialisation.")

        if O_indices is None:
            O_indices = self.aqua_O_indices
        if H_indices is None:
            H_indices = self.aqua_H_indices


        if include_substrate:
            non_H_indices = list(self.aqua_O_indices) + list(self.substrate_indices)
            voronoi_dictionary = {i: [] for i in non_H_indices}
            for H_index in H_indices:
                distances = self.distance_matrix[H_index][non_H_indices]
                smallest = np.min(distances)
                tolerance=1e-7
                closest_atom_index = np.where( abs (self.distance_matrix[H_index] - smallest )< tolerance )[0][0]
                voronoi_dictionary[closest_atom_index].append(int(H_index))
        else:
            voronoi_dictionary = {i: [] for i in O_indices}
            for H_index in H_indices:                       
                OH_distances=self.distance_matrix[H_index][list(O_indices)] 
                
                smallest = np.min(OH_distances)
                tolerance=1e-7
                closest_O_index = np.where( abs (self.distance_matrix[H_index] - smallest )< tolerance )[0][0]
                voronoi_dictionary[closest_O_index].append(int(H_index))


        if self.get_voronoi_dict is None:
            self.voronoi_dict = voronoi_dictionary

        return voronoi_dictionary



    def find_H_species(self,O_indices=None,H_indices=None,r_OH_ion=2,r_H2_threshold=2):
        """
        Finds species like H2 and lone H (detached from water molecules).
        Lone H defined as being more than r_OH_ion away from closest O atom.
        H2 defined as two lone H atoms being within r_H2_threshold of each other.
        """

        if O_indices is None:
            O_indices = self.O_indices
        if H_indices is None:
            H_indices = self.aqua_H_indices

        if self.voronoi_dict is None:
            voronoi_dict = self.get_voronoi_dict(O_indices,H_indices)
        else:
            voronoi_dict = self.voronoi_dict


        lone_H_indices = []

        for key, value in voronoi_dict.items():
            for index in value:
                distance = self.frame.get_distances(key,index,mic=True)[0]
                if distance > r_OH_ion:
                    lone_H_indices.append(index)

        H_map = {H_index : [] for H_index in lone_H_indices}

        for H_index in lone_H_indices:
            
            for H_index_2 in [H_index_2 for H_index_2 in lone_H_indices if H_index_2>H_index]:
                distance = self.frame.get_distances(H_index,H_index_2,mic=True)[0]
                if distance < r_H2_threshold:
                    H_map[H_index].append(H_index_2)


        return H_map



    def get_dissociation_statistics(self,O_indices=None,H_indices=None,include_substrate=False):

        if O_indices is None:
            O_indices = self.aqua_O_indices

        if H_indices is None:
            H_indices = self.aqua_H_indices

        H2O_count=0
        H3O_count=0
        OH_count=0
        O_count=0
        substrate_count=0


        voronoi_dict = self.get_voronoi_dict(include_substrate=include_substrate)

        for i in self.aqua_O_indices:

            local_H_list = voronoi_dict[i]
            num_H= len(local_H_list) 

            if num_H == 0:
                O_count += 1
            elif num_H == 1:
                OH_count += 1
            elif num_H == 2:
                H2O_count += 1
            elif num_H == 3:
                H3O_count += 1
            
        for i in self.substrate_O_indices:
            local_H_list = voronoi_dict[i]
            num_H= len(local_H_list) 

            if num_H == 1:
                substrate_count += 1
        
        return {
            "H2O_count": H2O_count,
            "H3O_count": H3O_count,
            "OH_count": OH_count,
            "O_count": O_count,
            "substrate_count": substrate_count,
        }


    def get_water_dipole_moment(self,water_O_index):
        """
        Assume water_O_index belongs to water, and not another species.
        """

        if self.voronoi_dict is None:
            voronoi_dict = self.get_voronoi_dict()
        else:
            voronoi_dict = self.voronoi_dict

        H_atom_indices = voronoi_dict[water_O_index]

        if len(H_atom_indices) != 2:
            raise ValueError(f"Water molecule with O index {water_O_index} does not have 2 H atoms in its voronoi region. It has {len(H_atom_indices)} H atoms.")
        
        OH1_vec = self.frame.get_distances(water_O_index,H_atom_indices[0],mic=True,vector=True)[0]
        OH2_vec = self.frame.get_distances(water_O_index,H_atom_indices[1],mic=True,vector=True)[0]
        dipole_vector = OH1_vec + OH2_vec 

        normalised_dip_vec =  dipole_vector / np.linalg.norm(dipole_vector)
        return normalised_dip_vec
    
    def get_water_euler_angles(self,water_O_indices=None):

        if water_O_indices is None:
            water_O_indices = self.aqua_O_indices

        H_vectors = self.get_water_H_vectors(water_O_indices)

        euler_angles = {}

        for O_index in water_O_indices:
            euler_angles[O_index] = []

            dipole_moment = ( H_vectors[O_index][0] + H_vectors[O_index][1] ) / 2
            delta = H_vectors[O_index][0] - dipole_moment


            cos_pitch = np.dot(dipole_moment, [0, 0, 1]) / (np.linalg.norm(dipole_moment))
            x_vec = np.cross(np.array([0,0,1]) , dipole_moment) / np.linalg.norm(dipole_moment) 
            cos_roll = np.dot ( 
                x_vec / np.linalg.norm(x_vec),
                delta / np.linalg.norm(delta)
            ) 

            pitch_angle = np.arccos(cos_pitch)
            roll_angle = np.arccos(cos_roll)

            euler_angles[O_index] = [pitch_angle, roll_angle]

        return euler_angles

    def get_water_H_vectors(self,water_O_indices=None):
        """
        Returns dict, with each O index as a key and a list of OH vectors as the values. 
        """

        if water_O_indices is None:
            water_O_indices = self.aqua_O_indices

        if self.voronoi_dict is None:
            voronoi_dict = self.get_voronoi_dict(water_O_indices)
        else:
            voronoi_dict = self.voronoi_dict
       
        H_vectors = {}

        for water_O_index in water_O_indices:

            H_atom_indices = voronoi_dict[water_O_index]

            for H_index in H_atom_indices:
                if water_O_index not in H_vectors:
                    H_vectors[water_O_index] = []
                
                OH_vec = self.frame.get_distances(water_O_index,H_index,mic=True,vector=True)[0]
                H_vectors[water_O_index].append(OH_vec)


        return H_vectors



    """ Methods for H-bond network analyis """


    def H_bond_geometry_check(self,
                              O_D_index,
                              H_index,
                              O_A_index,
                              ):
        
        if O_D_index == O_A_index:
            return False 
        
        
        # Checks if O_D donates H bond to O_A

        v_1 = self.frame.get_distances(O_D_index,H_index,mic=True,vector=True) [0]
        v_2 = self.frame.get_distances(O_D_index,O_A_index,mic=True,vector=True) [0]
        v_3 = self.frame.get_distances(H_index,O_A_index,mic=True,vector=True) [0]
        
        r_HO = np.linalg.norm(v_3) #H - Acceptor
        r_OH = np.linalg.norm(v_1) # Donor - H
        r_OO = np.linalg.norm(v_2)

        # Returns False if O_D is not the donor
        if r_HO < r_OH:
            return False

        cos_theta = np.dot(v_3,-v_1) / (np.linalg.norm(v_1) * np.linalg.norm(v_3))


        cos_theta_c = np.cos(self.theta_c * np.pi / 180)

    
        return r_OH < self.r_OH_c and cos_theta < cos_theta_c and r_OO < self.r_OO_c

    
    
    def check_H_bond(self,
                    O_D_index,O_A_index,
                    ):
    
        
        def find_local_H_indices(O_index):
            #Returns H_index vals for protons within r_OH_c of O_index
            distances = self.distance_matrix[O_index]
            H_distances = distances[list(self.H_indices)]
            local_H_distances = H_distances[[ 1e-7 < distance and distance < self.r_OH_c for distance in H_distances]]
            local_H_indices = np.where(np.isin(distances,local_H_distances))[0]
            return set(local_H_indices)

        #Finding protons in between O atoms
        local_O_H_indices_1 = find_local_H_indices(O_D_index)
        local_O_H_indices_2 = find_local_H_indices(O_A_index)
        if len(local_O_H_indices_1) ==0 and len(local_O_H_indices_2) ==0:
            # For substrate O atoms, for example
            return False
        common_H_indices = local_O_H_indices_1 & local_O_H_indices_2
        if len(common_H_indices)==0:
            return False

        #Finding 'bonding proton' which has shortest OHO path length
        

        path_lengths= {}

        for H_index in common_H_indices:
            length_O_D = self.distance_matrix[O_D_index][H_index]
            length_O_A = self.distance_matrix[H_index][O_A_index]
            path_length = length_O_D + length_O_A
            path_lengths[H_index] = path_length
        
    
        candidate_H_index = min(path_lengths, key=path_lengths.get)


        is_H_bond = self.H_bond_geometry_check(O_D_index,candidate_H_index,O_A_index)

        
        return is_H_bond
                    


    def get_H_bond_connectivity(self,O_donors_analyse=None,O_acceptors_analyse =None,directed=True):
        
        """
        Returns connectivity matrix as a dictionary of dictionaries.
        Form is {donator_index: {acceptor_index: is_H_bond = True/False}}.
        If directed is false then connectivity matrix is symmetric. Otherwise is_H_bond is only
        True if O_donator_idex donates to O_acceptor_index.
        """


        # Specify which Os to analyse
        # By default, only water O
        if O_donors_analyse is None:
            O_donors_analyse = self.aqua_O_indices
        if O_acceptors_analyse is None:
            O_acceptors_analyse = self.aqua_O_indices

        connectivity_matrix = {}

        for O_index_i in O_donors_analyse:
            for O_index_j in O_acceptors_analyse:

                if directed:
                    # Check if O_index_i is donor and O_index_j is acceptor
                    H_bond=self.check_H_bond(O_index_i,O_index_j)
                else:
                    H_bond=self.check_H_bond(O_index_i,O_index_j) or self.check_H_bond(O_index_j,O_index_i)

                if not O_index_i in connectivity_matrix:
                    connectivity_matrix[O_index_i] = {}
                    
                connectivity_matrix[O_index_i][O_index_j] = H_bond

                if H_bond and not directed:
                    if not O_index_j in connectivity_matrix:
                        connectivity_matrix[O_index_j] = {}
                    connectivity_matrix[O_index_j][O_index_i] = H_bond

        if directed:
            if self.directed_H_bond_connectivity is None:
                self.directed_H_bond_connectivity = connectivity_matrix

        else:
            if self.undirected_H_bond_connectivity is None:
                self.undirected_H_bond_connectivity = connectivity_matrix
                

        return connectivity_matrix




    def get_H_bond_clusters(self,O_analyse=None,directed=True):

        if O_analyse is None:
            O_analyse = self.O_indices
        
        O_analyse = list(O_analyse)


        if not directed:
            if self.undirected_H_bond_connectivity is None:
                H_bond_connectivity = self.get_H_bond_connectivity(O_analyse,directed=False)
            else:
                H_bond_connectivity = self.undirected_H_bond_connectivity
        
        if directed:
            if self.directed_H_bond_connectivity is None:
                H_bond_connectivity = self.get_H_bond_connectivity(O_analyse,directed=True)
            else:
                H_bond_connectivity = self.directed_H_bond_connectivity

        

        water_indices_dict={i:O_analyse[i] for i in range(len(O_analyse))}
        water_indices = list(water_indices_dict.keys())
        
        adjacency_matrix = np.zeros((len(water_indices),len(water_indices)),dtype=int)

        for i in range(len(water_indices)):
            for j in range(len(water_indices)):
                
                if H_bond_connectivity[water_indices_dict[i]][water_indices_dict[j]] == True:
                    adjacency_matrix[i][j] = 1
                else:
                    adjacency_matrix[i][j] = 0
            


        graph = csr_array(adjacency_matrix)

        n_components, labels = connected_components(csgraph=graph, directed=directed, return_labels=True)
        
        clusters = {}

        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
        
            clusters[label].append(water_indices_dict[i])


        return clusters
    