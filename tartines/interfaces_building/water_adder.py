import ase
import ase.io
import numpy as np
import copy


class WaterAdder:

    
    def __init__(self,init_structure,water_box_vectors):
        #mindist r_.. values taken from Xavi's code
        self.structure = copy.deepcopy(init_structure)
        #Adding tags to atoms
        self.structure.set_tags(np.arange(0,len(self.structure),1))

        self.water_box_vectors = water_box_vectors # ndarray of 3 arrays. each array -> water box primitive vectors
    def remove_atoms(self,indices):
        del self.structure[indices]
    
    def set_system(self,new_system):
        self.structure=new_system
        
    def get_system(self):
        return self.structure
    
    def find_neighbours(self, atom_index, r_cutoff):
        """
        Find the indices of the neighboring atoms within a specified distance cutoff. (Includes MIC).

        Parameters:
        - atom_index (int): The index of the atom for which to find neighbors.
        - r_cutoff (float): The distance cutoff for determining neighbors.

        Returns:
        - neighbours (numpy.ndarray): An array of indices representing the neighboring atoms.

        """
        neighbours = np.intersect1d(
            np.where(self.structure.get_all_distances(mic=True)[atom_index] < r_cutoff),
            np.where(self.structure.get_all_distances(mic=True)[atom_index] != 0)
        )
        return neighbours
    
      
    def atom_selector(self,atom_list,atom):
        """
        Outputs np.array of subset of indices from np.array 'atom_list' of type 'atom'.
        """
        filtered_list = atom_list[self.structure[atom_list].symbols==atom]
        return filtered_list
        
    
    def add_H2O(self,O_pos,H_pos=False,molecule_index=1):
        """
        Params
        ------
        -H_pos: list of numpy arrays of both H positions 
        -O_pos: O atom position, numpy array
        """

        if(H_pos!=False):
            H2O_disp = [H_pos[0],H_pos[1],np.array([0,0,0])] 
        else:
            H2O_r = 0.957 #OH bond length
            H2O_angle = 37.75/180 * np.pi #angle between x-axis and O-H displacemnet 
            H2O_disp = [H2O_r*np.array([-np.cos(H2O_angle),np.sin(H2O_angle),0]),
                        H2O_r*np.array([np.cos(H2O_angle),np.sin(H2O_angle),0]),
                        np.array([0,0,0])]

        water = ase.Atoms('H2O',positions = H2O_disp,tags=[molecule_index,molecule_index,molecule_index])
        euler_angles = np.random.rand(3) * 180 * [2,1,2]
        water.euler_rotate(euler_angles[0],euler_angles[1],euler_angles[2]) 
        new_pos = water.get_positions() + [ O_pos , O_pos , O_pos ]
        water.set_positions(new_pos)
        self.structure.extend(water)



    def add_OH(self,O_pos,H_pos=False,molecule_index=1):
        """
        Params
        ------
        -H_pos: list of numpy arrays of H position relative to O 
        -O_pos: O atom position, numpy array
        """

        if(H_pos!=False):
            HO_disp = [H_pos,np.array([0,0,0])] 
        else:
            oh_distance = 0.97
            HO_disp = [oh_distance*np.array([0,0,1]),
                        np.array([0,0,0])]

        OH = ase.Atoms('HO',positions = HO_disp,tags=[molecule_index,molecule_index])
        euler_angles = np.random.rand(3) * 180 * [2,1,2]
        OH.euler_rotate(euler_angles[0],euler_angles[1],euler_angles[2]) 
        new_pos = OH.get_positions() + [ O_pos , O_pos ]
        OH.set_positions(new_pos)
        self.structure.extend(OH)



    def add_H3O(self,O_pos,H_pos=False,molecule_index=1):
        """
        Params
        ------
        -H_pos: list of numpy arrays of H position relative to O 
        -O_pos: O atom position, numpy array
        """


        oh_distance = 0.98
        hoh_angle = 113.0
        gamma = np.deg2rad(hoh_angle)
        theta_deg = 74.34  
        theta = np.deg2rad(theta_deg)
        z = oh_distance * np.cos(theta)
        rho = oh_distance * np.sin(theta)
   

        if(H_pos!=False):
            HO_disp = [H_pos[0],H_pos[1],H_pos[2],np.array([0,0,0])] 
        else:
            HO_disp = []
            for k in range(3):
                phi = 2.0 * np.pi * k / 3.0      # 0°, 120°, 240°
                x = rho * np.cos(phi)
                y = rho * np.sin(phi)
                HO_disp.append(np.array([x, y, z]))

        HO_disp.append(np.array([0,0,0]))

        H3O = ase.Atoms('H3O',positions = HO_disp,tags=[molecule_index,molecule_index,molecule_index,molecule_index])
        euler_angles = np.random.rand(3) * 180 * [2,1,2]
        H3O.euler_rotate(euler_angles[0],euler_angles[1],euler_angles[2]) 
        new_pos = H3O.get_positions() + [ O_pos , O_pos , O_pos , O_pos]
        H3O.set_positions(new_pos)
        self.structure.extend(H3O)


                
    def fill_H2O(self,n_add,n_trials,r_cutoff=3,printing=False,r_OO=2.5,r_OH=1.5,r_HH=1.7,z_min=0):
        
        #z_min as an AD HOC addition to prevent waters from being added too close to the surface

        element_to_index = {'O':0 , 'H':1}
        cuttoff_matrix = np.array([ [r_OO , r_OH] , [r_OH , r_HH] ] )
        
        if n_add == 0 :
            return
        
        n_attempts=0
        n_added=0
        filling=True
            

        while(filling):
            success=False
            
            atoms=self.get_system()
            rand_coefs=np.random.rand(3)[:, np.newaxis]
            trial_point= np.sum( rand_coefs * self.water_box_vectors[:],axis=0)
            self.add_H2O(trial_point)
            


            O_index=len(atoms)-1
            H1_index=len(atoms)-2
            H2_index=len(atoms)-3


            water_positions = atoms.get_positions()[[O_index,H1_index,H2_index]]
            if water_positions[:,2].min() < z_min:
                self.remove_atoms([-1,-2,-3])
                continue

            #'Neighbours' are indices of atoms withn r_cutoff radius of the trial O atom
            if len(atoms) == 3:
                success = True
            else:
                trial_point_distances=atoms.get_distances(O_index,list(range(0,H2_index)), mic=True, vector=False)
                neighbours=np.array(range(0,H2_index))[trial_point_distances<r_cutoff]
                if(len(neighbours)==0):
                    success=True
                else:
                    
                    O_distances=atoms.get_distances(O_index,neighbours, mic=True, vector=False)
                    H1_distances=atoms.get_distances(H1_index,neighbours, mic=True, vector=False)
                    H2_distances=atoms.get_distances(H2_index,neighbours, mic=True, vector=False) 
                    
                    for neighbour_index in neighbours:
                        neighbour_element = atoms[neighbour_index].symbol
                        neighbour_element_index = element_to_index[neighbour_element]
                        #below are r_O and r_H: the smallest distances that this neighbour can have to an O or H atom
                        r_O = cuttoff_matrix[neighbour_element_index][0]
                        r_H = cuttoff_matrix[neighbour_element_index][1]
    
                        too_close = 0
                        too_close +=  np.sum([ distance < r_O for distance in O_distances])
                        too_close +=  np.sum([ distance < r_H for distance in H1_distances])
                        too_close += np.sum([ distance < r_H for distance in H2_distances])

                
                        if  too_close > 0:
                            success = False
                        
                        else:
                            success = True
                
            if(not success):
                self.remove_atoms([-1,-2,-3]) # removes the 3 water atoms we tried to add, but which didn't fit
            else:
                n_added+=1
                if(printing):
                    print(n_added," have now been added, after ",n_attempts," trials")
                if(n_added==n_add):
                    filling=False

            n_attempts+=1
            if(n_attempts==n_trials):
                filling=False
                if(printing):
                    print("Failed to fit enough waters")

        return n_added

    def get_tags(self, indices):
        """
        Returns the tags of the zeolite at the specified indices.

        Parameters:
        - indices (array): An arrray of indices specifying the zeolite(s) to retrieve tags from.

        Returns:
        - array: An array of tags corresponding to the zeolite(s) at the specified indices.
        """
        return self.structure.get_tags()[indices]

    def get_indices(self, tags):
        """
        Returns the indices of the given tags in the zeolite object.

        Parameters:
        tags (array): An array of tags to search for.

        Returns:
        list: An array of indices corresponding to the given tags.
        """

        tags=np.array(tags)
        return np.array( [np.where(self.structure.get_tags() == tag)[0][0] for tag in tags] ) 

    
    
