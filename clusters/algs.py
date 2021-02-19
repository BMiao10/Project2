import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import math
import random

class Clustering():
    """
    Parent class for hierarchical and partitioning clustering
    
    Attributes:
        self.k (int) : number of clusters to initialize, default=3
        self.clusters (list) : initialize an empty vector to contain cluster labels for clustering
        
    Parameters:
        k (int) : number of clusters to initialize, default=3
    
    """
    
    def __init__(self, k=3):
        """
        Constructor class for clustering
        """
        self.k = k
        self.clusters = []
    
    @staticmethod
    def calc_dist(values1, values2, how="bit"):
        """
        Calculates the distance between two arrays
        
        Parameters:
            values1 (np.array) : First array of values to calculate distance between
            values2 (np.array) : Second array of values to calculate distance between
            how (str) : Specifies what metric to return ("ham"ming, "euc"lidean, jac"card {for sparse data}, or "bit"-wise jaccard), default="bit"
        
        Returns:
            float, int : Distance (euclidian, manhattan, or jaccard based on specified metric) between the 2 arrays
            
        Raises:
            ValueError : Sample value arrays do not have the same length
        
        """

        # Calculate jaccard distance using array values with *sparse* representations.
        # The value arrays do not have to be the same length
        if how=="jac" : 
            return 1 - (len(np.intersect1d(values1, values2)) / len(np.union1d(values1, values2)))
            
        # For all other types of clustering, check that value arrays are the same length
        if len(values1) != len(values2):
            raise ValueError("Sample values must have same length to calculate distance by %s"%how)

        # Calculations for other distance metrics - must have dense representation
        if how=="ham" : 
            return np.bitwise_xor(values1, values2).sum()
            
        elif how=="bit" : 
            
            if np.bitwise_or(values1, values2).sum() == 0: 
                return 0
            
            return 1 - ((np.bitwise_and(values1, values2).sum() / np.bitwise_or(values1, values2).sum()))
        
        elif how=="euc":
            return math.sqrt(np.square(values1 - values2).sum())

     
class PartitionClustering(Clustering):
    """
    Perform partition clustering for a set of sample values
    
    Attributes:
    	self.k, self.clusters : inherited from parent Clustering class
        self.centroids (dict) : initialize empty dictionary to contain centroids
        
    Parameters:
        k (int) : number of clusters to initialize, default=3
    
    """
    def __init__(self, k=3):
        """
        Constructor class for partition clustering
        Inherits from Clustering class
        """
        super(PartitionClustering, self).__init__(k) 
        self.centroids = {}
   
    def cluster(self, samples, max_iter=10, dist_met="euc", preprocess=None):
        
        """
        Perform partitioning (kmeans++ & kmeans) clustering given a set of samples such as ligands

        Parameters:
            samples (list, np.array, set, or other iterable object) : Set of ligands to cluster
            max_iter (int) : maximum number of iterations for clustering if the algorithm does not converge, default=10
            dist_met (str) : distance metric to use for calculating distance between points and centroids, default="euc"
            preprocess (method) :  function that formats sample values to array / array-like object for clustering, default=None
        
        Returns:
            list : cluster labels whose indices correspond to the order of the orginal sample list
            
        Raises:
            ValueError : If number of clusters greater than number of samples 

        """
        
        # Check that the number of clusters does not exceed number of samples
        if self.k > len(samples):
            raise ValueError("Cannot have more clusters than samples!")
        
        # Extract onbits from Ligand objects and put them into an ndarray
        # Other preprocessing functions can be passed if the input is a set of other objects,
        # allowing this function to be data-type agnostic
        if preprocess is not None: 
            onbits = preprocess(samples)
        
        ### Use k-means++ to initialize self.k random centroids, with inspiration from
        ### https://www.real-statistics.com/multivariate-statistics/cluster-analysis/initializing-clusters-k-means/
        
        # Store a copy of the sample value data for k-means++
        # because values will be deleted from array as centroids are selected
        cent_samples = onbits.copy()
        
        # 1. Choose a random data point as first centroid c0 (and remove from further centroid selection)
        # Centroids are stored as index:sample for easy access
        rand = random.randint(0, len(cent_samples)-1)
        self.centroids[0] = cent_samples[rand]
        cent_samples = np.delete(cent_samples, rand, 0)
        
        # 2. For each data points that is not a centroid, find min distance to all chosen centroids
        # Each iteration, choose new centroid with random probability based on distance
        # Repeat until k centroids are selected 
        while len(self.centroids.keys()) < self.k:
            
            # get minimum distance from each sample to the centroids
            distances = []
            for curr_sample in cent_samples: 
                
                min_dist = np.inf
                for curr_cent in self.centroids.keys():
                    curr_dist = super().calc_dist(curr_sample, self.centroids[curr_cent], how=dist_met)
                    
                    if curr_dist < min_dist:
                        min_dist = curr_dist
                        
                distances.append(min_dist)
            
            # Normalize array of minimum distance values to 1 to create "probablilities"
            # The lower the distance (ie. the closer it is to a current centroid), the lower the probability
            # Values with a distance of 0 (identical to current centroid) will not be selected
            prob = np.array(distances)
            prob = prob / prob.sum()
            
            # Draw a random choice based on the probabilities
            choice = np.random.choice(range(len(cent_samples)), 1, p=prob)
            
            # Add the random choice of sample to centroids and remove it from further consideration 
            # I realized after writing this that it's not neccessary to delete the sample
            # since it will have 0 chance of being selected again, but it would decrease runtime 
            # (a tiny bit) during min distance calculations
            self.centroids[len(self.centroids.keys())] = cent_samples[choice[0]]
            cent_samples = np.delete(cent_samples, choice[0], 0)

        ### Use chosen centroids to perform k-means clustering
        
        # Keep track of ind for the number of iterations 
        # and not_converged to see whether algo has converged
        ind = 0
        not_converged = True
        
        # assign each sample to a cluster
        self.clusters = [0] * len(onbits)
        
        # clustering - adapted from https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
        # https://medium.com/@rishit.dagli/build-k-means-from-scratch-in-python-e46bf68aa875
        while ind < max_iter and not_converged:
            
            # calculate the distance from each sample to the centroid
            for sample_ind in range(len(onbits)):
                
                min_dist = np.inf
                for cent_ind in self.centroids.keys():
                    curr_dist = super().calc_dist(onbits[sample_ind], self.centroids[cent_ind], how=dist_met)
                    
                    # assign to lowest centroid
                    if curr_dist < min_dist:
                        self.clusters[sample_ind] = cent_ind
                        min_dist = curr_dist
            
            # if any of the centroids are empty, fix by assigning a random point to that centroid
            cent_keys, counts = np.unique(self.clusters, return_counts=True)
            cent_dict = dict(zip(cent_keys, counts))
            
            for centroid in self.centroids.keys():
                
                # search for centroids that are empty
                if centroid not in self.clusters: 
                    # put the first point from another cluster (that has 
                    # more than one point) into this current cluster
                    for index in range(len(self.clusters)):
                        if cent_dict[self.clusters[index]] > 1:
                            self.clusters[index] = centroid
                            break
                
            # store the previous clusters to check for convergence after updating
            prev_centroids = self.centroids
            
            # calculate mean of new clusters and update centroids
            for curr_cent in self.centroids.keys():
                
                # get the onbits for the centroids 
                curr_indices = list(np.where(np.array(self.clusters) == curr_cent)[0])
                curr_onbits = np.take(onbits, curr_indices, axis=0)
                
                # take the mean of the selected onbits and set that as the new centroid
                new_centroid = np.zeros(curr_onbits.shape[1])
                
                for curr_onbit in curr_onbits:
                    new_centroid += curr_onbit
                    
                self.centroids[centroid] = new_centroid / len(curr_onbits)
            
            # Check for convergence by comparing the old to new centroids
            # If old centroids are the same, then the algorithm is stopped
            # Adapted from https://stanford.edu/~cpiech/cs221/handouts/kmeans.html
            all_equal = 0
            
            for key in self.centroids.keys():
                if (prev_centroids[key] == self.centroids[key]).all():
                    all_equal += 1
                    
            if all_equal == len(self.centroids.keys()):
                not_converged = False
        
            # increment index tracking the number of iterations
            ind += 1
        
        return self.clusters
    
class HierarchicalClustering(Clustering):
    """
    Perform hierarchical clustering for a set of sample values  
    
    Attributes:
        self.k, self.clusters : inherited from parent Clustering class
        
    Parameters:
        k (int) : number of clusters to initialize, default=3
    """
    
    def __init__(self, k=3):
        """
        Constructor class for hierarchical clustering
        """
        super(HierarchicalClustering, self).__init__(k)
    
    def cluster(self, samples, linkage = "max", dist_met="bit", preprocess=None): 
        
        """
        Perform hierarchical clustering given a set of samples such as ligands

        Parameters:
            samples (list, np.array, set, or other iterable object) : Set of ligands to cluster
            linkage (str) : type of linkage to use when updating the distance matrix, default="max"
            dist_met (str) : distance metric to use for calculating distance between points and centroids, default="bit"
            preprocess (method) :  function that formats sample values to array / array-like object for clustering, default=None
        
        Returns:
            list : cluster labels whose indices correspond to orginal sample list
            
        Raises:
            ValueError : If number of clusters greater than number of samples 

        """

        # Check that the number of clusters does not exceed number of samples
        if self.k > len(samples):
            raise ValueError("Cannot have more clusters than samples!")
            
        # extract onbits from Ligand objects and put them into an ndarray
        if preprocess is not None: 
            samples = preprocess(samples)

        # Create matrix and labels list.
        # Initialize to infinity so all minimum distances 
        # are less than the default 
        dist_mat = np.ones([len(samples), len(samples)]) * np.inf
        
        # keep track of which matrix index values maps to which sample
        samples_dict = dict(zip(range(len(samples)), samples))
        
        # create distance matrix
        for ind1 in samples_dict.keys():
            for ind2 in samples_dict.keys():
                if ind1 != ind2:
                    
                    dist = super().calc_dist(samples_dict[ind1], samples_dict[ind2], how=dist_met)
                        
                    dist_mat[ind1][ind2] = np.inf
                    dist_mat[ind2][ind1] = dist
        
        # assign each sample to its own cluster first
        self.clusters = [[ind] for ind in samples_dict.keys()]
        
        # iterate until the number of clusters is reached
        while len(self.clusters) > self.k:

            # get the index of the minimum value 
            min_x = np.where(dist_mat == np.min(dist_mat))[0][0]
            min_y = np.where(dist_mat == np.min(dist_mat))[1][0]

            # Update the linkage for the two clusters
            # For single linkage, -1 is used for values in the bottom triangle instead
            # of np.inf, so that these values will be ignored
            for i in range(len(dist_mat)):
                if min_x != i & min_y !=i:
                    if linkage == 'average': # average linkage
                        dist_mat[i][min_x] = (dist_mat[i][min_x] + dist_mat[i][min_y]) / 2.0
                    elif linkage == 'min': # single linkage
                        dist_mat = np.where(dist_mat == np.inf, -1, dist_mat)
                        dist_mat[i][min_x] = min(dist_mat[i][min_x], dist_mat[i][min_y])
                        dist_mat = np.where(dist_mat == -1, np.inf, dist_mat)
                    elif linkage == 'max': # complete linkage
                        dist_mat[i][min_x] = max(dist_mat[i][min_x], dist_mat[i][min_y])
                    
            # get the clusters with minimum distance
            cluster_x = self.clusters[min_x]
            cluster_y = self.clusters[min_y]
            
            # combine the clusters and store the new cluster at index min_x (replacing it)
            self.clusters[min_x] = cluster_x + cluster_y
            
            # remove the old cluster that updated
            self.clusters.pop(min_y)
            dist_mat = np.delete(dist_mat, min_y, axis=0)
            dist_mat = np.delete(dist_mat, min_y, axis=1)
            
        # create dictionaries to map cluster labels to samples
        #sample_to_cluster = {}
        ind_to_clust = {}

        # assign cluster labels to samples
        for clust in range(len(self.clusters)):
            for ind in self.clusters[clust]:
                #sample_to_cluster[samples_dict[ind]] = clust
                ind_to_clust[ind] = clust
        
        # return vector of cluster labels for each sample
        self.clusters = [ind_to_clust[ind] for ind in range(len(samples))]
        return self.clusters
    

class Ligand():
    """
    Object containing representation and description of molecule
    
    Attributes:
        self.ligand_id (int): a unique ID to track the ligand
        self.score (list): Vina AutoDock score
        self.smile (str): SMILE-formatted string representation
        self.onbits (list, set, other iterable): sparse representation of ligand fingerprint
        
    Parameters:
        ligand_id (int) : a unique ID to track the ligand
        score (list) : Vina AutoDock score
        smile (str) : SMILE-formatted string representation
        onbits (list, set, other iterable) : sparse representation of ligand fingerprint
        bit_length (int) : the length of the bit vector for densification, default=1024
        densify (bool) : whether to densify the onbits upon initialization, default=True
    
    """
    
    def __init__(self, ligand_id, score, smile, onbits, bit_length=1024, densify=True):
        """
        Constructor method for Ligand class
        """
        self.ligand_id = ligand_id
        self.score = score
        self.smile = smile
        self.onbits = onbits
        self.dense = False
        
        if densify:
            self.densify(bit_length)
            self.dense = True

    def densify(self, length):
        """
        Converts sparse representation of ligand onbits to dense binary array
        
        Parameters:
            length (int) : the length of the dense representation
        
        """
        if self.dense: 
            print("onbits already dense")
        else:
            all_bits = self.onbits
            self.onbits = np.zeros(length, dtype=int)
            for bit in all_bits: 
                self.onbits[bit] = 1
            self.dense = True
        
    def sparsify(self):
        """
        Converts dense representation of ligand onbits to sparse array of indices where onbits occur
        """
        if not self.dense: 
            print("onbits already sparse")
        else: 
            self.onbits = np.where(self.onbits == 1)
    

def calcClusterQuality(samples, clust_labels, dist_met="bit", preprocess=None):
    """
    Measures the quality of a set of clusters using silhouette score

    Parameters:
        samples (np.ndarray) : sample values that map to labels by index
        clust_labels (np.array, list) : cluster labels for each of the samples
        dist_met (str) : distance metric to compare samples by when calculating silhouette score, default="bit"
        preprocess (method) : function to preprocess samples into sample x value array for easy manipulation, default=None

    Returns:
        float : silhouette score of clusters 
        
    """
    
    # preprocessing step for Ligands
    if preprocess is not None:
        samples = preprocess(samples)
    
    # create n x n distance matrix 
    dist_mat = np.ones([len(samples), len(samples)])
        
    for i in range(len(samples)):
        for j in range(len(samples)):
            dist_mat[i][j] = Clustering().calc_dist(samples[i], samples[j], how=dist_met)
    
    # calculate silhouette score for each point
    all_scores = []
    
    for index in range(len(samples)):
        
        # initialize variables in silhouette score calculation
        a_i = 0
        
        # initialize current point i and the cluster that i is in
        i = samples[index]
        curr_label = clust_labels[index]
        
        # get a(i) = get sum of distances from point i to all points in the same cluster and divide by n-1
        cluster = np.where(np.array(clust_labels) == curr_label)[0]
        
        # Set clusters of size 1 to have a silhouette score of 0
        if len(cluster) == 1:
            all_scores.append(0)
        else:
            # Continue to get a(i). The current point is removed from the calculation
            # although it shouldn't matter since it would have a distance of 0
            cluster = cluster[cluster != index]
            
            for curr_clust in cluster:
                a_i += dist_mat[index][curr_clust]
            
            a_i /= len(cluster) # divide by n - 1 (since point has been removed)
            
            # Get lowest average distance to other clusters, which is equivalent to 
            # b = sum of distances from point i to all points in the nearest cluster (divided by n)
            b_i = np.inf
            closest_clust = 0
            
            for label in clust_labels:
                if label != curr_label:
                    curr_clust = np.where(np.array(clust_labels) == label)[0]
                    b_dist = dist_mat[curr_clust].sum(axis=0)[index]
                    b_dist = b_dist / len(curr_clust)
                    
                    if b_dist < b_i:
                        b_i = b_dist
            
            # calculate silhouette score as (b - a) / max(a, b)
            all_scores.append((b_i - a_i) / max(b_i, a_i))

    # Average across all silhouette scores. The closer the scores are to 1, the better the cluster
    return np.mean(np.array(all_scores))    
    
def calcClusterSimilarity(labels1, labels2):
    """
    Calculates similarity between two different sets of clustering results
    
    Parameters:
        labels1, labels2 (list, np.array, or other list like object) : set of labels / results from a clustering method

    Returns: 
        float: Jaccard index between the two specified clustering results
    """
    
    # initialize varaiables for Jaccard Index calcuation
    f_11 = 0 # "True positive"
    f_10 = 0 # "False positive"
    f_01 = 0 # "False negative"
    
    # check whether each pair of items found in each cluster
    # Jaccard index = f11 / f01 + f10 + f11 = same / total (except negatives)
    for i in range(len(labels1)):
        for j in range(len(labels2)):
            pair_1 = labels1[i] == labels1[j]
            pair_2 = labels2[i] == labels2[j]
            
            if pair_1 and pair_2: # pairs of points are in same cluster in both 1 & 2
                f_11 += 1 
            elif pair_1 and not pair_2: # pairs of points are in same cluster in 1 but not 2
                f_10 +=1
            elif pair_2 and not pair_1: # pairs of points are in same cluster in 2 but not 1
                f_01 +=1
    
    # Return 1 if there are no common pairs between the samples. Sklearn's Jaccard metric allows
    # user input for either returning 0 or 1 in this case but I just chose 1 for simplicity and
    # the clusters are technically very similar in that they don't share any overlap
    # https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/metrics/_classification.py#L642
    if (f_01 + f_10 + f_11) == 0: 
        return 1
    
    return f_11 / (f_01 + f_10 + f_11)   

def readLigands(ligand_file, n=None):
    """
    Reads in csv file containing ligand information and converts data to list of Ligand objects
    
    Parameters:
       ligand_file (str, pathlike) : csv file containing ligand information
       n (int) : number of ligands to read in, default (=None) reads in all ligands in file
       
    Returns:
        list<Ligand> : Ligand objects read in from csv file
        
    """
    
    ligands = pd.read_csv(ligand_file)
    
    if n is not None: 
        ligands = ligands.loc[:(n-1)]

    ligands_list = []
    for ind in list(ligands.index):

        curr = ligands.loc[ind]
        onbits = [int(obj) for obj in curr["OnBits"].split(",")]

        object_ligands = Ligand(curr["LigandID"], curr["Score"], curr["SMILES"], onbits)
        ligands_list.append(object_ligands)
        
    return ligands_list

def preprocessLigands(samples):
    """
    Extracts onbits information from ligands as a preprocessing step to clustering or calculating other scores
    
    Parameters:
       samples (list<Ligands>) : list of Ligand objects with dense onbit representations
       
    Returns:
        np.ndarray : sample x onbit array that can be passed into clustering functions
        
    """
    onbits_list = []
        
    for curr_sample in samples:
        onbits_list.append(curr_sample.onbits)
        
    return np.stack(onbits_list, axis=0)





