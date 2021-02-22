
import pytest
from clusters import algs

def test_partitioning():
    
    # read in the ligands
    ligands = algs.readLigands("./ligand_information.csv", n=2)
    
    # Select some ligands for testing
    # 0 and 1 do not share any onbits so can be used to create artificial test cases
    ligand1 = ligands[0]
    ligand2 = ligands[1]
    ligand_subs = [ligand1, ligand2, ligand1, ligand2]
    
    # Create clustering object with k=2
    # Since we know these should be grouped into 2 groups
    part = algs.PartitionClustering(k=2)
    
    # check that k has been initialized correctly
    assert(part.k == 2)

    # get cluster results
    labels = part.cluster(ligand_subs, preprocess=algs.preprocessLigands)
    
    # test that identical objects are grouped together
    assert(labels[0] == labels[2])
    assert(labels[1] == labels[3])
    
    # test that non-identical objects are in separte clusters
    assert(labels[0] != labels[1])
    assert(labels[0] != labels[3])
    
    # test all clusters have at least one sample
    assert(len(np.where(labels == 0)) > 0)
    assert(len(np.where(labels == 1)) > 0)

def test_hierarchical():
    
    # read in the ligands
    ligands = algs.readLigands("./ligand_information.csv", n=2)
    
    # Select some ligands for testing
    # 0 and 1 do not share any onbits so can be used to create artificial test cases
    ligand_subs = ligands[0:2] + ligands[0:2] 
    
    # Create clustering object with k=2
    # Since we know these should be grouped into 2 groups
    hier = algs.HierarchicalClustering(k=2)
    
    # check that k has been initialized correctly
    assert(hier.k == 2)
    
     # get cluster results
    labels = hier.cluster(ligand_subs, preprocess=algs.preprocessLigands)
    
    # test that identical objects are grouped together
    assert(labels[0] == labels[2])
    assert(labels[1] == labels[3])
    
    # test that non-identical objects are in separte clusters
    assert(labels[0] != labels[1])
    assert(labels[0] != labels[3])
    
    # test all clusters have at least one sample
    assert(len(np.where(labels == 0)) > 0)
    assert(len(np.where(labels == 1)) > 0)

def test_clustSimilarity():
    
    # Create artificial labels for testing 
    # These represent perfect agreement between the two clustering results
    clust1 = [0,0,1,1]
    clust2 = [0,0,1,1]
    
    # test that identical clusterings show high similarity
    assert(algs.calcClusterSimilarity(clust1, clust2) == 1)
    
    # Create artificial labels for testing 
    # These represent only some agreement between the two clustering results
    clust1 = [0,1,0,1]
    clust2 = [0,0,1,1]
    
    # test that non-identical clusterings show lower similarity
    assert(algs.calcClusterSimilarity(clust1, clust2) == float(1/3))
    
    # Create artificial labels for testing 
    # These represent very little agreement between the two clustering results
    clust1 = [0, 1, 2, 3]
    clust2 = [0, 0, 0, 0]
    
    # test that non-identical clusterings show lower similarity
    assert(algs.calcClusterSimilarity(clust1, clust2) == 0.25)

def test_clustQuality():
    
    # Create artificial testing matrices
    # Preprocessing method is tested elsewhere
    samples = [[1,1,1,1], [1,1,1,1], [0,0,0,0], [0,0,0,0]]
    
    # test that bad clusters show low quality (silhouette score)
    bad_labels = [1,0,1,0]
    assert(algs.calcClusterQuality(samples, bad_labels) == -0.5)

    # test that good clusters show high quality (silhouette score)
    good_labels = [1,1,0,0]
    assert(algs.calcClusterQuality(samples, good_labels) == 1)


def test_preprocessing():
    
    # read in the ligands
    ligands = algs.readLigands("./ligand_information.csv")

    # subset the ligands (selection without replacement) to use for hierarchical clustering
    ligands_subs = ligands[0:10]

    # extract values to np.array using the preprocessing function
    onbits = algs.preprocessLigands(ligands_subs)

    # check for correct dimension
    assert(onbits.shape == (10,1024))

    # check for correct values in the new matrix
    assert(onbits[0][0] == 0)
    assert(onbits[0][489] == 1)
    assert(onbits[9][0] == 0)
    assert(onbits[9][650] == 1)

def test_calc_dist():
    
    # create test cases for Jaccard
    jac1 = [0, 1, 2, 3]
    jac2 = [0, 1, 2, 20, 30]
    jac3 = [4, 5, 6, 7]
    
    # check for correct Jaccard index (sparse representation)
    assert(algs.Clustering().calc_dist(jac1, jac1, how="jac") == 0) # all shared
    assert(algs.Clustering().calc_dist(jac1, jac2, how="jac") == 0.5) # half shared
    assert(algs.Clustering().calc_dist(jac1, jac3, how="jac") == 1) # none shared
    
    # create test cases for bitwise Jaccard (and euclidean)
    bit1 = np.array([0,0,0,0])
    bit2 = np.array([0,0,1,1])
    bit3 = np.array([1,1,1,1])
    
    # check for correct bitwise Jaccard 
    assert(algs.Clustering().calc_dist(bit1, bit1, how="bit") == 0) # all shared
    assert(algs.Clustering().calc_dist(bit2, bit3, how="bit") == 0.5) # half shared
    assert(algs.Clustering().calc_dist(bit1, bit2, how="bit") == 1) # half shared, half 0s (ignored)
    assert(algs.Clustering().calc_dist(bit1, bit3, how="bit") == 1) # none shared
    
    # check for correct Euclidean
    assert(algs.Clustering().calc_dist(bit1, bit1, how="euc") == 0) # all shared
    assert(algs.Clustering().calc_dist(bit1, bit3, how="euc") == 2) # none shared
    
    # create additional test cases for Euclidean
    euc1 = np.array([1, 2, 3, 4])
    euc2 = np.array([6, 7, 8, 9])
    
    # check for correct Euclidean between non-binary arrays
    assert(algs.Clustering().calc_dist(euc1, euc2, how="euc") == 10) 

def test_ligandIO():
    ligands = algs.readLigands("./ligand_information.csv")
    
    # check that all ligands have been read in
    assert(len(ligands) == 8524)
    
    # get first and last ligand
    ligand1 = ligands[0]
    ligand2 = ligands[-1]

    # check that the ligand onbits are densified correctly
    assert(len(ligand1.onbits) == 1024)
    assert(len(ligand2.onbits) == 1024)

    # check that the on bits are in fact on
    assert(list(np.where(ligand1.onbits == 1)[0]) == [360,489,915])
    assert(list(np.where(ligand2.onbits == 1)[0]) == [1,23,33,34,80,128,197,222,227,327,356,407,461,466, 474,482,486,564,570,
    	571,586,603,606,650,661,708,783, 807,836,846,881,893,947,953,981,990,1014,1017])

    # check that the not on bits are indeed off
    assert(ligand1.onbits[10] == 0)
    assert(ligand2.onbits[442] == 0)
    
    # check that the smiles have been read in correctly
    smile1 = "N#C"
    smile2 = "Ic1c(N(C(=O)C)C"
    
    assert(ligand1.smile == smile1)
    assert(ligand2.smile[:15] == smile2)
    
    # check that IDs are correct
    assert(ligand1.ligand_id == 0)
    assert(ligand2.ligand_id == 9055)
    
    # check that scores are correct
    assert(ligand1.score == -1.3)
    assert(ligand2.score == 73)

	
	


