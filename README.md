# Project 2 - Clustering and Drug Discovery
## Due 02/17/2021

![BuildStatus](https://github.com/ucsf-bmi-203-2021/Project2/workflows/HW2/badge.svg?event=push)

In this assignment, you will evaluate results from a high-throughput virtual screen against the SARS-CoV2 Spike protein / Human ACE2 interface.  There are two parts to this assignment and Part 2 requires completion of Part 1. We recommend reading through both Part 1 and Part 2 before beginning this assignment. 

* Part 1 - API and implementation
* Part 2 - Evaluating clustering

The data we are considering comes from [Smith and Smith, 2020](https://chemrxiv.org/articles/preprint/Repurposing_Therapeutics_for_the_Wuhan_Coronavirus_nCov-2019_Supercomputer-Based_Docking_to_the_Viral_S_Protein_and_Human_ACE2_Interface/11871402). In this study, they generated 6 Spike-Ace2 interface poses using MD simulations. They then docked ~10k small molecules against each protein conformation. Provided for you is the top (#1) pose for each ligand docked against one Spike-ACE2 interface conformation, as well as the corresponding SMILES string, AutoDock Vina score, and the “On” bits in the Extended Connectivity Fingerprint for that compound. These can all be found in ligand\_information.csv.

### File structure
```
.
├── README.md
├── Brenda_Miao_BMI203_HW2.ipynb
│   ...
├── clusters
│   ├── __main__.py
│   └── algs.py
│  
├── docs
│   └── build
│   	  └── html
│              └── index.html
│   ...
├── output
│   ├── 
│   ├── 
│   └── 
└── test
    └── test_clusters.py

```
### Sphinx documentation
Documentation for all my code can be found in the index.html file located in the docs folder as indicated in the file structure above

### Part I: Requirements & testing
Additional packages imported for Clusters.algs.py are updated in the requirements. Unit tests are located in test_clusters.py and passing indicated by the badge in this repository

### Part II
Code (with functions imported from clusters.algs.py), figures, and my answers for part II are located in the Jupyter notebook "Brenda_Miao_BMI203_HW2.ipynb" in the root folder. 
Output for certain questions, indicated in the Jupyter notebook markdown, are stored in output folder.


