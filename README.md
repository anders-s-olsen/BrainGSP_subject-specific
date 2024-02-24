# Subject-specific brain graph signal processing

A set of python scripts and functions to evaluate eigenmodes of individual and group-level connectomes, and inidividual and template surfaces. The methods are further described in the paper [Assessing Subject-Specific Structural Constraints on Brain Function via Graph Signal Processing (Unpublished)](). 

## Organization

Please note that the repository is not organized as a toolbox. The scripts may be freely used and adapted to the user's own needs, but some code adaptations should be expected. 

The repository contains the following key scripts:

* connectome_eigenmodes.py - thresholds connectome based on desired density, adds a local neighborhood graph, computes eigenmodes of normalized Laplacian
* connectome_construct_avg.py - computes a group-level connectome
* connectome_smooth.py - smoothes connectomes using the connectome-spatial-smoothing toolbox
* connectome_tck_to_npz.py - converts a tractography .tck file to a connectome saved as a numpy file using connectome-spatial-smoothing
* updated_mrtrix_tractography.sh - computes tractography, [original file here](https://github.com/sina-mansour/neural-identity/blob/master/codes/tractography/updated_mrtrix_tractography.sh)
* surface_eigenmodes.py - computes surface eigenmodes, [original file here](https://github.com/NSBLab/BrainEigenmodes/blob/main/surface_eigenmodes.py)
* reconstruct_data4.py - the main code for projecting functional data onto structural basis, discarding some eigenmodes, backprojecting, and evaluating reconstruction accuracy
* reconstruct_data_sorted.py - same as above, but where the eigenmodes are sorted according to their weight in the "full" 200-eigenmode linear model
* recon_accuracy_analysis.ipynb - plots of reconstruction accuracies
* brainmaps.ipynb - brain visualizations for methods figure

## Dependencies

Some of the work is a direct python translation of the [provided matlab code]([https://www.nature.com/articles/s41586-023-06098-1](https://github.com/NSBLab/BrainEigenmodes)) for the Nature-paper [Geometric Constraints on Human Brain Function](https://www.nature.com/articles/s41586-023-06098-1)

The surface eigenmode computation requires [LaPy](https://github.com/Deep-MI/LaPy/tree/main)

Connectome smoothing was performed using Connectome-spatial-smoothing [(code)](https://github.com/sina-mansour/connectome-spatial-smoothing), [(paper)](https://www.sciencedirect.com/science/article/pii/S1053811922000593)
