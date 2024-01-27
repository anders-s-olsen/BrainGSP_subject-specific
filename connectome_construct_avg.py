import numpy as np
import os
import scipy
import argparse
import matplotlib.pyplot as plt

def compute_mean_connectome(streamlines,fwhm,num_subs):
    if os.path.exists('results/avg_connectomes/avg_structural_connectome_'+streamlines+'_fwhm'+fwhm+'_'+str(num_subs)+'.npz'):
        print('Found average connectome for ' + str(num_subs) + ' subjects')
        return
    subject_list=np.loadtxt('subjectlists/subject_list_HCP_'+str(num_subs)+'.txt',dtype=str)
    if float(fwhm)>0:
        filename = '_smoothed_structural_connectome_'+streamlines+'_fwhm'+fwhm+'.npz'
    else:
        filename = '_unsmoothed_high_resolution_volumetric_probabilistic_track_endpoints_'+streamlines+'.tck_structural_connectivity.npz'
    npz_path = 'data/'
        
    # Define path to save the average connectome
    for subject in subject_list:
        npz_file = os.path.join(npz_path, subject + '/T1w/tractography/' \
                            + subject + filename)
        
        high_resolution_connectome = scipy.sparse.load_npz(npz_file)
            
        high_resolution_connectome = high_resolution_connectome[:29696, :29696] #extract only left hemisphere
        if subject == subject_list[0]:
            avg_connectome = high_resolution_connectome
        else:
            avg_connectome = avg_connectome + high_resolution_connectome
            
        print('Loaded connectome for subject ' + subject)

    # Calculate the average connectome
    avg_connectome = avg_connectome/len(subject_list)

    # u = scipy.sparse.load_npz('results/avg_connectomes/avg_structural_connectome_'+streamlines+'_fwhm'+fwhm+'_'+str(num_subs)+'.npz')
    # if np.sum(avg_connectome - u) != 0:
    #     print('New connectome!!')

    #     # Save the average connectome as a txt file
    scipy.sparse.save_npz('results/avg_connectomes/avg_structural_connectome_'+streamlines+'_fwhm'+fwhm+'_'+str(num_subs)+'.npz', avg_connectome)


def main(num_subs):

    # Load subject list
    for streamlines in ['20M']:
        for fwhm in ['0.0','2.0','4.0','6.0','8.0','10.0']:
            compute_mean_connectome(streamlines=streamlines,fwhm=fwhm,num_subs=num_subs)


if __name__ == '__main__':
    # add parser to get the number of subjects
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_subs', type=int, default=100)
    args = parser.parse_args()
    main(args.num_subs)