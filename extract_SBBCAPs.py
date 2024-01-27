import numpy as np
import nibabel as nib
import os
subjects = np.loadtxt('/media/miplab-nas2/Data/Anders/BGSP/subjectlists/subject_list_HCP_100.txt',dtype=str)
for sub in subjects:
    print('Reading subject '+sub)
    img_in = nib.load('/media/miplab-nas2/Data/Anders/BGSP/data/'+sub+'/T1w/Results/lh.rfMRI_REST1_LR.res1250.spaceT1w.detrend1_regMov1_zscore1.SSBCAPs_schaefer400yeo7_res1250_spaceT1w.gii')
    vertices = np.loadtxt('/media/miplab-nas2/Data/Anders/BGSP/data/'+sub+'/T1w/fsaverage_LR32k/'+sub+'.L.white_MSMAll.32k_fs_LR.vertex.txt',dtype=int)
    # check that no two vertices are the same
    # assert np.unique(vertices,axis=0).shape[0] == vertices.shape[0]
    # make new dir for data
    os.makedirs('/media/miplab-nas2/Data/Anders/BGSP/data/'+sub+'/T1w/Results/SBBCAPs',exist_ok=True)
    for reg in range(400):
        data = img_in.darrays[reg].data[vertices]
        np.savetxt('/media/miplab-nas2/Data/Anders/BGSP/data/'+sub+'/T1w/Results/SBBCAPs/lh.rfMRI_REST1_LR.res1250.spaceT1w.detrend1_regMov1_zscore1.SSBCAPs_schaefer400yeo7_res1250_spaceT1w_reg'+str(reg+1)+'.txt',data)