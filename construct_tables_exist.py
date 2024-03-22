import numpy as np
import pandas as pd
import os

def check_eigenmode_exists(num_subs):
    subject_list = np.loadtxt('subjectlists/subject_list_HCP_'+str(num_subs)+'.txt',dtype=str)
    df = pd.read_csv('basis_experiments_table'+str(num_subs)+'.csv')
    # add new column to df to indicate whether the file exists or not
    df['exists'] = False
    # iterate through df and check if file exists
    for idx,row in df.iterrows():
        if row.basis in ['subject-specific','ind conn/surf avg']:
            allex = []
            for sub in subject_list:
                if os.path.exists('data/'+str(sub)+'/T1w/tractography_eigenmodes/'+str(sub)+row.basis_file):
                    allex.append(True)
                else:
                    allex.append(False)
            if all(allex):
                df.loc[idx,'exists'] = True
        elif row.basis in ['ind. surface']:
            allex = []
            for sub in subject_list:
                if os.path.exists('data/'+str(sub)+'/T1w/fsaverage_LR32k/'+str(sub)+row.basis_file):
                    allex.append(True)
                else:
                    allex.append(False)
            if all(allex):
                df.loc[idx,'exists'] = True
        elif row.basis in ['template surface']:
            if os.path.exists('results/'+str(row.basis_file)):
                df.loc[idx,'exists'] = True
        elif row.basis in ['A_local']:
            if os.path.exists('results/'+str(row.basis_file)):
                df.loc[idx,'exists'] = True
        elif row.basis in ['avg connectome basis','avg basis flag','avg basis karcher opt','avg basis karcher Begelfor','avg basis procrustes']:
            if os.path.exists('results/avg_connectomes/'+str(row.basis_file)[:-4]+'_'+str(num_subs)+'.txt'):
                df.loc[idx,'exists'] = True
        elif row.basis in ['random_smoothed_basis']:
            allex = []
            for sub in subject_list:
                if os.path.exists('data/'+str(sub)+'/T1w/random_data/'+row.basis_file):
                    allex.append(True)
                else:
                    allex.append(False)
            if all(allex):
                df.loc[idx,'exists'] = True
    
    df.to_csv('basis_experiments_table'+str(num_subs)+'.csv',index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_subs', type=int, default=100)
    args = parser.parse_args()
    num_subs = args.num_subs

    check_eigenmode_exists(num_subs)