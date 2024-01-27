
#!/bin/bash

# Path to the subject list file
# subject_list="subjectlists/subject_list_HCP_100.txt"
subject_list="subjectlists/subject_list_HCP_100_rev.txt"
# reverse the order of subject_list

# Loop over each subject in the subject list
while IFS= read -r subject || [[ -n "$subject" ]]; do

    echo "Subject: $subject"

    # Check if either file1 or file2 exists
    if [[ -e "data/"$subject"/T1w/tractography/"$subject"_unsmoothed_high_resolution_volumetric_probabilistic_track_endpoints_20M.tck_structural_connectivity.npz" || -e "data/"$subject"/T1w/tractography/volumetric_probabilistic_tracks_20M.tck" ]]; then
        echo "Skipping subject: $subject"
        continue
    fi

    # Run the updated_mrtrix_tractography.sh script for each subject
    nice ./updated_mrtrix_tractography.sh "$subject" 20M
done < "$subject_list"
