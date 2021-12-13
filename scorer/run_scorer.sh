#!/usr/bin/env bash

PWD=`pwd`
inputdir=$1
# inputdir="single_pcd"

echo "Running shape generation from point cloud"
python --version
cmd=`python scorer/shape_generator.py --pathIn ${inputdir}`
# echo "Command is " ${inputdir}
# # retVal=$
if [[ ${cmd} -eq 0 ]]; then
    echo "Success generating shapes from pcd"
else
    echo "Error generating shapes from pcd"
fi


echo "Running generate multi-view images from mesh rendering"
#max 4 views if we use 3 perspective angles (up to 12 imgs supported in dataset)
views=4

outputdir="${inputdir}/test"

cmd=`scorer/generateviews.sh ${inputdir} ${views} ${outputdir}`
if [[ ${cmd} -eq 0 ]]; then
    echo "Success generating views"
else
    echo "Error generating views"
fi


echo "Running Scorer network on generated views"


angles=3
checkpointdir="scorer/checkpoint"
batch_size=12

cmd=`python scorer/test.py --imgdir ${inputdir} --angles ${angles} --views ${views} \
    --batchsize ${batch_size} --checkpointdir ${checkpointdir}`
if [[ ${cmd} -eq 0 ]]; then
    echo "Success generating views"
else
    echo "Error generating views"
fi

