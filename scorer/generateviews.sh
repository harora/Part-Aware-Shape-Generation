#!/bin/bash

#try to get the number of cores for multiprocessing in Linux and Mac.
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    number_cores=`grep -c ^processor /proc/cpuinfo`
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    number_cores=`sysctl -n hw.ncpu`
# elif [[ "$OSTYPE" == "cygwin" ]]; then
#     # POSIX compatibility layer and Linux environment emulation for Windows
# elif [[ "$OSTYPE" == "msys" ]]; then
#     # Lightweight shell and GNU utilities compiled for Windows (part of MinGW)
# elif [[ "$OSTYPE" == "win32" ]]; then
#     # I'm not sure this can happen.
# elif [[ "$OSTYPE" == "freebsd"* ]]; then
#     # ...
else
        # Unknown OS, number of processes will be 4.
    number_cores=4
fi

inputdir=$1
if [[ -z "$2" ]]; then
    noViews=12
    echo "No of views is 12"
else
    noViews=$2
    echo "No of views is ${noViews}" 
fi
outputdir=$3
# echo "Number of cores is ${number_cores}"
echo "Processing files at ${inputdir} using ${number_cores} processes"

find ${inputdir}/**/*.obj -print0 | xargs -0 -n1 -P${number_cores} -I {} blender --background --python scorer/render_blender.py -- --output_folder ${outputdir} --views ${noViews} {} > blender.log
