#!/bin/bash

FILENAME=opencl_profile_0.log

export COMPUTE_PROFILE=1
export COMPUTE_PROFILE_CSV=1
export COMPUTE_PROFILE_CONFIG=config.log
export COMPUTE_PROFILE_LOG=$FILENAME

./tea_leaf

sed -i 's/OPENCL/CUDA/g' $FILENAME
sed -i 's/ndrange/grid/g' $FILENAME
sed -i 's/workgroup/threadblock/g' $FILENAME
sed -i 's/stapmemperthreadblock/stasmemperblock/g' $FILENAME
sed -i 's/regperworkitem/regperthread/g' $FILENAME

