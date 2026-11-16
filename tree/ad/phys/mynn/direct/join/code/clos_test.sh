#!/bin/bash
# build and run code/clos_test.cpp from the tree (CPU only, seconds). clos_test.sh [BUILD_DIR]
set -u
J=$(cd "$(dirname "$0")/.." && pwd); O=${1:-$J/build/clos}; mkdir -p $O
sed -e "s/mixjac/mixjacd/g; s/MJ_/MJD_/g; s/float/double/g; s/mj_host_sqrt/mjd_host_sqrt/g" $J/blocks/mixjac.hpp > $O/mixjacd.hpp
cmp -s $O/mixjacd.hpp $J/code/mixjacd.hpp || { echo "code/mixjacd.hpp is not the transform of blocks/mixjac.hpp: regenerate it with the sed line of this script"; exit 2; }
g++ -std=c++17 -O1 -DJOIN_MIX -DJOIN_CLOSURE -I$J/blocks -I$J/code $J/code/clos_test.cpp -o $O/clos_test || exit 2
$O/clos_test
