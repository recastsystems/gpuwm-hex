#!/bin/bash
# build and run code/mix_test.cpp from the tree. code/mixjacd.hpp is blocks/mixjac.hpp with every float made
# double by the sed line below (the double mirror test A differences); this script refuses a committed copy that is not
# that transform of the current mixjac.hpp. CPU only. mix_test.sh [BUILD_DIR]
set -u
J=$(cd "$(dirname "$0")/.." && pwd); O=${1:-$J/build/mix}; mkdir -p $O
sed -e "s/mixjac/mixjacd/g; s/MJ_/MJD_/g; s/float/double/g; s/mj_host_sqrt/mjd_host_sqrt/g" $J/blocks/mixjac.hpp > $O/mixjacd.hpp
cmp -s $O/mixjacd.hpp $J/code/mixjacd.hpp || { echo "code/mixjacd.hpp is not the transform of blocks/mixjac.hpp: regenerate it with the sed line of this script"; exit 2; }
g++ -std=c++17 -O1 -DJOIN_MIX -I$J/blocks -I$J/code $J/code/mix_test.cpp -o $O/mix_test || exit 2
$O/mix_test
