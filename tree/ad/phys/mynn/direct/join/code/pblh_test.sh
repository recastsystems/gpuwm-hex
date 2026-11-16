#!/bin/bash
# build and run code/pblh_test.cpp from the tree (CPU only, seconds). pblh_test.sh [BUILD_DIR] [COLUMNS]
set -u
J=$(cd "$(dirname "$0")/.." && pwd); O=${1:-$J/build/pblh}; mkdir -p $O
g++ -std=c++17 -O1 -ffp-contract=off -I$J/blocks $J/code/pblh_test.cpp -o $O/pblh_test || exit 2
$O/pblh_test ${2:-4000}
