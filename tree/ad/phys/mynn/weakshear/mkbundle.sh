#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# A scheme bundle for AD_SCHEMES: BASE_BUNDLE copied, with the three boundary-layer images of IMAGE_DIR
# (rel.0.cubin the tangent, rel.1.cubin the taped reverse, rel.2.cubin the tape count) in its p3/. Taped route only:
# there is no tape-free image of the live mixing. Where the door table offers a route for them, the sensitivity command lays the same three images over the row's
# bundle itself (and moves the bundle's SHA256SUMS lines, which this script does not).
#   mkbundle.sh BASE_BUNDLE OUT_BUNDLE IMAGE_DIR
set -eu
cp -a "$1" "$2"; chmod -R u+w "$2"
cp -p "$3/rel.0.cubin" "$3/rel.1.cubin" "$3/rel.2.cubin" "$2/p3/"
echo "$2: p3 $(cd "$2/p3" && sha256sum rel.*.cubin | cut -c1-12 | tr "\n" " ")"
