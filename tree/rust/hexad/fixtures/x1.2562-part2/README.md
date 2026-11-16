# 2-rank x1.2562 rank-local meshes (multirank-2 --local-topology tests, src/e2e/topo.rs)

The x1.2562-mesh global fixtures sliced and renumbered by the partition l2g maps of a test machine <kernel set>
(part_sha256 in that file; halo_rings 3) with the remap law of tree/src/hexcore/partition_local_mesh_v841.py: values mapped
global->local, a reference absent from the rank clamped to local index 0, negative padding kept. Little-endian int32, gzip.

- p0.cells_on_edge.i32.gz: 9388 words, sha256 7f318cddc456235ff81e66e913a216bb121e2a081e09a0fc274aad715a867b58
- p0.edges_on_cell_w10.i32.gz: 15370 words, sha256 c22ae87175f998637654476a6287768b62127ada4f1aa923b968763a37c122ee
- p0.n_edges_on_cell.i32.gz: 1537 words, sha256 66ba6afcafbc3a5b2659583500f26ca58d2afa5d53863670b69c5c1b62a67332
- p0.vertices_on_edge.i32.gz: 9388 words, sha256 3be6135f3dde4e2d9ab0eed8765b86687bfee81ce47132844b08c1aa3d5c74a1
- p0.edges_on_vertex.i32.gz: 9474 words, sha256 e31351ef889a5a46d0645c0f9b624ea24cc156cfd2e178d9a1179751f97555cc
- p1.cells_on_edge.i32.gz: 9412 words, sha256 c484c0585b8bce1795a1ad4667f6e893d039ee38439a209b586421ed60bbca2d
- p1.edges_on_cell_w10.i32.gz: 15410 words, sha256 b3c2c477c0d2ed3b1df6f55e2826192e1b3bb7e5e2e0186fe47c81ca66892b99
- p1.n_edges_on_cell.i32.gz: 1541 words, sha256 94119dab1b6e8cd87f006df04a716316264cb8c0038a3a6600fffd83d6718932
- p1.vertices_on_edge.i32.gz: 9412 words, sha256 24e4c139490cb92293efe5d5b708859285f798ab38de953ba0d0274ab2fa7e28
- p1.edges_on_vertex.i32.gz: 9498 words, sha256 b2230c993a96647517780b2cb7850b593f032a35a428996b50a982eebd91c906
