# x1.2562 global mesh fixtures (multirank-2 --local-topology tests, src/e2e/topo.rs)

Little-endian int32, gzip, cut to the logical extent. Source: the 1-rank pkg18m2 plan (a test machine <kernel set>) recorded initial allocation payloads:

- cells_on_edge.i32.gz: allocation 24 (cells_on_edge), 15360 words, sha256 7c4b6c781f6001e70a03186e2b58639bf2e4455bd19184e39c0aa63db3f6df72; payload 7c4b6c781f6001e70a03186e2b58639bf2e4455bd19184e39c0aa63db3f6df72
- edges_on_cell_w10.i32.gz: allocation 27 (edges_on_cell), 25620 words, sha256 eedcaaed085bea9e56944986bd8375062b543cc4dcc4daffe415b530fc366dbc; payload 3179037500b1315bfea47bdc6bdf3ad2d52508c0689411a91b9609a336d833eb
- edges_on_cell_w6.i32.gz: allocation 74 (edges_on_cell), 15372 words, sha256 2310adab74aad8dc62f924232199e595abb4572b4dc77a4124b868dc6ab83d29; payload b8299eb028f5b0ed5fbbc7cae9c5528e67f3ccfde2bb6934bb76d5f3f71a8578
- n_edges_on_cell.i32.gz: allocation 29 (n_edges_on_cell), 2562 words, sha256 98c7fd5fdab896bb9350099b1bbf2532813c23b3b3c5e8491a7d7b94c33d18b8; payload bcf256e04a0f3fb882ab0a385588daf8b74786835e47198fd7432db7797a2c56
- vertices_on_edge.i32.gz: allocation 32 (vertices_on_edge), 15360 words, sha256 543c7cc9b6f1a806c8174b04d8747b16859070ae0a37604024e56e71bb631831; payload 543c7cc9b6f1a806c8174b04d8747b16859070ae0a37604024e56e71bb631831
- edges_on_vertex.i32.gz: allocation 36 (edges_on_vertex), 15360 words, sha256 3c63a50a5c13f7152cb74df6ada3a80481424976c617390aa6ed0cc1e9b704c8; payload 3c63a50a5c13f7152cb74df6ada3a80481424976c617390aa6ed0cc1e9b704c8
- cells_on_cell.i32.gz: allocation 30 (cells_on_cell), 25620 words, sha256 8bb479e0c97ed13605cc1fc099f41e84d950b50f17c83ad1311a0e28a8d7c230; payload 20126d89495fb4191cd6e0b08831a5d6efb3e6fbb6854aae089e3f1f07688427
