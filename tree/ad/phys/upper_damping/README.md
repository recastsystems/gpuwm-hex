# upper_damping: derivative rows for NOAA's upper damping

NOAA's hrrrv5 deck (rrfs-workflow 570c81fe parm/hrrrv5/namelist.atmosphere) runs two launches every step that the
Registry defaults never ran, so no kernel set had a row for them and the window refused any recording on the deck:

| row | forward (hexcore/cuda_upper_damping_v841.py) | derivative |
|---|---|---|
| upper_cam_f32 | kdiff = max(kdiff, floor(k)) on the top config_number_cam_damping_levels levels | executed arm: passes where kdiff > floor, 0 where the floor was taken (tie takes the floor) |
| upper_rayleigh_f32 | tend_u -= (rho_edge*u)*c(k) on the top levels | linear in rho_edge and u; elementwise reverse, no gather |

- `cuda/*.ad.cu`: the tangent (`__tl`) and adjoint (`__ad`) entries; `mkspecs.py`: their ad.entry/v1 specs and kernel-set rows;
  `build.sh OUT`: images (sm_89, sm_120, compute_80; --fmad=false --ftz=false) and specs into OUT; `add_rows.py [--copy] BASE OUT BUILD`:
  a kernel set = BASE plus the two rows.
- `kset/`: the two rows and their specs as `mkspecs.py` writes them. Each spec names the forward text the row is
  written against (`forward_source_sha256`); if that text changes, the rows are read against it again and rebuilt.
- The per-launch transpose check is the runner's replay with `AD_DOT_ONLY=upper_cam_f32,upper_rayleigh_f32` on a
  package recorded on the deck; the window check then reverses those launches.
