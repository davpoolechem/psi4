list(APPEND sources
  fmm_box.cc
  direct_fmm_tree.cc
  df_fmm_tree.cc
  fmm_shell_pair.cc
  fmm_tree.cc
  multipoles_helper.cc
  )

psi4_add_module(lib fmm sources)
