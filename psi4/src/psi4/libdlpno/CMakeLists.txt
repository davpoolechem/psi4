list(APPEND sources
  dlpnobase.cc
  mp2.cc
  ccsd.cc
  triples.cc
  wrapper.cc
  sparse.cc
  )
psi4_add_module(lib dlpno sources)

target_link_libraries(dlpno PUBLIC diis)
