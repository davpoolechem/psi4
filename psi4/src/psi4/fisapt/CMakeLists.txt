list(APPEND sources
  fisapt.cc
  local2.cc
  )
psi4_add_module(bin fisapt sources)

if(TARGET gauxc::gauxc)
  target_include_directories(fisapt
    PUBLIC
      $<TARGET_PROPERTY:gauxc::gauxc,INTERFACE_INCLUDE_DIRECTORIES>
    )
endif()

target_link_libraries(fisapt PUBLIC diis)
