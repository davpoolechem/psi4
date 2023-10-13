if(${ENABLE_gpu_dlpno_ccsd})
    find_package(gpu_dlpno_ccsd 0.1 CONFIG QUIET)
    if(${gpu_dlpno_ccsd_FOUND})
        get_property(_loc TARGET gpu_dlpno_ccsd::gpu_dlpno_ccsd PROPERTY LOCATION)
        message(STATUS "${Cyan}Found gpu_dlpno_ccsd${ColourReset}: ${_loc} (found version ${gpu_dlpno_ccsd_VERSION})")
        add_library(gpu_dlpno_ccsd_external INTERFACE)  # dummy

    else()
        if(${CMAKE_INSIST_FIND_PACKAGE_gpu_dlpno_ccsd})
            message(FATAL_ERROR "Suitable gpu_dlpno_ccsd could not be externally located as user insists")
        endif()

        include(ExternalProject)
        message(STATUS "Suitable gpu_dlpno_ccsd could not be located, ${Magenta}Building gpu_dlpno_ccsd${ColourReset} instead.")

        ExternalProject_Add(gpu_dlpno_ccsd_external
          DEPENDS psi4-core
          GIT_REPOSITORY https://github.com/davpoolechem/gpu_dlpno_ccsd
          GIT_TAG main
          CMAKE_ARGS  -DCMAKE_INSTALL_PREFIX=${STAGED_INSTALL_PREFIX}
                       -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                       -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                       -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                       -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
                       -DPYMOD_INSTALL_LIBDIR=${PYMOD_INSTALL_LIBDIR}
                       -DENABLE_XHOST=${ENABLE_XHOST}
                       -DENABLE_GENERIC=${ENABLE_GENERIC}
                       -DOpenMP_LIBRARY_DIRS=${OpenMP_LIBRARY_DIRS}
                       -Dpsi4_DIR=${STAGED_INSTALL_PREFIX}/share/cmake/psi4
                       -Dpybind11_DIR=${pybind11_DIR}
                       -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
                       -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
            CMAKE_CACHE_ARGS -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
                             -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
                             -DCMAKE_Fortran_FLAGS:STRING=${CMAKE_Fortran_FLAGS}
                             -DCMAKE_PREFIX_PATH:PATH=${STAGED_INSTALL_PREFIX})

        set(gpu_dlpno_ccsd_DIR ${STAGED_INSTALL_PREFIX}/share/cmake/gpu_dlpno_ccsd CACHE PATH "path to internally built gpu_dlpno_ccsdConfig.cmake" FORCE)
    endif()
else()
    add_library(gpu_dlpno_ccsd_external INTERFACE)  # dummy
endif()


