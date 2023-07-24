add_executable(main.xc main.cpp)
target_link_libraries(main.xc Kokkos::kokkos)

if(${adios2})
  if(${mpi})
    target_link_libraries(main.xc adios2::cxx11_mpi)
  else()
    target_link_libraries(main.xc adios2::cxx11)
  endif()

  add_compile_options("-D ADIOS2_ENABLED")
endif()

if(${mpi})
  target_link_libraries(main.xc MPI::MPI_CXX)
  target_include_directories(main.xc PUBLIC ${MPI_CXX_INCLUDE_PATH})
  add_compile_options("-D MPI_ENABLED")
endif()