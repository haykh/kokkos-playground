cmake_minimum_required(VERSION 3.16)
project(Kokkos-Adios2-MPI-Boilerplate LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(${Kokkos_ENABLE_CUDA})
  enable_language(CUDA)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Kokkos REQUIRED)

if(${adios2})
  find_package(adios2 REQUIRED)
  add_compile_options("-D ADIOS2_ENABLED")
endif()

if(${mpi})
  find_package(MPI REQUIRED)
  add_compile_options("-D MPI_ENABLED")
endif()

add_subdirectory(src)