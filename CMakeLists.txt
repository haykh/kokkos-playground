cmake_minimum_required(VERSION 3.16)
project(Kokkos-Adios2-MPI-Boilerplate LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(${Kokkos_ENABLE_CUDA})
  enable_language(CUDA)
endif()

if(${Kokkos_ENABLE_HIP})
  enable_language(HIP)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/Kokkos extern/Kokkos)
# set(Kokkos_SRC ${CMAKE_CURRENT_SOURCE_DIR}/extern/Kokkos CACHE PATH "Path to Kokkos src")
# set(Kokkos_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/extern/Kokkos CACHE PATH "Path to Kokkos build")

# find_package(Kokkos REQUIRED)

include(FetchContent)
FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG master
)
FetchContent_MakeAvailable(Kokkos)

if(${adios2})
  find_package(adios2 REQUIRED)
  add_compile_options("-D ADIOS2_ENABLED")
endif()

if(${mpi})
  find_package(MPI REQUIRED)
  add_compile_options("-D MPI_ENABLED")
endif()

add_subdirectory(src)
