cmake_minimum_required(VERSION 3.16)
project(CPP-Kokkos-Boilerplate LANGUAGES CXX CUDA)

include(FetchContent)
FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG master
)
FetchContent_MakeAvailable(Kokkos)

add_subdirectory(src)