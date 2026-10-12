cmake_minimum_required(VERSION 3.14)
project(hptp VERSION 1.0.0 LANGUAGES C CXX)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(HPTP_TOP_LEVEL ON)
else()
  set(HPTP_TOP_LEVEL OFF)
endif()
option(HPTP_BUILD_TESTS "Build the tests" ${HPTP_TOP_LEVEL})
option(HPTP_BUILD_BENCH "Build the C++ benchmark" ${HPTP_TOP_LEVEL})

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# Header-only C++ library: #include "hptp.hpp".
add_library(hptp_headers INTERFACE)
add_library(hptp::headers ALIAS hptp_headers)
target_include_directories(hptp_headers INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                                  $<INSTALL_INTERFACE:include>)
target_compile_features(hptp_headers INTERFACE cxx_std_17)
target_link_libraries(hptp_headers INTERFACE Threads::Threads)

# Compiled library with the C interface of hptp.h (static by default; -DBUILD_SHARED_LIBS=ON for a shared one).
add_library(hptp src/hptp.cpp)
add_library(hptp::hptp ALIAS hptp)
target_link_libraries(hptp PUBLIC hptp_headers)
target_compile_definitions(hptp PRIVATE HPTP_BUILDING)
if(BUILD_SHARED_LIBS)
  target_compile_definitions(hptp PUBLIC HPTP_SHARED)
  set_target_properties(hptp PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()
set_target_properties(hptp PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

if(HPTP_BUILD_TESTS)
  enable_testing()
  add_executable(test_permute tests/test_permute.cpp)
  target_link_libraries(test_permute PRIVATE hptp_headers)
  add_test(NAME test_permute COMMAND test_permute)
  add_executable(test_permute_small_blocks tests/test_permute.cpp)
  target_link_libraries(test_permute_small_blocks PRIVATE hptp_headers)
  target_compile_definitions(test_permute_small_blocks PRIVATE HPTP_ROW_BLOCK_BYTES=64 HPTP_SHORT_ROW_BYTES=24
                             HPTP_WIDE_STRIDE_BYTES=96 HPTP_WIDE_TILE_COLS=3 HPTP_COL_TILE_BYTES=40
                             HPTP_BLOCK_BYTES=200)
  add_test(NAME test_permute_small_blocks COMMAND test_permute_small_blocks)
  add_executable(test_permute_no_swar tests/test_permute.cpp)
  target_link_libraries(test_permute_no_swar PRIVATE hptp_headers)
  target_compile_definitions(test_permute_no_swar PRIVATE HPTP_NO_SWAR)
  add_test(NAME test_permute_no_swar COMMAND test_permute_no_swar 600)
  add_executable(test_capi tests/test_capi.c)
  target_link_libraries(test_capi PRIVATE hptp)
  set_target_properties(test_capi PROPERTIES LINKER_LANGUAGE CXX)
  add_test(NAME test_capi COMMAND test_capi)
endif()

if(HPTP_BUILD_BENCH)
  add_executable(bench_permute bench/bench_permute.cpp)
  target_link_libraries(bench_permute PRIVATE hptp_headers)
endif()

include(GNUInstallDirs)
install(TARGETS hptp hptp_headers EXPORT hptpTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/hptp.h include/hptp.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT hptpTargets NAMESPACE hptp:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hptp)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/hptpConfig.cmake
     "include(CMakeFindDependencyMacro)\nfind_dependency(Threads)\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/hptpTargets.cmake\")\n")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hptpConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hptp)
