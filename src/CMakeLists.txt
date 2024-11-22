#
# Library for integral reading
#

project(integrals-lib)

add_library(integrals
  chunk_reader.cpp	
  reader.cpp
  buffer.cpp
)

find_package(Eigen3 REQUIRED)
find_package(MPI REQUIRED)
find_package(BLAS)

if (BLAS_FOUND)
    if (BLAS_LIBRARIES MATCHES ".*Accelerate")
        message(INFO " Using Apple BLAS")
    endif ()
    if (BLAS_LIBRARIES MATCHES ".*openblas")
        message(INFO " Using OpenBLAS")
        add_compile_definitions(USE_GEMM3M)
    endif ()
    add_compile_definitions(EIGEN_USE_BLAS)
endif ()



target_link_libraries(integrals PUBLIC GREEN::H5PP GREEN::NDARRAY GREEN::PARAMS Eigen3::Eigen MPI::MPI_CXX)
if(BLAS_FOUND)
    target_link_libraries(integrals PUBLIC BLAS::BLAS)
endif(BLAS_FOUND)
target_include_directories(integrals PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
