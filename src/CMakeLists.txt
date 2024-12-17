#
# Library for integral reading
#

project(integrals-lib)

add_library(integrals
  chunk_reader.cpp	
  reader.cpp
  buffer.cpp
)

target_link_libraries(integrals PUBLIC GREEN::H5PP GREEN::NDARRAY GREEN::PARAMS Eigen3::Eigen MPI::MPI_CXX)
if(BLAS_FOUND)
    target_link_libraries(integrals PUBLIC BLAS::BLAS)
endif(BLAS_FOUND)
target_include_directories(integrals PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
