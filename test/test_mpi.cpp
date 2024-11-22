#include <catch2/catch_test_macros.hpp>
#include<iostream>
#include <mpi.h>

TEST_CASE("Size", "[MPI]") {
  int size=0;
  int rank=0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if(size==1){ std::cerr<<"MPI size is 1. most tests and possible problems will occur with MPI sizes >1"<<std::endl; return;}


  REQUIRE(size>=2);  //if this throws run the tests with mpirun -np X, X>=2. 
}
