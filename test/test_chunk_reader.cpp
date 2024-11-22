#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "green/integrals/buffered_reader/chunk_reader.hpp"
#include <mpi.h>
#include <chrono>
#include <thread>
#include <filesystem>


TEST_CASE("ReadFakeData","[reader]") {

  int chunks_per_file=336;
  int total_files=36;
  int nao=26;
  int naux=200;
  int number_of_keys=chunks_per_file*total_files;

  reader c("no_such_file", number_of_keys, naux, nao); //test these numbers

  Eigen::VectorXd buffer(naux*nao*nao*2);
  c.read_key(0, &(buffer[0]));

  REQUIRE(buffer[0]==42);
  REQUIRE(buffer[naux]== 42.);
  REQUIRE(buffer[naux*nao*nao*2-1]==42.);
}

TEST_CASE("Init","[chunk_reader]") {

  chunk_reader c;
}
//these tests are straight from the Si 6x6x6 example
TEST_CASE("InitBasePath","[chunk_reader]") {

  int chunks_per_file=336;
  int total_files=36;
  int nao=26;
  int naux=200;
  int number_of_keys=chunks_per_file*total_files;

  if(!std::filesystem::exists(TEST_PATH)){ std::cerr<<"hdf5 data not found. aborting test"<<std::endl; return;}

  chunk_reader c(TEST_PATH, number_of_keys, naux, nao); //test these numbers

  REQUIRE(c.chunk_indices()[0]==0);
  REQUIRE(c.chunk_indices()[1]==336);
  REQUIRE(c.chunk_indices()[35]==11760);
}

TEST_CASE("ReadSomething","[chunk_reader]") {
  int chunks_per_file=336;
  int total_files=36;
  int nao=26;
  int naux=200;
  int number_of_keys=chunks_per_file*total_files;

  if(!std::filesystem::exists(TEST_PATH)){ std::cerr<<"hdf5 data not found. aborting test"<<std::endl; return;}
  chunk_reader c(TEST_PATH, number_of_keys, naux, nao,true); //test these numbers

  Eigen::VectorXd data(c.element_size());
  c.read_key(0, &(data[0]));
  REQUIRE_THAT(data[0], Catch::Matchers::WithinAbs(5.26945, 1.e-5));
  c.read_key(335, &(data[0]));
  c.read_key(336, &(data[0]));
  c.read_key(2000, &(data[0]));
}
