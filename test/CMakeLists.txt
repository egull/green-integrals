project(integrals_test)

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.3 # or a later release
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_executable(integrals_test
test_chunk_reader.cpp
test_access_counter.cpp		#test_hash.cpp
test_age_out_buffer.cpp		test_main.cpp
test_buffer.cpp			test_mpi.cpp
test_bufferedSI.cpp		test_shared_memory_region.cpp
)
target_compile_definitions(integrals_test PRIVATE TEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(integrals_test
        PRIVATE
        Catch2::Catch2WithMain
        GREEN::INTEGRALS)

include(CTest)
include(Catch)
catch_discover_tests(integrals_test)

