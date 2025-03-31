# Add the CUDA source file explicitly to distinguish it from the rest
set(CUDA_SOURCES
    physical_GPU_groupby.cu
)

add_library_unity(
  duckdb_operator_aggregate
  OBJECT
  aggregate_object.cpp
  distinct_aggregate_data.cpp
  physical_hash_aggregate.cpp
  grouped_aggregate_data.cpp
  physical_partitioned_aggregate.cpp
  physical_perfecthash_aggregate.cpp
  physical_ungrouped_aggregate.cpp
  physical_window.cpp
  physical_streaming_window.cpp)

# Add CUDA sources separately to ensure proper handling
target_sources(duckdb_operator_aggregate PRIVATE ${CUDA_SOURCES})

 #Add required compiler flags
 target_compile_options(duckdb_operator_aggregate PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
     --expt-relaxed-constexpr
     --expt-extended-lambda
 >
)

set_target_properties(duckdb_operator_aggregate PROPERTIES
    # CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "61"  # Set appropriate architectures for your hardware
)


# Link Warpcore and Threads
target_link_libraries(duckdb_operator_aggregate PUBLIC warpcore Threads::Threads CUDA::cudart)

# Set CUDA standard for the target
# set_target_properties(duckdb_operator_aggregate PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)


set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_operator_aggregate>
    PARENT_SCOPE)
