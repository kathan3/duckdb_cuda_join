# Add the CUDA source file explicitly to distinguish it from the rest
set(CUDA_SOURCES
    physical_GPU_join.cu
)

# Add Unity Build for other files
add_library_unity(
  duckdb_operator_join
  OBJECT
  outer_join_marker.cpp
  physical_asof_join.cpp
  physical_blockwise_nl_join.cpp
  physical_comparison_join.cpp
  physical_cross_product.cpp
  physical_delim_join.cpp
  physical_left_delim_join.cpp
  physical_hash_join.cpp
  physical_iejoin.cpp
  physical_join.cpp
  physical_nested_loop_join.cpp
  perfect_hash_join_executor.cpp
  physical_piecewise_merge_join.cpp
  physical_positional_join.cpp
  physical_range_join.cpp
  physical_right_delim_join.cpp
)

# Add CUDA sources separately to ensure proper handling
target_sources(duckdb_operator_join PRIVATE ${CUDA_SOURCES})

 #Add required compiler flags
 target_compile_options(duckdb_operator_join PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
     --expt-relaxed-constexpr
     --expt-extended-lambda
 >
)

set_target_properties(duckdb_operator_join PROPERTIES
    # CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "61"  # Set appropriate architectures for your hardware
)


# Link Warpcore and Threads
target_link_libraries(duckdb_operator_join PUBLIC warpcore Threads::Threads CUDA::cudart)

# Set CUDA standard for the target
# set_target_properties(duckdb_operator_join PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_operator_join>
    PARENT_SCOPE)