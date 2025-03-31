#include "duckdb/execution/operator/aggregate/physical_GPU_groupby.hpp"

#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/parallel/base_pipeline_event.hpp"
#include "duckdb/parallel/interrupt.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/parallel/executor_task.hpp"

#include <warpcore/multi_value_hash_table.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

// -----------------------------------------------------------------------------
// Warpcore definitions
// -----------------------------------------------------------------------------
using gpu_key_t = uint64_t;
using gpu_val_t = uint64_t;

using gpu_ht_t = warpcore::MultiValueHashTable<
    gpu_key_t,
    gpu_val_t,
    std::numeric_limits<gpu_key_t>::max(),     // empty key sentinel
    std::numeric_limits<gpu_key_t>::max() - 1  // tombstone sentinel
>;

// Simple CUDA kernel to compute counts per group
__global__ void compute_counts(const warpcore::index_t *d_begin_offsets,
                               const warpcore::index_t *d_end_offsets,
                               gpu_val_t *d_counts,
                               int num_groups)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_groups) {
		// #values in this group = d_end_offsets[idx] - d_begin_offsets[idx]
		d_counts[idx] = d_end_offsets[idx] - d_begin_offsets[idx];
	}
}

namespace duckdb {

// -----------------------------------------------------------------------------
// Global Sink State
// -----------------------------------------------------------------------------
struct GPUGroupByGlobalSinkState : public GlobalSinkState {
	GPUGroupByGlobalSinkState(ClientContext &context, const vector<LogicalType> &types_p)
	    : finalized(false), build_size(0) {
		build_chunk.Initialize(Allocator::Get(context), types_p);
	}

	// We won't free device pointers in the destructor because we do it in Finalize()
	// If you want a safety check, you can do so carefully (checking for null).
	~GPUGroupByGlobalSinkState() override {
		// e.g., if still allocated, free them. But typically we do it in Finalize()
	}

	//! Single chunk for all data
	DataChunk build_chunk;
	idx_t build_size;

	//! CPU side vectors
	std::vector<gpu_key_t> host_keys; // the group keys from DuckDB
	std::vector<gpu_val_t> host_vals; // some "value" (row_id, etc.)

	//! GPU memory
	gpu_key_t         *d_keys          = nullptr;
	gpu_val_t         *d_vals          = nullptr;
	gpu_key_t         *d_group_keys    = nullptr;
	warpcore::index_t *d_begin_offsets = nullptr;
	warpcore::index_t *d_end_offsets   = nullptr;
	gpu_val_t         *d_group_vals    = nullptr;
	gpu_val_t         *d_counts        = nullptr;

	//! Warpcore hash table
	std::unique_ptr<gpu_ht_t> gpu_hash_table;
	warpcore::index_t num_groups = 0;  // how many unique groups found
	gpu_val_t         num_values = 0;  // total #values across all groups

	//! We'll store final results on CPU after retrieving from GPU
	std::vector<uint64_t> final_group_keys;
	std::vector<uint64_t> final_group_counts;

	//! concurrency
	std::mutex lock;
	bool finalized;
};

// -----------------------------------------------------------------------------
// Local Sink State
// -----------------------------------------------------------------------------
struct GPUGroupByLocalSinkState : public LocalSinkState {
	GPUGroupByLocalSinkState(const vector<LogicalType> &types_p) : local_size(0) {
		local_chunk.Initialize(Allocator::DefaultAllocator(), types_p);
	}
	DataChunk local_chunk;
	idx_t local_size;
};

// -----------------------------------------------------------------------------
// Global/Local Source State
// -----------------------------------------------------------------------------
struct GPUGroupByGlobalSourceState : public GlobalSourceState {
	GPUGroupByGlobalSourceState() : done(false), current_idx(0) {}
	bool done;
	idx_t current_idx;
};

struct GPUGroupByLocalSourceState : public LocalSourceState {
};

// -----------------------------------------------------------------------------
// Operator Constructor
// -----------------------------------------------------------------------------
PhysicalGPUGroupBy::PhysicalGPUGroupBy(vector<LogicalType> types_p,
                                       vector<unique_ptr<Expression>> groups_p,
                                       idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::GPU_GROUP_BY, std::move(types_p), estimated_cardinality) {
	for (auto &expr : groups_p) {
		if (expr->type == ExpressionType::BOUND_REF) {
			auto &bound_ref = expr->Cast<BoundReferenceExpression>();
			groupby_columns.push_back(bound_ref.index);
		} else {
			throw NotImplementedException("Only BoundReferenceExpression supported in GPUGroupBy!");
		}
	}
	if (groupby_columns.size() != 1) {
		throw NotImplementedException("Only single grouping column is supported in this example.");
	}
}

// -----------------------------------------------------------------------------
// SINK interface
// -----------------------------------------------------------------------------
unique_ptr<GlobalSinkState> PhysicalGPUGroupBy::GetGlobalSinkState(ClientContext &context) const {
	auto &child_types = children[0]->GetTypes();
	return make_uniq<GPUGroupByGlobalSinkState>(context, child_types);
}

unique_ptr<LocalSinkState> PhysicalGPUGroupBy::GetLocalSinkState(ExecutionContext &context) const {
	auto &child_types = children[0]->GetTypes();
	return make_uniq<GPUGroupByLocalSinkState>(child_types);
}

SinkResultType PhysicalGPUGroupBy::Sink(ExecutionContext &context, DataChunk &chunk,
                                        OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<GPUGroupByLocalSinkState>();
	if (!chunk.size()) {
		return SinkResultType::NEED_MORE_INPUT;
	}
	// Accumulate all rows into a local chunk
	lstate.local_chunk.Append(chunk, true);
	lstate.local_size += chunk.size();
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalGPUGroupBy::Combine(ExecutionContext &context,
                                                  OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<GPUGroupByGlobalSinkState>();
	auto &lstate = input.local_state.Cast<GPUGroupByLocalSinkState>();

	if (lstate.local_size == 0) {
		return SinkCombineResultType::FINISHED;
	}
	{
		std::lock_guard<std::mutex> lock(gstate.lock);
		// Combine local_chunk => global build_chunk
		gstate.build_chunk.Append(lstate.local_chunk, true);
		gstate.build_size += lstate.local_size;
	}
	return SinkCombineResultType::FINISHED;
}

// -----------------------------------------------------------------------------
// FINALIZE (SINGLE-THREADED)
// -----------------------------------------------------------------------------
SinkFinalizeType PhysicalGPUGroupBy::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                              OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<GPUGroupByGlobalSinkState>();

	// no data => short circuit
	if (gstate.build_size == 0) {
		gstate.finalized = true;
		return SinkFinalizeType::READY;
	}

	// -----------------------------------------------------------------------------
	// 1) Convert build_chunk => CPU host_keys/host_vals
	// -----------------------------------------------------------------------------
	idx_t group_col_idx = groupby_columns[0];
	auto &group_col = gstate.build_chunk.data[group_col_idx];

	UnifiedVectorFormat group_data;
	group_col.ToUnifiedFormat(gstate.build_size, group_data);

	auto keys_ptr = reinterpret_cast<const int32_t*>(group_data.data);
	gstate.host_keys.reserve(gstate.build_size);
	gstate.host_vals.reserve(gstate.build_size);

	for (idx_t i = 0; i < gstate.build_size; i++) {
		auto sel_idx = group_data.sel->get_index(i);
		if (!group_data.validity.RowIsValid(sel_idx)) {
			// skip nulls
			continue;
		}
		gpu_key_t k = static_cast<gpu_key_t>(keys_ptr[sel_idx]);
		gstate.host_keys.push_back(k);

		// For example, store row_id as the "value"
		gstate.host_vals.push_back((gpu_val_t)i);
	}
	idx_t final_count = gstate.host_keys.size();
	if (final_count == 0) {
		// all rows were null
		gstate.finalized = true;
		return SinkFinalizeType::READY;
	}

	// -----------------------------------------------------------------------------
	// 2) Copy (key, val) to GPU
	// -----------------------------------------------------------------------------
	cudaMalloc(&gstate.d_keys, sizeof(gpu_key_t) * final_count);
	cudaMalloc(&gstate.d_vals, sizeof(gpu_val_t) * final_count);

	cudaMemcpy(gstate.d_keys, gstate.host_keys.data(), final_count * sizeof(gpu_key_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gstate.d_vals, gstate.host_vals.data(), final_count * sizeof(gpu_val_t), cudaMemcpyHostToDevice);

	// -----------------------------------------------------------------------------
	// 3) Build Warpcore hash table
	// -----------------------------------------------------------------------------
	float load_factor = 0.9f;
	uint64_t capacity = static_cast<uint64_t>(static_cast<double>(final_count) / load_factor);

	gstate.gpu_hash_table = std::make_unique<gpu_ht_t>(capacity);
	gstate.gpu_hash_table->init(); // initialize internal structures

	// Insert all pairs
	gstate.gpu_hash_table->insert(gstate.d_keys, gstate.d_vals, final_count);
	cudaDeviceSynchronize();

	// -----------------------------------------------------------------------------
	// 4) retrieve_all => get group keys + group "value-lists" offsets
	// -----------------------------------------------------------------------------
	cudaMalloc(&gstate.d_group_keys,    sizeof(gpu_key_t)         * capacity);
	cudaMalloc(&gstate.d_begin_offsets, sizeof(warpcore::index_t) * capacity);
	cudaMalloc(&gstate.d_end_offsets,   sizeof(warpcore::index_t) * capacity);
	cudaMalloc(&gstate.d_group_vals,    sizeof(gpu_val_t)         * final_count);
	cudaMalloc(&gstate.d_counts,        sizeof(gpu_val_t)         * capacity);

	gstate.num_groups = 0;   // will be updated by retrieve_all
	gstate.num_values = 0;   // total # of values across all groups

	gstate.gpu_hash_table->retrieve_all(
	    gstate.d_group_keys,
	    gstate.num_groups,
	    gstate.d_begin_offsets,
	    gstate.d_end_offsets,
	    gstate.d_group_vals,
	    gstate.num_values,
	    0 // stream ID
	);
	cudaDeviceSynchronize();

	// -----------------------------------------------------------------------------
	// 5) compute_counts => run a simple kernel
	// -----------------------------------------------------------------------------
	int threads = 256;
	int blocks  = (gstate.num_groups + threads - 1) / threads;
	compute_counts<<<blocks, threads>>>(gstate.d_begin_offsets,
	                                   gstate.d_end_offsets,
	                                   gstate.d_counts,
	                                   gstate.num_groups);
	cudaDeviceSynchronize();

	if (gstate.num_groups == 0) {
		// no unique keys found
		gstate.finalized = true;
		return SinkFinalizeType::READY;
	}

	// -----------------------------------------------------------------------------
	// 6) Copy final group results back to CPU
	// -----------------------------------------------------------------------------
	// We'll store them in final_group_keys/final_group_counts
	gstate.final_group_keys.resize(gstate.num_groups);
	gstate.final_group_counts.resize(gstate.num_groups);

	cudaMemcpy(gstate.final_group_keys.data(),
	           gstate.d_group_keys,
	           sizeof(gpu_key_t) * gstate.num_groups,
	           cudaMemcpyDeviceToHost);

	cudaMemcpy(gstate.final_group_counts.data(),
	           gstate.d_counts,
	           sizeof(gpu_val_t) * gstate.num_groups,
	           cudaMemcpyDeviceToHost);

	// If you want, you could also gather the group values from d_group_vals,
	// but in this example we're only counting how many values are in each group.

	// -----------------------------------------------------------------------------
	// 7) Immediately free GPU memory to be memory-efficient
	// -----------------------------------------------------------------------------
	gstate.gpu_hash_table.reset(); // destroys Warpcore HT
	cudaFree(gstate.d_keys);
	cudaFree(gstate.d_vals);
	cudaFree(gstate.d_group_keys);
	cudaFree(gstate.d_begin_offsets);
	cudaFree(gstate.d_end_offsets);
	cudaFree(gstate.d_group_vals);
	cudaFree(gstate.d_counts);

	gstate.d_keys          = nullptr;
	gstate.d_vals          = nullptr;
	gstate.d_group_keys    = nullptr;
	gstate.d_begin_offsets = nullptr;
	gstate.d_end_offsets   = nullptr;
	gstate.d_group_vals    = nullptr;
	gstate.d_counts        = nullptr;

	gstate.finalized = true;
	return SinkFinalizeType::READY;
}

// -----------------------------------------------------------------------------
// SOURCE interface
// -----------------------------------------------------------------------------
unique_ptr<GlobalSourceState> PhysicalGPUGroupBy::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<GPUGroupByGlobalSourceState>();
}

unique_ptr<LocalSourceState> PhysicalGPUGroupBy::GetLocalSourceState(ExecutionContext &context,
                                                                     GlobalSourceState &gstate) const {
	return make_uniq<GPUGroupByLocalSourceState>();
}

SourceResultType PhysicalGPUGroupBy::GetData(ExecutionContext &context, DataChunk &chunk,
                                             OperatorSourceInput &input) const {
	auto &sink = sink_state->Cast<GPUGroupByGlobalSinkState>();
	auto &gstate = input.global_state.Cast<GPUGroupByGlobalSourceState>();

	// If no data or not finalized, we're done
	if (!sink.finalized || sink.build_size == 0) {
		chunk.SetCardinality(0);
		return SourceResultType::FINISHED;
	}

	// If we've already produced all groups
	if (gstate.done && gstate.current_idx >= sink.final_group_keys.size()) {
		chunk.SetCardinality(0);
		return SourceResultType::FINISHED;
	}

	// We simply read from CPU vectors now (final_group_keys, final_group_counts)
	idx_t start_idx = gstate.current_idx;
	idx_t end_idx   = std::min<idx_t>(start_idx + STANDARD_VECTOR_SIZE, sink.final_group_keys.size());
	idx_t out_count = end_idx - start_idx;

	chunk.Initialize(Allocator::DefaultAllocator(), this->types);
	auto &group_col = chunk.data[0];
	auto &count_col = chunk.data[1];

	auto *gptr = FlatVector::GetData<int32_t>(group_col);
	auto *cptr = FlatVector::GetData<int64_t>(count_col);

	for (idx_t i = 0; i < out_count; i++) {
		idx_t idx = start_idx + i;
		gptr[i]   = (int32_t)sink.final_group_keys[idx];
		cptr[i]   = (int64_t)sink.final_group_counts[idx];
	}
	chunk.SetCardinality(out_count);

	gstate.current_idx += out_count;
	if (gstate.current_idx >= sink.final_group_keys.size()) {
		gstate.done = true;
	}
	return (out_count > 0) ? SourceResultType::HAVE_MORE_OUTPUT
	                      : SourceResultType::FINISHED;
}

// -----------------------------------------------------------------------------
// EXPLAIN
// -----------------------------------------------------------------------------
InsertionOrderPreservingMap<string> PhysicalGPUGroupBy::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["GPU GroupBy"] = "Warpcore MultiValueHashTable-based single key, COUNT(*) aggregator";
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

} // namespace duckdb