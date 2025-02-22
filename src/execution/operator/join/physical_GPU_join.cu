//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/join/physical_GPU_join.cu
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/operator/join/physical_GPU_join.hpp"
#include "duckdb/execution/operator/join/physical_comparison_join.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

// NEW INCLUDES for CachingOperatorState
#include "duckdb/execution/physical_operator.hpp" // for CachingOperatorState

// *** Use MultiValueHashTable here instead of SingleValueHashTable ***
#include <warpcore/multi_value_hash_table.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <unordered_map>

#include <chrono>
#include <cstdio>
#include <string>

// Simple RAII timer: starts timing on construction and prints elapsed time on destruction.
struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    std::string name;
    
    Timer(const std::string &name) : name(name) {
        start = std::chrono::high_resolution_clock::now();
    }
    
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        // Print elapsed time in microseconds; adjust as needed.
        printf("Timer [%s]: %ld microseconds\n", name.c_str(), duration);
    }
};

namespace duckdb {

// ------------------------------------------------------------------------
// We mirror the same approach as "PhysicalHashJoin" does in DuckDB
// ------------------------------------------------------------------------

// A simplistic GPU hash table for single integer (64-bit) keys
using key_t = uint64_t;
using value_t = uint64_t;
// *** Replaced SingleValueHashTable with MultiValueHashTable ***
using hash_table_t = warpcore::MultiValueHashTable<key_t, value_t,
                              std::numeric_limits<key_t>::max(),      // empty key
                              std::numeric_limits<key_t>::max() - 1>;    // tombstone key  


//===--------------------------------------------------------------------===//
// Constructor
//===--------------------------------------------------------------------===//

PhysicalKathanJoin::PhysicalKathanJoin(LogicalOperator &op,
                                       unique_ptr<PhysicalOperator> left,
                                       unique_ptr<PhysicalOperator> right,
                                       vector<JoinCondition> cond,
                                       JoinType join_type,
                                       const vector<idx_t> &left_projection_map,
                                       const vector<idx_t> &right_projection_map,
                                       idx_t estimated_cardinality)
    : PhysicalComparisonJoin(op, PhysicalOperatorType::GPU_JOIN, std::move(cond), join_type, estimated_cardinality) {
	printf("=== PhysicalGPUJoin: Constructor called ===\n");

	children.push_back(std::move(left));
	children.push_back(std::move(right));

	// We now support multiple conditions (not just one).
    // For each condition, check if the RHS is a BoundReferenceExpression
       for (auto &condition : conditions) {
        if (condition.right->GetExpressionClass() == ExpressionClass::BOUND_REF) {
            auto &bound_ref = condition.right->Cast<BoundReferenceExpression>();
            printf(">>> Found build key at column index = %lld\n", (long long)bound_ref.index);
			auto &kathan = condition.left->Cast<BoundReferenceExpression>();
			printf(">>> Found probe key at column index = %lld\n", (long long)kathan.index);
            build_key_indices.push_back(bound_ref.index);
        } else {
            // Some other expression type on the RHS => not supported
            throw NotImplementedException("RHS must be a BoundReferenceExpression for GPU join keys");
        }
    }


	// Collect condition types
	unordered_map<idx_t, idx_t> build_columns_in_conditions;
	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
		condition_types.push_back(condition.left->return_type);
		if (condition.right->GetExpressionClass() == ExpressionClass::BOUND_REF) {
			build_columns_in_conditions.emplace(condition.right->Cast<BoundReferenceExpression>().index, cond_idx);
		}
	}

	auto &lhs_input_types = children[0]->GetTypes();

	// Create a projection map for the LHS if none was provided
	lhs_output_columns.col_idxs = left_projection_map;
	if (lhs_output_columns.col_idxs.empty()) {
		lhs_output_columns.col_idxs.reserve(lhs_input_types.size());
		for (idx_t i = 0; i < lhs_input_types.size(); i++) {
			lhs_output_columns.col_idxs.emplace_back(i);
		}
	}
	for (auto &lhs_col : lhs_output_columns.col_idxs) {
		auto &lhs_col_type = lhs_input_types[lhs_col];
		lhs_output_columns.col_types.push_back(lhs_col_type);
	}

	// Create a projection map for the RHS if none was provided
	auto &rhs_input_types = children[1]->GetTypes();
	auto right_projection_map_copy = right_projection_map;
	if (right_projection_map_copy.empty()) {
		right_projection_map_copy.reserve(rhs_input_types.size());
		for (idx_t i = 0; i < rhs_input_types.size(); i++) {
			right_projection_map_copy.emplace_back(i);
		}
	}

	// Now fill payload expressions/types and RHS columns/types
	for (auto &rhs_col : right_projection_map_copy) {
		auto &rhs_col_type = rhs_input_types[rhs_col];
		auto it = build_columns_in_conditions.find(rhs_col);
		if (it == build_columns_in_conditions.end()) {
			// This rhs column is not a join key
			payload_columns.col_idxs.push_back(rhs_col);
			payload_columns.col_types.push_back(rhs_col_type);
			rhs_output_columns.col_idxs.push_back(condition_types.size() + payload_columns.col_types.size() - 1);
		} else {
			// This rhs column is a join key
			rhs_output_columns.col_idxs.push_back(it->second);
		}
		rhs_output_columns.col_types.push_back(rhs_col_type);
	}
}

PhysicalKathanJoin::PhysicalKathanJoin(LogicalOperator &op,
                                       unique_ptr<PhysicalOperator> left,
                                       unique_ptr<PhysicalOperator> right,
                                       vector<JoinCondition> cond,
                                       JoinType join_type,
                                       idx_t estimated_cardinality)
    : PhysicalKathanJoin(op, std::move(left), std::move(right), std::move(cond), join_type, {}, {}, estimated_cardinality) {
}

//===--------------------------------------------------------------------===//
// Global/Local Sink States
//===--------------------------------------------------------------------===//
struct KathanJoinGlobalSinkState : public GlobalSinkState {
	~KathanJoinGlobalSinkState() override {
		// Free GPU memory
		if (d_keys) cudaFree(d_keys);
		if (d_vals) cudaFree(d_vals);
	}
	// CPU row-wise build side
	vector<vector<Value>> build_rows;
	idx_t build_size = 0;

	// GPU stuff
	key_t *d_keys = nullptr;
	value_t *d_vals = nullptr;
	unique_ptr<hash_table_t> gpu_hash_table;

	bool finalized = false;
};

struct KathanJoinLocalSinkState : public LocalSinkState {
	~KathanJoinLocalSinkState() override {
	}
	vector<vector<Value>> local_build_rows;
};

//===--------------------------------------------------------------------===//
// Operator State (Probe Side)
//===--------------------------------------------------------------------===//

class KathanJoinOperatorState : public CachingOperatorState {
public:
	KathanJoinOperatorState(ClientContext &context, const vector<JoinCondition> &cond_p)
	    : probe_executor(context) {
		// single join key example
		for (auto &c : cond_p) {
			probe_executor.AddExpression(*c.left);
		}
		vector<LogicalType> key_types;
		key_types.push_back(cond_p[0].left->return_type);
		join_keys.Initialize(Allocator::Get(context), key_types);
	}

	~KathanJoinOperatorState() override {
		// free GPU memory if allocated
		if (d_probe_keys) cudaFree(d_probe_keys);
		if (d_begin_offsets) cudaFree(d_begin_offsets);
		if (d_end_offsets) cudaFree(d_end_offsets);
		if (d_matched_ids) cudaFree(d_matched_ids);
	}

	ExpressionExecutor probe_executor;
	DataChunk join_keys; // single-col chunk for LHS key

	// LHS columns
	DataChunk lhs_output;

	// GPU allocations for probe
	bool gpu_alloc = false;
	key_t *d_probe_keys = nullptr;

	// For multi-value retrieval
	warpcore::index_t *d_begin_offsets = nullptr;
	warpcore::index_t *d_end_offsets = nullptr;
	value_t *d_matched_ids = nullptr;

public:
	void Reset() {
		join_keys.Reset();
		lhs_output.Reset();
	}
	void Finalize(const PhysicalOperator &op, ExecutionContext &context) override {
		// optionally flush profiler
	}
};

//===--------------------------------------------------------------------===//
// Overridden Methods
//===--------------------------------------------------------------------===//

unique_ptr<GlobalSinkState> PhysicalKathanJoin::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<KathanJoinGlobalSinkState>();
}

unique_ptr<LocalSinkState> PhysicalKathanJoin::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<KathanJoinLocalSinkState>();
}

// Sink
SinkResultType PhysicalKathanJoin::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<KathanJoinGlobalSinkState>();
	auto &lstate = input.local_state.Cast<KathanJoinLocalSinkState>();

	for (idx_t row_idx = 0; row_idx < chunk.size(); row_idx++) {
		vector<Value> row;
		row.reserve(chunk.ColumnCount());
		for (idx_t col_idx = 0; col_idx < chunk.ColumnCount(); col_idx++) {
			row.push_back(chunk.GetValue(col_idx, row_idx));
		}
		lstate.local_build_rows.push_back(std::move(row));
	}
	return SinkResultType::NEED_MORE_INPUT;
}

// Combine
SinkCombineResultType PhysicalKathanJoin::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<KathanJoinGlobalSinkState>();
	auto &lstate = input.local_state.Cast<KathanJoinLocalSinkState>();

	for (auto &row : lstate.local_build_rows) {
		gstate.build_rows.push_back(std::move(row));
	}
	lstate.local_build_rows.clear();
	return SinkCombineResultType::FINISHED;
}

// Finalize
SinkFinalizeType PhysicalKathanJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                              OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<KathanJoinGlobalSinkState>();
	gstate.build_size = gstate.build_rows.size();
	if (gstate.build_size == 0 && EmptyResultIfRHSIsEmpty()) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	 // Suppose we only handle the first key index for now:
    if (build_key_indices.empty()) {
        throw InternalException("No build key indices found");
    }
    if (build_key_indices.size() > 1) {
        printf("WARNING: multiple join conditions, but using only the first.\n");
    }

    idx_t key_idx = build_key_indices[0];
	// Build GPU HT
	vector<key_t> h_keys;
	vector<value_t> h_vals;
	h_keys.reserve(gstate.build_size);
	h_vals.reserve(gstate.build_size);

	for (idx_t i = 0; i < gstate.build_size; i++) {
		auto &val = gstate.build_rows[i][key_idx];
		if (!val.IsNull()) {
			uint64_t k = val.GetValue<uint64_t>();
			h_keys.push_back(k);
			h_vals.push_back((value_t)i);
		}
	}
	idx_t final_cnt = h_keys.size();
	if (final_cnt == 0 && EmptyResultIfRHSIsEmpty()) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}
	cudaMalloc(&gstate.d_keys, sizeof(key_t) * final_cnt);
	cudaMalloc(&gstate.d_vals, sizeof(value_t) * final_cnt);
	cudaMemcpy(gstate.d_keys, h_keys.data(), final_cnt * sizeof(key_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gstate.d_vals, h_vals.data(), final_cnt * sizeof(value_t), cudaMemcpyHostToDevice);

	float load_factor = 0.9f;
	uint64_t capacity = (uint64_t)(final_cnt / load_factor);
	gstate.gpu_hash_table = make_uniq<hash_table_t>(capacity /* can specify max_values_per_key if needed */);

	gstate.gpu_hash_table->init();
	gstate.gpu_hash_table->insert(gstate.d_keys, gstate.d_vals, final_cnt);
	cudaDeviceSynchronize();

	gstate.finalized = true;
	return SinkFinalizeType::READY;
}

// OperatorState
unique_ptr<OperatorState> PhysicalKathanJoin::GetOperatorState(ExecutionContext &context) const {
	auto state = make_uniq<KathanJoinOperatorState>(context.client, conditions);

	// Initialize "lhs_output" chunk
	state->lhs_output.Initialize(Allocator::Get(context.client), lhs_output_columns.col_types);

	// Optionally, set caching
	state->initialized = false;
	state->can_cache_chunk = true;

	return std::move(state);
}

// ExecuteInternal (Probe side)
OperatorResultType PhysicalKathanJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input,
                                                       DataChunk &chunk, GlobalOperatorState &gstate_p,
                                                       OperatorState &state_p) const {
	auto &sink = sink_state->Cast<KathanJoinGlobalSinkState>();
	auto &op_state = state_p.Cast<KathanJoinOperatorState>();

	if (!sink.finalized) {
		return OperatorResultType::FINISHED; // Build not ready
	}
	if (sink.build_size == 0 && EmptyResultIfRHSIsEmpty()) {
		return OperatorResultType::FINISHED; // no output if empty RHS
	}
	if (input.size() == 0) {
		return OperatorResultType::FINISHED;
	}

	op_state.Reset();

	// 1) Project LHS columns that we want to pass through
	op_state.lhs_output.ReferenceColumns(input, lhs_output_columns.col_idxs);

	// 2) Evaluate join keys for the LHS
	op_state.probe_executor.Execute(input, op_state.join_keys);

	idx_t size = op_state.join_keys.size();
	if (size == 0) {
		chunk.SetCardinality(0);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// GPU allocation for probe side
	if (!op_state.gpu_alloc) {
		cudaMalloc(&op_state.d_probe_keys, sizeof(key_t) * STANDARD_VECTOR_SIZE);
		cudaMalloc(&op_state.d_begin_offsets, sizeof(warpcore::index_t) * STANDARD_VECTOR_SIZE);
		cudaMalloc(&op_state.d_end_offsets,   sizeof(warpcore::index_t) * STANDARD_VECTOR_SIZE);
		// We'll allocate matched_ids array after we get total_num_matches from the "dry run"
		op_state.gpu_alloc = true;
	}

	// Copy keys to GPU
	idx_t col_for_probe = 0; // or whichever condition is "primary"
	vector<key_t> h_probe_keys(size);
	for (idx_t i = 0; i < size; i++) {
		h_probe_keys[i] = op_state.join_keys.GetValue(0, i).GetValue<uint64_t>();
	}
	cudaMemcpy(op_state.d_probe_keys, h_probe_keys.data(), size * sizeof(key_t), cudaMemcpyHostToDevice);

	//----------------------------------------------------------------------
	// 3) Multi-value retrieve: DRY RUN
	//----------------------------------------------------------------------
	warpcore::index_t total_num_matches = 0;

	sink.gpu_hash_table->retrieve(op_state.d_probe_keys, // keys
	                              size,                   // number of keys
	                              op_state.d_begin_offsets,
	                              op_state.d_end_offsets,
	                              /*values_out=*/nullptr, // null => dry run
	                              total_num_matches       // number of matches
	);
	cudaDeviceSynchronize();

	if (total_num_matches == 0) {
		// no matches => produce empty chunk
		chunk.SetCardinality(0);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	//----------------------------------------------------------------------
	// 4) Allocate memory to hold all matched row IDs
	//----------------------------------------------------------------------
	if (op_state.d_matched_ids) {
		cudaFree(op_state.d_matched_ids);
		op_state.d_matched_ids = nullptr;
	}
	cudaMalloc(&op_state.d_matched_ids, sizeof(value_t) * total_num_matches);

	//----------------------------------------------------------------------
	// 5) Perform actual retrieval
	//----------------------------------------------------------------------
	sink.gpu_hash_table->retrieve(op_state.d_probe_keys,
	                              size,
	                              op_state.d_begin_offsets,
	                              op_state.d_end_offsets,
	                              op_state.d_matched_ids,
	                              total_num_matches);
	cudaDeviceSynchronize();

	//----------------------------------------------------------------------
	// 6) Copy offsets and matched IDs back to host
	//----------------------------------------------------------------------
	vector<warpcore::index_t> begin_offsets_h(size);
	vector<warpcore::index_t> end_offsets_h(size);
	vector<value_t> matched_ids_h(total_num_matches);

	cudaMemcpy(begin_offsets_h.data(), op_state.d_begin_offsets,
	           size * sizeof(warpcore::index_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(end_offsets_h.data(),   op_state.d_end_offsets,
	           size * sizeof(warpcore::index_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(matched_ids_h.data(),   op_state.d_matched_ids,
	           total_num_matches * sizeof(value_t), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	//----------------------------------------------------------------------
	// 7) Construct the output chunk
	//----------------------------------------------------------------------
	chunk.Destroy();
	chunk.Initialize(Allocator::DefaultAllocator(), this->types);

	idx_t out_idx = 0;
	for (idx_t i = 0; i < size; i++) {
		auto b_begin = begin_offsets_h[i];
		auto b_end   = end_offsets_h[i];

		// Each offset in [b_begin, b_end) is a matched row_id in build_rows
		for (auto off = b_begin; off < b_end && out_idx < STANDARD_VECTOR_SIZE; off++) {
			auto build_row_id = matched_ids_h[off];

			// [A] Copy LHS columns
			idx_t col_offset = 0;
			for (idx_t c = 0; c < lhs_output_columns.col_idxs.size(); c++) {
				chunk.SetValue(col_offset, out_idx, op_state.lhs_output.GetValue(c, i));
				col_offset++;
			}

			// [B] Copy RHS columns from build_rows
			auto &row = sink.build_rows[build_row_id];
			for (idx_t c = 0; c < rhs_output_columns.col_idxs.size(); c++) {
				auto out_col_idx = rhs_output_columns.col_idxs[c];
				chunk.SetValue(col_offset, out_idx, row[out_col_idx]);
				col_offset++;
			}
			out_idx++;
		}
		// If we wanted to handle >STANDARD_VECTOR_SIZE, we’d break out and produce partial output
	}
	chunk.SetCardinality(out_idx);

	return (out_idx == 0) ? OperatorResultType::FINISHED : OperatorResultType::NEED_MORE_INPUT;
}

// Optional Debug Info
InsertionOrderPreservingMap<string> PhysicalKathanJoin::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Join Type"] = EnumUtil::ToString(join_type);
	string conds;
	for (auto &cond : conditions) {
		if (!conds.empty()) {
			conds += " AND ";
		}
		conds += cond.left->GetName() + " " +
		         ExpressionTypeToString(cond.comparison) + " " +
		         cond.right->GetName();
	}
	result["Conditions"] = conds;
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

} // namespace duckdb
