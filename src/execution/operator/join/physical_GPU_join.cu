//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/join/physical_GPU_join.cu
//
//===----------------------------------------------------------------------===//
//
// A minimal single-column INNER GPU hash join operator that uses:
//
// 1) One large DataChunk (build_chunk) to store the build side in the sink phase.
// 2) Warpcore's MultiValueHashTable for GPU-based hashing (supports duplicates).
// 3) Directly slices the build side with a SelectionVector instead of building a pos_map_list.
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/operator/join/physical_GPU_join.hpp"
#include "duckdb/execution/operator/join/physical_comparison_join.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/types/data_chunk.hpp"

#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/physical_operator.hpp" // CachingOperatorState

// Warpcore multi-value hash table
#include <warpcore/multi_value_hash_table.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <limits>
#include <unordered_map>

namespace duckdb {

//-----------------------------------------------------------------------------
// GPU Key/Value and Warpcore Hash Table
//-----------------------------------------------------------------------------
using gpu_key_t = uint64_t;
using gpu_val_t = uint64_t;
using gpu_hash_table_t = warpcore::MultiValueHashTable<gpu_key_t, gpu_val_t,
                              std::numeric_limits<gpu_key_t>::max(),
                              std::numeric_limits<gpu_key_t>::max() - 1>;

//-----------------------------------------------------------------------------
// Global Sink State
//-----------------------------------------------------------------------------
struct KathanJoinGlobalSinkState : public GlobalSinkState {
	// We store the entire build side in ONE large DataChunk
	DataChunk build_chunk;

	gpu_key_t *d_keys = nullptr;
	gpu_val_t *d_vals = nullptr;
	unique_ptr<gpu_hash_table_t> gpu_hash_table;

	idx_t build_size = 0;
	bool finalized = false;

	KathanJoinGlobalSinkState(ClientContext &context, const vector<LogicalType> &build_types_p) {
		// Initialize a chunk with the build types. 
		// By default, a DataChunk has capacity = STANDARD_VECTOR_SIZE, 
		// but we can append with resize = true to grow beyond that limit if needed.
		build_chunk.Initialize(Allocator::Get(context), build_types_p);
	}

	~KathanJoinGlobalSinkState() override {
		if (d_keys) cudaFree(d_keys);
		if (d_vals) cudaFree(d_vals);
	}
};

//-----------------------------------------------------------------------------
// Local Sink State
//-----------------------------------------------------------------------------
struct KathanJoinLocalSinkState : public LocalSinkState {
};

//-----------------------------------------------------------------------------
// Operator (Probe) State - Derived from CachingOperatorState
// but sets can_cache_chunk = false
//-----------------------------------------------------------------------------
class KathanJoinOperatorState : public CachingOperatorState {
public:
	KathanJoinOperatorState(ClientContext &context, const vector<JoinCondition> &cond_p)
	    : probe_executor(context) {
		// We'll evaluate the LHS join keys
		D_ASSERT(!cond_p.empty());
		for (auto &c : cond_p) {
			probe_executor.AddExpression(*c.left);
		}
		// single-col chunk for the GPU join key
		vector<LogicalType> key_types;
		key_types.push_back(cond_p[0].left->return_type);
		join_keys.Initialize(Allocator::Get(context), key_types);

		// *** Turn OFF caching even though we inherit from CachingOperatorState
		can_cache_chunk = false;
	}

	~KathanJoinOperatorState() override {
		if (d_probe_keys) cudaFree(d_probe_keys);
		if (d_begin_offsets) cudaFree(d_begin_offsets);
		if (d_end_offsets) cudaFree(d_end_offsets);
		if (d_matched_ids) cudaFree(d_matched_ids);
	}

	void Reset() {
		join_keys.Reset();
		lhs_output.Reset();
	}

	void Finalize(const PhysicalOperator &op, ExecutionContext &context) override {
		// No caching => no final flush
	}

public:
	ExpressionExecutor probe_executor;
	DataChunk join_keys;   // LHS key chunk
	DataChunk lhs_output;  // LHS projected columns

	// GPU allocations
	bool gpu_alloc = false;
	gpu_key_t *d_probe_keys = nullptr;
	warpcore::index_t *d_begin_offsets = nullptr;
	warpcore::index_t *d_end_offsets   = nullptr;
	gpu_val_t *d_matched_ids = nullptr;
};

//-----------------------------------------------------------------------------
// PhysicalKathanJoin Implementation
//-----------------------------------------------------------------------------
PhysicalKathanJoin::PhysicalKathanJoin(LogicalOperator &op,
                                       unique_ptr<PhysicalOperator> left,
                                       unique_ptr<PhysicalOperator> right,
                                       vector<JoinCondition> cond,
                                       JoinType join_type,
                                       const vector<idx_t> &left_projection_map,
                                       const vector<idx_t> &right_projection_map,
                                       idx_t estimated_cardinality)
    : PhysicalComparisonJoin(op, PhysicalOperatorType::GPU_JOIN, std::move(cond), join_type, estimated_cardinality) {

    printf("physical_GPU_join is called\n");
	children.push_back(std::move(left));
	children.push_back(std::move(right));

	// Collect build key indices from conditions
	for (auto &cnd : conditions) {
		if (cnd.right->GetExpressionClass() == ExpressionClass::BOUND_REF) {
			auto &rhs_ref = cnd.right->Cast<BoundReferenceExpression>();
			build_key_indices.push_back(rhs_ref.index);
		} else {
			throw NotImplementedException("GPU join needs BoundReferenceExpression on RHS");
		}
	}

	// For each condition, store condition types
	unordered_map<idx_t, idx_t> build_columns_in_conditions;
	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &cond_ = conditions[cond_idx];
		condition_types.push_back(cond_.left->return_type);
		if (cond_.right->GetExpressionClass() == ExpressionClass::BOUND_REF) {
			build_columns_in_conditions.emplace(cond_.right->Cast<BoundReferenceExpression>().index, cond_idx);
		}
	}

	// LHS projection
	auto &lhs_types = children[0]->GetTypes();
	lhs_output_columns.col_idxs = left_projection_map;
	if (lhs_output_columns.col_idxs.empty()) {
		lhs_output_columns.col_idxs.reserve(lhs_types.size());
		for (idx_t i = 0; i < lhs_types.size(); i++) {
			lhs_output_columns.col_idxs.push_back(i);
		}
	}
	for (auto col_idx : lhs_output_columns.col_idxs) {
		lhs_output_columns.col_types.push_back(lhs_types[col_idx]);
	}

	// RHS projection
	auto &rhs_types = children[1]->GetTypes();
	auto right_map_copy = right_projection_map;
	if (right_map_copy.empty()) {
		right_map_copy.reserve(rhs_types.size());
		for (idx_t i = 0; i < rhs_types.size(); i++) {
			right_map_copy.push_back(i);
		}
	}
	for (auto rhs_col : right_map_copy) {
		auto &rhs_col_type = rhs_types[rhs_col];
		auto it = build_columns_in_conditions.find(rhs_col);
		if (it == build_columns_in_conditions.end()) {
			// This is a payload column
			payload_columns.col_idxs.push_back(rhs_col);
			payload_columns.col_types.push_back(rhs_col_type);
			// The output index for this column is the index after the condition columns
			rhs_output_columns.col_idxs.push_back(condition_types.size() + payload_columns.col_types.size() - 1);
		} else {
			// This is part of the join key (for reference in the output)
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
    : PhysicalComparisonJoin(op, PhysicalOperatorType::GPU_JOIN, std::move(cond), join_type, estimated_cardinality) {
	children.push_back(std::move(left));
	children.push_back(std::move(right));
}

//-----------------------------------------------------------------------------
// Global/Local Sink
//-----------------------------------------------------------------------------
unique_ptr<GlobalSinkState> PhysicalKathanJoin::GetGlobalSinkState(ClientContext &context) const {
	auto &rhs_types = children[1]->GetTypes();
	return make_uniq<KathanJoinGlobalSinkState>(context, rhs_types);
}

unique_ptr<LocalSinkState> PhysicalKathanJoin::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<KathanJoinLocalSinkState>();
}

// In the sink, we now just append all incoming chunks into "build_chunk"
SinkResultType PhysicalKathanJoin::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<KathanJoinGlobalSinkState>();

	// Append to our single build_chunk. "true" => allow resizing if needed
	gstate.build_chunk.Append(chunk, true);
	gstate.build_size += chunk.size();

	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalKathanJoin::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
	return SinkCombineResultType::FINISHED;
}

// Finalize: build Warpcore hash table from the single build_chunk
SinkFinalizeType PhysicalKathanJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                              OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<KathanJoinGlobalSinkState>();

	// If build side is empty and it's an inner join, no output
	if (gstate.build_size == 0 && EmptyResultIfRHSIsEmpty()) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}
	if (build_key_indices.empty()) {
		throw InternalException("No build keys found for GPU join");
	}

	// For simplicity, we support a single key column
	idx_t primary_key_idx = build_key_indices[0];

	// Collect (key, row_id) from build_chunk
	vector<gpu_key_t> h_keys;
	vector<gpu_val_t> h_vals;
	h_keys.reserve(gstate.build_size);
	h_vals.reserve(gstate.build_size);

	auto &key_col = gstate.build_chunk.data[primary_key_idx];
	UnifiedVectorFormat key_data;
	key_col.ToUnifiedFormat(gstate.build_size, key_data);
	auto keys_ptr = reinterpret_cast<const uint64_t*>(key_data.data);

	for (idx_t i = 0; i < gstate.build_size; i++) {
		auto sel_idx = key_data.sel->get_index(i);
		if (!key_data.validity.RowIsValid(sel_idx)) {
			// skip null key in an inner join
			continue;
		}
		gpu_key_t k = keys_ptr[sel_idx];
		h_keys.push_back(k);
		h_vals.push_back((gpu_val_t) i); // row_id = i
	}

	idx_t final_cnt = h_keys.size();
	if (final_cnt == 0 && EmptyResultIfRHSIsEmpty()) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	// GPU allocate arrays
	cudaMalloc(&gstate.d_keys, sizeof(gpu_key_t) * final_cnt);
	cudaMalloc(&gstate.d_vals, sizeof(gpu_val_t) * final_cnt);
	cudaMemcpy(gstate.d_keys, h_keys.data(), final_cnt * sizeof(gpu_key_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gstate.d_vals, h_vals.data(), final_cnt * sizeof(gpu_val_t), cudaMemcpyHostToDevice);

	// Build hash table
	float load_factor = 0.9f;
	uint64_t capacity = (uint64_t)((double)final_cnt / load_factor);
	gstate.gpu_hash_table = make_uniq<gpu_hash_table_t>(capacity);
	gstate.gpu_hash_table->init();
	gstate.gpu_hash_table->insert(gstate.d_keys, gstate.d_vals, final_cnt);
	cudaDeviceSynchronize();

	gstate.finalized = true;
	return SinkFinalizeType::READY;
}

//-----------------------------------------------------------------------------
// GetOperatorState (for the probe)
//-----------------------------------------------------------------------------
unique_ptr<OperatorState> PhysicalKathanJoin::GetOperatorState(ExecutionContext &context) const {
	auto state = make_uniq<KathanJoinOperatorState>(context.client, conditions);
	state->lhs_output.Initialize(Allocator::Get(context.client), lhs_output_columns.col_types);
	return std::move(state);
}

//-----------------------------------------------------------------------------
// ExecuteInternal (probe + produce join results)
//-----------------------------------------------------------------------------
OperatorResultType PhysicalKathanJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                       GlobalOperatorState &gstate, OperatorState &state) const {
	auto &sink = sink_state->Cast<KathanJoinGlobalSinkState>();
	auto &op_state = state.Cast<KathanJoinOperatorState>();
	// printf("Input size: %llu\n", input.size());
	if (!sink.finalized) {
		return OperatorResultType::FINISHED;
	}
	if (sink.build_size == 0 && EmptyResultIfRHSIsEmpty()) {
		// For an inner join, no matches possible
		return OperatorResultType::FINISHED;
	}
	if (input.size() == 0) {
		return OperatorResultType::FINISHED;
	}
	op_state.Reset();

	//----------------------------------------------------------------------
	// 1) Project LHS columns
	//----------------------------------------------------------------------
	op_state.lhs_output.ReferenceColumns(input, lhs_output_columns.col_idxs);

	//----------------------------------------------------------------------
	// 2) Evaluate single join key
	//----------------------------------------------------------------------
	op_state.probe_executor.Execute(input, op_state.join_keys);
	idx_t size = op_state.join_keys.size();
	if (size == 0) {
		chunk.SetCardinality(0);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	//----------------------------------------------------------------------
	// 3) GPU allocate if needed
	//----------------------------------------------------------------------
	if (!op_state.gpu_alloc) {
		cudaMalloc(&op_state.d_probe_keys,   sizeof(gpu_key_t) * STANDARD_VECTOR_SIZE);
		cudaMalloc(&op_state.d_begin_offsets,sizeof(warpcore::index_t) * STANDARD_VECTOR_SIZE);
		cudaMalloc(&op_state.d_end_offsets,  sizeof(warpcore::index_t) * STANDARD_VECTOR_SIZE);
		op_state.gpu_alloc = true;
	}

	//----------------------------------------------------------------------
	// 4) Copy probe keys from CPU to GPU
	//----------------------------------------------------------------------
	vector<gpu_key_t> h_probe_keys(size);
	for (idx_t i = 0; i < size; i++) {
		h_probe_keys[i] = op_state.join_keys.GetValue(0, i).GetValue<uint64_t>();
	}
	cudaMemcpy(op_state.d_probe_keys, h_probe_keys.data(), size * sizeof(gpu_key_t), cudaMemcpyHostToDevice);

	//----------------------------------------------------------------------
	// 5) Dry-run retrieve for counting total matches
	//----------------------------------------------------------------------
	warpcore::index_t total_num_matches = 0;
	sink.gpu_hash_table->retrieve(op_state.d_probe_keys, size,
	                              op_state.d_begin_offsets, op_state.d_end_offsets,
	                              nullptr, total_num_matches);
	cudaDeviceSynchronize();

	if (total_num_matches == 0) {
		chunk.SetCardinality(0);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	//----------------------------------------------------------------------
	// 6) Allocate matched IDs
	//----------------------------------------------------------------------
	if (op_state.d_matched_ids) {
		cudaFree(op_state.d_matched_ids);
		op_state.d_matched_ids = nullptr;
	}
	cudaMalloc(&op_state.d_matched_ids, sizeof(gpu_val_t) * total_num_matches);

	//----------------------------------------------------------------------
	// 7) Actual retrieve
	//----------------------------------------------------------------------
	sink.gpu_hash_table->retrieve(op_state.d_probe_keys, size,
	                              op_state.d_begin_offsets, op_state.d_end_offsets,
	                              op_state.d_matched_ids, total_num_matches);
	cudaDeviceSynchronize();

	//----------------------------------------------------------------------
	// 8) Copy matched offsets back to CPU
	//----------------------------------------------------------------------
	vector<warpcore::index_t> begin_offsets_h(size), end_offsets_h(size);
	vector<gpu_val_t> matched_ids_h(total_num_matches);

	cudaMemcpy(begin_offsets_h.data(), op_state.d_begin_offsets, size * sizeof(warpcore::index_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(end_offsets_h.data(),   op_state.d_end_offsets,   size * sizeof(warpcore::index_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(matched_ids_h.data(),   op_state.d_matched_ids, total_num_matches * sizeof(gpu_val_t), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	//----------------------------------------------------------------------
	// 9) Flatten matches: build_indices & probe_indices
	//----------------------------------------------------------------------
	vector<idx_t> build_indices;
	vector<idx_t> probe_indices;
	build_indices.reserve(total_num_matches);
	probe_indices.reserve(total_num_matches);

	idx_t match_count = 0;
	for (idx_t i = 0; i < size; i++) {
		auto b_begin = begin_offsets_h[i];
		auto b_end   = end_offsets_h[i];
		for (auto off = b_begin; off < b_end; off++) {
			build_indices.push_back(matched_ids_h[off]);
			probe_indices.push_back(i);
			match_count++;
		}
	}
	// (In production code, you'd likely produce multiple chunks if match_count > STANDARD_VECTOR_SIZE.)
	if (match_count > STANDARD_VECTOR_SIZE) {
		match_count = STANDARD_VECTOR_SIZE; // naive truncation for this example
	}

	//----------------------------------------------------------------------
	// 10) Construct output chunk
	//----------------------------------------------------------------------
	chunk.Destroy();
	chunk.Initialize(Allocator::DefaultAllocator(), this->types);

	// [A] Slice LHS
	SelectionVector probe_sel(STANDARD_VECTOR_SIZE);
	for (idx_t i = 0; i < match_count; i++) {
		probe_sel.set_index(i, probe_indices[i]);
	}
	for (idx_t col_idx = 0; col_idx < lhs_output_columns.col_types.size(); col_idx++) {
		auto &dest_vec = chunk.data[col_idx];
		dest_vec.Reference(op_state.lhs_output.data[col_idx]);
		dest_vec.Slice(probe_sel, match_count);
	}

	// [B] Slice RHS from the single build_chunk
	idx_t rhs_offset = lhs_output_columns.col_types.size();

	// Build a selection vector for the build side
	SelectionVector build_sel(STANDARD_VECTOR_SIZE);
	for (idx_t i = 0; i < match_count; i++) {
		build_sel.set_index(i, build_indices[i]);
	}

	for (idx_t col_i = 0; col_i < rhs_output_columns.col_types.size(); col_i++) {
		auto out_col_idx = rhs_offset + col_i;
		auto build_col_idx = rhs_output_columns.col_idxs[col_i];

		// Refer to the build column, then slice
		auto &dest_vec = chunk.data[out_col_idx];
		dest_vec.Reference(sink.build_chunk.data[build_col_idx]);
		dest_vec.Slice(build_sel, match_count);
	}

	chunk.SetCardinality(match_count);
	return (match_count == 0) ? OperatorResultType::FINISHED : OperatorResultType::NEED_MORE_INPUT;
}

//-----------------------------------------------------------------------------
// Optional Debug Info
//-----------------------------------------------------------------------------
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
