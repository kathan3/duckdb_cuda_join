#pragma once

#include "duckdb/execution/operator/join/physical_comparison_join.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/operator/logical_join.hpp"

namespace duckdb {

// We mimic the style of PhysicalHashJoin::JoinProjectionColumns
struct KathanJoinProjectionColumns {
	vector<idx_t> col_idxs;
	vector<LogicalType> col_types;
};

class PhysicalKathanJoin : public PhysicalComparisonJoin {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::GPU_JOIN;

public:
	PhysicalKathanJoin(
	    LogicalOperator &op, 
	    unique_ptr<PhysicalOperator> left, 
	    unique_ptr<PhysicalOperator> right,
	    vector<JoinCondition> cond,
	    JoinType join_type,
	    const vector<idx_t> &left_projection_map,
	    const vector<idx_t> &right_projection_map,
	    idx_t estimated_cardinality);

	PhysicalKathanJoin(
	    LogicalOperator &op, 
	    unique_ptr<PhysicalOperator> left, 
	    unique_ptr<PhysicalOperator> right,
	    vector<JoinCondition> cond,
	    JoinType join_type,
	    idx_t estimated_cardinality);

	// The types of the join keys (similar to condition_types in PhysicalHashJoin)
	vector<LogicalType> condition_types;

	// The projection info for LHS, RHS, and the "payload" from the build side
	KathanJoinProjectionColumns lhs_output_columns;
	KathanJoinProjectionColumns rhs_output_columns;
	KathanJoinProjectionColumns payload_columns;

public:
	// We'll add minimal overrides that you can expand
	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return false; // or true if you intend parallel
	}

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	// Operator Execution
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;

protected:
	OperatorResultType ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                                   GlobalOperatorState &gstate, OperatorState &state) const override;
private:
	
	vector<idx_t> build_key_indices;

public:
	// Example of how you might set up a hashtable initializer, similar to HashJoin
	// (the actual implementation is up to you).
	// unique_ptr<JoinHashTable> InitializeHashTable(ClientContext &context) const;

	// Optional debug info
	InsertionOrderPreservingMap<string> ParamsToString() const override;
};

} // namespace duckdb
