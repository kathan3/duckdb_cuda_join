#pragma once
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_operator_states.hpp"

#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"


namespace duckdb {

// ---------------------------------------------------------------------------
// Minimal join‑condition struct
// ---------------------------------------------------------------------------
struct KathanJoinCondition {
	unique_ptr<Expression> left;
	unique_ptr<Expression> right;
	ExpressionType comparison;
};

void KathanReorderConditions(vector<KathanJoinCondition> &conditions);
bool KathanEmptyResultIfRHSIsEmpty(JoinType join_type);

// ---------------------------------------------------------------------------
// Projection helper
// ---------------------------------------------------------------------------
struct KathanJoinProjectionColumns {
	vector<idx_t> col_idxs;
	vector<LogicalType> col_types;
};

// GPU helpers (defined in .cu)
struct BuildGPU;
struct ProbeGPU;

// ---------------------------------------------------------------------------
// GLOBAL SINK  (build side)
// ---------------------------------------------------------------------------
struct KathanJoinGlobalSinkState : public GlobalSinkState {
	DataChunk build_chunk;
	idx_t build_size = 0;

	BuildGPU *build_gpu = nullptr;
	bool finalized      = false;

	// ------------- accumulated timers -------------
	double sink_cpu_ms               = 0.0;
	double finalize_hostprep_ms      = 0.0;
	double finalize_gpu_h2d_ms       = 0.0;
	double finalize_gpu_build_ms     = 0.0;
	double finalize_overall_ms       = 0.0;

	~KathanJoinGlobalSinkState() override;
};

// ---------------------------------------------------------------------------
// GLOBAL OPERATOR  (probe side)
// ---------------------------------------------------------------------------
struct KathanJoinGlobalOperatorState : public GlobalOperatorState {
	DataChunk probe_chunk;
	idx_t probe_size = 0;

	ProbeGPU *probe_gpu = nullptr;

	vector<idx_t> build_indices;
	vector<idx_t> probe_indices;
	idx_t match_count   = 0;
	idx_t output_offset = 0;

	bool finished_join  = false;

	// ------------- accumulated timers -------------
	double execute_cpu_ms            = 0.0;
	double probe_gpu_h2d_ms          = 0.0;
	double probe_hostprep_ms         = 0.0;
	double retrieve_gpu_ms1           = 0.0;
	double retrieve_gpu_ms2 =            0.0;
	double d2h_ms                    = 0.0;
	double flatten_cpu_ms            = 0.0;
	double output_cpu_ms             = 0.0;
	bool printed_times               = false;

	~KathanJoinGlobalOperatorState() override;
};

// LOCAL states (unchanged)
struct KathanJoinLocalSinkState : public LocalSinkState {};
struct KathanJoinOperatorState   : public OperatorState   {};

// ---------------------------------------------------------------------------
// PhysicalKathanJoin declaration
// ---------------------------------------------------------------------------
class PhysicalKathanJoin : public PhysicalOperator {
public:
	static constexpr PhysicalOperatorType TYPE = PhysicalOperatorType::KATHAN_JOIN;

	// two ctor overloads (cond as JoinCondition or KathanJoinCondition)
	PhysicalKathanJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
	                   unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond,
	                   JoinType join_type, const vector<idx_t> &left_projection_map,
	                   const vector<idx_t> &right_projection_map,
	                   idx_t estimated_cardinality);

	PhysicalKathanJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
	                   unique_ptr<PhysicalOperator> right,
	                   vector<KathanJoinCondition> cond, JoinType join_type,
	                   idx_t estimated_cardinality);

	// ---- members ----
	JoinType join_type;
	bool build_key_is_varchar = false;
	vector<KathanJoinCondition> conditions;
	vector<LogicalType> condition_types;
	vector<idx_t> build_key_indices;
	KathanJoinProjectionColumns lhs_output_columns;
	KathanJoinProjectionColumns rhs_output_columns;

	// ---- overrides ----
	bool IsSink() const override { return true; }
	bool ParallelSink() const override { return false; }
	bool IsSource() const override { return false; }

	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;

	// Sink
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	// Probe
	bool RequiresFinalExecute() const override { return true; }
	unique_ptr<GlobalOperatorState> GetGlobalOperatorState(ClientContext &context) const override;
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;
	OperatorFinalizeResultType FinalExecute(ExecutionContext &context, DataChunk &chunk,
	                                        GlobalOperatorState &gstate, OperatorState &state) const override;

	InsertionOrderPreservingMap<string> ParamsToString() const override;
};

} // namespace duckdb
