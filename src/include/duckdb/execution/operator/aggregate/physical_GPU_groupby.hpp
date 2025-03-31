#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/parser/group_by_node.hpp"

namespace duckdb {

//! A demonstration GPU-based group-by operator, storing the entire input in one chunk
class PhysicalGPUGroupBy : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::GPU_GROUP_BY;

public:
	PhysicalGPUGroupBy(vector<LogicalType> types_p,
	                   vector<unique_ptr<Expression>> groups_p,
	                   idx_t estimated_cardinality);

	// Single grouping column index in the input
	vector<idx_t> groupby_columns;

public:
	//------------------------------------------
	// SINK INTERFACE
	//------------------------------------------
	bool IsSink() const override {
		return true;
	}

	bool ParallelOperator() const override {
		return true;
	}
	
	bool ParallelSink() const override {
		return true;
	}

	bool SinkOrderDependent() const override {
		return false;
	}

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk,
	                    OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context,
	                              OperatorSinkCombineInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event,
	                          ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	//------------------------------------------
	// SOURCE INTERFACE
	//------------------------------------------
	bool IsSource() const override{
		return true;
	};
	bool ParallelSource() const override{
		return true;
	};
	OrderPreservationType SourceOrder() const override{
		return OrderPreservationType::NO_ORDER;
	};

	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;
	SourceResultType GetData(ExecutionContext &context, DataChunk &chunk,
	                         OperatorSourceInput &input) const override;

	//------------------------------------------
	// MISC
	//------------------------------------------
	InsertionOrderPreservingMap<string> ParamsToString() const override;
};

} // namespace duckdb
