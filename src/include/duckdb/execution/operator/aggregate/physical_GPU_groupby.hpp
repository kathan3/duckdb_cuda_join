#pragma once
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/parser/group_by_node.hpp"

namespace duckdb {

//─────────────────── aggregate catalogue ───────────────────────────────────
enum class GPUAggKind { COUNT_STAR, SUM, MIN, MAX, AVG };

struct GPUAggSpec {
	GPUAggKind  kind;
	idx_t       input_col;          // invalid for COUNT(*)
	LogicalType val_type;           // BIGINT or HUGEINT
};

//─────────────────── operator ──────────────────────────────────────────────
class PhysicalGPUGroupBy : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::GPU_GROUP_BY;

	PhysicalGPUGroupBy(vector<LogicalType>                       /* final output types */,
	                   vector<unique_ptr<Expression>> groups_p,   /* GROUP BY list      */
	                   vector<unique_ptr<Expression>> aggregates, /* Agg list           */
	                   idx_t estimated_cardinality);

	//──── plan metadata ────────────────────────────────────────────────────
	vector<idx_t>      groupby_columns;   // ≥1 key columns
	vector<LogicalType> key_types;        // logical types per key
	vector<bool>       key_is_varchar;    // VARCHAR flags per key
	vector<GPUAggSpec> agg_specs;         // one per SQL aggregate

	//──── SINK interface ───────────────────────────────────────────────────
	bool IsSink()                  const override { return true;  }
	bool ParallelOperator()        const override { return false; }
	bool ParallelSink()            const override { return false; }
	bool SinkOrderDependent()      const override { return false; }

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &)           const override;
	unique_ptr<LocalSinkState>  GetLocalSinkState(ExecutionContext &)         const override;
	SinkResultType  Sink(ExecutionContext &, DataChunk &, OperatorSinkInput&) const override;
	SinkCombineResultType Combine(ExecutionContext &, OperatorSinkCombineInput&) const override;
	SinkFinalizeType Finalize(Pipeline &, Event &, ClientContext &,
	                          OperatorSinkFinalizeInput&)                    const override;

	//──── SOURCE interface ─────────────────────────────────────────────────
	bool IsSource()               const override { return true;  }
	bool ParallelSource()         const override { return false; }
	OrderPreservationType SourceOrder() const override { return OrderPreservationType::NO_ORDER; }

	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &) const override;
	unique_ptr<LocalSourceState>  GetLocalSourceState(ExecutionContext &,
	                                                  GlobalSourceState &) const override;
	SourceResultType GetData(ExecutionContext &, DataChunk &, OperatorSourceInput&) const override;

	//──── misc ─────────────────────────────────────────────────────────────
	InsertionOrderPreservingMap<string> ParamsToString() const override;
};

} // namespace duckdb
