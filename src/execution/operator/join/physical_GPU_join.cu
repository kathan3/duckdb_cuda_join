//===----------------------------------------------------------------------===//
// physical_GPU_join.cu  –  GPU hash-join with BIGINT *or* VARCHAR keys
//                             (timed version)
//===----------------------------------------------------------------------===//
#include "duckdb/execution/operator/join/physical_GPU_join.hpp"

#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/planner/operator/logical_join.hpp"

#include <warpcore/multi_value_hash_table.cuh>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace duckdb {

// ────────────────────────────────────────────────────────────────────────────
// 0)  Hash helpers (host-side)
// ────────────────────────────────────────────────────────────────────────────
using gpu_key_t = uint64_t;
using gpu_val_t = uint64_t;
static constexpr gpu_key_t EMPTY_KEY  = std::numeric_limits<gpu_key_t>::max();
static constexpr gpu_key_t TOMBSTONE  = EMPTY_KEY - 1;

static inline gpu_key_t FNV1a64(const char *data, size_t len) {
	const gpu_key_t OFFSET = 0xcbf29ce484222325ULL;
	const gpu_key_t PRIME  = 0x100000001b3ULL;
	gpu_key_t h = OFFSET;
	for (size_t i = 0; i < len; ++i) {
		h ^= static_cast<unsigned char>(data[i]);
		h *= PRIME;
	}
	return h;
}

// ────────────────────────────────────────────────────────────────────────────
// 1)  GPU typedefs & helper structs
// ────────────────────────────────────────────────────────────────────────────
using gpu_hash_table_t =
    warpcore::MultiValueHashTable<gpu_key_t, gpu_val_t, EMPTY_KEY, TOMBSTONE>;

struct BuildGPU {
	gpu_key_t *d_keys = nullptr;
	gpu_val_t *d_vals = nullptr;
	std::unique_ptr<gpu_hash_table_t> warpcore_table;
	~BuildGPU() {
		if (d_keys) cudaFree(d_keys);
		if (d_vals) cudaFree(d_vals);
	}
};
struct ProbeGPU { ~ProbeGPU() {} };

// ────────────────────────────────────────────────────────────────────────────
// 2)  Timer helpers
// ────────────────────────────────────────────────────────────────────────────
using clock_t = std::chrono::high_resolution_clock;
static float TimeGPUOp(const std::function<void()> &op) {
	cudaEvent_t s, e;
	cudaEventCreate(&s);
	cudaEventCreate(&e);
	cudaEventRecord(s);
	op();
	cudaEventRecord(e);
	cudaEventSynchronize(e);
	float ms = 0.0f;
	cudaEventElapsedTime(&ms, s, e);
	cudaEventDestroy(s);
	cudaEventDestroy(e);
	return ms;
}

// ────────────────────────────────────────────────────────────────────────────
// 3)  Small utility functions (unchanged)
// ────────────────────────────────────────────────────────────────────────────
void KathanReorderConditions(vector<KathanJoinCondition> &conds) {
	bool seenneq = false, ordered = true;
	for (auto &c : conds) {
		bool eq = c.comparison == ExpressionType::COMPARE_EQUAL ||
		          c.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM;
		if (eq && seenneq) { ordered = false; break; }
		if (!eq) seenneq = true;
	}
	if (!ordered) {
		vector<KathanJoinCondition> eqs, others;
		for (auto &c : conds)
			((c.comparison == ExpressionType::COMPARE_EQUAL ||
			  c.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM)
			     ? eqs
			     : others)
			    .push_back(std::move(c));
		conds.clear();
		conds.insert(conds.end(), std::make_move_iterator(eqs.begin()),
		             std::make_move_iterator(eqs.end()));
		conds.insert(conds.end(), std::make_move_iterator(others.begin()),
		             std::make_move_iterator(others.end()));
	}
}
bool KathanEmptyResultIfRHSIsEmpty(JoinType jt) {
	switch (jt) {
	case JoinType::INNER:
	case JoinType::RIGHT:
	case JoinType::SEMI:
	case JoinType::RIGHT_SEMI:
	case JoinType::RIGHT_ANTI:
		return true;
	default:
		return false;
	}
}

// ────────────────────────────────────────────────────────────────────────────
// 4)  Destructor definitions
// ────────────────────────────────────────────────────────────────────────────
KathanJoinGlobalSinkState::~KathanJoinGlobalSinkState() { delete build_gpu; }
KathanJoinGlobalOperatorState::~KathanJoinGlobalOperatorState() {
	delete probe_gpu;
}

// ────────────────────────────────────────────────────────────────────────────
// 5)  Constructors
//     (only change: remember whether the build-side key is VARCHAR)
// ────────────────────────────────────────────────────────────────────────────
PhysicalKathanJoin::PhysicalKathanJoin(
    LogicalOperator &op, unique_ptr<PhysicalOperator> left,
	unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond,
   JoinType jt, const vector<idx_t> &lproj,
    const vector<idx_t> &rproj, idx_t est_card)
    : PhysicalOperator(PhysicalOperatorType::KATHAN_JOIN, op.types, est_card),
      join_type(jt) {

	children.push_back(std::move(left));
	children.push_back(std::move(right));

	for (auto &c : cond) {
		KathanJoinCondition k;
		k.left       = c.left->Copy();
		k.right      = c.right->Copy();
		k.comparison = c.comparison;
		conditions.push_back(std::move(k));
	}
	KathanReorderConditions(conditions);

	/* ---- single build key only ---- */
	auto &rhs_ref = conditions[0].right->Cast<BoundReferenceExpression>();
	build_key_indices.push_back(rhs_ref.index);
	build_key_is_varchar =
	    conditions[0].left->return_type == LogicalType::VARCHAR;
	condition_types.push_back(conditions[0].left->return_type);

	/* ---- projection maps (unchanged) ---- */
	auto &ltypes = children[0]->GetTypes();
	if (lproj.empty()) {
		for (idx_t i = 0; i < ltypes.size(); i++) {
			lhs_output_columns.col_idxs.push_back(i);
			lhs_output_columns.col_types.push_back(ltypes[i]);
		}
	} else {
		for (auto i : lproj) {
			lhs_output_columns.col_idxs.push_back(i);
			lhs_output_columns.col_types.push_back(ltypes[i]);
		}
	}
	auto &rtypes = children[1]->GetTypes();
	if (rproj.empty()) {
		for (idx_t i = 0; i < rtypes.size(); i++) {
			rhs_output_columns.col_idxs.push_back(i);
			rhs_output_columns.col_types.push_back(rtypes[i]);
		}
	} else {
		for (auto i : rproj) {
			rhs_output_columns.col_idxs.push_back(i);
			rhs_output_columns.col_types.push_back(rtypes[i]);
		}
	}
}

/*  Secondary constructor (Logical conditions already converted) */
    PhysicalKathanJoin::PhysicalKathanJoin(
    LogicalOperator &op, unique_ptr<PhysicalOperator> left,
    unique_ptr<PhysicalOperator> right, vector<KathanJoinCondition> cond,
    JoinType jt, idx_t est_card)
    : PhysicalOperator(PhysicalOperatorType::KATHAN_JOIN, op.types, est_card),
      join_type(jt), conditions(std::move(cond)) {

	children.push_back(std::move(left));
	children.push_back(std::move(right));
	KathanReorderConditions(conditions);

	auto &rhs_ref = conditions[0].right->Cast<BoundReferenceExpression>();
	build_key_indices.push_back(rhs_ref.index);
	build_key_is_varchar =
	    conditions[0].left->return_type == LogicalType::VARCHAR;
	condition_types.push_back(conditions[0].left->return_type);

	auto &ltypes = children[0]->GetTypes();
	for (idx_t i = 0; i < ltypes.size(); i++) {
		lhs_output_columns.col_idxs.push_back(i);
		lhs_output_columns.col_types.push_back(ltypes[i]);
	}
	auto &rtypes = children[1]->GetTypes();
	for (idx_t i = 0; i < rtypes.size(); i++) {
		rhs_output_columns.col_idxs.push_back(i);
		rhs_output_columns.col_types.push_back(rtypes[i]);
	}
}

// ────────────────────────────────────────────────────────────────────────────
// 6)  Pipeline wiring (unchanged)
// ────────────────────────────────────────────────────────────────────────────
void PhysicalKathanJoin::BuildPipelines(Pipeline &curr, MetaPipeline &meta) {
	meta.GetState().AddPipelineOperator(curr, *this);
	auto &build_meta = meta.CreateChildMetaPipeline(curr, *this);
	build_meta.Build(*children[1]);
	children[0]->BuildPipelines(curr, meta);
}

// ────────────────────────────────────────────────────────────────────────────
// 7)  Sink (build side)
//     *MODIFIED*: BIGINT → copy directly; VARCHAR → hash with FNV1a64
// ────────────────────────────────────────────────────────────────────────────
unique_ptr<GlobalSinkState>
PhysicalKathanJoin::GetGlobalSinkState(ClientContext &ctx) const {
	auto g = make_uniq<KathanJoinGlobalSinkState>();
	g->build_chunk.Initialize(Allocator::Get(ctx), children[1]->GetTypes());
	g->build_gpu = new BuildGPU();
	return std::move(g);
}
unique_ptr<LocalSinkState>
PhysicalKathanJoin::GetLocalSinkState(ExecutionContext &) const {
	return make_uniq<KathanJoinLocalSinkState>();
}

SinkResultType PhysicalKathanJoin::Sink(ExecutionContext &, DataChunk &chunk,
                                        OperatorSinkInput &in) const {
	auto &g = in.global_state.Cast<KathanJoinGlobalSinkState>();

	auto t0 = clock_t::now();
	g.build_chunk.Append(chunk, true);
	g.build_size += chunk.size();
	g.sink_cpu_ms +=
	    std::chrono::duration<double, std::milli>(clock_t::now() - t0).count();
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalKathanJoin::Combine(ExecutionContext &,
                                                  OperatorSinkCombineInput &) const {
	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalKathanJoin::Finalize(Pipeline &, Event &, ClientContext &,
                                              OperatorSinkFinalizeInput &in) const {

	auto &g = in.global_state.Cast<KathanJoinGlobalSinkState>();
	if (g.build_size == 0 && KathanEmptyResultIfRHSIsEmpty(join_type))
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;

	// ─── host prep ───────────────────────────────────────────────────
	auto cpu0 = clock_t::now();
	idx_t key_idx = build_key_indices[0];
	auto &kcol = g.build_chunk.data[key_idx];
	UnifiedVectorFormat kvf;
	kcol.ToUnifiedFormat(g.build_size, kvf);

	vector<gpu_key_t> h_keys;
	vector<gpu_val_t> h_vals;
	h_keys.reserve(g.build_size);
	h_vals.reserve(g.build_size);

	if (!build_key_is_varchar) {
		/* BIGINT key */
		auto kptr = reinterpret_cast<const gpu_key_t *>(kvf.data);
		for (idx_t i = 0; i < g.build_size; i++) {
			auto s = kvf.sel->get_index(i);
			if (!kvf.validity.RowIsValid(s)) continue;
			h_keys.push_back(kptr[s]);
			h_vals.push_back((gpu_val_t)i);
		}
	} else {
		/* VARCHAR key → hash */
		auto kptr = reinterpret_cast<const string_t *>(kvf.data);
		for (idx_t i = 0; i < g.build_size; i++) {
			auto s = kvf.sel->get_index(i);
			if (!kvf.validity.RowIsValid(s)) continue;
			const string_t &str = kptr[s];
			h_keys.push_back(FNV1a64(str.GetDataUnsafe(), str.GetSize()));
			h_vals.push_back((gpu_val_t)i);
		}
		
	}
	g.finalize_hostprep_ms =
	    std::chrono::duration<double, std::milli>(clock_t::now() - cpu0).count();

	std::cout << "Total non-null build keys: " << h_keys.size() << std::endl;		
	if (h_keys.empty() && KathanEmptyResultIfRHSIsEmpty(join_type))
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;

	// ─── H2D ─────────────────────────────────────────────────────────
	auto &bgpu = *g.build_gpu;
	g.finalize_gpu_h2d_ms = TimeGPUOp([&] {
		cudaMalloc(&bgpu.d_keys, sizeof(gpu_key_t) * h_keys.size());
		cudaMalloc(&bgpu.d_vals, sizeof(gpu_val_t) * h_vals.size());
		cudaMemcpy(bgpu.d_keys, h_keys.data(),
		           sizeof(gpu_key_t) * h_keys.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(bgpu.d_vals, h_vals.data(),
		           sizeof(gpu_val_t) * h_vals.size(), cudaMemcpyHostToDevice);
	});

	// ─── build hash table (Warpcore) ─────────────────────────────────
	g.finalize_gpu_build_ms = TimeGPUOp([&] {
		float lf     = 0.9f;
		uint64_t cap = (uint64_t)((double)h_keys.size() / lf);
		bgpu.warpcore_table = std::make_unique<gpu_hash_table_t>(cap);
		bgpu.warpcore_table->init();
		bgpu.warpcore_table->insert(bgpu.d_keys, bgpu.d_vals, h_keys.size());
		cudaDeviceSynchronize();
	});

	g.finalized           = true;
    g.finalize_overall_ms = g.finalize_hostprep_ms +
	                        g.finalize_gpu_h2d_ms + g.finalize_gpu_build_ms;


	std::cout << "==========  KathanJoin BUILD (Sink/Finalize) ==========\n";
	std::cout << "[CPU] total Sink()                     = " << g.sink_cpu_ms
	          << " ms\n";
	std::cout << "[CPU] host prep                        = "
	          << g.finalize_hostprep_ms << " ms\n";
	std::cout << "[GPU] host→device                      = "
	          << g.finalize_gpu_h2d_ms << " ms\n";
	std::cout << "[GPU] build Warpcore                   = "
	          << g.finalize_gpu_build_ms << " ms\n";
	std::cout << "=======================================================\n";
	return SinkFinalizeType::READY;
}

// ────────────────────────────────────────────────────────────────────────────
// 8)  Probe-side state & Execute()
//     *MODIFIED*: hash VARCHAR probe keys the same way
// ────────────────────────────────────────────────────────────────────────────
unique_ptr<GlobalOperatorState>
PhysicalKathanJoin::GetGlobalOperatorState(ClientContext &) const {
	auto g = make_uniq<KathanJoinGlobalOperatorState>();
	g->probe_gpu = new ProbeGPU();
	return std::move(g);
}
unique_ptr<OperatorState>
PhysicalKathanJoin::GetOperatorState(ExecutionContext &) const {
	return make_uniq<KathanJoinOperatorState>();
}

OperatorResultType PhysicalKathanJoin::Execute(ExecutionContext &ctx, DataChunk &in,
                                               DataChunk &out,
                                               GlobalOperatorState &g_p,
                                               OperatorState &) const {
	auto &g = g_p.Cast<KathanJoinGlobalOperatorState>();
	auto t0 = clock_t::now();

	if (!in.size()) {
		out.SetCardinality(0);
		return OperatorResultType::NEED_MORE_INPUT;
	}
	if (g.probe_chunk.data.empty())
		g.probe_chunk.Initialize(Allocator::Get(ctx.client),
		                         children[0]->GetTypes());
	g.probe_chunk.Append(in, true);
	g.probe_size += in.size();

	g.execute_cpu_ms +=
	    std::chrono::duration<double, std::milli>(clock_t::now() - t0).count();
	out.SetCardinality(0);
	return OperatorResultType::NEED_MORE_INPUT;
}

// ────────────────────────────────────────────────────────────────────────────
// 9)  FinalExecute – retrieve matches, emit chunks
//     *MODIFIED*: VARCHAR probe hashing
// ────────────────────────────────────────────────────────────────────────────
OperatorFinalizeResultType PhysicalKathanJoin::FinalExecute(
    ExecutionContext &, DataChunk &chunk, GlobalOperatorState &g_p,
    OperatorState &) const {

	auto &sink = sink_state->Cast<KathanJoinGlobalSinkState>();
	auto &g    = g_p.Cast<KathanJoinGlobalOperatorState>();

	if (!sink.finalized ||
	    (sink.build_size == 0 && KathanEmptyResultIfRHSIsEmpty(join_type))) {
		chunk.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	// ── 1) first entry: build match index buffers ────────────────────
	if (!g.finished_join) {
		auto total_start = clock_t::now();

		auto hostprep_start = clock_t::now();
		vector<gpu_key_t> h_probe;
		h_probe.reserve(g.probe_size);

		auto &pcol = g.probe_chunk.data[0];
		UnifiedVectorFormat pvf;
		pcol.ToUnifiedFormat(g.probe_size, pvf);

		if (!build_key_is_varchar) {
			auto pptr = reinterpret_cast<const gpu_key_t *>(pvf.data);
			for (idx_t i = 0; i < g.probe_size; i++) {
				auto s = pvf.sel->get_index(i);
				h_probe.push_back(pvf.validity.RowIsValid(s) ?
				                  pptr[s] : EMPTY_KEY);
			}
		} else {
			auto pptr = reinterpret_cast<const string_t *>(pvf.data);
			for (idx_t i = 0; i < g.probe_size; i++) {
				auto s = pvf.sel->get_index(i);
				if (!pvf.validity.RowIsValid(s)) {
					h_probe.push_back(EMPTY_KEY);
				} else {
					const string_t &str = pptr[s];
					h_probe.push_back(
					    FNV1a64(str.GetDataUnsafe(), str.GetSize()));
				}
			}
			
		}
		g.probe_hostprep_ms =
		    std::chrono::duration<double, std::milli>(
		        clock_t::now() - hostprep_start)
		        .count();
		
std::cout << "Total non-null probe keys: " << h_probe.size() << std::endl;
		/* ── host→device ─────────────────────────────────────────── */
		gpu_key_t *d_probe = nullptr;
		g.probe_gpu_h2d_ms = TimeGPUOp([&] {
			cudaMalloc(&d_probe, sizeof(gpu_key_t) * h_probe.size());
			cudaMemcpy(d_probe, h_probe.data(),
			           sizeof(gpu_key_t) * h_probe.size(),
			           cudaMemcpyHostToDevice);
		});

		/* ── first retrieve: counts ─────────────────────────────── */
		warpcore::index_t *d_beg = nullptr, *d_end = nullptr;
		cudaMalloc(&d_beg, sizeof(warpcore::index_t) * h_probe.size());
		cudaMalloc(&d_end, sizeof(warpcore::index_t) * h_probe.size());
		warpcore::index_t tot_matches = 0;

		g.retrieve_gpu_ms1 = TimeGPUOp([&] {
			sink.build_gpu->warpcore_table->retrieve(
			    d_probe, h_probe.size(), d_beg, d_end, nullptr, tot_matches);
			cudaDeviceSynchronize();
		});

		if (tot_matches == 0) {
			if (!g.printed_times) {
				g.printed_times = true;
				std::cout
				    << "==========  KathanJoin PROBE (Execute/FinalExecute) "
				       "=========\n";
				std::cout << "[CPU] Execute() buffering            = "
				          << g.execute_cpu_ms << " ms\n";
				std::cout << "[CPU] probe host prep                 = "
				          << g.probe_hostprep_ms << " ms\n";
				std::cout << "[GPU] probe keys host→device         = "
				          << g.probe_gpu_h2d_ms << " ms\n";
				std::cout << "[GPU] retrieve() count               = "
				          << g.retrieve_gpu_ms1 << " ms\n";
				std::cout << "[GPU] retrieve() join               = "
				          << g.retrieve_gpu_ms2 << " ms\n";


				 /* ── NEW: section totals ─────────────────────────────── */
				double probe_total_ms =
					g.execute_cpu_ms + g.probe_hostprep_ms
					+ g.probe_gpu_h2d_ms + g.retrieve_gpu_ms1 + g.retrieve_gpu_ms2;

				double build_total_ms =
					sink.sink_cpu_ms + sink.finalize_overall_ms;

				double grand_total_ms = build_total_ms + probe_total_ms;

					std::cout << "[TOT] GRAND total                       = "
					<< grand_total_ms << " ms\n";
				std::cout
				    << "================================================================\n";
			}
			chunk.SetCardinality(0);
			g.finished_join = true;
			cudaFree(d_probe);
			cudaFree(d_beg);
			cudaFree(d_end);
			return OperatorFinalizeResultType::FINISHED;
		}

		/* ── second retrieve: materialise ids ───────────────────── */
		gpu_val_t *d_ids = nullptr;
		cudaMalloc(&d_ids, sizeof(gpu_val_t) * tot_matches);
		g.retrieve_gpu_ms2 += TimeGPUOp([&] {
			sink.build_gpu->warpcore_table->retrieve(
			    d_probe, h_probe.size(), d_beg, d_end, d_ids, tot_matches);
			cudaDeviceSynchronize();
		});

		/* ── copy back to host ──────────────────────────────────── */
		vector<warpcore::index_t> h_beg(h_probe.size()),
		    h_end(h_probe.size());
		vector<gpu_val_t> h_ids(tot_matches);

		auto d2h_start = clock_t::now();
		cudaMemcpy(h_beg.data(), d_beg,
		           sizeof(warpcore::index_t) * h_beg.size(),
		           cudaMemcpyDeviceToHost);
		cudaMemcpy(h_end.data(), d_end,
		           sizeof(warpcore::index_t) * h_end.size(),
		           cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ids.data(), d_ids, sizeof(gpu_val_t) * h_ids.size(),
		           cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		g.d2h_ms = std::chrono::duration<double, std::milli>(
		               clock_t::now() - d2h_start)
		               .count();

		cudaFree(d_probe);
		cudaFree(d_beg);
		cudaFree(d_end);
		cudaFree(d_ids);

		/* ── flatten match lists ───────────────────────────────── */
		auto flat_start = clock_t::now();
		g.build_indices.reserve(tot_matches);
		g.probe_indices.reserve(tot_matches);
		for (idx_t i = 0; i < h_probe.size(); i++) {
			for (auto off = h_beg[i]; off < h_end[i]; off++) {
				g.probe_indices.push_back(i);
				g.build_indices.push_back(h_ids[off]);
			}
		}
		g.match_count   = g.build_indices.size();
		g.output_offset = 0;
		g.flatten_cpu_ms = std::chrono::duration<double, std::milli>(
		                       clock_t::now() - flat_start)
		                       .count();

		g.finished_join = true;
		g.output_cpu_ms += std::chrono::duration<double, std::milli>(
		                       clock_t::now() - total_start)
		                       .count();
	}

	// ── 2) emit a chunk ──────────────────────────────────────────────
	idx_t remain = g.match_count - g.output_offset;
	if (remain == 0) {
		chunk.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	idx_t emit = std::min<idx_t>(remain, STANDARD_VECTOR_SIZE);


	chunk.Destroy();
	chunk.Initialize(Allocator::DefaultAllocator(), this->types);
		auto slice_start = clock_t::now();
	SelectionVector p_sel(STANDARD_VECTOR_SIZE), b_sel(STANDARD_VECTOR_SIZE);
	for (idx_t i = 0; i < emit; i++) {
		auto gi = g.output_offset + i;
		p_sel.set_index(i, g.probe_indices[gi]);
		b_sel.set_index(i, g.build_indices[gi]);
	}

	/* LHS */
	for (idx_t c = 0; c < lhs_output_columns.col_types.size(); c++) {
		auto &d = chunk.data[c];
		d.Reference(g.probe_chunk.data[lhs_output_columns.col_idxs[c]]);
		d.Slice(p_sel, emit);
	}
	/* RHS */
	idx_t rhs_offs = lhs_output_columns.col_types.size();
	for (idx_t c = 0; c < rhs_output_columns.col_types.size(); c++) {
		auto &d = chunk.data[rhs_offs + c];
		d.Reference(sink.build_chunk.data[rhs_output_columns.col_idxs[c]]);
		d.Slice(b_sel, emit);
	}
	chunk.SetCardinality(emit);

	g.output_offset += emit;
	g.output_cpu_ms += std::chrono::duration<double, std::milli>(
	                       clock_t::now() - slice_start)
	                       .count();

	/* ------------------------------------------------------------------ */
	/*  COLLISION CHECK – abort the query if a hash collision is detected */
	/* ------------------------------------------------------------------ */
	if (build_key_is_varchar) {
		const idx_t key_col_idx = 0;                      // join key column in output

		auto &lhs_key_vec = g.probe_chunk.data[0];
		auto &rhs_key_vec = sink.build_chunk.data[build_key_indices[0]];

		for (idx_t i = 0; i < emit; i++) {
			auto lhs_idx = p_sel.get_index(i);
			auto rhs_idx = b_sel.get_index(i);

			auto lhs_str = FlatVector::GetData<string_t>(lhs_key_vec)[lhs_idx];
			auto rhs_str = FlatVector::GetData<string_t>(rhs_key_vec)[rhs_idx];

			if (lhs_str != rhs_str) {
				throw InternalException(
					"GPU KathanJoin detected a hash collision between "
					"probe key '%s' and build key '%s'. "
					"Join aborted to guarantee correctness.",
					lhs_str.GetDataUnsafe(), rhs_str.GetDataUnsafe());
			}
		}
	}

	/* ── 3) print timers once ─────────────────────────────────────── */
	if (g.output_offset >= g.match_count && !g.printed_times) {
		g.printed_times = true;
		std::cout
		    << "==========  KathanJoin PROBE (Execute/FinalExecute) =========\n";
		std::cout << "[CPU] Execute() buffering            = "
		          << g.execute_cpu_ms << " ms\n";
		std::cout << "[CPU] probe host prep                 = "
		          << g.probe_hostprep_ms << " ms\n";
		std::cout << "[GPU] probe keys host→device         = "
		          << g.probe_gpu_h2d_ms << " ms\n";
		std::cout << "[GPU] retrieve() count             = "
		          << g.retrieve_gpu_ms1 << " ms\n";
		std::cout << "[GPU] retrieve() join             = "
		          << g.retrieve_gpu_ms2 << " ms\n";
		std::cout << "[GPU] Total retrieve time                = "
		          << (g.retrieve_gpu_ms1 + g.retrieve_gpu_ms2) << " ms\n";
		std::cout << "[CPU] device→host ids                = " << g.d2h_ms
		          << " ms\n";
		std::cout << "[CPU] flatten match lists            = " << g.flatten_cpu_ms
		          << " ms\n";
		std::cout << "[CPU] chunk slicing/output           = " << g.output_cpu_ms
		          << " ms\n";
		double probe_total_ms =
			g.execute_cpu_ms   + g.probe_hostprep_ms + g.probe_gpu_h2d_ms
				+ g.retrieve_gpu_ms1 + g.retrieve_gpu_ms2 + g.d2h_ms           + g.flatten_cpu_ms
				+ g.output_cpu_ms;

		double build_total_ms =
			sink.sink_cpu_ms + sink.finalize_overall_ms;   // value from step 1	
		double grand_total_ms = build_total_ms + probe_total_ms;	  
		std::cout << "[TOT] GRAND total                       = "
          << grand_total_ms << " ms\n";
		std::cout
		    << "================================================================\n";
	}

	return g.output_offset < g.match_count
	           ? OperatorFinalizeResultType::HAVE_MORE_OUTPUT
	           : OperatorFinalizeResultType::FINISHED;
}

// ────────────────────────────────────────────────────────────────────────────
// 10) Explain (unchanged)
// ────────────────────────────────────────────────────────────────────────────
InsertionOrderPreservingMap<string>
PhysicalKathanJoin::ParamsToString() const {
	InsertionOrderPreservingMap<string> res;
	res["Join Type"] = EnumUtil::ToString(join_type);
	string conds;
	for (idx_t i = 0; i < conditions.size(); i++) {
		if (i) conds += " AND ";
		conds += conditions[i].left->GetName() + " " +
		         ExpressionTypeToString(conditions[i].comparison) + " " +
		         conditions[i].right->GetName();
	}
	res["Conditions"] = conds;
	SetEstimatedCardinality(res, estimated_cardinality);
	return res;
}

} // namespace duckdb
