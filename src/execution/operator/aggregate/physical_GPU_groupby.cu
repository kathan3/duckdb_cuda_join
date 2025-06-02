#include "duckdb/execution/operator/aggregate/physical_GPU_groupby.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/hugeint.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <climits>

using U64 = unsigned long long;
static constexpr U64 HT_EMPTY = ULLONG_MAX;            // hash-table sentinel

//──────────────── host helpers ────────────────────────────────────────────
static size_t NextPow2(size_t v) {
	if (v <= 2) return 2;
	--v;
	v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
#if UINTPTR_MAX > 0xffffffff
	v |= v >> 32;
#endif
	return ++v;
}

static inline U64 FNV1a64(const char *data, size_t len) {
	const U64 OFFSET = 0xcbf29ce484222325ULL;
	const U64 PRIME  = 0x100000001b3ULL;
	U64 h = OFFSET;
	for (size_t i = 0; i < len; ++i) {
		h ^= static_cast<unsigned char>(data[i]);
		h *= PRIME;
	}
	return h;
}

static const char *KindName(duckdb::GPUAggKind k) {
	using namespace duckdb;
	switch (k) {
	case GPUAggKind::COUNT_STAR: return "COUNT";
	case GPUAggKind::SUM:        return "SUM";
	case GPUAggKind::MIN:        return "MIN";
	case GPUAggKind::MAX:        return "MAX";
	case GPUAggKind::AVG:        return "AVG";
	default:                     return "?";
	}
}

//──────────────── device helpers ─────────────────────────────────────────
__device__ __forceinline__ U64 murmur64_u64(U64 k) {   // fmix round
	k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
	k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
	k ^= k >> 33; return k;
}

//──────── constant metadata (small) ───────────────────────────────────────
constexpr int MAX_SPECS = 16;
constexpr int MAX_KEYS  = 16;

__constant__ duckdb::GPUAggKind d_kind[MAX_SPECS];
__constant__ int               d_val_slot[MAX_SPECS];
__device__  int                d_nspecs;
__device__  int                d_nkeys;   // # key columns at runtime

//──────────────── kernel 1: hash key tuples --------------------------------
__global__ void kernel_hash_keys(const U64 * const *row_keys,
                                 size_t n, int nkeys,
                                 U64 *row_hash)     // OUT
{
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n) return;

	U64 h = 0xc70f6907ULL ^ (8ULL * nkeys);   // seed ^ len(bytes)
#pragma unroll
	for (int k = 0; k < nkeys; ++k)
		h = murmur64_u64(h ^ row_keys[k][tid]);

	row_hash[tid] = h;
}

static constexpr U64 HT_BUSY  = ULLONG_MAX - 1ULL;   // new sentinel

__global__ void kernel_insert_multi(
        const U64 *row_hash,
        const U64 * const *row_keys,
        const int64_t * const *row_vals,
        size_t n, size_t cap,
        U64 **ht_keys,
        U64 *ht_cnt,
        U64 **ht_val_arr)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    size_t pos = row_hash[tid] & (cap - 1);

    while (true) {
        U64 prev = atomicCAS(&ht_keys[0][pos], HT_EMPTY, HT_BUSY);

        /* 1 ─ empty slot, we locked it */
        if (prev == HT_EMPTY) {
            #pragma unroll
            for (int k = 1; k < d_nkeys; ++k)
                ht_keys[k][pos] = row_keys[k][tid];

            __threadfence();                        // release

            atomicExch(&ht_keys[0][pos], row_keys[0][tid]);
            break;
        }

        /* 2 ─ slot still initialising, spin */
        if (prev == HT_BUSY)
            continue;

        /* 3 ─ READY slot: compare */
        if (prev == row_keys[0][tid]) {

            __threadfence();                        // acquire  ← NEW

            bool same = true;
            #pragma unroll
            for (int k = 1; k < d_nkeys && same; ++k)
                same &= (ht_keys[k][pos] == row_keys[k][tid]);
            if (same) break;
        }
        pos = (pos + 1) & (cap - 1);
    }

    /* ---- aggregate updates (unchanged) ---- */
    atomicAdd(&ht_cnt[pos], 1ULL);
    #pragma unroll
    for (int i = 0; i < d_nspecs; ++i) {
        auto kind = d_kind[i];
        if (kind == duckdb::GPUAggKind::COUNT_STAR) continue;

        U64 *dst  = ht_val_arr[d_val_slot[i]] + pos;
        int64_t v = row_vals[d_val_slot[i]][tid];

        if (kind == duckdb::GPUAggKind::SUM ||
            kind == duckdb::GPUAggKind::AVG)
            atomicAdd(dst, (U64)v);
        else if (kind == duckdb::GPUAggKind::MIN)
            atomicMin(dst, (U64)v);
        else
            atomicMax(dst, (U64)v);
    }
}

//──────────────── kernel 3: compact hash table -----------------------------
__global__ void kernel_compact_multi(
    const U64 * const *ht_keys,  const U64 *ht_cnt,
    U64 **ht_val_arr,            size_t cap,
    U64 **out_keys,              U64 *out_cnt,
    U64 **out_vals,              U64 *d_nout)
{
	size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= cap) return;
	if (ht_keys[0][pos] == HT_EMPTY) return;

	U64 idx = atomicAdd(d_nout, 1ULL);

#pragma unroll
	for (int k = 0; k < d_nkeys; ++k)
		out_keys[k][idx] = ht_keys[k][pos];

	out_cnt[idx] = ht_cnt[pos];

#pragma unroll
	for (int i = 0; i < d_nspecs; ++i)
		if (d_kind[i] != duckdb::GPUAggKind::COUNT_STAR)
			out_vals[d_val_slot[i]][idx] = ht_val_arr[d_val_slot[i]][pos];
}

//──────────────── sink-state struct ---------------------------------------
namespace duckdb {
struct GPUGroupByGlobalSinkState : public GlobalSinkState {
	explicit GPUGroupByGlobalSinkState(ClientContext &ctx,
	                                   const vector<LogicalType> &child_types)
	    : build_size(0), finalized(false) {
		build_chunk.Initialize(Allocator::Get(ctx), child_types);
	}

	DataChunk build_chunk;
	idx_t     build_size;

	// host-side accumulators
	vector<vector<U64>>              host_key_cols;     // [key][row]
	vector<vector<string>>           host_key_strs;     // [key]
	vector<unordered_map<U64,size_t>> hash2str;         // [key]
	vector<vector<int64_t>>          host_vals;         // [valSpec][row]

	// device pointers
	vector<U64*>     d_row_keys;     // [key]
	U64 **d_row_keys_arr = nullptr;
	U64 *d_row_hash      = nullptr;  // NEW

	vector<int64_t*> d_val_cols;     // [valSpec]
	U64 **d_row_vals_arr = nullptr;

	vector<U64*> d_ht_keys;          // [key]
	U64 **d_ht_keys_arr = nullptr;
	U64 *d_ht_cnt   = nullptr;
	vector<U64*> d_ht_val;           // [valSpec]
	U64 **d_ht_val_arr = nullptr;

	vector<U64*> d_out_keys;         // [key]
	U64 **d_out_keys_arr = nullptr;
	U64 *d_out_cnt  = nullptr;
	vector<U64*> d_out_val;          // [valSpec]
	U64 **d_out_val_arr  = nullptr;
	U64 *d_ngroups = nullptr;

	size_t ht_cap   = 0;
	U64    h_ngroups = 0;

	// final CPU result
	vector<vector<U64>>   final_key_cols;      // [key][g]
	vector<vector<string>> final_key_str_cols; // [key][g]
	vector<U64>           final_cnt;
	vector<vector<U64>>   final_val;           // [valSpec][g]

	bool finalized;
};
} // namespace duckdb

//──────────────── constructor ---------------------------------------------
namespace duckdb {
PhysicalGPUGroupBy::PhysicalGPUGroupBy(vector<LogicalType> out_types,
                                       vector<unique_ptr<Expression>> groups,
                                       vector<unique_ptr<Expression>> aggs,
                                       idx_t est_card)
    : PhysicalOperator(TYPE, std::move(out_types), est_card) {
	//── keys ------------------------------------------------------------
	if (groups.empty())
		throw NotImplementedException("GPUGroupBy: at least one key required");

	for (auto &expr : groups) {
		if (expr->type != ExpressionType::BOUND_REF)
			throw NotImplementedException("GPUGroupBy: keys must be bound refs");

		auto &ref = expr->Cast<BoundReferenceExpression>();
		key_types.push_back(ref.return_type);
		groupby_columns.push_back(ref.index);

		bool vc = ref.return_type == LogicalType::VARCHAR;
		if (!vc && ref.return_type != LogicalType::BIGINT)
			throw NotImplementedException("GPUGroupBy: key must be BIGINT/VARCHAR");
		key_is_varchar.push_back(vc);
	}
	if (key_types.size() > MAX_KEYS)
		throw NotImplementedException("GPUGroupBy: too many keys (max %d)", MAX_KEYS);

	//── aggregates ------------------------------------------------------
	for (auto &expr : aggs) {
		auto &a  = expr->Cast<BoundAggregateExpression>();
		auto  fn = StringUtil::Lower(a.function.name);

		if (fn == "count_star") {
			agg_specs.push_back({GPUAggKind::COUNT_STAR,
			                     DConstants::INVALID_INDEX,
			                     LogicalType::BIGINT});
		} else if (fn == "sum" || fn == "sum_no_overflow") {
			agg_specs.push_back({GPUAggKind::SUM,
			                     a.children[0]->Cast<BoundReferenceExpression>().index,
			                     a.return_type});
		} else if (fn == "min") {
			agg_specs.push_back({GPUAggKind::MIN,
			                     a.children[0]->Cast<BoundReferenceExpression>().index,
			                     a.return_type});
		} else if (fn == "max") {
			agg_specs.push_back({GPUAggKind::MAX,
			                     a.children[0]->Cast<BoundReferenceExpression>().index,
			                     a.return_type});
		} else if (fn == "avg" || fn == "avg_no_overflow") {
			agg_specs.push_back({GPUAggKind::AVG,
			                     a.children[0]->Cast<BoundReferenceExpression>().index,
			                     a.return_type});
		} else {
			throw NotImplementedException("GPUGroupBy: aggregate "+fn+" not supported");
		}
	}

	//── build final output schema --------------------------------------
	vector<LogicalType> schema;
	for (auto &t : key_types) schema.emplace_back(t);
	for (auto &spec : agg_specs) {
		switch (spec.kind) {
		case GPUAggKind::COUNT_STAR: schema.emplace_back(LogicalType::BIGINT);  break;
		case GPUAggKind::SUM:
		case GPUAggKind::MIN:
		case GPUAggKind::MAX:        schema.emplace_back(spec.val_type);        break;
		case GPUAggKind::AVG:        schema.emplace_back(LogicalType::DOUBLE);  break;
		}
	}
	this->types = std::move(schema);

	//── console banner --------------------------------------------------
	printf("[GPUGroupBy] keys:");
	for (auto &t : key_types) printf(" %s", t.ToString().c_str());
	printf("  | aggs:");
	for (auto &s : agg_specs) printf(" %s", KindName(s.kind));
	printf("\n");
}
} // namespace duckdb

//──────────────── sink plumbing -------------------------------------------
namespace duckdb {
unique_ptr<LocalSinkState>
PhysicalGPUGroupBy::GetLocalSinkState(ExecutionContext &) const {
	return make_uniq<LocalSinkState>();
}
unique_ptr<GlobalSinkState>
PhysicalGPUGroupBy::GetGlobalSinkState(ClientContext &ctx) const {
	return make_uniq<GPUGroupByGlobalSinkState>(ctx, children[0]->GetTypes());
}
SinkResultType PhysicalGPUGroupBy::Sink(ExecutionContext &,
                                        DataChunk &chunk,
                                        OperatorSinkInput &in) const {
	auto &gs = in.global_state.Cast<GPUGroupByGlobalSinkState>();
	if (!chunk.size()) return SinkResultType::NEED_MORE_INPUT;
	gs.build_chunk.Append(chunk, true);
	gs.build_size += chunk.size();
	return SinkResultType::NEED_MORE_INPUT;
}
SinkCombineResultType PhysicalGPUGroupBy::Combine(ExecutionContext &,
                                                  OperatorSinkCombineInput&) const {
	return SinkCombineResultType::FINISHED;
}
} // namespace duckdb

//──────────────── SINK finalisation ---------------------------------------
namespace duckdb {
SinkFinalizeType PhysicalGPUGroupBy::Finalize(Pipeline &, Event &, ClientContext &,
                                              OperatorSinkFinalizeInput &in) const {
	auto &gs = in.global_state.Cast<GPUGroupByGlobalSinkState>();
	if (gs.build_size == 0) { gs.finalized = true; return SinkFinalizeType::READY; }

	const int key_cnt = key_types.size();

	//── map spec → value slot ------------------------------------------
	int val_specs = 0;
	vector<int> spec2slot(agg_specs.size(), -1);
	for (size_t i = 0; i < agg_specs.size(); ++i)
		if (agg_specs[i].kind != GPUAggKind::COUNT_STAR)
			spec2slot[i] = val_specs++;
	gs.host_vals.assign(val_specs, {});
	for (auto &v : gs.host_vals) v.reserve(gs.build_size);

	//── flatten key vectors once ---------------------------------------
	gs.host_key_cols.assign(key_cnt, {});
	gs.host_key_strs.assign(key_cnt, {});
	gs.hash2str.assign(key_cnt, {});

	vector<const int64_t *> key_int_ptr(key_cnt, nullptr);
	vector<const string_t *> key_str_ptr(key_cnt, nullptr);
	vector<const ValidityMask*> key_valid(key_cnt, nullptr);

	for (int k = 0; k < key_cnt; ++k) {
		auto col = groupby_columns[k];
		auto &vec = gs.build_chunk.data[col];
		vec.Flatten(gs.build_chunk.size());
		key_valid[k] = &FlatVector::Validity(vec);

		if (!key_is_varchar[k])
			key_int_ptr[k] = FlatVector::GetData<int64_t>(vec);
		else
			key_str_ptr[k] = FlatVector::GetData<string_t>(vec);
	}

	//── flatten value vectors ------------------------------------------
	vector<const int64_t *>     val_ptrs(val_specs, nullptr);
	vector<const ValidityMask*> val_masks(val_specs, nullptr);
	for (size_t spec = 0; spec < agg_specs.size(); ++spec) {
		if (agg_specs[spec].kind == GPUAggKind::COUNT_STAR) continue;
		int slot = spec2slot[spec];
		int col  = agg_specs[spec].input_col;

		auto &vec = gs.build_chunk.data[col];
		vec.Flatten(gs.build_chunk.size());
		val_ptrs[slot]  = FlatVector::GetData<int64_t>(vec);
		val_masks[slot] = &FlatVector::Validity(vec);
	}

	//── copy rows into host buffers ------------------------------------
	for (idx_t i = 0; i < gs.build_size; ++i) {
		bool any_null = false;
		for (int k = 0; k < key_cnt && !any_null; ++k)
			any_null = !key_valid[k]->RowIsValid(i);
		if (any_null) continue;  // NULL key => skip row

		for (int k = 0; k < key_cnt; ++k) {
			if (!key_is_varchar[k]) {
				gs.host_key_cols[k].push_back((U64)key_int_ptr[k][i]);
			} else {
				string str = key_str_ptr[k][i].GetString();
				U64 h      = FNV1a64(str.data(), str.size());
				if (gs.hash2str[k].find(h) == gs.hash2str[k].end()) {
					gs.hash2str[k][h] = gs.host_key_strs[k].size();
					gs.host_key_strs[k].push_back(std::move(str));
				}
				gs.host_key_cols[k].push_back(h);
			}
		}

		for (int s = 0; s < val_specs; ++s) {
			bool valid = val_masks[s]->RowIsValid(i);
			gs.host_vals[s].push_back(valid ? val_ptrs[s][i] : 0);
		}
	}
	size_t N = gs.host_key_cols[0].size();
	if (N == 0) { gs.finalized = true; return SinkFinalizeType::READY; }

	//── upload key columns ---------------------------------------------
	gs.d_row_keys.resize(key_cnt, nullptr);
	for (int k = 0; k < key_cnt; ++k) {
		cudaMalloc(&gs.d_row_keys[k], N * sizeof(U64));
		cudaMemcpy(gs.d_row_keys[k], gs.host_key_cols[k].data(),
		           N * sizeof(U64), cudaMemcpyHostToDevice);
	}
	cudaMalloc(&gs.d_row_keys_arr, key_cnt * sizeof(U64 *));
	cudaMemcpy(gs.d_row_keys_arr, gs.d_row_keys.data(),
	           key_cnt * sizeof(U64 *), cudaMemcpyHostToDevice);

	//── compute composite hash per row ---------------------------------
	cudaMalloc(&gs.d_row_hash, N * sizeof(U64));
	int TPB = 256;
	kernel_hash_keys<<<(N + TPB - 1) / TPB, TPB>>>(
	    (const U64 * const *)gs.d_row_keys_arr, N, key_cnt, gs.d_row_hash);
	cudaDeviceSynchronize();

	//── upload value columns -------------------------------------------
	gs.d_val_cols.resize(val_specs, nullptr);
	for (int s = 0; s < val_specs; ++s) {
		cudaMalloc(&gs.d_val_cols[s], N * sizeof(int64_t));
		cudaMemcpy(gs.d_val_cols[s], gs.host_vals[s].data(),
		           N * sizeof(int64_t), cudaMemcpyHostToDevice);
	}
	if (val_specs) {
		cudaMalloc(&gs.d_row_vals_arr, val_specs * sizeof(int64_t *));
		cudaMemcpy(gs.d_row_vals_arr, gs.d_val_cols.data(),
		           val_specs * sizeof(int64_t *), cudaMemcpyHostToDevice);
	}

	//── build hash table -----------------------------------------------
	gs.ht_cap = NextPow2(N * 2);
	gs.d_ht_keys.resize(key_cnt, nullptr);
	for (int k = 0; k < key_cnt; ++k) {
		cudaMalloc(&gs.d_ht_keys[k], gs.ht_cap * sizeof(U64));
		cudaMemset(gs.d_ht_keys[k], 0xFF, gs.ht_cap * sizeof(U64));
	}
	cudaMalloc(&gs.d_ht_keys_arr, key_cnt * sizeof(U64 *));
	cudaMemcpy(gs.d_ht_keys_arr, gs.d_ht_keys.data(),
	           key_cnt * sizeof(U64 *), cudaMemcpyHostToDevice);

	cudaMalloc(&gs.d_ht_cnt, gs.ht_cap * sizeof(U64));
	cudaMemset(gs.d_ht_cnt, 0x00, gs.ht_cap * sizeof(U64));

	gs.d_ht_val.resize(val_specs, nullptr);
	for (int s = 0; s < val_specs; ++s) {
		cudaMalloc(&gs.d_ht_val[s], gs.ht_cap * sizeof(U64));
		bool is_min = false;
		for (size_t spec = 0; spec < agg_specs.size(); ++spec)
			if (spec2slot[spec] == s &&
			    agg_specs[spec].kind == GPUAggKind::MIN)
				is_min = true;
		cudaMemset(gs.d_ht_val[s], is_min ? 0xFF : 0x00,
		           gs.ht_cap * sizeof(U64));
	}
	if (val_specs) {
		cudaMalloc(&gs.d_ht_val_arr, val_specs * sizeof(U64 *));
		cudaMemcpy(gs.d_ht_val_arr, gs.d_ht_val.data(),
		           val_specs * sizeof(U64 *), cudaMemcpyHostToDevice);
	}

	//── copy constant metadata to device -------------------------------
	int nspecs = (int)agg_specs.size();
	std::vector<GPUAggKind> h_kind(nspecs);
	std::vector<int>        h_slot(nspecs);
	for (size_t i = 0; i < agg_specs.size(); ++i) {
		h_kind[i] = agg_specs[i].kind;
		h_slot[i] = spec2slot[i];
	}
	cudaMemcpyToSymbol(d_kind,     h_kind.data(), nspecs * sizeof(GPUAggKind));
	cudaMemcpyToSymbol(d_val_slot, h_slot.data(), nspecs * sizeof(int));
	cudaMemcpyToSymbol(d_nspecs,   &nspecs, sizeof(int));
	cudaMemcpyToSymbol(d_nkeys,    &key_cnt, sizeof(int));

	//── INSERT kernel ---------------------------------------------------
	kernel_insert_multi<<<(N + TPB - 1) / TPB, TPB>>>(
	    gs.d_row_hash,
	    (const U64 * const *)gs.d_row_keys_arr,
	    (const int64_t * const *)gs.d_row_vals_arr,
	    N, gs.ht_cap,
	    gs.d_ht_keys_arr, gs.d_ht_cnt, gs.d_ht_val_arr);
	cudaDeviceSynchronize();

	//── COMPACT phase ---------------------------------------------------
	gs.d_out_keys.resize(key_cnt, nullptr);
	for (int k = 0; k < key_cnt; ++k)
		cudaMalloc(&gs.d_out_keys[k], gs.ht_cap * sizeof(U64));
	cudaMalloc(&gs.d_out_keys_arr, key_cnt * sizeof(U64 *));
	cudaMemcpy(gs.d_out_keys_arr, gs.d_out_keys.data(),
	           key_cnt * sizeof(U64 *), cudaMemcpyHostToDevice);

	cudaMalloc(&gs.d_out_cnt, gs.ht_cap * sizeof(U64));
	gs.d_out_val.resize(val_specs, nullptr);
	for (int s = 0; s < val_specs; ++s)
		cudaMalloc(&gs.d_out_val[s], gs.ht_cap * sizeof(U64));
	if (val_specs) {
		cudaMalloc(&gs.d_out_val_arr, val_specs * sizeof(U64 *));
		cudaMemcpy(gs.d_out_val_arr, gs.d_out_val.data(),
		           val_specs * sizeof(U64 *), cudaMemcpyHostToDevice);
	}

	cudaMalloc(&gs.d_ngroups, sizeof(U64));
	U64 zero = 0ULL;
	cudaMemcpy(gs.d_ngroups, &zero, sizeof(U64), cudaMemcpyHostToDevice);

	kernel_compact_multi<<<(gs.ht_cap + TPB - 1) / TPB, TPB>>>(
	    (const U64 * const *)gs.d_ht_keys_arr, gs.d_ht_cnt,
	    gs.d_ht_val_arr, gs.ht_cap,
	    gs.d_out_keys_arr, gs.d_out_cnt,
	    gs.d_out_val_arr, gs.d_ngroups);
	cudaDeviceSynchronize();

	//── download dense arrays ------------------------------------------
	cudaMemcpy(&gs.h_ngroups, gs.d_ngroups,
	           sizeof(U64), cudaMemcpyDeviceToHost);
	if (!gs.h_ngroups) { gs.finalized = true; return SinkFinalizeType::READY; }

	gs.final_key_cols.resize(key_cnt);
	for (int k = 0; k < key_cnt; ++k) {
		gs.final_key_cols[k].resize(gs.h_ngroups);
		cudaMemcpy(gs.final_key_cols[k].data(), gs.d_out_keys[k],
		           gs.h_ngroups * sizeof(U64), cudaMemcpyDeviceToHost);
	}
	gs.final_cnt.resize(gs.h_ngroups);
	cudaMemcpy(gs.final_cnt.data(), gs.d_out_cnt,
	           gs.h_ngroups * sizeof(U64), cudaMemcpyDeviceToHost);

	gs.final_val.resize(val_specs);
	for (int s = 0; s < val_specs; ++s) {
		gs.final_val[s].resize(gs.h_ngroups);
		cudaMemcpy(gs.final_val[s].data(), gs.d_out_val[s],
		           gs.h_ngroups * sizeof(U64), cudaMemcpyDeviceToHost);
	}

	//── rebuild VARCHAR keys -------------------------------------------
	gs.final_key_str_cols.resize(key_cnt);
	for (int k = 0; k < key_cnt; ++k) if (key_is_varchar[k]) {
		gs.final_key_str_cols[k].resize(gs.h_ngroups);
		for (idx_t g = 0; g < gs.h_ngroups; ++g) {
			U64 h = gs.final_key_cols[k][g];
			auto it = gs.hash2str[k].find(h);
			gs.final_key_str_cols[k][g] =
			    (it != gs.hash2str[k].end())
			    ? gs.host_key_strs[k][it->second]
			    : "<hash-collision>";
		}
	}

	// free temp hash
	cudaFree(gs.d_row_hash);

	gs.finalized = true;
	return SinkFinalizeType::READY;
}
} // namespace duckdb

//──────────────── SOURCE side ---------------------------------------------
namespace duckdb {
unique_ptr<GlobalSourceState>
PhysicalGPUGroupBy::GetGlobalSourceState(ClientContext &) const {
	return make_uniq<GlobalSourceState>();
}
unique_ptr<LocalSourceState>
PhysicalGPUGroupBy::GetLocalSourceState(ExecutionContext &,
                                        GlobalSourceState &) const {
	return make_uniq<LocalSourceState>();
}

SourceResultType PhysicalGPUGroupBy::GetData(ExecutionContext &,
                                             DataChunk &chunk,
                                             OperatorSourceInput &) const {
	auto &gs = sink_state->Cast<GPUGroupByGlobalSinkState>();
	static idx_t read_pos = 0;
	if (!gs.finalized || read_pos >= gs.h_ngroups) {
		chunk.SetCardinality(0);
		return SourceResultType::FINISHED;
	}

	idx_t take = std::min<idx_t>(STANDARD_VECTOR_SIZE, gs.h_ngroups - read_pos);
	chunk.Initialize(Allocator::DefaultAllocator(), this->types);

	const int key_cnt = key_types.size();
	idx_t col = 0;

	//── emit keys -------------------------------------------------------
	for (int k = 0; k < key_cnt; ++k) {
		if (!key_is_varchar[k]) {
			auto *dst = FlatVector::GetData<int64_t>(chunk.data[col++]);
			for (idx_t i = 0; i < take; ++i)
				dst[i] = (int64_t)gs.final_key_cols[k][read_pos+i];
		} else {
			auto &vec = chunk.data[col++];
			auto *dst = FlatVector::GetData<string_t>(vec);
			for (idx_t i = 0; i < take; ++i)
				dst[i] = StringVector::AddString(vec,
				            gs.final_key_str_cols[k][read_pos+i]);
		}
	}

	//── map spec→slot ---------------------------------------------------
	int val_specs = 0;
	vector<int> spec2slot(agg_specs.size(), -1);
	for (size_t i = 0; i < agg_specs.size(); ++i)
		if (agg_specs[i].kind != GPUAggKind::COUNT_STAR)
			spec2slot[i] = val_specs++;

	//── emit aggregates -------------------------------------------------
	for (size_t spec = 0; spec < agg_specs.size(); ++spec) {
		auto kind = agg_specs[spec].kind;
		switch (kind) {
		case GPUAggKind::COUNT_STAR: {
			auto *dst = FlatVector::GetData<int64_t>(chunk.data[col++]);
			for (idx_t i = 0; i < take; ++i)
				dst[i] = (int64_t)gs.final_cnt[read_pos+i];
			break;
		}
		case GPUAggKind::SUM: {
			if (agg_specs[spec].val_type.id() == LogicalTypeId::HUGEINT) {
				auto *dst = FlatVector::GetData<hugeint_t>(chunk.data[col++]);
				auto &arr = gs.final_val[spec2slot[spec]];
				for (idx_t i = 0; i < take; ++i)
					dst[i] = Hugeint::Convert((int64_t)arr[read_pos+i]);
			} else {
				auto *dst = FlatVector::GetData<int64_t>(chunk.data[col++]);
				auto &arr = gs.final_val[spec2slot[spec]];
				for (idx_t i = 0; i < take; ++i)
					dst[i] = (int64_t)arr[read_pos+i];
			}
			break;
		}
		case GPUAggKind::MIN:
		case GPUAggKind::MAX: {
			auto *dst = FlatVector::GetData<int64_t>(chunk.data[col++]);
			auto &arr = gs.final_val[spec2slot[spec]];
			for (idx_t i = 0; i < take; ++i)
				dst[i] = (int64_t)arr[read_pos+i];
			break;
		}
		case GPUAggKind::AVG: {
			auto *dst = FlatVector::GetData<double>(chunk.data[col++]);
			auto &sum_arr = gs.final_val[spec2slot[spec]];
			for (idx_t i = 0; i < take; ++i)
				dst[i] = double(sum_arr[read_pos+i]) /
				         double(gs.final_cnt[read_pos+i]);
			break;
		}
		}
	}

	chunk.SetCardinality(take);
	read_pos += take;
	return SourceResultType::HAVE_MORE_OUTPUT;
}
} // namespace duckdb

//──────────────── ParamsToString ------------------------------------------
namespace duckdb {
InsertionOrderPreservingMap<string>
PhysicalGPUGroupBy::ParamsToString() const {
	InsertionOrderPreservingMap<string> m;
	m["GPU GroupBy"] = std::to_string(key_types.size()) + " keys";
	return m;
}
} // namespace duckdb
