package executor

import (
	"fmt"
	"os"
	"sort"

	"overgo/internal/tensor"
	"overgo/internal/tensor/planner"
)

func dumpResourceMemory(resources *executorResources) {
	if os.Getenv("OVERGO_MEMORY_PLAN") == "" || resources == nil {
		return
	}
	var pool, free uint64
	for _, lease := range resources.buffers.allocations {
		pool += lease.size
	}
	for size, pointers := range resources.buffers.free {
		free += size * uint64(len(pointers))
	}
	var staging, scores uint64
	if resources.blas != nil {
		staging, scores = resources.blas.stagingBytes, resources.blas.scoreBytes
	}
	fmt.Fprintf(os.Stderr, "[memory-resources] arena=%d staging=%d scores=%d q8=%d pool=%d free=%d\n",
		resources.arenaSize, staging, scores, resources.q8Input.stagingBytes, pool, free)
}

// OVERGO_OP_COUNTS=1: dump per-graph launch composition at compile time.
func dumpOpCounts(compiled *CompiledGraph) {
	if os.Getenv("OVERGO_OP_COUNTS") == "" && os.Getenv("OVERGO_MEMORY_PLAN") == "" {
		return
	}
	if os.Getenv("OVERGO_MEMORY_PLAN") != "" {
		allocations := make([]planner.Allocation, 0, len(compiled.memory.Allocations))
		for _, allocation := range compiled.memory.Allocations {
			allocations = append(allocations, allocation)
		}
		sort.Slice(allocations, func(i, j int) bool { return allocations[i].Size > allocations[j].Size })
		fmt.Fprintf(os.Stderr, "[memory-plan] arena=%d allocations=%d\n", compiled.memory.ArenaSize, len(allocations))
		for _, allocation := range allocations[:min(20, len(allocations))] {
			fmt.Fprintf(os.Stderr, "[memory-plan]   id=%d op=%s size=%d live=%d..%d shape=%v\n",
				allocation.Tensor.ID, allocation.Tensor.Op, allocation.Size, allocation.First, allocation.Last, allocation.Tensor.Shape.Dims)
		}
	}
	if os.Getenv("OVERGO_OP_COUNTS") == "" {
		return
	}
	counts := make(map[string]int)
	launches := 0
	for _, node := range compiled.order {
		if node.Op == tensor.OpInput {
			continue
		}
		if _, skipped := compiled.skipped[node]; skipped {
			counts["skipped:"+node.Op.String()]++
			continue
		}
		if _, elided := compiled.elided[node]; elided {
			counts["elided:"+node.Op.String()]++
			continue
		}
		if _, aliases, err := tensor.ResolveStorageView(node); err == nil && aliases {
			counts["alias:"+node.Op.String()]++
			continue
		}
		key := node.Op.String()
		fusion := compiled.nodes[compiled.orderIndexes[node]].fusion
		if fusion != nil {
			switch fusion.kind {
			case compiledFusionWeightedRMS:
				key = "fused:weighted_rms"
				if fusion.weightedRMS.addLeft != nil {
					key = "fused:weighted_rms_add"
				}
			case compiledFusionActivatedGate:
				key = "fused:activated_gate"
			case compiledFusionWeightedRMSGate:
				key = "fused:weighted_rms_gate"
			case compiledFusionQ8ArgmaxPartials, compiledFusionQ8ArgmaxReduction:
				key = "fused:q8_argmax"
			case compiledFusionBF16Gate:
				key = "fused:bf16_gate"
			case compiledFusionBF16ProjAdd:
				key = "fused:bf16_projection_add"
			case compiledFusionBF16Append:
				key = "fused:bf16_append"
			case compiledFusionBF16ArgmaxPartials, compiledFusionBF16ArgmaxReduction:
				key = "fused:bf16_argmax"
			case compiledFusionRopeAppend:
				key = "fused:rope_append"
			}
		} else if node.Op == tensor.OpMulMat {
			left, right := node.Inputs[0], node.Inputs[1]
			key = fmt.Sprintf("mul_mat[%s %dx%d rhs=%d]",
				left.Type, left.Shape.Dims[0], left.Shape.Dims[1], right.Shape.Dims[1])
		}
		counts[key]++
		launches++
	}
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	fmt.Fprintf(os.Stderr, "[op-counts] graph nodes=%d launched=%d\n", len(compiled.order), launches)
	for _, key := range keys {
		fmt.Fprintf(os.Stderr, "[op-counts]   %-40s %d\n", key, counts[key])
	}
}
