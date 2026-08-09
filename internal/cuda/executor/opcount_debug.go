package executor

import (
	"fmt"
	"os"
	"sort"

	"overgo/internal/tensor"
)

// OVERGO_OP_COUNTS=1: dump per-graph launch composition at compile time.
func dumpOpCounts(compiled *CompiledGraph) {
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
		switch {
		case compiled.weightedRMS != nil && func() bool { _, ok := compiled.weightedRMS[node]; return ok }():
			key = "fused:weighted_rms"
			if compiled.weightedRMS[node].addLeft != nil {
				key = "fused:weighted_rms_add"
			}
		case compiled.activatedGate != nil && func() bool { _, ok := compiled.activatedGate[node]; return ok }():
			key = "fused:activated_gate"
		case compiled.weightedRMSGate != nil && func() bool { _, ok := compiled.weightedRMSGate[node]; return ok }():
			key = "fused:weighted_rms_gate"
		case compiled.q8Argmax != nil && func() bool { _, ok := compiled.q8Argmax[node]; return ok }():
			key = "fused:q8_argmax"
		case compiled.bf16Gate != nil && func() bool { _, ok := compiled.bf16Gate[node]; return ok }():
			key = "fused:bf16_gate"
		case compiled.bf16ProjAdd != nil && func() bool { _, ok := compiled.bf16ProjAdd[node]; return ok }():
			key = "fused:bf16_projection_add"
		case compiled.bf16Append != nil && func() bool { _, ok := compiled.bf16Append[node]; return ok }():
			key = "fused:bf16_append"
		case compiled.bf16Argmax != nil && func() bool { _, ok := compiled.bf16Argmax[node]; return ok }():
			key = "fused:bf16_argmax"
		case compiled.ropeAppend != nil && func() bool { _, ok := compiled.ropeAppend[node]; return ok }():
			key = "fused:rope_append"
		case node.Op == tensor.OpMulMat:
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
