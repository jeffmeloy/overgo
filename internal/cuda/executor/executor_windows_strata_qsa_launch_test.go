//go:build windows

package executor

import (
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// Strata's launches the parity programs call and no served step does yet: the fp16 and int8 gathers, the gate and a
// rope scaling's defaults.

// launchStrataKVGather gathers step[width] selected cells named by ids through the page table into fp16 scratch
// `[id][kv_head][head_dim]`, dequantizing int8 cells: kv_gather_step or kv_gather_q8_step. The grid is sized by
// maxIDs, the capacity; the kernels read the real count from the step state.
func launchStrataKVGather(
	state *device.State, functions functionSet, pools strataKVPools, table, ids, step driver.DevicePtr, maxIDs uint32,
	geometry strataKVGeometry, keyScratch, valueScratch driver.DevicePtr, int8 bool,
) error {
	if err := geometry.validate(int8); err != nil {
		return err
	}
	if maxIDs == 0 {
		return nil
	}
	heads, width, page := geometry.kvHeads, geometry.headDim, geometry.pageSize
	total := maxIDs * uint32(heads) * (uint32(width) / strataGatherValues)
	grid := driver.Dim3{X: (total + strataGatherThreads - 1) / strataGatherThreads, Y: 1, Z: 1}
	block := driver.Dim3{X: strataGatherThreads, Y: 1, Z: 1}
	if !int8 {
		return launchGridABI(state, functions[kernelStrataKvGatherF16], grid, block,
			&pools.key, &pools.value, &table, &ids, &step, &heads, &width, &page, &keyScratch, &valueScratch)
	}
	return launchGridABI(state, functions[kernelStrataKvGatherQ8], grid, block,
		&pools.key, &pools.value, &pools.keyScale, &pools.valueScale, &table, &ids, &step, &heads, &width, &page,
		&keyScratch, &valueScratch)
}

// launchStrataGateApply is qsa.cu's qsa_gate_apply_f32: attention times the sigmoid of the second half of each
// head's query-and-gate block, in double, unrounded.
func launchStrataGateApply(state *device.State, functions functionSet, attention, queryAndGate driver.DevicePtr, heads, width int32, output driver.DevicePtr) error {
	return launch1DABI(state, functions[kernelStrataGateApplyF32], uint32(heads*width), &attention, &queryAndGate, &heads, &width, &output)
}

func newStrataRopeScaling(kind int, factor, extFactor float64) strataRopeScaling {
	return strataRopeScaling{kind: kind, freqBase: strataFreqBaseDefault, factor: factor, extFactor: extFactor, attnFactor: 1,
		originalContext: 262144, betaFast: 32, betaSlow: 1}
}
