package executor

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor/dtype"
)

// Strata's grouped native experts (kernels/cuda/strata_iq_experts.cuh, ported from Strata at 296b59d2, MIT:
// src/kernels/cuda/iq_kernels.cu): native_expert_grouped's CUDA layout (exp_mode 0, a CUDA card's default) at
// Flash-Next's formats, its host side translated. STRATA_EXPERT_V2 and V2K, STRATA_GROUPED_V1, STRATA_OLD_IQ_MMVQ and
// STRATA_NO_SUB16_GU are off in the bench, so only the multi-entry kernels with their sub-warp rows are launched.

// native_expert_grouped's geometry, which its kernels' indexing fixes.
const (
	strataExpertThreads        = 256  // every kernel's block
	strataExpertGURows         = 8    // GU_ROWS: gate/up rows a block, a warp each
	strataExpertSub16Rows      = 16   // the 16-lane gate/up rows: a block's rows where n_embd is 2,560
	strataExpertSub16Embedding = 2560 // the n_embd the 16-lane gate/up rows take (xb == 80)
	strataExpertSubDownFF      = 640  // the n_ff the sub-warp down rows take (hb == 20)
	strataExpertSubDownRows    = 32   // IQ4_NL's 8-lane down rows: a block's rows at that n_ff
	strataExpertDownRows       = 8    // the other down kernels' rows a block, a warp each
	strataExpertScratchAlign   = 256  // native_expert_scratch_bytes' alignment of each part
)

// strataNativeExpertLayout is iq_kernels.hpp's NativeExpertLayout in its C layout (72 bytes), passed by value: an
// expert blob's gate rows, up rows and down rows, each as the GGUF stores them.
type strataNativeExpertLayout struct {
	GUType, DType  int32
	Embedding, FF  int64
	GURow, DRow    uint64 // bytes a row
	UpOff, DownOff uint64 // byte offsets inside the blob
	Bytes          uint64 // the whole blob
	TailOff        uint64 // NVFP4's scales at the blob's end; 0 = none
}

// strataNativeExpertLayoutOf is native_expert_layout: the blob of an expert whose gate and up rows are guType over
// embedding values and whose down rows are dType over ff values.
func strataNativeExpertLayoutOf(guType, dType dtype.Type, embedding, ff int64) (strataNativeExpertLayout, error) {
	guRow, err := guType.StorageBytes(uint64(embedding), uint64(embedding))
	if err != nil {
		return strataNativeExpertLayout{}, err
	}
	dRow, err := dType.StorageBytes(uint64(ff), uint64(ff))
	if err != nil {
		return strataNativeExpertLayout{}, err
	}
	layout := strataNativeExpertLayout{
		GUType: int32(guType), DType: int32(dType), Embedding: embedding, FF: ff, GURow: guRow, DRow: dRow,
		UpOff: uint64(ff) * guRow,
	}
	layout.DownOff = strataGateUpParts * layout.UpOff
	layout.Bytes = layout.DownOff + uint64(embedding)*dRow
	return layout, nil
}

// strataNativeExpertScratchBytes is native_expert_scratch_bytes: the gate, up and SwiGLU rows of capEntries entries,
// then their q8_1 image, each part 256-byte aligned.
func strataNativeExpertScratchBytes(capEntries, ff int64) uint64 {
	return strataExpertScratchParts*strataExpertAlign(uint64(capEntries)*uint64(ff)*f32ByteSize) +
		strataExpertAlign(uint64(capEntries)*uint64(ff/strataNativeQ8K)*strataNativeQ81Bytes)
}

func strataExpertAlign(bytes uint64) uint64 {
	return (bytes + strataExpertScratchAlign - 1) &^ (strataExpertScratchAlign - 1)
}

// launchStrataNativeExpertGrouped is native_expert_grouped: group g's blob at device address grpPtr[g], its entries
// [grpStart[g], grpStart[g+1]) reading token entTok[e]'s q8_1 activation (embedding/32 blocks a token in xq) and
// writing row entDst[e] of out (embedding floats); the count of groups is read on the device. gridGroups (1 ..
// capGroups; 0 = capGroups) groups run side by side, a block row each striding over the rest.
func launchStrataNativeExpertGrouped(state *device.State, functions functionSet, layout strataNativeExpertLayout,
	grpPtr, grpStart, nGroups, entDst, entTok driver.DevicePtr, capGroups, capEntries int64, xq, scratch, out driver.DevicePtr,
	gridGroups int64) error {
	if capGroups <= 0 || capEntries <= 0 {
		return nil
	}
	if layout.FF%strataNativeQ8K != 0 || layout.TailOff != 0 {
		return fmt.Errorf("native_expert_grouped: n_ff %d", layout.FF)
	}
	part := strataExpertAlign(uint64(capEntries) * uint64(layout.FF) * f32ByteSize)
	gate := scratch
	up := scratch + driver.DevicePtr(part)
	hq := scratch + driver.DevicePtr(strataExpertScratchParts*part)
	gy := gridGroups
	if gridGroups <= 0 || gridGroups > capGroups {
		gy = capGroups
	}
	var gu kernelFunctionID
	switch dtype.Type(layout.GUType) {
	case dtype.IQ3S:
		gu = kernelStrataNativeGuMultiIq3S
	case dtype.IQ4XS:
		gu = kernelStrataNativeGuMultiIq4Xs
	default:
		return fmt.Errorf("native_expert_grouped: gate/up type %d", layout.GUType)
	}
	// Both ported gate/up formats have a Split: the 16-lane rows where n_embd is 2,560.
	guRows := int64(strataExpertGURows)
	if layout.Embedding == strataExpertSub16Embedding {
		guRows = strataExpertSub16Rows
	}
	threads := driver.Dim3{X: strataExpertThreads, Y: 1, Z: 1}
	if err := launchGridABI(state, functions[gu], driver.Dim3{X: uint32((strataGateUpParts*layout.FF + guRows - 1) / guRows), Y: uint32(gy), Z: 1},
		threads, &grpPtr, &grpStart, &nGroups, &entTok, &xq, &layout, &gate, &up); err != nil {
		return err
	}
	nh := capEntries * layout.FF
	ff := int32(layout.FF)
	if err := launchGridABI(state, functions[kernelStrataSwigluQ81Entries],
		driver.Dim3{X: uint32((nh + strataExpertThreads - 1) / strataExpertThreads), Y: 1, Z: 1}, threads,
		&gate, &up, &grpStart, &nGroups, &ff, &hq); err != nil {
		return err
	}
	var down kernelFunctionID
	dRows := int64(strataExpertDownRows)
	switch dtype.Type(layout.DType) {
	case dtype.IQ4NL:
		down = kernelStrataNativeDownMultiIq4Nl
		if layout.FF == strataExpertSubDownFF {
			dRows = strataExpertSubDownRows
		}
	case dtype.Q8_0:
		down = kernelStrataNativeDownQ80
	default:
		return fmt.Errorf("native_expert_grouped: down type %d", layout.DType)
	}
	return launchGridABI(state, functions[down],
		driver.Dim3{X: uint32((layout.Embedding + dRows - 1) / dRows), Y: uint32(gy), Z: 1}, threads,
		&grpPtr, &grpStart, &nGroups, &entDst, &hq, &layout, &out)
}
