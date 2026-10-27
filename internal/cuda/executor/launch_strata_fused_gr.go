package executor

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// Strata's hyper-connection read and write (kernels/cuda/strata_fused_gr.cuh and strata_native_gr_postops.cuh, ported
// from Strata at 296b59d2, MIT: src/kernels/cuda/fused_gr.cu and native_gr_postops.cu): fused_gr_read_multi over the
// GGUF's Q8_0 projections (STRATA_HC_Q8, launch_q8) and gr_write_multi through native_gr_post_multi (gr_native_mmvf,
// which --native sets), their host sides translated.

// fused_gr.cu's compiled geometry: Flash-Next's hyper-connections.
const (
	strataGREmbedding  = 2560                           // N: n_embd
	strataGRStreams    = 4                              // HC: the streams
	strataGRRank       = 320                            // LR: hc_lr
	strataGRMaxTokens  = 8                              // kFusedGrMaxT
	strataGRThreads    = 256                            // THREADS
	strataGRUpBlocks   = 160                            // UPM_BLOCKS: n_embd over UPM_COLS
	strataGRRowGroups  = 10                             // Q8_RG: the down rows over (WARPS * Q8_RPW); one more group holds the inject rows
	strataGRChunks     = 16                             // Q8_NKC: the streams' columns in chunks of Q8_KC
	strataGRPartialRow = strataGRRank + strataGRStreams // PR: a (token, chunk)'s partial rows, the down's and the inject's
)

// strataFusedGRArgs is fused_gr.hpp's FusedGrArgs, field for field in its C layout (152 bytes): a token's operands
// of a read.
type strataFusedGRArgs struct {
	R, ROut                                     driver.DevicePtr
	Apply                                       uint8
	_                                           [strataCPadByte]byte
	BoPrev, InjPrev, WNorm, WDown, WUp, WInject driver.DevicePtr
	Eps                                         float32
	_                                           [strataCPadWord]byte
	Lo, Rs, InjectOut, Mixed                    driver.DevicePtr
	Q8Down, Q8Up, Q8Inject                      driver.DevicePtr
	Q8Mixed, Q8Count                            driver.DevicePtr
}

// strataGRMulti is fused_gr.cu's GrMulti in its C layout (1,232 bytes): a read's tokens, passed by value.
type strataGRMulti struct {
	A  [strataGRMaxTokens]strataFusedGRArgs
	Xn driver.DevicePtr
	T  int32
	_  [strataCPadWord]byte
}

// strataGRDownQ8 and strataGRUpQ8 are launch_q8_t's instances by token count, from one.
var (
	strataGRDownQ8 = [strataGRMaxTokens]kernelFunctionID{
		kernelStrataGrDownQ8T1, kernelStrataGrDownQ8T2, kernelStrataGrDownQ8T3, kernelStrataGrDownQ8T4,
		kernelStrataGrDownQ8T5, kernelStrataGrDownQ8T6, kernelStrataGrDownQ8T7, kernelStrataGrDownQ8T8,
	}
	strataGRUpQ8 = [strataGRMaxTokens]kernelFunctionID{
		kernelStrataGrUpQ8T1, kernelStrataGrUpQ8T2, kernelStrataGrUpQ8T3, kernelStrataGrUpQ8T4,
		kernelStrataGrUpQ8T5, kernelStrataGrUpQ8T6, kernelStrataGrUpQ8T7, kernelStrataGrUpQ8T8,
	}
)

// launchStrataFusedGRReadQ8 is fused_gr_read_multi with Q8_0 projections (launch_q8): the read of the tokens' args, one
// args a token, with their down and up projections and norm shared, over scratch (the down's partial rows, then its
// sums of squares).
func launchStrataFusedGRReadQ8(state *device.State, functions functionSet, args []strataFusedGRArgs, scratch driver.DevicePtr) error {
	tokens := len(args)
	if tokens < 1 || tokens > strataGRMaxTokens || scratch == 0 {
		return fmt.Errorf("fused_gr_read_multi: invalid arguments")
	}
	var multi strataGRMulti
	for index, token := range args {
		if token.R == 0 || token.WNorm == 0 || token.Q8Down == 0 || token.Q8Up == 0 || token.Lo == 0 || token.Rs == 0 ||
			token.Mixed == 0 || (token.WInject != 0 && token.InjectOut == 0) ||
			(token.Apply != 0 && (token.BoPrev == 0 || token.InjPrev == 0 || token.ROut == 0)) ||
			token.Q8Down != args[0].Q8Down || token.Q8Up != args[0].Q8Up || token.WInject != args[0].WInject ||
			token.WNorm != args[0].WNorm || token.Q8Inject != args[0].Q8Inject {
			return fmt.Errorf("fused_gr_read_multi: invalid arguments for token %d", index)
		}
		multi.A[index] = token
		// S26 STRATA_QFUSE (opt-in, not ported): no token writes a q8_1 image of its mixed row.
		multi.A[index].Q8Mixed, multi.A[index].Q8Count = 0, 0
	}
	multi.Xn, multi.T = scratch, int32(tokens)
	part := scratch
	sums := scratch + driver.DevicePtr(uint64(tokens)*strataGRChunks*strataGRPartialRow*f32ByteSize)
	if err := launchGridABI(state, functions[strataGRDownQ8[tokens-1]],
		driver.Dim3{X: strataGRRowGroups + 1, Y: strataGRChunks, Z: 1}, driver.Dim3{X: strataGRThreads, Y: 1, Z: 1},
		&multi, &part, &sums); err != nil {
		return err
	}
	return launchGridABI(state, functions[strataGRUpQ8[tokens-1]],
		driver.Dim3{X: strataGRUpBlocks, Y: 1, Z: 1}, driver.Dim3{X: strataGRThreads, Y: 1, Z: 1},
		&multi, &part, &sums)
}

// f32ByteSize is a float's bytes.
const f32ByteSize = 4

// launchStrataNativeGRPost is native_gr_post_multi (gr_write_multi under gr_native_mmvf): each of tokens contiguous
// residuals of streams rows of embedding values gains its block output weighted by 2 sigmoid(inject / streams), into
// output (which may be the residual).
func launchStrataNativeGRPost(state *device.State, functions functionSet, residual, blockOut, inject, output driver.DevicePtr,
	embedding, streams, tokens int32) error {
	if tokens <= 0 || embedding <= 0 || streams <= 0 {
		return fmt.Errorf("native GR postops require positive bounded dimensions")
	}
	scale := 1 / float32(streams)
	values := uint32(embedding) * uint32(streams)
	return launchGridABI(state, functions[kernelStrataNativeGrPost],
		driver.Dim3{X: (values + strataGRThreads - 1) / strataGRThreads, Y: uint32(tokens), Z: 1},
		driver.Dim3{X: strataGRThreads, Y: 1, Z: 1}, &residual, &blockOut, &inject, &output, &embedding, &streams, &scale)
}
