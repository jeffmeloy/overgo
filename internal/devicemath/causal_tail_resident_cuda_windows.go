//go:build windows

package devicemath

import (
	"fmt"
	"math"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// FrozenCausalTailPlanSizes lists concurrent resident/session allocations.
func FrozenCausalTailPlanSizes(seq, vocab, hidden int) []uint64 {
	bytes := func(elements int) uint64 { return uint64(elements) * f32Bytes }
	return []uint64{
		bytes(vocab * hidden),                       // tied lexical table
		bytes(hidden), bytes(hidden), bytes(hidden), // final norm W/G/M
		bytes(seq),               // token ids
		bytes(seq * hidden),      // gathered input
		bytes(seq * hidden),      // final normalized
		bytes((seq - 1) * vocab), // logits / dLogits
		bytes(seq - 1),           // row losses
		bytes(seq * hidden),      // normalized gradient
		bytes(seq * hidden),      // hidden gradient
	}
}

// StackForwardBackwardResidentFrozenCausal runs frozen lexical gather/head/CE
// and the trainable stack/final norm in one CUDA session.
func StackForwardBackwardResidentFrozenCausal(
	worker *device.Worker,
	tokens []int,
	dW, dG driver.DevicePtr,
	offsets []LayerTensorOffsets,
	normWeights []LayerNormPair,
	invFreq []float32,
	dHead, dFinalNorm, dFinalNormGrad driver.DevicePtr,
	seq, vocab, hidden, heads, kvHeads, headDim, intermediate int,
	rmsEps float64,
) (float64, []LayerNormPair, error) {
	if seq < 2 || len(tokens) != seq || vocab <= 0 {
		return 0, nil, fmt.Errorf("resident causal tail: tokens=%d seq=%d vocab=%d", len(tokens), seq, vocab)
	}
	ids := make([]uint32, seq)
	for index, token := range tokens {
		if token < 0 || token >= vocab {
			return 0, nil, fmt.Errorf("resident causal tail: token %d at %d outside vocab %d", token, index, vocab)
		}
		ids[index] = uint32(token)
	}
	dims := newLayerDims(seq, hidden, heads, kvHeads, headDim, intermediate, rmsEps)
	if len(offsets) == 0 || len(offsets) != len(normWeights) || heads%kvHeads != 0 || len(invFreq) != headDim/2 {
		return 0, nil, fmt.Errorf("resident causal tail: invalid stack geometry")
	}
	normGrads := make([]LayerNormPair, len(offsets))
	for index := range normGrads {
		normGrads[index] = LayerNormPair{InLN: make([]float32, hidden), PostLN: make([]float32, hidden)}
	}
	lossRows := make([]float32, seq-1)
	err := withCUDABLAS(worker, func(session *cudaBLAS) error {
		ops, err := newLayerOps(session, stackFnNames, dims, invFreq)
		if err != nil {
			return err
		}
		weights, gradients, err := residentLayerPointers(session, dW, dG, offsets, normWeights, hidden)
		if err != nil {
			return err
		}
		idPtr, err := session.alloc(len(ids))
		if err != nil {
			return err
		}
		if err := session.state.Driver.MemcpyHtoD(idPtr, driver.Bytes(ids)); err != nil {
			return err
		}
		embeds, err := session.alloc(seq * hidden)
		if err != nil {
			return err
		}
		gather, err := session.function("embedding_gather_rows_f32")
		if err != nil {
			return err
		}
		rowsU, widthU := uint32(seq), uint32(hidden)
		if err := session.launch1D(gather, rowsU*widthU,
			unsafe.Pointer(&dHead), unsafe.Pointer(&idPtr), unsafe.Pointer(&embeds),
			unsafe.Pointer(&rowsU), unsafe.Pointer(&widthU)); err != nil {
			return err
		}
		lossPtr := driver.DevicePtr(0)
		_, err = ops.runStackDevice(embeds, weights, gradients, func(final driver.DevicePtr) (driver.DevicePtr, error) {
			var dNormed driver.DevicePtr
			var tailErr error
			lossPtr, dNormed, tailErr = frozenCausalTail(session, final, idPtr, dHead, dFinalNorm, dFinalNormGrad, seq, vocab, hidden, rmsEps)
			if tailErr != nil {
				return 0, tailErr
			}
			return residentCausalHiddenGradient(session, dNormed, final, dFinalNorm, dFinalNormGrad, seq, hidden, rmsEps)
		})
		if err != nil {
			return err
		}
		downloads := make([]cudaDownload, 0, len(offsets)*2+1)
		for index := range offsets {
			downloads = append(downloads,
				cudaDownload{normGrads[index].InLN, gradients[index].dInLN},
				cudaDownload{normGrads[index].PostLN, gradients[index].dPostLN})
		}
		downloads = append(downloads, cudaDownload{lossRows, lossPtr})
		return session.finish(downloads...)
	})
	if err != nil {
		return 0, nil, err
	}
	var loss float64
	for _, value := range lossRows {
		loss += float64(value)
	}
	return loss, normGrads, nil
}

func residentLayerPointers(session *cudaBLAS, dW, dG driver.DevicePtr, offsets []LayerTensorOffsets, norms []LayerNormPair, hidden int) ([]layerWeightPtrs, []layerGradPtrs, error) {
	weights := make([]layerWeightPtrs, len(offsets))
	gradients := make([]layerGradPtrs, len(offsets))
	for index, offset := range offsets {
		if len(norms[index].InLN) != hidden || len(norms[index].PostLN) != hidden {
			return nil, nil, fmt.Errorf("resident layer %d norm width mismatch", index)
		}
		if err := session.state.Driver.MemcpyHtoD(ptrAt(dW, offset.InLN), driver.Bytes(norms[index].InLN)); err != nil {
			return nil, nil, err
		}
		if err := session.state.Driver.MemcpyHtoD(ptrAt(dW, offset.PostLN), driver.Bytes(norms[index].PostLN)); err != nil {
			return nil, nil, err
		}
		weights[index] = layerWeightPtrs{
			inLN: ptrAt(dW, offset.InLN), postLN: ptrAt(dW, offset.PostLN),
			q: ptrAt(dW, offset.Q), k: ptrAt(dW, offset.K), v: ptrAt(dW, offset.V), o: ptrAt(dW, offset.O),
			gate: ptrAt(dW, offset.Gate), up: ptrAt(dW, offset.Up), down: ptrAt(dW, offset.Down),
		}
		gradients[index] = layerGradPtrs{
			dInLN: ptrAt(dG, offset.InLN), dPostLN: ptrAt(dG, offset.PostLN),
			dQ: ptrAt(dG, offset.Q), dK: ptrAt(dG, offset.K), dV: ptrAt(dG, offset.V), dO: ptrAt(dG, offset.O),
			dGate: ptrAt(dG, offset.Gate), dUp: ptrAt(dG, offset.Up), dDown: ptrAt(dG, offset.Down),
		}
	}
	return weights, gradients, nil
}

func frozenCausalTail(session *cudaBLAS, final, ids, head, finalNorm, finalNormGrad driver.DevicePtr, seq, vocab, hidden int, eps float64) (driver.DevicePtr, driver.DevicePtr, error) {
	normed, err := session.alloc(seq * hidden)
	if err != nil {
		return 0, 0, err
	}
	normKernel, err := session.function("weighted_rms_norm_f32")
	if err != nil {
		return 0, 0, err
	}
	widthU, rowsU, epsF := uint32(hidden), uint32(seq), float32(eps)
	if err := session.launch1D(normKernel, rowsU*deviceBlockThreads,
		unsafe.Pointer(&final), unsafe.Pointer(&finalNorm), unsafe.Pointer(&normed),
		unsafe.Pointer(&widthU), unsafe.Pointer(&rowsU), unsafe.Pointer(&epsF)); err != nil {
		return 0, 0, err
	}
	predictions := seq - 1
	logits, err := session.alloc(predictions * vocab)
	if err != nil {
		return 0, 0, err
	}
	if err := session.gemm(false, true, predictions, hidden, vocab, normed, head, logits); err != nil {
		return 0, 0, err
	}
	losses, err := session.alloc(predictions)
	if err != nil {
		return 0, 0, err
	}
	ce, err := session.function("softmax_ce_grad_rows_f32")
	if err != nil {
		return 0, 0, err
	}
	targets := ids + driver.DevicePtr(4)
	predU, vocabU, scale := uint32(predictions), uint32(vocab), float32(1)/float32(predictions)
	if err := session.launch1D(ce, predU,
		unsafe.Pointer(&logits), unsafe.Pointer(&targets), unsafe.Pointer(&losses),
		unsafe.Pointer(&predU), unsafe.Pointer(&vocabU), unsafe.Pointer(&scale)); err != nil {
		return 0, 0, err
	}
	dNormed, err := session.alloc(seq * hidden)
	if err != nil {
		return 0, 0, err
	}
	if err := session.state.Driver.MemsetD32Async(dNormed, 0, uint64(seq*hidden), session.state.Stream); err != nil {
		return 0, 0, err
	}
	if err := session.gemm(false, false, predictions, vocab, hidden, logits, head, dNormed); err != nil {
		return 0, 0, err
	}
	if err := session.state.Driver.MemsetD32Async(finalNormGrad, 0, uint64(hidden), session.state.Stream); err != nil {
		return 0, 0, err
	}
	return losses, dNormed, nil
}

func residentCausalHiddenGradient(session *cudaBLAS, dNormed, final, finalNorm, finalNormGrad driver.DevicePtr, seq, hidden int, eps float64) (driver.DevicePtr, error) {
	if uint64(seq) > math.MaxUint32 || uint64(hidden) > math.MaxUint32 {
		return 0, fmt.Errorf("resident causal tail: dimensions exceed ABI")
	}
	dHidden, err := session.alloc(seq * hidden)
	if err != nil {
		return 0, err
	}
	backward, err := session.function("rms_norm_backward_f32")
	if err != nil {
		return 0, err
	}
	rowsU, widthU, epsF := uint32(seq), uint32(hidden), float32(eps)
	if err := session.launch1D(backward, rowsU,
		unsafe.Pointer(&dNormed), unsafe.Pointer(&final), unsafe.Pointer(&finalNorm),
		unsafe.Pointer(&dHidden), unsafe.Pointer(&finalNormGrad),
		unsafe.Pointer(&rowsU), unsafe.Pointer(&widthU), unsafe.Pointer(&epsF)); err != nil {
		return 0, err
	}
	return dHidden, nil
}
