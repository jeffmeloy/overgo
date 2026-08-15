//go:build windows

package devicemath

import (
	"context"
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

// FrozenCausalTrainingSession retains compiled modules and training buffers.
type FrozenCausalTrainingSession struct {
	worker                            *device.Worker
	cuda                              *cudaBLAS
	ops                               *layerOps
	dW, dG                            driver.DevicePtr
	dHead, dFinalNorm, dFinalNormGrad driver.DevicePtr
	idPtr, embeds                     driver.DevicePtr
	offsets                           []LayerTensorOffsets
	seq, vocab, hidden                int
	rmsEps                            float64
	closed                            bool
}

func NewFrozenCausalTrainingSession(
	worker *device.Worker,
	dW, dG driver.DevicePtr,
	offsets []LayerTensorOffsets,
	invFreq []float32,
	dHead, dFinalNorm, dFinalNormGrad driver.DevicePtr,
	seq, vocab, hidden, heads, kvHeads, headDim, intermediate int,
	rmsEps float64,
) (*FrozenCausalTrainingSession, error) {
	if worker == nil || seq < 2 || vocab <= 0 || len(offsets) == 0 || heads%kvHeads != 0 || len(invFreq) != headDim/2 {
		return nil, fmt.Errorf("resident causal session: invalid authority or geometry")
	}
	result := &FrozenCausalTrainingSession{
		worker: worker, dW: dW, dG: dG, dHead: dHead, dFinalNorm: dFinalNorm, dFinalNormGrad: dFinalNormGrad,
		offsets: append([]LayerTensorOffsets(nil), offsets...), seq: seq, vocab: vocab, hidden: hidden, rmsEps: rmsEps,
	}
	err := worker.Do(context.Background(), func(state *device.State) error {
		scope := &cudaScope{state: state}
		session, err := openCUDABLAS(scope)
		if err != nil {
			return err
		}
		session.bf16Operands = true
		result.cuda = session
		dims := newLayerDims(seq, hidden, heads, kvHeads, headDim, intermediate, rmsEps)
		if result.ops, err = newLayerOps(session, stackFnNames, dims, invFreq); err != nil {
			return err
		}
		if result.idPtr, err = session.alloc(seq); err != nil {
			return err
		}
		result.embeds, err = session.alloc(seq * hidden)
		return err
	})
	if err != nil {
		if result.cuda != nil {
			_ = worker.Do(context.Background(), func(_ *device.State) error {
				return result.cuda.close()
			})
		}
		return nil, err
	}
	return result, nil
}

func (s *FrozenCausalTrainingSession) Run(tokens []int, vectorWeights []LayerTrainVectors) (float64, []LayerTrainVectors, error) {
	if s == nil || s.closed || len(tokens) != s.seq || len(vectorWeights) != len(s.offsets) {
		return 0, nil, fmt.Errorf("resident causal session: invalid run input")
	}
	ids := make([]uint32, s.seq)
	for index, token := range tokens {
		if token < 0 || token >= s.vocab {
			return 0, nil, fmt.Errorf("resident causal tail: token %d at %d outside vocab %d", token, index, s.vocab)
		}
		ids[index] = uint32(token)
	}
	dims := s.ops.d
	vectorGrads := make([]LayerTrainVectors, len(s.offsets))
	for index := range vectorGrads {
		vectorGrads[index] = LayerTrainVectors{InLN: make([]float32, s.hidden), PostLN: make([]float32, s.hidden)}
		if s.offsets[index].AttnBias {
			vectorGrads[index].QBias = make([]float32, dims.width)
			vectorGrads[index].KBias = make([]float32, dims.kvWidth)
			vectorGrads[index].VBias = make([]float32, dims.kvWidth)
		}
	}
	lossRows := make([]float32, s.seq-1)
	err := s.worker.Do(context.Background(), func(_ *device.State) error {
		weights, gradients, err := residentLayerPointers(s.cuda, s.dW, s.dG, s.offsets, vectorWeights, dims)
		if err != nil {
			return err
		}
		if err := s.cuda.state.Driver.MemcpyHtoD(s.idPtr, driver.Bytes(ids)); err != nil {
			return err
		}
		gather, err := s.cuda.function("embedding_gather_rows_f32")
		if err != nil {
			return err
		}
		rowsU, widthU := uint32(s.seq), uint32(s.hidden)
		if err := s.cuda.launch1D(gather, rowsU*widthU,
			unsafe.Pointer(&s.dHead), unsafe.Pointer(&s.idPtr), unsafe.Pointer(&s.embeds),
			unsafe.Pointer(&rowsU), unsafe.Pointer(&widthU)); err != nil {
			return err
		}
		lossPtr := driver.DevicePtr(0)
		_, err = s.ops.runStackDevice(s.embeds, weights, gradients, func(final driver.DevicePtr) (driver.DevicePtr, error) {
			var dNormed driver.DevicePtr
			var tailErr error
			lossPtr, dNormed, tailErr = frozenCausalTail(s.cuda, final, s.idPtr, s.dHead, s.dFinalNorm, s.dFinalNormGrad, s.seq, s.vocab, s.hidden, s.rmsEps)
			if tailErr != nil {
				return 0, tailErr
			}
			return residentCausalHiddenGradient(s.cuda, dNormed, final, s.dFinalNorm, s.dFinalNormGrad, s.seq, s.hidden, s.rmsEps)
		})
		if err != nil {
			return err
		}
		downloads := make([]cudaDownload, 0, len(s.offsets)*2+1)
		for index := range s.offsets {
			downloads = append(downloads,
				cudaDownload{vectorGrads[index].InLN, gradients[index].dInLN},
				cudaDownload{vectorGrads[index].PostLN, gradients[index].dPostLN})
			if s.offsets[index].AttnBias {
				downloads = append(downloads,
					cudaDownload{vectorGrads[index].QBias, gradients[index].dQBias},
					cudaDownload{vectorGrads[index].KBias, gradients[index].dKBias},
					cudaDownload{vectorGrads[index].VBias, gradients[index].dVBias})
			}
		}
		downloads = append(downloads, cudaDownload{lossRows, lossPtr})
		return s.cuda.finish(downloads...)
	})
	if err != nil {
		return 0, nil, err
	}
	var loss float64
	for _, value := range lossRows {
		loss += float64(value)
	}
	return loss, vectorGrads, nil
}

func (s *FrozenCausalTrainingSession) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	return s.worker.Do(context.Background(), func(_ *device.State) error { return s.cuda.close() })
}

func residentLayerPointers(session *cudaBLAS, dW, dG driver.DevicePtr, offsets []LayerTensorOffsets, vectors []LayerTrainVectors, dims layerDims) ([]layerWeightPtrs, []layerGradPtrs, error) {
	weights := make([]layerWeightPtrs, len(offsets))
	gradients := make([]layerGradPtrs, len(offsets))
	for index, offset := range offsets {
		if len(vectors[index].InLN) != dims.hidden || len(vectors[index].PostLN) != dims.hidden {
			return nil, nil, fmt.Errorf("resident layer %d norm width mismatch", index)
		}
		if err := session.state.Driver.MemcpyHtoD(ptrAt(dW, offset.InLN), driver.Bytes(vectors[index].InLN)); err != nil {
			return nil, nil, err
		}
		if err := session.state.Driver.MemcpyHtoD(ptrAt(dW, offset.PostLN), driver.Bytes(vectors[index].PostLN)); err != nil {
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
		if offset.AttnBias {
			for _, spec := range []struct {
				weightOffset, gradientOffset int
				values                       []float32
				weight, gradient             *driver.DevicePtr
			}{{offset.QBias, offset.QBias, vectors[index].QBias, &weights[index].qBias, &gradients[index].dQBias},
				{offset.KBias, offset.KBias, vectors[index].KBias, &weights[index].kBias, &gradients[index].dKBias},
				{offset.VBias, offset.VBias, vectors[index].VBias, &weights[index].vBias, &gradients[index].dVBias}} {
				if err := session.state.Driver.MemcpyHtoD(ptrAt(dW, spec.weightOffset), driver.Bytes(spec.values)); err != nil {
					return nil, nil, err
				}
				*spec.weight = ptrAt(dW, spec.weightOffset)
				*spec.gradient = ptrAt(dG, spec.gradientOffset)
			}
		}
	}
	return weights, gradients, nil
}

type frozenCausalTailBuffers struct {
	seq, vocab, hidden                       int
	normed, logits, losses, dNormed, dHidden driver.DevicePtr
}

func (session *cudaBLAS) frozenTailBuffers(seq, vocab, hidden int) (*frozenCausalTailBuffers, error) {
	if session.frozenTail != nil {
		b := session.frozenTail
		if b.seq != seq || b.vocab != vocab || b.hidden != hidden {
			return nil, fmt.Errorf("resident causal tail: retained geometry changed")
		}
		return b, nil
	}
	b := &frozenCausalTailBuffers{seq: seq, vocab: vocab, hidden: hidden}
	var err error
	for _, spec := range []struct {
		pointer *driver.DevicePtr
		count   int
	}{{&b.normed, seq * hidden}, {&b.logits, (seq - 1) * vocab}, {&b.losses, seq - 1}, {&b.dNormed, seq * hidden}, {&b.dHidden, seq * hidden}} {
		if *spec.pointer, err = session.alloc(spec.count); err != nil {
			return nil, err
		}
	}
	session.frozenTail = b
	return b, nil
}

func frozenCausalTail(session *cudaBLAS, final, ids, head, finalNorm, finalNormGrad driver.DevicePtr, seq, vocab, hidden int, eps float64) (driver.DevicePtr, driver.DevicePtr, error) {
	buffers, err := session.frozenTailBuffers(seq, vocab, hidden)
	if err != nil {
		return 0, 0, err
	}
	normed := buffers.normed
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
	logits := buffers.logits
	if err := session.gemm(false, true, predictions, hidden, vocab, normed, head, logits); err != nil {
		return 0, 0, err
	}
	losses := buffers.losses
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
	dNormed := buffers.dNormed
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
	if session.frozenTail == nil || session.frozenTail.seq != seq || session.frozenTail.hidden != hidden {
		return 0, fmt.Errorf("resident causal tail: hidden buffer unavailable")
	}
	dHidden := session.frozenTail.dHidden
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
