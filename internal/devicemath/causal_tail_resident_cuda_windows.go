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

// CausalLexical names the resident lexical tensors of a causal training
// session: the token embedding the stack reads, the output head and final
// norm the tail reads, and the gradients the step writes. A tied head is the
// embedding's own buffer. A frozen table leaves EmbeddingGrad and HeadGrad
// zero; a trained table names both, and they are one buffer exactly when the
// head is tied.
type CausalLexical struct {
	Embedding, Head, FinalNorm             driver.DevicePtr
	EmbeddingGrad, HeadGrad, FinalNormGrad driver.DevicePtr
}

func (l CausalLexical) frozen() bool { return l.HeadGrad == 0 }

func (l CausalLexical) valid() bool {
	if l.Embedding == 0 || l.Head == 0 || l.FinalNorm == 0 || l.FinalNormGrad == 0 {
		return false
	}
	if l.frozen() {
		return l.EmbeddingGrad == 0
	}
	return l.EmbeddingGrad != 0 && (l.Embedding == l.Head) == (l.EmbeddingGrad == l.HeadGrad)
}

// CausalTrainingSession retains the compiled modules and buffers of a dense
// causal training step that runs wholly on the device: the token lookup, the
// layer stack forward and backward, and the lexical tail.
type CausalTrainingSession struct {
	worker             *device.Worker
	cuda               *cudaBLAS
	ops                *layerOps
	dW, dG             driver.DevicePtr
	lexical            CausalLexical
	idPtr, embeds      driver.DevicePtr
	offsets            []LayerTensorOffsets
	seq, vocab, hidden int
	rmsEps             float64
	closed             bool
}

// NewCausalTrainingSession opens a session over the resident layer buffers
// dW and dG and the lexical tensors. A frozen table runs with BF16 GEMM
// operands, the protocol its adaptive leadership was matched under; a
// trained table keeps the F32 operands its host-parity bounds hold.
func NewCausalTrainingSession(
	worker *device.Worker,
	dW, dG driver.DevicePtr,
	offsets []LayerTensorOffsets,
	invFreq []float32,
	lexical CausalLexical,
	seq, vocab, hidden, heads, kvHeads, headDim, intermediate int,
	rmsEps float64,
) (*CausalTrainingSession, error) {
	if worker == nil || seq < 2 || vocab <= 0 || len(offsets) == 0 || heads%kvHeads != 0 || len(invFreq) != headDim/2 || !lexical.valid() {
		return nil, fmt.Errorf("resident causal session: invalid authority or geometry")
	}
	result := &CausalTrainingSession{
		worker: worker, dW: dW, dG: dG, lexical: lexical,
		offsets: append([]LayerTensorOffsets(nil), offsets...), seq: seq, vocab: vocab, hidden: hidden, rmsEps: rmsEps,
	}
	err := worker.Do(context.Background(), func(state *device.State) error {
		scope := &cudaScope{state: state}
		session, err := openCUDABLAS(scope)
		if err != nil {
			return err
		}
		session.bf16Operands = lexical.frozen()
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

// Run takes one step's forward and backward over tokens. It returns the
// mean next-token loss and the layer vector gradients; the matrix, final
// norm and (for a trained table) lexical gradients stay resident.
func (s *CausalTrainingSession) Run(tokens []int, vectorWeights []LayerTrainVectors) (float64, []LayerTrainVectors, error) {
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
			unsafe.Pointer(&s.lexical.Embedding), unsafe.Pointer(&s.idPtr), unsafe.Pointer(&s.embeds),
			unsafe.Pointer(&rowsU), unsafe.Pointer(&widthU)); err != nil {
			return err
		}
		lossPtr := driver.DevicePtr(0)
		embeddingInput, err := s.ops.runStackDevice(s.embeds, weights, gradients, func(final driver.DevicePtr) (driver.DevicePtr, error) {
			var dNormed driver.DevicePtr
			var tailErr error
			lossPtr, dNormed, tailErr = causalTail(s.cuda, final, s.idPtr, s.lexical, s.seq, s.vocab, s.hidden, s.rmsEps)
			if tailErr != nil {
				return 0, tailErr
			}
			return residentCausalHiddenGradient(s.cuda, dNormed, final, s.lexical.FinalNorm, s.lexical.FinalNormGrad, s.seq, s.hidden, s.rmsEps)
		})
		if err != nil {
			return err
		}
		if !s.lexical.frozen() {
			if err := s.scatterEmbeddingGradient(embeddingInput); err != nil {
				return err
			}
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

// scatterEmbeddingGradient adds each position's embedding-input gradient
// row onto its token's row of the embedding gradient. The tied head's
// gradient already holds the tail's contribution; an untied embedding's
// gradient starts from zero. Repeated tokens accumulate by atomic adds, as
// the norm-weight gradients of the stack do.
func (s *CausalTrainingSession) scatterEmbeddingGradient(embeddingInput driver.DevicePtr) error {
	if s.lexical.EmbeddingGrad != s.lexical.HeadGrad {
		if err := s.cuda.state.Driver.MemsetD32Async(s.lexical.EmbeddingGrad, 0, uint64(s.vocab*s.hidden), s.cuda.state.Stream); err != nil {
			return err
		}
	}
	scatter, err := s.cuda.function("indexed_row_scatter_add_f32")
	if err != nil {
		return err
	}
	rowsU, widthU := uint32(s.seq), uint32(s.hidden)
	return s.cuda.launch1D(scatter, rowsU*widthU,
		unsafe.Pointer(&embeddingInput), unsafe.Pointer(&s.idPtr), unsafe.Pointer(&s.lexical.EmbeddingGrad),
		unsafe.Pointer(&rowsU), unsafe.Pointer(&widthU))
}

// Close releases the session's modules and buffers.
func (s *CausalTrainingSession) Close() error {
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
		biasOK := !offset.AttnBias || len(vectors[index].QBias) == dims.width && len(vectors[index].KBias) == dims.kvWidth && len(vectors[index].VBias) == dims.kvWidth
		if len(vectors[index].InLN) != dims.hidden || len(vectors[index].PostLN) != dims.hidden || !biasOK {
			return nil, nil, fmt.Errorf("resident layer %d vector width mismatch", index)
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

type causalTailBuffers struct {
	seq, vocab, hidden                       int
	normed, logits, losses, dNormed, dHidden driver.DevicePtr
}

func (s *cudaBLAS) causalTailBuffers(seq, vocab, hidden int) (*causalTailBuffers, error) {
	if s.causalTail != nil {
		b := s.causalTail
		if b.seq != seq || b.vocab != vocab || b.hidden != hidden {
			return nil, fmt.Errorf("resident causal tail: retained geometry changed")
		}
		return b, nil
	}
	b := &causalTailBuffers{seq: seq, vocab: vocab, hidden: hidden}
	var err error
	for _, spec := range []struct {
		pointer *driver.DevicePtr
		count   int
	}{{&b.normed, seq * hidden}, {&b.logits, (seq - 1) * vocab}, {&b.losses, seq - 1}, {&b.dNormed, seq * hidden}, {&b.dHidden, seq * hidden}} {
		if *spec.pointer, err = s.alloc(spec.count); err != nil {
			return nil, err
		}
	}
	s.causalTail = b
	return b, nil
}

// causalTail runs the final norm, the output head and the next-token
// cross-entropy on the final hidden states, and their backward to the
// normed states. A trained table's head gradient is written here; the
// final-norm gradient is cleared for the hidden-state backward to fill.
func causalTail(session *cudaBLAS, final, ids driver.DevicePtr, lexical CausalLexical, seq, vocab, hidden int, eps float64) (driver.DevicePtr, driver.DevicePtr, error) {
	buffers, err := session.causalTailBuffers(seq, vocab, hidden)
	if err != nil {
		return 0, 0, err
	}
	head, finalNorm, finalNormGrad := lexical.Head, lexical.FinalNorm, lexical.FinalNormGrad
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
	if !lexical.frozen() {
		if err := session.gemm(true, false, vocab, predictions, hidden, logits, normed, lexical.HeadGrad); err != nil {
			return 0, 0, err
		}
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
	if session.causalTail == nil || session.causalTail.seq != seq || session.causalTail.hidden != hidden {
		return 0, fmt.Errorf("resident causal tail: hidden buffer unavailable")
	}
	dHidden := session.causalTail.dHidden
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
