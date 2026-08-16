package thoughtbank

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// Full ThoughtBankLM forward: embed, run the hyper-connection block stack while
// reading the thought bank as fast weights, collapse the streams, write one new
// thought, and project to logits. Ported from adaptive's
// go/extmodel/fast_weight_bank_lm.go + hyper_connection_block.go at b8fef3cc2.

// FastWeightBankLMWeights is the whole model.
type FastWeightBankLMWeights struct {
	VocabSize, DModel, NLayers, NHC, MemDim, MaxMem int

	Embed []float32 // [vocab, d_model]; lm_head is TIED to this when LMHead nil
	// LMHead is nil when tied. A non-nil value is used as an untied head, so the
	// tie is a property of the weights rather than a branch in the forward.
	LMHead  []float32
	AOutNet []float32 // [n_hc, d_model]
	NormOut []float32 // [d_model]
	Blocks  []*HyperConnectionBlockWeights
	Write   *thoughtWriteWeights
	NormEps float64
}

// thoughtWriteWeights is the thought-bank write head.
type thoughtWriteWeights struct {
	WriteCtxQ     []float32 // [1, d_model] attention-pool scorer
	WriteGate     []float32 // [mem_dim, d_model] per-dimension content gate
	WriteGateBias []float32 // [mem_dim]
	ThoughtHead   []float32 // [mem_dim, d_model]
	NormWrite     []float32 // [mem_dim]
	WriteDecision []float32 // [1, d_model] scalar write/skip logit
	WriteDecBias  []float32 // [1]
}

// fastWeightBankLMOutput carries what a step needs: logits, the bank to carry
// into the next turn, and the averaged balance auxiliary.
type fastWeightBankLMOutput struct {
	Logits  []float32 // [seq, vocab]
	MemBank []float32 // [slots', mem_dim]
	Slots   int
	// HText is the collapsed, normalised hidden the LM head reads -- [seq,
	// d_model]. It is the model's text_hidden organ output.
	HText       []float32
	BalanceLoss float64
}

// FastWeightBankLMForward runs one sequence.
//
// initMem may be empty; the reference then seeds the bank with random slots,
// which this deliberately does not do (a random seed makes the forward a
// non-function of its inputs). Callers wanting the seeded behaviour pass the
// seed slots in explicitly.
func FastWeightBankLMForward(ids []int32, initMem []float32, slots int, w *FastWeightBankLMWeights) (*fastWeightBankLMOutput, error) {
	return fastWeightBankLMForward(ids, initMem, slots, w, nil)
}

func fastWeightBankLMForward(ids []int32, initMem []float32, slots int, w *FastWeightBankLMWeights, capture func(int, []float32)) (*fastWeightBankLMOutput, error) {
	seq := len(ids)
	if seq == 0 {
		return nil, fmt.Errorf("model: empty input")
	}
	d, n := w.DModel, w.NHC
	flat := n * d
	if len(w.Blocks) != w.NLayers {
		return nil, fmt.Errorf("model: %d blocks, want %d", len(w.Blocks), w.NLayers)
	}

	// Embed, then broadcast the token into every hyper-connection stream.
	x := make([]float32, seq*flat)
	for t, id := range ids {
		if int(id) < 0 || int(id) >= w.VocabSize {
			return nil, fmt.Errorf("model: token %d out of range [0,%d)", id, w.VocabSize)
		}
		row := w.Embed[int(id)*d : (int(id)+1)*d]
		for s := 0; s < n; s++ {
			copy(x[t*flat+s*d:t*flat+(s+1)*d], row)
		}
	}

	bank := initMem
	var totalBal float64
	for i, blk := range w.Blocks {
		var attentionInput func([]float32)
		if capture != nil {
			attentionInput = func(input []float32) { capture(i, input) }
		}
		out, bal, err := hyperConnectionBlockForward(x, seq, slots, bank, blk, attentionInput)
		if err != nil {
			return nil, fmt.Errorf("model: block %d: %w", i, err)
		}
		x = out
		totalBal += bal
	}
	totalBal /= float64(w.NLayers)

	// Collapse the streams with the model's own learned weights, then norm.
	hText := collapseStreams(x, seq, n, d, w.AOutNet)
	hText = rmsNormNew(hText, w.NormOut, seq, d, w.NormEps)

	// Write one new thought and FIFO-evict past capacity.
	newBank, newSlots, err := writeThoughtSlot(hText, bank, seq, slots, w)
	if err != nil {
		return nil, err
	}

	head := w.LMHead
	if head == nil {
		head = w.Embed // weight tying
	}
	logits := make([]float32, seq*w.VocabSize)
	vocab := w.VocabSize
	hostmath.ParallelRangeF64(seq*vocab, d, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			t, v := i/vocab, i%vocab
			logits[i] = float32(dot(head[v*d:(v+1)*d], hText[t*d:(t+1)*d]))
		}
	})

	return &fastWeightBankLMOutput{
		Logits: logits, MemBank: newBank, Slots: newSlots,
		HText: hText, BalanceLoss: totalBal,
	}, nil
}

// writeThoughtSlot produces one gated thought vector from an attention pool over
// the sequence and appends it to the bank.
//
// The gate is a PRODUCT of two learned quantities: a scalar alpha deciding
// whether to write at all, and a per-dimension p deciding which features of the
// thought survive.
func writeThoughtSlot(hText, bank []float32, seq, slots int, w *FastWeightBankLMWeights) ([]float32, int, error) {
	d, md := w.DModel, w.MemDim
	wr := w.Write
	if wr == nil {
		return bank, slots, nil
	}

	// Attention pool over positions.
	scores := make([]float64, seq)
	maxS := math.Inf(-1)
	for t := 0; t < seq; t++ {
		scores[t] = dot(wr.WriteCtxQ[:d], hText[t*d:(t+1)*d])
		if scores[t] > maxS {
			maxS = scores[t]
		}
	}
	var sum float64
	for t := 0; t < seq; t++ {
		scores[t] = math.Exp(scores[t] - maxS)
		sum += scores[t]
	}
	ctx := make([]float32, d)
	for t := 0; t < seq; t++ {
		a := scores[t] / sum
		for j := 0; j < d; j++ {
			ctx[j] += float32(a * float64(hText[t*d+j]))
		}
	}

	thought := make([]float32, md)
	for i := 0; i < md; i++ {
		thought[i] = float32(dot(wr.ThoughtHead[i*d:(i+1)*d], ctx))
	}
	thought = rmsNormNew(thought, wr.NormWrite, 1, md, w.NormEps)

	alpha := sigmoid(dot(wr.WriteDecision[:d], ctx) + float64(wr.WriteDecBias[0]))

	slot := make([]float32, md)
	for i := 0; i < md; i++ {
		p := sigmoid(dot(wr.WriteGate[i*d:(i+1)*d], ctx) + float64(wr.WriteGateBias[i]))
		slot[i] = float32(alpha * p * float64(thought[i]))
	}

	appended := make([]float32, 0, (slots+1)*md)
	appended = append(appended, bank...)
	appended = append(appended, slot...)
	total := slots + 1
	if total > w.MaxMem {
		drop := total - w.MaxMem
		appended = appended[drop*md:]
		total = w.MaxMem
	}
	return appended, total, nil
}

// HyperConnectionBlockWeights is one DualModalBlock.
type HyperConnectionBlockWeights struct {
	NHC, DModel int
	Attn        *CompressedHybridAttnWeights
	MoE         *sharedRoutedMoEWeights
	Bank        *FastWeightBankWeights

	MHCAttn *HyperConnectionWeights
	MHCMoE  *HyperConnectionWeights

	NormAttn  []float32 // [d_model] applied INSIDE the mHC attention wrapper
	NormMoE   []float32 // [d_model] applied INSIDE the mHC MoE wrapper
	ACrossNet []float32 // [n_hc, d_model] collapse for the bank read
	ReadBank  bool
	NormEps   float64
}

// HyperConnectionBlockForward runs one block over a single sequence.
//
// x holds seq x n_hc x d_model values; bank holds slots x mem_dim (may be
// empty). Returns the updated residual streams and the MoE balance auxiliary.
func HyperConnectionBlockForward(x []float32, seq, slots int, bank []float32, w *HyperConnectionBlockWeights) ([]float32, float64, error) {
	return hyperConnectionBlockForward(x, seq, slots, bank, w, nil)
}

func hyperConnectionBlockForward(x []float32, seq, slots int, bank []float32, w *HyperConnectionBlockWeights, captureAttentionInput func([]float32)) ([]float32, float64, error) {
	n, d := w.NHC, w.DModel
	flat := n * d
	if len(x) != seq*flat {
		return nil, 0, fmt.Errorf("block: input has %d values, want %d", len(x), seq*flat)
	}

	// 1. Attention, wrapped by mHC. The pre-norm lives INSIDE the wrapper: mHC
	// collapses the streams first, then the sub-layer normalises what it got.
	var attnErr error
	out, err := HyperConnectionForward(x, seq, w.MHCAttn, func(in []float32, rows, dd int) []float32 {
		normed := rmsNormNew(in, w.NormAttn, rows, dd, w.NormEps)
		if captureAttentionInput != nil {
			captureAttentionInput(normed)
		}
		y, e := CompressedHybridAttentionForward(normed, rows, w.Attn)
		if e != nil {
			attnErr = e
			return make([]float32, rows*dd)
		}
		return y
	})
	if err != nil {
		return nil, 0, err
	}
	if attnErr != nil {
		return nil, 0, attnErr
	}

	// 2. Fast-weight bank read, applied as a delta broadcast across streams. The
	// collapse here uses its OWN learned weights (A_cross_net), not the model's
	// output collapse.
	if w.ReadBank && slots > 0 && len(bank) > 0 {
		h0 := collapseStreams(out, seq, n, d, w.ACrossNet)
		h1, e := FastWeightBankRead(h0, bank, seq, slots, w.Bank)
		if e != nil {
			return nil, 0, e
		}
		for t := 0; t < seq; t++ {
			for s := 0; s < n; s++ {
				for j := 0; j < d; j++ {
					out[t*flat+s*d+j] += h1[t*d+j] - h0[t*d+j]
				}
			}
		}
	}

	// 3. MoE, wrapped by mHC.
	var bal float64
	var moeErr error
	out, err = HyperConnectionForward(out, seq, w.MHCMoE, func(in []float32, rows, dd int) []float32 {
		normed := rmsNormNew(in, w.NormMoE, rows, dd, w.NormEps)
		y, b, e := SharedRoutedMoEForward(normed, rows, w.MoE)
		if e != nil {
			moeErr = e
			return make([]float32, rows*dd)
		}
		bal = b
		return y
	})
	if err != nil {
		return nil, 0, err
	}
	if moeErr != nil {
		return nil, 0, moeErr
	}
	return out, bal, nil
}

// collapseStreams reduces [seq, n_hc, d] to [seq, d] by a softmax over streams
// whose logits come from the streams' MEAN, not from each stream separately.
func collapseStreams(x []float32, seq, n, d int, proj []float32) []float32 {
	flat := n * d
	out := make([]float32, seq*d)
	mean := make([]float32, d)
	logits := make([]float64, n)
	for t := 0; t < seq; t++ {
		for j := 0; j < d; j++ {
			var acc float64
			for s := 0; s < n; s++ {
				acc += float64(x[t*flat+s*d+j])
			}
			mean[j] = float32(acc / float64(n))
		}
		maxL := math.Inf(-1)
		for s := 0; s < n; s++ {
			logits[s] = dot(proj[s*d:(s+1)*d], mean)
			if logits[s] > maxL {
				maxL = logits[s]
			}
		}
		var sum float64
		for s := 0; s < n; s++ {
			logits[s] = math.Exp(logits[s] - maxL)
			sum += logits[s]
		}
		for s := 0; s < n; s++ {
			a := logits[s] / sum
			for j := 0; j < d; j++ {
				out[t*d+j] += float32(a * float64(x[t*flat+s*d+j]))
			}
		}
	}
	return out
}
