package thoughtbank

import (
	"fmt"
	"math"
	"sort"

	"overgo/internal/hostmath"
)

// O(n) incremental single-token decode for the fast-weight-bank LM. Ported from
// adaptive's go/extmodel/fast_weight_bank_lm_decode.go at b8fef3cc2.
//
// FastWeightBankLMForward stays the golden oracle. This path reproduces it
// token-for-token while doing O(1) work per step w.r.t. sequence length (aside
// from attending over cached blocks + the window), by exploiting three facts:
//
//  1. Everything except attention is per-token: embed, the mHC wrappers,
//     Sinkhorn, stream collapse, the bank read (bank fixed during a forward),
//     the MoE router+experts, the output norm, and the LM head all read one
//     position and write one. So each is invoked with rows=1 here.
//  2. Attention is block-causal + sliding-window: token t reads compression
//     block b only for b < t/M plus the n_win raw tokens strictly before t. A
//     completed block's compressed KV is stable once the block fills and is
//     cached.
//  3. The thought WRITE happens once per FORWARD, AFTER the stack, over the
//     whole sequence's collapsed hidden, and feeds only MemBank -- never the
//     logits of the same forward. So decode holds the bank fixed across steps
//     and replays writeThoughtSlot once over the accumulated hidden.
//
// The per-token arithmetic is reproduced in the reference's exact order, so the
// incremental logits are bit-for-bit the full forward's.

// fwbAttnLayerCache is one attention layer's decode cache.
type fwbAttnLayerCache struct {
	w *CompressedHybridAttnWeights

	// comp holds the RAW (un-normed) compressed KV of every COMPLETED block;
	// comp[b] is [d_head]. The CSA indexer scores raw comp; both schemes RMSNorm
	// a block only when it is placed into the attention K/V.
	comp [][]float32

	// curBlock accumulates the raw attention-input tokens of the block currently
	// filling; prevBlock is the just-completed block, retained for the CSA
	// overlap (block b's compression also reads block b-1's tokens).
	curBlock  []float32
	prevBlock []float32

	// Sliding-window ring over the last n_win tokens: winK is RMSNorm'd (the
	// reference norms window K but NOT window V), winV is raw. Slot for token s
	// is s % n_win.
	winK []float32
	winV []float32
}

func newFwbAttnLayerCache(w *CompressedHybridAttnWeights) *fwbAttnLayerCache {
	return &fwbAttnLayerCache{
		w:    w,
		winK: make([]float32, w.NWin*w.DHead),
		winV: make([]float32, w.NWin*w.DHead),
	}
}

// FastWeightBankLMDecodeState carries everything a step needs across tokens.
type FastWeightBankLMDecodeState struct {
	w      *FastWeightBankLMWeights
	bank   []float32 // fixed across the decode, exactly as the reference read is
	slots  int
	pos    int // tokens processed so far == absolute position of the next token
	layers []*fwbAttnLayerCache
	hText  []float32 // [pos, d_model], appended one row per step
}

// Pos reports how many tokens have been decoded so far.
func (s *FastWeightBankLMDecodeState) Pos() int { return s.pos }

// FastWeightBankLMDecodeInit prefills the caches over promptIDs and returns the
// state plus the LAST prompt position's logits.
func FastWeightBankLMDecodeInit(w *FastWeightBankLMWeights, promptIDs []int32, initMem []float32, slots int) (*FastWeightBankLMDecodeState, []float32, error) {
	if len(promptIDs) == 0 {
		return nil, nil, fmt.Errorf("decode: empty prompt")
	}
	if len(w.Blocks) != w.NLayers {
		return nil, nil, fmt.Errorf("decode: %d blocks, want %d", len(w.Blocks), w.NLayers)
	}
	if slots > 0 && len(initMem) != slots*w.MemDim {
		return nil, nil, fmt.Errorf("decode: bank has %d values, want %d", len(initMem), slots*w.MemDim)
	}
	s := &FastWeightBankLMDecodeState{
		w: w, bank: initMem, slots: slots,
		layers: make([]*fwbAttnLayerCache, len(w.Blocks)),
	}
	for i, blk := range w.Blocks {
		s.layers[i] = newFwbAttnLayerCache(blk.Attn)
	}
	attentionInputs := make([][]float32, len(w.Blocks))
	output, err := fastWeightBankLMForward(promptIDs, initMem, slots, w, func(layer int, input []float32) {
		attentionInputs[layer] = input
	})
	if err != nil {
		return nil, nil, err
	}
	for layer, cache := range s.layers {
		if err := cache.prefill(attentionInputs[layer], len(promptIDs)); err != nil {
			return nil, nil, fmt.Errorf("decode: prefill layer %d: %w", layer, err)
		}
	}
	s.pos = len(promptIDs)
	s.hText = append(s.hText, output.HText...)
	start := (len(promptIDs) - 1) * w.VocabSize
	last := append([]float32(nil), output.Logits[start:start+w.VocabSize]...)
	return s, last, nil
}

// FastWeightBankLMDecodeStep advances ONE token and returns that position's
// logits, updating the caches in place.
func FastWeightBankLMDecodeStep(s *FastWeightBankLMDecodeState, next int32) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("decode: nil state")
	}
	return s.stepToken(next)
}

// MemBank replays the single post-hoc thought write over the hidden accumulated
// so far, returning the bank the reference forward would carry out.
func (s *FastWeightBankLMDecodeState) MemBank() ([]float32, int, error) {
	if s.pos == 0 {
		return s.bank, s.slots, nil
	}
	return writeThoughtSlot(s.hText, s.bank, s.pos, s.slots, s.w)
}

// stepToken runs the whole model for one token and returns its logits.
func (s *FastWeightBankLMDecodeState) stepToken(id int32) ([]float32, error) {
	w := s.w
	d, n := w.DModel, w.NHC
	flat := n * d
	if int(id) < 0 || int(id) >= w.VocabSize {
		return nil, fmt.Errorf("decode: token %d out of range [0,%d)", id, w.VocabSize)
	}

	x := make([]float32, flat)
	row := w.Embed[int(id)*d : (int(id)+1)*d]
	for st := 0; st < n; st++ {
		copy(x[st*d:(st+1)*d], row)
	}

	for i, blk := range w.Blocks {
		out, err := s.layers[i].decodeBlock(x, s.bank, s.slots, s.pos, blk)
		if err != nil {
			return nil, fmt.Errorf("decode: block %d: %w", i, err)
		}
		x = out
	}

	hText := collapseStreamVector(x, n, d, w.AOutNet)
	hText = rmsNormVector(hText, w.NormOut, d, w.NormEps)
	s.hText = append(s.hText, hText...)

	head := w.LMHead
	if head == nil {
		head = w.Embed // weight tying
	}
	vocab := w.VocabSize
	logits := make([]float32, vocab)
	hostmath.ParallelRangeF64(vocab, d, func(lo, hi int) {
		for v := lo; v < hi; v++ {
			logits[v] = float32(dot(head[v*d:(v+1)*d], hText))
		}
	})

	s.pos++
	return logits, nil
}

// decodeBlock is HyperConnectionBlockForward for a single row, with the
// attention sub-layer served from the layer cache.
func (c *fwbAttnLayerCache) decodeBlock(x, bank []float32, slots, pos int, w *HyperConnectionBlockWeights) ([]float32, error) {
	n, d := w.NHC, w.DModel

	var attnErr error
	out, err := hyperConnectionVector(x, w.MHCAttn, func(in []float32, rows, dd int) []float32 {
		normed := rmsNormNew(in, w.NormAttn, rows, dd, w.NormEps)
		y, e := c.attnStep(normed, pos)
		if e != nil {
			attnErr = e
			return make([]float32, rows*dd)
		}
		return y
	})
	if err != nil {
		return nil, err
	}
	if attnErr != nil {
		return nil, attnErr
	}

	if w.ReadBank && slots > 0 && len(bank) > 0 {
		h0 := collapseStreamVector(out, n, d, w.ACrossNet)
		h1, e := fastWeightBankReadVector(h0, bank, slots, w.Bank)
		if e != nil {
			return nil, e
		}
		for st := 0; st < n; st++ {
			for j := 0; j < d; j++ {
				out[st*d+j] += h1[j] - h0[j]
			}
		}
	}

	var moeErr error
	out, err = hyperConnectionVector(out, w.MHCMoE, func(in []float32, rows, dd int) []float32 {
		normed := rmsNormNew(in, w.NormMoE, rows, dd, w.NormEps)
		y, _, e := SharedRoutedMoEForward(normed, rows, w.MoE)
		if e != nil {
			moeErr = e
			return make([]float32, rows*dd)
		}
		return y
	})
	if err != nil {
		return nil, err
	}
	if moeErr != nil {
		return nil, moeErr
	}
	return out, nil
}

// attnStep is CompressedHybridAttentionForward's per-token body for one token at
// absolute position pos, reading completed blocks + window from the cache.
func (c *fwbAttnLayerCache) attnStep(h []float32, pos int) ([]float32, error) {
	w := c.w
	d, dh, nh := w.DModel, w.DHead, w.NHeads
	dlq := w.DLatentQ
	m := w.M
	blockOfT := pos / m
	nDone := len(c.comp)
	if nDone != blockOfT {
		return nil, fmt.Errorf("attn cache: %d completed blocks at pos %d, want %d", nDone, pos, blockOfT)
	}

	// Query: low-rank d -> d_latent_q -> n_heads*d_head, RoPE at pos, then q_norm.
	cq := make([]float32, dlq)
	for i := 0; i < dlq; i++ {
		cq[i] = float32(dot(w.WDq[i*d:(i+1)*d], h))
	}
	q := make([]float32, nh*dh)
	for i := 0; i < nh*dh; i++ {
		q[i] = float32(dot(w.WUq[i*dlq:(i+1)*dlq], cq))
	}
	applyRotarySinglePos(q, nh, dh, pos)
	q = rmsNormNew(q, w.QNorm, nh, dh, w.NormEps)

	// Select block entries. CSA: lightning-indexer top-k over completed blocks.
	// HCA: every completed block, in index order. Both are causal-exact because
	// nDone == blockOfT is precisely the set token t may see.
	var selBlocks []int
	if w.Sparse {
		sel := w.TopK
		if sel > nDone {
			sel = nDone
		}
		scores := make([]float64, nDone)
		order := make([]int, nDone)
		for b := 0; b < nDone; b++ {
			order[b] = b
		}
		invSqrtDh := 1.0 / math.Sqrt(float64(dh))
		for ih := 0; ih < w.NIdxHeads; ih++ {
			hw := dot(w.WW[ih*d:(ih+1)*d], h)
			for b := 0; b < nDone; b++ {
				var acc float64
				cb := c.comp[b]
				for e := 0; e < dh; e++ {
					qi := dot(w.WIq[(ih*dh+e)*dlq:(ih*dh+e+1)*dlq], cq)
					acc += qi * float64(cb[e])
				}
				acc *= invSqrtDh
				var zero float64
				if acc < zero {
					acc = zero // ReLU
				}
				scores[b] += hw * acc
			}
		}
		sort.SliceStable(order, func(a, b int) bool { return scores[order[a]] > scores[order[b]] })
		selBlocks = order[:sel]
	} else {
		selBlocks = make([]int, nDone)
		for b := 0; b < nDone; b++ {
			selBlocks[b] = b
		}
	}
	sel := len(selBlocks)
	nkv := sel + w.NWin

	kAll := make([]float32, nkv*dh)
	vAll := make([]float32, nkv*dh)
	for i, b := range selBlocks {
		nb := rmsNormVector(c.comp[b], w.KVNorm, dh, w.NormEps)
		copy(kAll[i*dh:(i+1)*dh], nb)
		copy(vAll[i*dh:(i+1)*dh], nb)
	}
	// Sliding window: slot j holds token pos-n_win+j (zero if negative).
	for j := 0; j < w.NWin; j++ {
		src := pos - w.NWin + j
		dst := (sel + j) * dh
		if src < 0 {
			continue // zero-filled: contributes exp(0-max) to the denominator
		}
		ring := (src % w.NWin) * dh
		copy(kAll[dst:dst+dh], c.winK[ring:ring+dh])
		copy(vAll[dst:dst+dh], c.winV[ring:ring+dh])
	}

	// Multi-query attention with the sink softmax, reproduced in the reference's
	// exact entry order so the float32 accumulation matches bit-for-bit.
	invSqrt := 1.0 / math.Sqrt(float64(dh))
	logits := make([]float32, nh*nkv)
	weights := make([]float32, nh*nkv)
	for hd := 0; hd < nh; hd++ {
		qv := q[hd*dh : (hd+1)*dh]
		for nn := 0; nn < nkv; nn++ {
			logits[hd*nkv+nn] = float32(dot(kAll[nn*dh:(nn+1)*dh], qv) * invSqrt)
		}
	}
	attentionSinkSoftmaxVector(weights, logits, w.SinkLogits, nh, nkv)
	headOut := make([]float32, nh*dh)
	for hd := 0; hd < nh; hd++ {
		dst := headOut[hd*dh : (hd+1)*dh]
		for nn := 0; nn < nkv; nn++ {
			a := float64(weights[hd*nkv+nn])
			if a == 0 {
				continue
			}
			for e := 0; e < dh; e++ {
				dst[e] += float32(a * float64(vAll[nn*dh+e]))
			}
		}
	}
	out, err := groupedOutputVector(headOut, nh, dh, w)
	if err != nil {
		return nil, err
	}

	// Incorporate this token AFTER it has been read (window excludes t; the block
	// it belongs to is never attended by t and is compressed only once full).
	if err := c.appendToken(pos, h); err != nil {
		return nil, err
	}
	return out, nil
}

func (c *fwbAttnLayerCache) prefill(input []float32, rows int) error {
	d := c.w.DModel
	if len(input) != rows*d {
		return fmt.Errorf("attention input has %d values, want %d", len(input), rows*d)
	}
	for position := range rows {
		if err := c.appendToken(position, input[position*d:(position+1)*d]); err != nil {
			return err
		}
	}
	return nil
}

func (c *fwbAttnLayerCache) appendToken(pos int, hidden []float32) error {
	w := c.w
	d, dh := w.DModel, w.DHead
	windowKey, windowValue := make([]float32, dh), make([]float32, dh)
	for feature := range dh {
		windowKey[feature] = float32(dot(w.WWk[feature*d:(feature+1)*d], hidden))
		windowValue[feature] = float32(dot(w.WWv[feature*d:(feature+1)*d], hidden))
	}
	windowKey = rmsNormVector(windowKey, w.KVNorm, dh, w.NormEps)
	c.pushWindow(pos, windowKey, windowValue)
	return c.appendBlockToken(hidden)
}

// pushWindow stores this token's window K/V at its ring slot.
func (c *fwbAttnLayerCache) pushWindow(pos int, wk, wv []float32) {
	dh := c.w.DHead
	slot := (pos % c.w.NWin) * dh
	copy(c.winK[slot:slot+dh], wk)
	copy(c.winV[slot:slot+dh], wv)
}

// appendBlockToken appends the raw attention-input token to the filling block
// and, when it reaches M tokens, compresses it (reusing the reference
// compressors so the compressed KV is bit-identical) and rolls the buffers.
func (c *fwbAttnLayerCache) appendBlockToken(h []float32) error {
	w := c.w
	d, dh, m := w.DModel, w.DHead, w.M
	c.curBlock = append(c.curBlock, h...)
	if len(c.curBlock) < m*d {
		return nil
	}

	var comp []float32
	var err error
	if w.Sparse {
		if c.prevBlock == nil {
			// Block 0: phantom predecessor, exactly the reference's block 0.
			blocks := len(c.curBlock) / (m * d)
			comp, err = CompressKVSparse(c.curBlock, w.WKVa, w.WKVb, w.WZa, w.WZb, w.PosA, w.PosB, blocks, m, d, dh)
		} else {
			// Block b>0 is index 1 of a [prev, cur] pair; its series-b reads prev.
			cat := make([]float32, len(c.prevBlock)+len(c.curBlock))
			copy(cat, c.prevBlock)
			copy(cat[m*d:], c.curBlock)
			var two []float32
			blocks := len(cat) / (m * d)
			two, err = CompressKVSparse(cat, w.WKVa, w.WKVb, w.WZa, w.WZb, w.PosA, w.PosB, blocks, m, d, dh)
			if err == nil {
				comp = two[len(two)-dh:]
			}
		}
	} else {
		blocks := len(c.curBlock) / (m * d)
		comp, err = CompressKVHeavy(c.curBlock, w.WKV, w.WZ, w.Pos, blocks, m, d, dh)
	}
	if err != nil {
		return err
	}
	block := make([]float32, dh)
	copy(block, comp[:dh])
	c.comp = append(c.comp, block)

	c.prevBlock = c.curBlock
	c.curBlock = make([]float32, 0, m*d)
	return nil
}

// applyRotarySinglePos rotates q ([n_heads, d_head]) at one absolute position,
// reproducing NewRotaryCache+ApplyRotaryInto: split-half pairing, base 10000,
// and the cache's float32 round-trip on cos/sin.
func applyRotarySinglePos(q []float32, nh, dh, pos int) {
	half := dh / 2
	for i := 0; i < half; i++ {
		invFreq := 1.0 / math.Pow(RotaryBase, float64(i)/float64(half))
		angle := float64(pos) * invFreq
		cs := float64(float32(math.Cos(angle)))
		sn := float64(float32(math.Sin(angle)))
		for hd := 0; hd < nh; hd++ {
			base := hd * dh
			xe := float64(q[base+i])
			xo := float64(q[base+half+i])
			q[base+i] = float32(xe*cs - xo*sn)
			q[base+half+i] = float32(xe*sn + xo*cs)
		}
	}
}
