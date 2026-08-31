package thoughtbank

import (
	"fmt"
	"math"
	"sort"
)

// Full hybrid-attention forward. Even layers run CSA, odd layers HCA; both share
// the query path, the sliding window, the multi-query core, and the grouped
// output projection, and differ in how they build the compressed KV summary they
// attend over. Ported from adaptive's go/extmodel/compressed_hybrid_attention.go
// at b8fef3cc2.

// CompressedHybridAttnWeights carries one attention layer's parameters. CSA-only
// and HCA-only fields are both present; Sparse selects which set is read, so a
// caller binding a checkpoint layer needs one type for what the reference models
// as two classes with one contract.
type CompressedHybridAttnWeights struct {
	Sparse bool // true = CSA (even layers), false = HCA (odd layers)

	DModel, NHeads, DHead int
	M, NWin, DLatentQ     int
	NGroups               int
	TopK, NIdxHeads       int // CSA only

	// Compression. CSA uses the a/b pairs; HCA uses WKV/WZ/Pos.
	WKVa, WKVb, WZa, WZb, PosA, PosB []float32
	WKV, WZ, Pos                     []float32

	WDq []float32 // [d_latent_q, d_model]
	WUq []float32 // [n_heads*d_head, d_latent_q]
	WIq []float32 // CSA: [n_idx_heads*d_head, d_latent_q]
	WW  []float32 // CSA: [n_idx_heads, d_model]

	WWk []float32 // [d_head, d_model]
	WWv []float32 // [d_head, d_model]

	OutGroup [][]float32 // n_groups x [d_model/n_groups, (n_heads/n_groups)*d_head]
	OutProj  []float32   // [d_model, d_model]

	QNorm      []float32 // [d_head]
	KVNorm     []float32 // [d_head]
	SinkLogits []float32 // [n_heads]
	NormEps    float64
}

// blockCount returns the number of compression blocks for seq tokens, padding up
// to a whole block exactly as the reference does.
func (w *CompressedHybridAttnWeights) blockCount(seq int) int {
	return (seq + w.M - 1) / w.M
}

// CompressedHybridAttentionForward runs one attention layer over a single
// sequence. h holds seq x d_model values; the result has the same shape.
func CompressedHybridAttentionForward(h []float32, seq int, w *CompressedHybridAttnWeights) ([]float32, error) {
	if len(h) != seq*w.DModel {
		return nil, fmt.Errorf("attention: input has %d values, want %d", len(h), seq*w.DModel)
	}
	d, dh, nh := w.DModel, w.DHead, w.NHeads
	nBlocks := w.blockCount(seq)

	// Zero-pad to a whole block; the reference pads with zeros and the
	// compression softmax sees them, so they are not simply ignored.
	padded := make([]float32, nBlocks*w.M*d)
	copy(padded, h)

	var comp []float32
	var err error
	if w.Sparse {
		comp, err = CompressKVSparse(padded, w.WKVa, w.WKVb, w.WZa, w.WZb, w.PosA, w.PosB,
			nBlocks, w.M, d, dh)
	} else {
		comp, err = CompressKVHeavy(padded, w.WKV, w.WZ, w.Pos, nBlocks, w.M, d, dh)
		// HCA normalises the compressed summary immediately; CSA normalises only
		// the entries it selects, AFTER the indexer has scored the raw ones.
		if err == nil {
			comp = rmsNormNew(comp, w.KVNorm, nBlocks, dh, w.NormEps)
		}
	}
	if err != nil {
		return nil, err
	}

	// Queries: low-rank d -> d_latent_q -> n_heads*d_head, then RoPE, then norm.
	cq := make([]float32, seq*w.DLatentQ)
	for t := range seq {
		tok := h[t*d : (t+1)*d]
		for i := 0; i < w.DLatentQ; i++ {
			cq[t*w.DLatentQ+i] = float32(dot(w.WDq[i*d:(i+1)*d], tok))
		}
	}
	q := make([]float32, seq*nh*dh)
	for t := range seq {
		c := cq[t*w.DLatentQ : (t+1)*w.DLatentQ]
		for i := range nh * dh {
			q[t*nh*dh+i] = float32(dot(w.WUq[i*w.DLatentQ:(i+1)*w.DLatentQ], c))
		}
	}
	// RoPE applies per HEAD over the time axis, so the head axis must lead.
	qh := make([]float32, seq*nh*dh)
	for t := range seq {
		for hd := range nh {
			copy(qh[(hd*seq+t)*dh:(hd*seq+t+1)*dh], q[(t*nh+hd)*dh:(t*nh+hd+1)*dh])
		}
	}
	rope := NewRotaryCache(seq, dh)
	ApplyRotaryInto(qh, qh, nh, seq, dh, rope)
	for t := range seq {
		for hd := range nh {
			copy(q[(t*nh+hd)*dh:(t*nh+hd+1)*dh], qh[(hd*seq+t)*dh:(hd*seq+t+1)*dh])
		}
	}
	q = rmsNormNew(q, w.QNorm, seq*nh, dh, w.NormEps)

	// Sliding window: for token t the window is the n_win tokens STRICTLY BEFORE
	// t (the reference left-pads by n_win and gathers t..t+n_win-1, excluding t).
	wk := make([]float32, seq*dh)
	wv := make([]float32, seq*dh)
	for t := range seq {
		tok := h[t*d : (t+1)*d]
		for e := range dh {
			wk[t*dh+e] = float32(dot(w.WWk[e*d:(e+1)*d], tok))
			wv[t*dh+e] = float32(dot(w.WWv[e*d:(e+1)*d], tok))
		}
	}
	wk = rmsNormNew(wk, w.KVNorm, seq, dh, w.NormEps)

	sel := w.TopK
	if !w.Sparse {
		sel = nBlocks // HCA attends every block, masked causally
	} else {
		sel = min(sel, nBlocks)
	}
	nkv := sel + w.NWin

	kAll := make([]float32, nkv*dh)
	vAll := make([]float32, nkv*dh)
	logits := make([]float32, nh*nkv)
	weights := make([]float32, nh*nkv)
	negativeInfinity := float32(math.Inf(-nkv))
	headOut := make([]float32, seq*nh*dh)
	scores := make([]float64, nBlocks)
	order := make([]int, nBlocks)
	valid := make([]bool, sel)

	for t := range seq {
		blockOfT := t / w.M
		clear(valid)

		if w.Sparse {
			// Lightning indexer: multi-head ReLU affinity, aggregated by learned
			// per-head scalars. It scores the RAW compressed blocks.
			clear(scores)
			for b := range nBlocks {
				order[b] = b
			}
			c := cq[t*w.DLatentQ : (t+1)*w.DLatentQ]
			tok := h[t*d : (t+1)*d]
			for ih := 0; ih < w.NIdxHeads; ih++ {
				hw := dot(w.WW[ih*d:(ih+1)*d], tok)
				for b := range nBlocks {
					var acc float64
					for e := range dh {
						qi := dot(w.WIq[(ih*dh+e)*w.DLatentQ:(ih*dh+e+1)*w.DLatentQ], c)
						acc += qi * float64(comp[b*dh+e])
					}
					acc /= math.Sqrt(float64(dh))
					var zero float64
					acc = max(acc, zero) // ReLU
					scores[b] += hw * acc
				}
			}
			// Causal at BLOCK granularity: token t sees block j only if j < t/m.
			eligible := order[:blockOfT]
			sort.SliceStable(eligible, func(a, b int) bool { return scores[eligible[a]] > scores[eligible[b]] })
			for i := range min(sel, len(eligible)) {
				b := eligible[i]
				valid[i] = true
				src := comp[b*dh : (b+1)*dh]
				n := rmsNormVector(src, w.KVNorm, dh, w.NormEps)
				copy(kAll[i*dh:(i+1)*dh], n)
				copy(vAll[i*dh:(i+1)*dh], n)
			}
		} else {
			for b := range nBlocks {
				if blockOfT > b {
					valid[b] = true
					copy(kAll[b*dh:(b+1)*dh], comp[b*dh:(b+1)*dh])
					copy(vAll[b*dh:(b+1)*dh], comp[b*dh:(b+1)*dh])
				} else {
					clear(kAll[b*dh : (b+1)*dh])
					clear(vAll[b*dh : (b+1)*dh])
				}
			}
		}

		for j := 0; j < w.NWin; j++ {
			src := t - w.NWin + j
			dst := (sel + j) * dh
			if src < 0 {
				clear(kAll[dst : dst+dh])
				clear(vAll[dst : dst+dh])
				continue
			}
			copy(kAll[dst:dst+dh], wk[src*dh:(src+1)*dh])
			copy(vAll[dst:dst+dh], wv[src*dh:(src+1)*dh])
		}

		// Multi-query: every head reads the same K/V.
		invSqrt := 1.0 / math.Sqrt(float64(dh))
		for hd := range nh {
			qv := q[(t*nh+hd)*dh : (t*nh+hd+1)*dh]
			for n := range nkv {
				if n < sel && !valid[n] {
					logits[hd*nkv+n] = negativeInfinity
					continue
				}
				logits[hd*nkv+n] = float32(dot(kAll[n*dh:(n+1)*dh], qv) * invSqrt)
			}
		}
		attentionSinkSoftmaxVector(weights, logits, w.SinkLogits, nh, nkv)
		for hd := range nh {
			dst := headOut[(t*nh+hd)*dh : (t*nh+hd+1)*dh]
			clear(dst)
			for n := range nkv {
				a := float64(weights[hd*nkv+n])
				if a == 0 {
					continue
				}
				for e := range dh {
					dst[e] += float32(a * float64(vAll[n*dh+e]))
				}
			}
		}
	}

	return groupedOutputProjection(headOut, seq, nh, dh, w)
}

// groupedOutputProjection concatenates per-group projections of the head outputs
// and applies the final projection. Heads are split into n_groups contiguous
// runs; each group's heads are flattened and projected to d_model/n_groups, so
// the concatenation is d_model wide again.
func groupedOutputProjection(headOut []float32, seq, nh, dh int, w *CompressedHybridAttnWeights) ([]float32, error) {
	if w.NGroups <= 0 || nh%w.NGroups != 0 {
		return nil, fmt.Errorf("attention: %d heads do not split into %d groups", nh, w.NGroups)
	}
	hpg := nh / w.NGroups
	dg := w.DModel / w.NGroups
	if len(w.OutGroup) != w.NGroups {
		return nil, fmt.Errorf("attention: %d group projections, want %d", len(w.OutGroup), w.NGroups)
	}
	cat := make([]float32, seq*w.DModel)
	for t := range seq {
		for g := 0; g < w.NGroups; g++ {
			in := headOut[(t*nh+g*hpg)*dh : (t*nh+(g+1)*hpg)*dh]
			gw := w.OutGroup[g]
			for o := range dg {
				cat[t*w.DModel+g*dg+o] = float32(dot(gw[o*hpg*dh:(o+1)*hpg*dh], in))
			}
		}
	}
	out := make([]float32, seq*w.DModel)
	for t := range seq {
		row := cat[t*w.DModel : (t+1)*w.DModel]
		for o := 0; o < w.DModel; o++ {
			out[t*w.DModel+o] = float32(dot(w.OutProj[o*w.DModel:(o+1)*w.DModel], row))
		}
	}
	return out, nil
}

func groupedOutputVector(headOut []float32, nh, dh int, weights *CompressedHybridAttnWeights) ([]float32, error) {
	return groupedOutputProjection(headOut, len(headOut)/(nh*dh), nh, dh, weights)
}
