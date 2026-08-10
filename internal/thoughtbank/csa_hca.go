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
	for t := 0; t < seq; t++ {
		tok := h[t*d : (t+1)*d]
		for i := 0; i < w.DLatentQ; i++ {
			cq[t*w.DLatentQ+i] = float32(dot(w.WDq[i*d:(i+1)*d], tok))
		}
	}
	q := make([]float32, seq*nh*dh)
	for t := 0; t < seq; t++ {
		c := cq[t*w.DLatentQ : (t+1)*w.DLatentQ]
		for i := 0; i < nh*dh; i++ {
			q[t*nh*dh+i] = float32(dot(w.WUq[i*w.DLatentQ:(i+1)*w.DLatentQ], c))
		}
	}
	// RoPE applies per HEAD over the time axis, so the head axis must lead.
	qh := make([]float32, seq*nh*dh)
	for t := 0; t < seq; t++ {
		for hd := 0; hd < nh; hd++ {
			copy(qh[(hd*seq+t)*dh:(hd*seq+t+1)*dh], q[(t*nh+hd)*dh:(t*nh+hd+1)*dh])
		}
	}
	rope := NewRotaryCache(seq, dh)
	ApplyRotaryInto(qh, qh, nh, seq, dh, rope)
	for t := 0; t < seq; t++ {
		for hd := 0; hd < nh; hd++ {
			copy(q[(t*nh+hd)*dh:(t*nh+hd+1)*dh], qh[(hd*seq+t)*dh:(hd*seq+t+1)*dh])
		}
	}
	q = rmsNormNew(q, w.QNorm, seq*nh, dh, w.NormEps)

	// Sliding window: for token t the window is the n_win tokens STRICTLY BEFORE
	// t (the reference left-pads by n_win and gathers t..t+n_win-1, excluding t).
	wk := make([]float32, seq*dh)
	wv := make([]float32, seq*dh)
	for t := 0; t < seq; t++ {
		tok := h[t*d : (t+1)*d]
		for e := 0; e < dh; e++ {
			wk[t*dh+e] = float32(dot(w.WWk[e*d:(e+1)*d], tok))
			wv[t*dh+e] = float32(dot(w.WWv[e*d:(e+1)*d], tok))
		}
	}
	wk = rmsNormNew(wk, w.KVNorm, seq, dh, w.NormEps)

	sel := w.TopK
	if !w.Sparse {
		sel = nBlocks // HCA attends every block, masked causally
	} else if sel > nBlocks {
		sel = nBlocks
	}
	nkv := sel + w.NWin

	kAll := make([]float32, nkv*dh)
	vAll := make([]float32, nkv*dh)
	logits := make([]float32, nh*nkv)
	weights := make([]float32, nh*nkv)
	headOut := make([]float32, seq*nh*dh)
	scores := make([]float64, nBlocks)
	order := make([]int, nBlocks)
	valid := make([]bool, sel)

	for t := 0; t < seq; t++ {
		blockOfT := t / w.M
		for i := range valid {
			valid[i] = false
		}

		if w.Sparse {
			// Lightning indexer: multi-head ReLU affinity, aggregated by learned
			// per-head scalars. It scores the RAW compressed blocks.
			for b := 0; b < nBlocks; b++ {
				scores[b] = 0
				order[b] = b
			}
			c := cq[t*w.DLatentQ : (t+1)*w.DLatentQ]
			tok := h[t*d : (t+1)*d]
			for ih := 0; ih < w.NIdxHeads; ih++ {
				hw := dot(w.WW[ih*d:(ih+1)*d], tok)
				for b := 0; b < nBlocks; b++ {
					var acc float64
					for e := 0; e < dh; e++ {
						qi := dot(w.WIq[(ih*dh+e)*w.DLatentQ:(ih*dh+e+1)*w.DLatentQ], c)
						acc += qi * float64(comp[b*dh+e])
					}
					acc /= math.Sqrt(float64(dh))
					if acc < 0 {
						acc = 0 // ReLU
					}
					scores[b] += hw * acc
				}
			}
			// Causal at BLOCK granularity: token t sees block j only if j < t/m.
			for b := 0; b < nBlocks; b++ {
				if blockOfT <= b {
					scores[b] = math.Inf(-1)
				}
			}
			sort.SliceStable(order, func(a, b int) bool { return scores[order[a]] > scores[order[b]] })
			for i := 0; i < sel; i++ {
				b := order[i]
				if math.IsInf(scores[b], -1) {
					continue // stays invalid; masked below
				}
				valid[i] = true
				src := comp[b*dh : (b+1)*dh]
				n := rmsNormNew(src, w.KVNorm, 1, dh, w.NormEps)
				copy(kAll[i*dh:(i+1)*dh], n)
				copy(vAll[i*dh:(i+1)*dh], n)
			}
		} else {
			for b := 0; b < nBlocks; b++ {
				if blockOfT > b {
					valid[b] = true
					copy(kAll[b*dh:(b+1)*dh], comp[b*dh:(b+1)*dh])
					copy(vAll[b*dh:(b+1)*dh], comp[b*dh:(b+1)*dh])
				} else {
					for e := 0; e < dh; e++ {
						kAll[b*dh+e], vAll[b*dh+e] = 0, 0
					}
				}
			}
		}

		for j := 0; j < w.NWin; j++ {
			src := t - w.NWin + j
			dst := (sel + j) * dh
			if src < 0 {
				for e := 0; e < dh; e++ {
					kAll[dst+e], vAll[dst+e] = 0, 0
				}
				continue
			}
			copy(kAll[dst:dst+dh], wk[src*dh:(src+1)*dh])
			copy(vAll[dst:dst+dh], wv[src*dh:(src+1)*dh])
		}

		// Multi-query: every head reads the same K/V.
		invSqrt := 1.0 / math.Sqrt(float64(dh))
		for hd := 0; hd < nh; hd++ {
			qv := q[(t*nh+hd)*dh : (t*nh+hd+1)*dh]
			for n := 0; n < nkv; n++ {
				if n < sel && !valid[n] {
					logits[hd*nkv+n] = float32(math.Inf(-1))
					continue
				}
				logits[hd*nkv+n] = float32(dot(kAll[n*dh:(n+1)*dh], qv) * invSqrt)
			}
		}
		AttentionSinkSoftmaxInto(weights, logits, w.SinkLogits, 1, nh, nkv)
		for hd := 0; hd < nh; hd++ {
			dst := headOut[(t*nh+hd)*dh : (t*nh+hd+1)*dh]
			for e := range dst {
				dst[e] = 0
			}
			for n := 0; n < nkv; n++ {
				a := float64(weights[hd*nkv+n])
				if a == 0 {
					continue
				}
				for e := 0; e < dh; e++ {
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
	for t := 0; t < seq; t++ {
		for g := 0; g < w.NGroups; g++ {
			in := headOut[(t*nh+g*hpg)*dh : (t*nh+(g+1)*hpg)*dh]
			gw := w.OutGroup[g]
			for o := 0; o < dg; o++ {
				cat[t*w.DModel+g*dg+o] = float32(dot(gw[o*hpg*dh:(o+1)*hpg*dh], in))
			}
		}
	}
	out := make([]float32, seq*w.DModel)
	for t := 0; t < seq; t++ {
		row := cat[t*w.DModel : (t+1)*w.DModel]
		for o := 0; o < w.DModel; o++ {
			out[t*w.DModel+o] = float32(dot(w.OutProj[o*w.DModel:(o+1)*w.DModel], row))
		}
	}
	return out, nil
}
