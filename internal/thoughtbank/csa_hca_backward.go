package thoughtbank

import (
	"fmt"
	"math"
	"sort"
)

// CompressedHybridAttnGradients mirrors CompressedHybridAttnWeights. HCA
// populates WKV/WZ/Pos; CSA populates the a/b compression set. The lightning
// indexer (WIq, WW) is DETACHED in the reference -- topk indices feed only a
// hard gather and a boolean mask, neither differentiable -- so its gradients are
// structurally zero and left nil.
type CompressedHybridAttnGradients struct {
	DH []float32 // [seq, d_model]

	DWKV, DWZ, DPos          []float32 // HCA compression
	DWKVa, DWKVb, DWZa, DWZb []float32 // CSA compression
	DPosA, DPosB             []float32 // CSA compression

	DWDq, DWUq  []float32
	DWWk, DWWv  []float32
	DOutGroup   [][]float32
	DOutProj    []float32
	DQNorm      []float32
	DKVNorm     []float32
	DSinkLogits []float32
}

// CompressedHybridAttentionBackward differentiates CompressedHybridAttentionForward
// for both HCA (Sparse=false) and CSA (Sparse=true). It composes the already-
// bracketed component backwards -- linearBackward, the negated-sin ApplyRotaryInto
// (RoPE VJP), rmsNormBackwardAndWeightGrad, AttentionSinkSoftmaxBackward, and
// CompressKVHeavy/SparseBackward -- so the new code is wiring at the seams.
//
// HCA normalises the whole compressed summary once and attends every causal
// block; dcomp is the direct sum of dK+dV over attending tokens. CSA leaves the
// summary un-normalised, selects the top-k blocks per token with the (detached)
// lightning indexer, and normalises only the SELECTED entries; comp receives
// gradient ONLY through the selected-entry gather, recomputed here to know where
// to scatter, then routed through the per-entry kv_norm. The RoPE VJP reuses
// ApplyRotaryInto with sin negated (a rotation's transpose is the rotation by
// -theta). KVNorm is shared across the compression norm and the window-key norm
// (CSA also across every per-entry selected norm), so DKVNorm accumulates.
func CompressedHybridAttentionBackward(h, dOut []float32, seq int, w *CompressedHybridAttnWeights) (*CompressedHybridAttnGradients, error) {
	d, dh, nh := w.DModel, w.DHead, w.NHeads
	if len(h) != seq*d {
		return nil, fmt.Errorf("attention backward: h has %d values, want %d", len(h), seq*d)
	}
	if len(dOut) != seq*d {
		return nil, fmt.Errorf("attention backward: dOut has %d values, want %d", len(dOut), seq*d)
	}
	if w.NGroups <= 0 || nh%w.NGroups != 0 {
		return nil, fmt.Errorf("attention backward: %d heads do not split into %d groups", nh, w.NGroups)
	}
	nBlocks := w.blockCount(seq)
	sel := nBlocks
	if w.Sparse {
		sel = w.TopK
		if sel > nBlocks {
			sel = nBlocks
		}
	}
	nkv := sel + w.NWin
	invSqrt := 1.0 / math.Sqrt(float64(dh))
	eps := w.NormEps
	hpg := nh / w.NGroups
	dg := d / w.NGroups

	// ---- Recompute forward intermediates from inputs (no stale state) ----
	padded := make([]float32, nBlocks*w.M*d)
	copy(padded, h)
	var compRaw []float32
	var err error
	if w.Sparse {
		compRaw, err = CompressKVSparse(padded, w.WKVa, w.WKVb, w.WZa, w.WZb, w.PosA, w.PosB, nBlocks, w.M, d, dh)
	} else {
		compRaw, err = CompressKVHeavy(padded, w.WKV, w.WZ, w.Pos, nBlocks, w.M, d, dh)
	}
	if err != nil {
		return nil, err
	}
	comp := compRaw
	if !w.Sparse {
		comp = rmsNormNew(compRaw, w.KVNorm, nBlocks, dh, eps)
	}

	cq := make([]float32, seq*w.DLatentQ)
	for t := 0; t < seq; t++ {
		tok := h[t*d : (t+1)*d]
		for i := 0; i < w.DLatentQ; i++ {
			cq[t*w.DLatentQ+i] = float32(dot(w.WDq[i*d:(i+1)*d], tok))
		}
	}
	qLow := make([]float32, seq*nh*dh)
	for t := 0; t < seq; t++ {
		c := cq[t*w.DLatentQ : (t+1)*w.DLatentQ]
		for i := 0; i < nh*dh; i++ {
			qLow[t*nh*dh+i] = float32(dot(w.WUq[i*w.DLatentQ:(i+1)*w.DLatentQ], c))
		}
	}
	rope := NewRotaryCache(seq, dh)
	qh := make([]float32, seq*nh*dh)
	for t := 0; t < seq; t++ {
		for hd := 0; hd < nh; hd++ {
			copy(qh[(hd*seq+t)*dh:(hd*seq+t+1)*dh], qLow[(t*nh+hd)*dh:(t*nh+hd+1)*dh])
		}
	}
	ApplyRotaryInto(qh, qh, nh, seq, dh, rope)
	qPreNorm := make([]float32, seq*nh*dh)
	for t := 0; t < seq; t++ {
		for hd := 0; hd < nh; hd++ {
			copy(qPreNorm[(t*nh+hd)*dh:(t*nh+hd+1)*dh], qh[(hd*seq+t)*dh:(hd*seq+t+1)*dh])
		}
	}
	q := rmsNormNew(qPreNorm, w.QNorm, seq*nh, dh, eps)

	wkRaw := make([]float32, seq*dh)
	wv := make([]float32, seq*dh)
	for t := 0; t < seq; t++ {
		tok := h[t*d : (t+1)*d]
		for e := 0; e < dh; e++ {
			wkRaw[t*dh+e] = float32(dot(w.WWk[e*d:(e+1)*d], tok))
			wv[t*dh+e] = float32(dot(w.WWv[e*d:(e+1)*d], tok))
		}
	}
	wk := rmsNormNew(wkRaw, w.KVNorm, seq, dh, eps)

	// ---- Accumulators ----
	g := &CompressedHybridAttnGradients{
		DH:          make([]float32, seq*d),
		DWDq:        make([]float32, len(w.WDq)),
		DWUq:        make([]float32, len(w.WUq)),
		DWWk:        make([]float32, len(w.WWk)),
		DWWv:        make([]float32, len(w.WWv)),
		DOutProj:    make([]float32, len(w.OutProj)),
		DQNorm:      make([]float32, len(w.QNorm)),
		DKVNorm:     make([]float32, len(w.KVNorm)),
		DSinkLogits: make([]float32, len(w.SinkLogits)),
	}
	if w.Sparse {
		g.DWKVa = make([]float32, len(w.WKVa))
		g.DWKVb = make([]float32, len(w.WKVb))
		g.DWZa = make([]float32, len(w.WZa))
		g.DWZb = make([]float32, len(w.WZb))
		g.DPosA = make([]float32, len(w.PosA))
		g.DPosB = make([]float32, len(w.PosB))
	} else {
		g.DWKV = make([]float32, len(w.WKV))
		g.DWZ = make([]float32, len(w.WZ))
		g.DPos = make([]float32, len(w.Pos))
	}
	g.DOutGroup = make([][]float32, len(w.OutGroup))
	for gi := range w.OutGroup {
		g.DOutGroup[gi] = make([]float32, len(w.OutGroup[gi]))
	}

	dq := make([]float32, seq*nh*dh)
	dcomp := make([]float32, nBlocks*dh) // CSA scatters here through per-entry kv_norm; HCA sums directly
	dwk := make([]float32, seq*dh)
	dwv := make([]float32, seq*dh)

	// Per-token scratch.
	kAll := make([]float32, nkv*dh)
	vAll := make([]float32, nkv*dh)
	logits := make([]float32, nh*nkv)
	weights := make([]float32, nh*nkv)
	headOutT := make([]float32, nh*dh)
	catT := make([]float32, d)
	dHeadOutT := make([]float32, nh*dh)
	dcatT := make([]float32, d)
	dkAll := make([]float64, nkv*dh)
	dvAll := make([]float64, nkv*dh)
	dweights := make([]float32, nh*nkv)
	valid := make([]bool, sel)
	slotBlock := make([]int, sel) // CSA: which comp block sits in slot i (-1 if none)
	scores := make([]float64, nBlocks)
	order := make([]int, nBlocks)

	for t := 0; t < seq; t++ {
		blockOfT := t / w.M
		for i := 0; i < sel; i++ {
			valid[i] = false
			slotBlock[i] = -1
		}

		if w.Sparse {
			// Lightning indexer (recompute for SELECTION only; detached, no grad).
			c := cq[t*w.DLatentQ : (t+1)*w.DLatentQ]
			tok := h[t*d : (t+1)*d]
			for b := 0; b < nBlocks; b++ {
				scores[b] = 0
				order[b] = b
			}
			for ih := 0; ih < w.NIdxHeads; ih++ {
				hw := dot(w.WW[ih*d:(ih+1)*d], tok)
				for b := 0; b < nBlocks; b++ {
					var acc float64
					for e := 0; e < dh; e++ {
						qi := dot(w.WIq[(ih*dh+e)*w.DLatentQ:(ih*dh+e+1)*w.DLatentQ], c)
						acc += qi * float64(compRaw[b*dh+e])
					}
					acc /= math.Sqrt(float64(dh))
					if acc < 0 {
						acc = 0
					}
					scores[b] += hw * acc
				}
			}
			for b := 0; b < nBlocks; b++ {
				if blockOfT <= b {
					scores[b] = math.Inf(-1)
				}
			}
			sort.SliceStable(order, func(a, b int) bool { return scores[order[a]] > scores[order[b]] })
			for i := 0; i < sel; i++ {
				b := order[i]
				if math.IsInf(scores[b], -1) {
					continue
				}
				valid[i] = true
				slotBlock[i] = b
				n := rmsNormNew(comp[b*dh:(b+1)*dh], w.KVNorm, 1, dh, eps)
				copy(kAll[i*dh:(i+1)*dh], n)
				copy(vAll[i*dh:(i+1)*dh], n)
			}
		} else {
			for b := 0; b < nBlocks; b++ {
				if blockOfT > b {
					valid[b] = true
					slotBlock[b] = b
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
			dst := headOutT[hd*dh : (hd+1)*dh]
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

		// ---- Grouped output projection, forward (cat) then backward for row t ----
		for gi := 0; gi < w.NGroups; gi++ {
			in := headOutT[gi*hpg*dh : (gi+1)*hpg*dh]
			gw := w.OutGroup[gi]
			for o := 0; o < dg; o++ {
				catT[gi*dg+o] = float32(dot(gw[o*hpg*dh:(o+1)*hpg*dh], in))
			}
		}
		dOutT := dOut[t*d : (t+1)*d]
		for i := range dcatT {
			dcatT[i] = 0
		}
		for o2 := 0; o2 < d; o2++ {
			go2 := float64(dOutT[o2])
			if go2 == 0 {
				continue
			}
			opRow := w.OutProj[o2*d : (o2+1)*d]
			dopRow := g.DOutProj[o2*d : (o2+1)*d]
			for i := 0; i < d; i++ {
				dcatT[i] += float32(go2 * float64(opRow[i]))
				dopRow[i] += float32(go2 * float64(catT[i]))
			}
		}
		for e := range dHeadOutT {
			dHeadOutT[e] = 0
		}
		for gi := 0; gi < w.NGroups; gi++ {
			gw := w.OutGroup[gi]
			dgw := g.DOutGroup[gi]
			base := gi * hpg * dh
			for o := 0; o < dg; o++ {
				dco := float64(dcatT[gi*dg+o])
				if dco == 0 {
					continue
				}
				gwRow := gw[o*hpg*dh : (o+1)*hpg*dh]
				dgwRow := dgw[o*hpg*dh : (o+1)*hpg*dh]
				for k := 0; k < hpg*dh; k++ {
					dHeadOutT[base+k] += float32(dco * float64(gwRow[k]))
					dgwRow[k] += float32(dco * float64(headOutT[base+k]))
				}
			}
		}

		// ---- Attention core backward using dHeadOutT ----
		for i := range dkAll {
			dkAll[i] = 0
			dvAll[i] = 0
		}
		for hd := 0; hd < nh; hd++ {
			dHO := dHeadOutT[hd*dh : (hd+1)*dh]
			for n := 0; n < nkv; n++ {
				a := float64(weights[hd*nkv+n])
				dwsum := 0.0
				for e := 0; e < dh; e++ {
					dwsum += float64(dHO[e]) * float64(vAll[n*dh+e])
					dvAll[n*dh+e] += a * float64(dHO[e])
				}
				dweights[hd*nkv+n] = float32(dwsum)
			}
		}
		dLogits, dSinkT := AttentionSinkSoftmaxBackward(dweights, weights, logits, w.SinkLogits, 1, nh, nkv)
		for hd := 0; hd < nh; hd++ {
			g.DSinkLogits[hd] += dSinkT[hd]
		}
		for hd := 0; hd < nh; hd++ {
			qv := q[(t*nh+hd)*dh : (t*nh+hd+1)*dh]
			dqv := dq[(t*nh+hd)*dh : (t*nh+hd+1)*dh]
			for n := 0; n < nkv; n++ {
				if n < sel && !valid[n] {
					continue
				}
				dl := float64(dLogits[hd*nkv+n]) * invSqrt
				if dl == 0 {
					continue
				}
				for e := 0; e < dh; e++ {
					dqv[e] += float32(dl * float64(kAll[n*dh+e]))
					dkAll[n*dh+e] += dl * float64(qv[e])
				}
			}
		}

		// Route dkAll/dvAll to their sources.
		for i := 0; i < sel; i++ {
			if !valid[i] {
				continue
			}
			b := slotBlock[i]
			if w.Sparse {
				// Selected entry was kv_norm(comp[b]); route dK+dV through the
				// per-entry norm to dcomp[b], accumulating DKVNorm.
				dsel := make([]float32, dh)
				for e := 0; e < dh; e++ {
					dsel[e] = float32(dkAll[i*dh+e] + dvAll[i*dh+e]) // key == value
				}
				dcb, dkvn := rmsNormBackwardAndWeightGrad(comp[b*dh:(b+1)*dh], w.KVNorm, dsel, 1, dh, eps)
				for e := 0; e < dh; e++ {
					dcomp[b*dh+e] += dcb[e]
					g.DKVNorm[e] += dkvn[e]
				}
			} else {
				// HCA: comp is pre-normed, so dcomp accumulates directly.
				for e := 0; e < dh; e++ {
					dcomp[b*dh+e] += float32(dkAll[i*dh+e] + dvAll[i*dh+e])
				}
			}
		}
		for j := 0; j < w.NWin; j++ {
			src := t - w.NWin + j
			if src < 0 {
				continue
			}
			n := sel + j
			for e := 0; e < dh; e++ {
				dwk[src*dh+e] += float32(dkAll[n*dh+e])
				dwv[src*dh+e] += float32(dvAll[n*dh+e])
			}
		}
	}

	// ---- Query path backward: q_norm -> RoPE^T -> W_uq -> W_dq ----
	dqPreNorm, dQNorm := rmsNormBackwardAndWeightGrad(qPreNorm, w.QNorm, dq, seq*nh, dh, eps)
	copy(g.DQNorm, dQNorm)
	negSin := make([]float32, len(rope.Sin))
	for i := range rope.Sin {
		negSin[i] = -rope.Sin[i]
	}
	negRope := &rotaryCache{Cos: rope.Cos, Sin: negSin, Half: rope.Half}
	dqh := make([]float32, seq*nh*dh)
	for t := 0; t < seq; t++ {
		for hd := 0; hd < nh; hd++ {
			copy(dqh[(hd*seq+t)*dh:(hd*seq+t+1)*dh], dqPreNorm[(t*nh+hd)*dh:(t*nh+hd+1)*dh])
		}
	}
	ApplyRotaryInto(dqh, dqh, nh, seq, dh, negRope)
	dqLow := make([]float32, seq*nh*dh)
	for t := 0; t < seq; t++ {
		for hd := 0; hd < nh; hd++ {
			copy(dqLow[(t*nh+hd)*dh:(t*nh+hd+1)*dh], dqh[(hd*seq+t)*dh:(hd*seq+t+1)*dh])
		}
	}
	dcq, dWUq, _ := linearBackward(cq, w.WUq, dqLow, seq, w.DLatentQ, nh*dh, false)
	copy(g.DWUq, dWUq)
	dHq, dWDq, _ := linearBackward(h, w.WDq, dcq, seq, d, w.DLatentQ, false)
	copy(g.DWDq, dWDq)
	for i := range dHq {
		g.DH[i] += dHq[i]
	}

	// ---- Window path backward ----
	dwkRaw, dKVNormWk := rmsNormBackwardAndWeightGrad(wkRaw, w.KVNorm, dwk, seq, dh, eps)
	for i := range g.DKVNorm {
		g.DKVNorm[i] += dKVNormWk[i]
	}
	dHwk, dWWk, _ := linearBackward(h, w.WWk, dwkRaw, seq, d, dh, false)
	copy(g.DWWk, dWWk)
	for i := range dHwk {
		g.DH[i] += dHwk[i]
	}
	dHwv, dWWv, _ := linearBackward(h, w.WWv, dwv, seq, d, dh, false)
	copy(g.DWWv, dWWv)
	for i := range dHwv {
		g.DH[i] += dHwv[i]
	}

	// ---- Compression path backward ----
	var dcompRaw []float32
	if w.Sparse {
		// dcomp is already the gradient of the RAW summary (per-entry norms were
		// reversed inside the loop), so it flows straight into the sparse backward.
		dcompRaw = dcomp
		cg, err := CompressKVSparseBackward(padded, w.WKVa, w.WKVb, w.WZa, w.WZb, w.PosA, w.PosB, dcompRaw, nBlocks, w.M, d, dh)
		if err != nil {
			return nil, err
		}
		copy(g.DWKVa, cg.DWKVa)
		copy(g.DWKVb, cg.DWKVb)
		copy(g.DWZa, cg.DWZa)
		copy(g.DWZb, cg.DWZb)
		copy(g.DPosA, cg.DPosA)
		copy(g.DPosB, cg.DPosB)
		for i := 0; i < seq*d; i++ {
			g.DH[i] += cg.DHPad[i]
		}
	} else {
		var dKVNormComp []float32
		dcompRaw, dKVNormComp = rmsNormBackwardAndWeightGrad(compRaw, w.KVNorm, dcomp, nBlocks, dh, eps)
		for i := range g.DKVNorm {
			g.DKVNorm[i] += dKVNormComp[i]
		}
		cg, err := CompressKVHeavyBackward(padded, w.WKV, w.WZ, w.Pos, dcompRaw, nBlocks, w.M, d, dh)
		if err != nil {
			return nil, err
		}
		copy(g.DWKV, cg.DWKV)
		copy(g.DWZ, cg.DWZ)
		copy(g.DPos, cg.DPos)
		for i := 0; i < seq*d; i++ {
			g.DH[i] += cg.DHPad[i]
		}
	}

	return g, nil
}
