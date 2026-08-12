//go:build windows

package densecausal

import (
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/devicemath"
)

// ropeInvF32 converts the rope inverse-frequency table to fp32 for the device.
func ropeInvF32(invFreq []float64) []float32 {
	out := make([]float32, len(invFreq))
	for i, v := range invFreq {
		out[i] = float32(v)
	}
	return out
}

// deviceCausalSoftmaxGQA computes the per-head causal softmax p[nh*seq*seq] the
// device MHA consumes, matching hostmath.CausalAttentionBackward's scoring
// (scale=1 -- the score scale is already folded into qScaled -- GQA kv=h/group,
// keys 0..qi). p[h*seq*seq + qi*seq + m].
func deviceCausalSoftmaxGQA(q, k []float32, seq, nh, nkv, hd int) []float32 {
	group := nh / nkv
	p := make([]float32, nh*seq*seq)
	row := make([]float64, seq)
	for h := 0; h < nh; h++ {
		kv := h / group
		for qi := 0; qi < seq; qi++ {
			nkeys := qi + 1
			mx := math.Inf(-1)
			for m := 0; m < nkeys; m++ {
				var dot float64
				for x := 0; x < hd; x++ {
					dot += float64(q[(qi*nh+h)*hd+x]) * float64(k[(m*nkv+kv)*hd+x])
				}
				row[m] = dot
				if dot > mx {
					mx = dot
				}
			}
			var sum float64
			for m := 0; m < nkeys; m++ {
				row[m] = math.Exp(row[m] - mx)
				sum += row[m]
			}
			for m := 0; m < nkeys; m++ {
				p[h*seq*seq+qi*seq+m] = float32(row[m] / sum)
			}
		}
	}
	return p
}

// deviceLayerBackward is the device counterpart to layerBackward: it recomputes
// the forward on host for intermediates, then runs every VJP on the GPU via
// internal/devicemath, matching layerBackward's math (and weight-grad slots)
// exactly. Returns dx (the layer's input gradient) and writes weight grads into
// g. Attention bias is not yet supported (AttnBias must be false).
func (m *Model) deviceLayerBackward(worker *device.Worker, index int, x, dOut []float32, cache layerCache, invFreq []float64, seq int, g Grads) ([]float32, error) {
	d := m.Dims
	if d.AttnBias {
		return nil, fmt.Errorf("deviceLayerBackward: attention bias not supported yet")
	}
	l, err := m.layerWeights(index)
	if err != nil {
		return nil, err
	}
	prefix := fmt.Sprintf("model.layers.%d.", index)
	width := d.Heads * d.HeadDim
	kvWidth := d.KVHeads * d.HeadDim
	scale := float32(1 / math.Sqrt(float64(d.HeadDim)))
	invF32 := ropeInvF32(invFreq)

	// Forward intermediates come from the cache (saved by layerForwardCached) --
	// no host recompute.
	xn, tr, h2, hn := cache.xn, cache.tr, cache.h2, cache.hn
	gate, up, a, hMLP := cache.gate, cache.up, cache.a, cache.hMLP
	p := deviceCausalSoftmaxGQA(tr.qScaled, tr.kRoped, seq, d.Heads, d.KVHeads, d.HeadDim)

	// --- MLP branch backward ---
	mlp, err := devicemath.GatedMLPBackwardTResident(worker, hn, l.gate, l.up, l.down, gate, a, up, hMLP, dOut, seq, d.Hidden, d.Intermediate)
	if err != nil {
		return nil, err
	}
	copy(g.slot(prefix+"mlp.gate_proj.weight", d.Intermediate*d.Hidden), mlp.DWGate)
	copy(g.slot(prefix+"mlp.up_proj.weight", d.Intermediate*d.Hidden), mlp.DWUp)
	copy(g.slot(prefix+"mlp.down_proj.weight", d.Hidden*d.Intermediate), mlp.DWDown)
	dPost, dwPost, err := devicemath.RMSNormBackward(worker, h2, l.postLN, mlp.DX, seq, d.Hidden, d.RMSEps)
	if err != nil {
		return nil, err
	}
	copy(g.slot(prefix+"post_attention_layernorm.weight", d.Hidden), dwPost)
	dh2 := make([]float32, seq*d.Hidden)
	for i := range dh2 {
		dh2[i] = dOut[i] + dPost[i]
	}

	// --- Attention branch backward ---
	dAttnCore, dWo, err := devicemath.LinearBackwardT(worker, tr.attnCore, l.o, dh2, seq, width, d.Hidden)
	if err != nil {
		return nil, err
	}
	copy(g.slot(prefix+"self_attn.o_proj.weight", d.Hidden*width), dWo)
	dq, dk, dv, err := devicemath.MultiHeadAttentionBackwardResident(worker, tr.qScaled, tr.kRoped, tr.v, p, dAttnCore, seq, d.Heads, d.KVHeads, d.HeadDim, 1.0)
	if err != nil {
		return nil, err
	}
	for i := range dq {
		dq[i] *= scale
	}
	if dq, err = devicemath.RoPEHalfBackward(worker, dq, invF32, seq, d.Heads, d.HeadDim); err != nil {
		return nil, err
	}
	if dk, err = devicemath.RoPEHalfBackward(worker, dk, invF32, seq, d.KVHeads, d.HeadDim); err != nil {
		return nil, err
	}
	dXnQ, dWq, err := devicemath.LinearBackwardT(worker, xn, l.q, dq, seq, d.Hidden, width)
	if err != nil {
		return nil, err
	}
	dXnK, dWk, err := devicemath.LinearBackwardT(worker, xn, l.k, dk, seq, d.Hidden, kvWidth)
	if err != nil {
		return nil, err
	}
	dXnV, dWv, err := devicemath.LinearBackwardT(worker, xn, l.v, dv, seq, d.Hidden, kvWidth)
	if err != nil {
		return nil, err
	}
	copy(g.slot(prefix+"self_attn.q_proj.weight", width*d.Hidden), dWq)
	copy(g.slot(prefix+"self_attn.k_proj.weight", kvWidth*d.Hidden), dWk)
	copy(g.slot(prefix+"self_attn.v_proj.weight", kvWidth*d.Hidden), dWv)
	dXn := make([]float32, seq*d.Hidden)
	for i := range dXn {
		dXn[i] = dXnQ[i] + dXnK[i] + dXnV[i]
	}

	dInn, dwIn, err := devicemath.RMSNormBackward(worker, x, l.inLN, dXn, seq, d.Hidden, d.RMSEps)
	if err != nil {
		return nil, err
	}
	copy(g.slot(prefix+"input_layernorm.weight", d.Hidden), dwIn)
	dx := make([]float32, seq*d.Hidden)
	for i := range dx {
		dx[i] = dh2[i] + dInn[i]
	}
	return dx, nil
}
