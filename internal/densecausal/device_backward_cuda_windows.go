//go:build windows

package densecausal

import (
	"fmt"

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

// deviceLayerBackward is the device counterpart to layerBackward: it runs the
// whole layer backward in ONE resident cudaBLAS session
// (devicemath.LayerBackwardResident) consuming the forward cache -- every VJP and
// intermediate gradient stays device-resident, no per-op round-trips. Returns dx
// (the layer's input gradient) and writes weight grads into g, matching
// layerBackward's math and slots. Attention bias is not yet supported.
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
	fc := devicemath.LayerForwardCache{
		Xn: cache.xn, QScaled: cache.tr.qScaled, KRoped: cache.tr.kRoped, V: cache.tr.v, AttnCore: cache.tr.attnCore,
		H2: cache.h2, Hn: cache.hn, Gate: cache.gate, Up: cache.up, A: cache.a, HMLP: cache.hMLP,
	}
	w := devicemath.LayerForwardWeights{
		InLN: l.inLN, PostLN: l.postLN, Q: l.q, K: l.k, V: l.v, O: l.o, Gate: l.gate, Up: l.up, Down: l.down,
	}
	r, err := devicemath.LayerBackwardResident(worker, x, dOut, fc, w, ropeInvF32(invFreq), seq, d.Hidden, d.Heads, d.KVHeads, d.HeadDim, d.Intermediate, d.RMSEps)
	if err != nil {
		return nil, err
	}
	copy(g.slot(prefix+"mlp.gate_proj.weight", d.Intermediate*d.Hidden), r.DWGate)
	copy(g.slot(prefix+"mlp.up_proj.weight", d.Intermediate*d.Hidden), r.DWUp)
	copy(g.slot(prefix+"mlp.down_proj.weight", d.Hidden*d.Intermediate), r.DWDown)
	copy(g.slot(prefix+"post_attention_layernorm.weight", d.Hidden), r.DWPostLN)
	copy(g.slot(prefix+"self_attn.o_proj.weight", d.Hidden*width), r.DWO)
	copy(g.slot(prefix+"self_attn.q_proj.weight", width*d.Hidden), r.DWQ)
	copy(g.slot(prefix+"self_attn.k_proj.weight", kvWidth*d.Hidden), r.DWK)
	copy(g.slot(prefix+"self_attn.v_proj.weight", kvWidth*d.Hidden), r.DWV)
	copy(g.slot(prefix+"input_layernorm.weight", d.Hidden), r.DWInLN)
	return r.DX, nil
}
