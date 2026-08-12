package hostmath

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"overgo/internal/testutil"
)

// hybridGolden mirrors fixtures/qwen35_hybrid_grad_golden.json
// (schema qwen35_hybrid_grad_golden/v1): torch autograd grads of
// L=sum(dOut*layer(x)) for both hybrid decoder-layer variants, the oracle for
// the host HybridDecoderLayerBackward VJP.
type hybridWeight struct {
	Shape  []int     `json:"shape"`
	Values []float64 `json:"values"`
	Grad   []float64 `json:"grad"`
}

type hybridGolden struct {
	Schema string `json:"schema"`
	Dims   struct {
		T, H, Inter, HD int
		Eps             float64
		Linear          struct{ Hk, Hv, K int }
		Attention       struct {
			Heads, Kv int
			Theta     float64
		}
	} `json:"dims"`
	X         hybridWeight            `json:"x"`
	XAttn     hybridWeight            `json:"x_attn"`
	DOut      []float64               `json:"dOut"`
	OutLinear []float64               `json:"out_linear"`
	OutAttn   []float64               `json:"out_attention"`
	Linear    map[string]hybridWeight `json:"linear"`
	Attention map[string]hybridWeight `json:"attention"`
}

func f32(v []float64) []float32 {
	s := make([]float32, len(v))
	for i, x := range v {
		s[i] = float32(x)
	}
	return s
}

// TestHybridDecoderLayerGradParity checks host forward output and the host VJP
// against the torch autograd golden, for both mix variants. This is the
// cross-implementation parity backing the FD self-check.
func TestHybridDecoderLayerGradParity(t *testing.T) {
	path := testutil.FixturePath(t, "qwen35_hybrid_grad_golden.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("UNAVAILABLE: %s absent; grad parity NOT verified (run py -3.12 scripts/gen_qwen35_hybrid_grad_golden.py)", path)
	}
	var g hybridGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	T, H, inter, hd, eps := g.Dims.T, g.Dims.H, g.Dims.Inter, g.Dims.HD, g.Dims.Eps
	dOut := f32(g.DOut)

	// forward output tolerance (f64 torch vs f32-store/f64-accum host) and grad
	// tolerance; both generous since storage differs but math matches.
	const outTol, gradTol = 2e-4, 3e-3
	cmp := func(t *testing.T, name string, got []float32, want []float64) {
		var maxd float64
		for i := range got {
			if d := math.Abs(float64(got[i]) - want[i]); d > maxd {
				maxd = d
			}
		}
		if maxd > gradTol {
			t.Errorf("%s: max|host-torch| %.3e > %.1e", name, maxd, gradTol)
		} else {
			t.Logf("%-13s max|host-torch| %.3e", name, maxd)
		}
	}
	cmpOut := func(t *testing.T, got []float32, want []float64) {
		var maxd float64
		for i := range got {
			if d := math.Abs(float64(got[i]) - want[i]); d > maxd {
				maxd = d
			}
		}
		if maxd > outTol {
			t.Errorf("forward out: max|host-torch| %.3e > %.1e", maxd, outTol)
		} else {
			t.Logf("forward out   max|host-torch| %.3e", maxd)
		}
	}

	t.Run("linear_attention", func(t *testing.T) {
		m := g.Linear
		hk, hv := g.Dims.Linear.Hk, g.Dims.Linear.Hv
		w := HybridLayerWeights{
			InputNorm: f32(m["InputNorm"].Values), PostNorm: f32(m["PostNorm"].Values), IsLinear: true,
			MLP: QwenMLPWeights{Gate: f32(m["mlp.Gate"].Values), Up: f32(m["mlp.Up"].Values), Down: f32(m["mlp.Down"].Values)},
			GDN: GatedDeltaMixWeights{
				Wq: f32(m["mix.Wq"].Values), Wk: f32(m["mix.Wk"].Values), Wv: f32(m["mix.Wv"].Values),
				ConvQ: f32(m["mix.ConvQ"].Values), ConvK: f32(m["mix.ConvK"].Values), ConvV: f32(m["mix.ConvV"].Values),
				ConvBiasQ: f32(m["mix.ConvBiasQ"].Values), ConvBiasK: f32(m["mix.ConvBiasK"].Values), ConvBiasV: f32(m["mix.ConvBiasV"].Values),
				Wbeta: f32(m["mix.Wbeta"].Values), Walpha: f32(m["mix.Walpha"].Values),
				TimeStep: f32(m["mix.TimeStep"].Values), A: f32(m["mix.A"].Values),
				Wz: f32(m["mix.Wz"].Values), Norm: f32(m["mix.Norm"].Values), Wout: f32(m["mix.Wout"].Values),
			},
		}
		d := HybridLayerDims{Tokens: T, Hidden: H, Inter: inter, Eps: eps,
			GDN: GatedDeltaMixDims{Tokens: T, Hidden: H, KeyHeads: hk, ValueHeads: hv, HeadDim: hd, ConvK: g.Dims.Linear.K, OutDim: H, Eps: eps}}
		x := f32(g.X.Values)
		state := f32(m["mix.state"].Values)
		out, c := HybridDecoderLayerForward(x, w, d, state)
		cmpOut(t, out, g.OutLinear)
		gr := HybridDecoderLayerBackward(x, w, d, state, dOut, c)
		cmp(t, "dX", gr.DX, g.X.Grad)
		cmp(t, "dInputNorm", gr.DInputNorm, m["InputNorm"].Grad)
		cmp(t, "dPostNorm", gr.DPostNorm, m["PostNorm"].Grad)
		cmp(t, "dMLP.Gate", gr.DMLP.Gate, m["mlp.Gate"].Grad)
		cmp(t, "dMLP.Up", gr.DMLP.Up, m["mlp.Up"].Grad)
		cmp(t, "dMLP.Down", gr.DMLP.Down, m["mlp.Down"].Grad)
		cmp(t, "dWq", gr.DGDN.DWq, m["mix.Wq"].Grad)
		cmp(t, "dWk", gr.DGDN.DWk, m["mix.Wk"].Grad)
		cmp(t, "dWv", gr.DGDN.DWv, m["mix.Wv"].Grad)
		cmp(t, "dConvQ", gr.DGDN.DConvQ, m["mix.ConvQ"].Grad)
		cmp(t, "dConvK", gr.DGDN.DConvK, m["mix.ConvK"].Grad)
		cmp(t, "dConvV", gr.DGDN.DConvV, m["mix.ConvV"].Grad)
		cmp(t, "dConvBiasQ", gr.DGDN.DConvBiasQ, m["mix.ConvBiasQ"].Grad)
		cmp(t, "dConvBiasK", gr.DGDN.DConvBiasK, m["mix.ConvBiasK"].Grad)
		cmp(t, "dConvBiasV", gr.DGDN.DConvBiasV, m["mix.ConvBiasV"].Grad)
		cmp(t, "dWbeta", gr.DGDN.DWbeta, m["mix.Wbeta"].Grad)
		cmp(t, "dWalpha", gr.DGDN.DWalpha, m["mix.Walpha"].Grad)
		cmp(t, "dTimeStep", gr.DGDN.DTimeStep, m["mix.TimeStep"].Grad)
		cmp(t, "dA", gr.DGDN.DA, m["mix.A"].Grad)
		cmp(t, "dWz", gr.DGDN.DWz, m["mix.Wz"].Grad)
		cmp(t, "dNorm", gr.DGDN.DNorm, m["mix.Norm"].Grad)
		cmp(t, "dWout", gr.DGDN.DWout, m["mix.Wout"].Grad)
		cmp(t, "dState", gr.DState, m["mix.state"].Grad)
	})

	t.Run("full_attention", func(t *testing.T) {
		m := g.Attention
		heads, kv := g.Dims.Attention.Heads, g.Dims.Attention.Kv
		w := HybridLayerWeights{
			InputNorm: f32(m["InputNorm"].Values), PostNorm: f32(m["PostNorm"].Values), IsLinear: false,
			MLP: QwenMLPWeights{Gate: f32(m["mlp.Gate"].Values), Up: f32(m["mlp.Up"].Values), Down: f32(m["mlp.Down"].Values)},
			Attn: AttentionMixWeights{
				Wq: f32(m["mix.Wq"].Values), Wk: f32(m["mix.Wk"].Values), Wv: f32(m["mix.Wv"].Values), Wo: f32(m["mix.Wo"].Values),
				QNorm: f32(m["mix.QNorm"].Values), KNorm: f32(m["mix.KNorm"].Values),
			},
		}
		d := HybridLayerDims{Tokens: T, Hidden: H, Inter: inter, Eps: eps,
			Attn: AttentionMixDims{Tokens: T, Hidden: H, Heads: heads, KVHeads: kv, HeadDim: hd, RopeTheta: g.Dims.Attention.Theta, Eps: eps}}
		x := f32(g.XAttn.Values)
		out, c := HybridDecoderLayerForward(x, w, d, nil)
		cmpOut(t, out, g.OutAttn)
		gr := HybridDecoderLayerBackward(x, w, d, nil, dOut, c)
		cmp(t, "dX", gr.DX, g.XAttn.Grad)
		cmp(t, "dInputNorm", gr.DInputNorm, m["InputNorm"].Grad)
		cmp(t, "dPostNorm", gr.DPostNorm, m["PostNorm"].Grad)
		cmp(t, "dMLP.Gate", gr.DMLP.Gate, m["mlp.Gate"].Grad)
		cmp(t, "dMLP.Up", gr.DMLP.Up, m["mlp.Up"].Grad)
		cmp(t, "dMLP.Down", gr.DMLP.Down, m["mlp.Down"].Grad)
		cmp(t, "dWq", gr.DAttn.Wq, m["mix.Wq"].Grad)
		cmp(t, "dWk", gr.DAttn.Wk, m["mix.Wk"].Grad)
		cmp(t, "dWv", gr.DAttn.Wv, m["mix.Wv"].Grad)
		cmp(t, "dWo", gr.DAttn.Wo, m["mix.Wo"].Grad)
		cmp(t, "dQNorm", gr.DAttn.QNorm, m["mix.QNorm"].Grad)
		cmp(t, "dKNorm", gr.DAttn.KNorm, m["mix.KNorm"].Grad)
	})
}
