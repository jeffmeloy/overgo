package hybridtrain

import (
	"math/rand"
	"overgo/internal/hostmath"
)

import "testing"

// TestLayerStreamPlansPartitionSlabs proves the per-layer stream plans are an
// exact partition of the compiled matrix and vector plans: contiguous windows
// in layer order, no gap, no overlap, and every group claimed by exactly one
// layer. This is the layout contract the layer-streamed training lane steps
// through, so a binding-order drift fails here before it can corrupt a run.
func TestLayerStreamPlansPartitionSlabs(t *testing.T) {
	cfg := StackConfig{
		Types:  []LayerKind{FullAttention, LinearAttention, LinearAttention, FullAttention},
		Tokens: 3, Hidden: 8, Inter: 12, Eps: 1e-6,
		Heads: 2, KVHeads: 1, HeadDim: 4, RopeDim: 4, RopeTheta: 10000,
		GDNKeyHeads: 2, GDNValueHeads: 2, GDNHeadDim: 4, GDNConvK: 3,
	}
	m, err := BuildModel(cfg, 7)
	if err != nil {
		t.Fatal(err)
	}
	plans, err := m.layerStreamPlans()
	if err != nil {
		t.Fatal(err)
	}
	if len(plans) != len(cfg.Types) {
		t.Fatalf("plans cover %d layers, want %d", len(plans), len(cfg.Types))
	}
	matNext, vecNext := 0, 0
	for li, p := range plans {
		if p.matOff != matNext || p.matSize <= 0 {
			t.Fatalf("layer %d matrix window [%d,+%d) does not continue at %d", li, p.matOff, p.matSize, matNext)
		}
		if p.vecOff != vecNext || p.vecSize <= 0 {
			t.Fatalf("layer %d vector window [%d,+%d) does not continue at %d", li, p.vecOff, p.vecSize, vecNext)
		}
		if p.plan.ParameterCount() != p.matSize {
			t.Fatalf("layer %d sub-plan covers %d of %d window elements", li, p.plan.ParameterCount(), p.matSize)
		}
		matNext += p.matSize
		vecNext += p.vecSize
	}
	if matNext != m.MatrixParamCount() {
		t.Fatalf("matrix windows cover %d of %d elements", matNext, m.MatrixParamCount())
	}
	if vecNext != m.VectorParamCount() {
		t.Fatalf("vector windows cover %d of %d elements", vecNext, m.VectorParamCount())
	}

	budget, err := m.LayerStreamedBudget()
	if err != nil {
		t.Fatal(err)
	}
	maxLayer := 0
	for _, p := range plans {
		maxLayer = max(maxLayer, p.matSize)
	}
	if budget.ScratchElems != maxLayer || budget.ScratchElems >= budget.MasterElems {
		t.Fatalf("scratch bound %d, want largest layer %d strictly below masters %d",
			budget.ScratchElems, maxLayer, budget.MasterElems)
	}
	if budget.MasterElems != m.MatrixParamCount() || budget.MomentumElems != m.MatrixParamCount() {
		t.Fatalf("budget masters/momentum %d/%d, want %d", budget.MasterElems, budget.MomentumElems, m.MatrixParamCount())
	}
}

// BuildModel constructs a deterministic mixed stack.
func BuildModel(cfg StackConfig, seed int64) (*Model, error) {
	m := &Model{Cfg: cfg}
	b := &builder{m: m, rng: rand.New(rand.NewSource(seed))}
	H, inter := cfg.Hidden, cfg.Inter

	// Stable field addresses for deferred aliases.
	m.Weights = make([]hostmath.HybridLayerWeights, len(cfg.Types))
	m.Dims = make([]hostmath.HybridLayerDims, len(cfg.Types))
	m.States = make([][]float32, len(cfg.Types))

	for li, kind := range cfg.Types {
		w := &m.Weights[li]
		w.IsLinear = kind == LinearAttention
		p := func(s string) string { return name(li, s) }
		// Shared norm and MLP groups.
		b.vec(&w.InputNorm, p("input_norm"), H)
		b.vec(&w.PostNorm, p("post_norm"), H)
		b.mat(&w.MLP.Gate, p("mlp.gate"), inter, H)
		b.mat(&w.MLP.Up, p("mlp.up"), inter, H)
		b.mat(&w.MLP.Down, p("mlp.down"), H, inter)

		switch kind {
		case FullAttention:
			qDim, kvDim := cfg.Heads*cfg.HeadDim, cfg.KVHeads*cfg.HeadDim
			b.mat(&w.Attn.Wq, p("attn.q"), qDim, H)
			b.mat(&w.Attn.Wk, p("attn.k"), kvDim, H)
			b.mat(&w.Attn.Wv, p("attn.v"), kvDim, H)
			b.mat(&w.Attn.Wo, p("attn.o"), H, qDim)
			b.vec(&w.Attn.QNorm, p("attn.qnorm"), cfg.HeadDim)
			b.vec(&w.Attn.KNorm, p("attn.knorm"), cfg.HeadDim)
		case LinearAttention:
			keyDim := cfg.GDNKeyHeads * cfg.GDNHeadDim
			valDim := cfg.GDNValueHeads * cfg.GDNHeadDim
			hv, K := cfg.GDNValueHeads, cfg.GDNConvK
			b.mat(&w.GDN.Wq, p("gdn.q"), keyDim, H)
			b.mat(&w.GDN.Wk, p("gdn.k"), keyDim, H)
			b.mat(&w.GDN.Wv, p("gdn.v"), valDim, H)
			b.mat(&w.GDN.Wbeta, p("gdn.beta"), hv, H)
			b.mat(&w.GDN.Walpha, p("gdn.alpha"), hv, H)
			b.mat(&w.GDN.Wz, p("gdn.z"), valDim, H)
			b.mat(&w.GDN.Wout, p("gdn.out"), H, valDim)
			// Small vectors stay host-resident.
			b.vec(&w.GDN.ConvQ, p("gdn.convq"), keyDim*K)
			b.vec(&w.GDN.ConvK, p("gdn.convk"), keyDim*K)
			b.vec(&w.GDN.ConvV, p("gdn.convv"), valDim*K)
			b.vec(&w.GDN.ConvBiasQ, p("gdn.cbq"), keyDim)
			b.vec(&w.GDN.ConvBiasK, p("gdn.cbk"), keyDim)
			b.vec(&w.GDN.ConvBiasV, p("gdn.cbv"), valDim)
			b.vec(&w.GDN.TimeStep, p("gdn.dt"), hv)
			b.vec(&w.GDN.A, p("gdn.a"), hv)
			b.vec(&w.GDN.Norm, p("gdn.norm"), cfg.GDNHeadDim)
		}
		m.Dims[li] = cfg.layerDims(kind)
		if kind == LinearAttention {
			m.States[li] = make([]float32, cfg.GDNValueHeads*cfg.GDNHeadDim*cfg.GDNHeadDim)
		}
	}
	// Publish final slab aliases.
	b.rebindAliases()

	m.X = make([]float32, cfg.Tokens*H)
	for i := range m.X {
		m.X[i] = float32(b.rng.NormFloat64() * 0.3)
	}
	m.Target = make([]float32, cfg.Tokens*H)
	for i := range m.Target {
		m.Target[i] = float32(b.rng.NormFloat64() * 0.3)
	}

	if err := b.finish(); err != nil {
		return nil, err
	}
	return m, nil
}
