// Package hybridtrain assembles a multi-layer qwen3.5-style HYBRID decoder stack
// (a mix of full_attention and linear_attention/GDN layers) into a training loop
// and provides a host reference trajectory (this file) plus a device resident
// trajectory (hybrid_train_device_cuda_windows.go). The host reference is the
// parity oracle: it runs the same N-layer stack the device loop runs, K steps of
// the same optimizer, entirely on host with hostmath.HybridDecoderLayerForward /
// HybridDecoderLayerBackward.
package hybridtrain

import (
	"math/rand"

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
)

// LayerKind selects a layer's mix variant.
type LayerKind int

const (
	FullAttention LayerKind = iota
	LinearAttention
)

// StackConfig is a small, fully explicit hybrid model: every dimension comes from
// here (no magics), and Types lists the per-layer mix variant (qwen3.5-style
// layer_types). Dims are shared across layers; the GDN and attention geometries
// are the two mix variants' shapes.
type StackConfig struct {
	Types                 []LayerKind
	Tokens, Hidden, Inter int
	Eps                   float64
	// full_attention geometry.
	Heads, KVHeads, HeadDim int
	RopeDim                 int
	RopeTheta               float64
	// linear_attention (GDN) geometry.
	GDNKeyHeads, GDNValueHeads, GDNHeadDim, GDNConvK int
}

// matDesc describes one resident-Muon matrix: its element range in the flat matrix
// weight buffer and its [rows,cols] geometry (weights stored [out,in]).
type matDesc struct {
	name       string
	off, size  int
	rows, cols int
}

// Model is one built hybrid stack: the per-layer weights (whose matrix slices ALIAS
// the flat matW buffer and whose vector slices alias vecW, so an optimizer step on
// the flat buffers is seen by the forward with no scatter), the per-layer dims and
// GDN input states, the fixed input x and loss target, and the optimizer layouts.
type Model struct {
	Cfg     StackConfig
	Weights []hostmath.HybridLayerWeights
	Dims    []hostmath.HybridLayerDims
	States  [][]float32
	X       []float32
	Target  []float32

	matW    []float32 // flat resident-Muon matrix weights (layer slices alias this)
	vecW    []float32 // flat host-Sign vector params (layer slices alias this)
	mats    []matDesc // resident-Muon matrices, in canonical stack order
	matPlan optimizer.Plan
	vecPlan optimizer.Plan
}

// MatrixParamCount / VectorParamCount expose the two optimizer buffers' sizes.
func (m *Model) MatrixParamCount() int { return len(m.matW) }
func (m *Model) VectorParamCount() int { return len(m.vecW) }

// layerDims builds the per-layer HybridLayerDims for a layer kind from the config.
func (c StackConfig) layerDims(kind LayerKind) hostmath.HybridLayerDims {
	d := hostmath.HybridLayerDims{Tokens: c.Tokens, Hidden: c.Hidden, Inter: c.Inter, Eps: c.Eps}
	switch kind {
	case FullAttention:
		d.Attn = hostmath.AttentionMixDims{
			Tokens: c.Tokens, Hidden: c.Hidden, Heads: c.Heads, KVHeads: c.KVHeads, HeadDim: c.HeadDim,
			RopeDim: c.RopeDim, RopeTheta: c.RopeTheta, Eps: c.Eps,
		}
	case LinearAttention:
		d.GDN = hostmath.GatedDeltaMixDims{
			Tokens: c.Tokens, Hidden: c.Hidden, KeyHeads: c.GDNKeyHeads, ValueHeads: c.GDNValueHeads,
			HeadDim: c.GDNHeadDim, ConvK: c.GDNConvK, OutDim: c.Hidden, Eps: c.Eps,
		}
	}
	return d
}

// builder accumulates the flat weight buffers while wiring each layer's slices as
// aliases into them, recording matrix descriptors and optimizer group specs.
type builder struct {
	m        *Model
	rng      *rand.Rand
	matSpecs []optimizer.GroupSpec
	vecSpecs []optimizer.GroupSpec
}

// randn draws n N(0,0.3) weights (well-conditioned small init, matching the layer
// tests) directly appended into dst; returns the freshly-appended sub-slice.
func (b *builder) randn(dst *[]float32, n int) []float32 {
	start := len(*dst)
	for i := 0; i < n; i++ {
		*dst = append(*dst, float32(b.rng.NormFloat64()*0.3))
	}
	return (*dst)[start : start+n]
}

// mat appends a [rows,cols] matrix into matW, records it as a resident-Muon group,
// and returns its aliasing slice.
func (b *builder) mat(name string, rows, cols int) []float32 {
	off := len(b.m.matW)
	s := b.randn(&b.m.matW, rows*cols)
	b.m.mats = append(b.m.mats, matDesc{name: name, off: off, size: rows * cols, rows: rows, cols: cols})
	b.matSpecs = append(b.matSpecs, optimizer.GroupSpec{Name: name, Start: off, End: off + rows*cols, Rows: rows, Cols: cols})
	return s
}

// vec appends an n-element host-Sign param into vecW (recorded as a 1xN Sign group)
// and returns its aliasing slice.
func (b *builder) vec(name string, n int) []float32 {
	off := len(b.m.vecW)
	s := b.randn(&b.m.vecW, n)
	b.vecSpecs = append(b.vecSpecs, optimizer.GroupSpec{Name: name, Start: off, End: off + n, Rows: 1, Cols: n})
	return s
}

// BuildModel constructs the hybrid stack deterministically from seed. Two calls with
// the same seed and config produce bit-identical initial weights -- the host and
// device trajectories start from the same point.
func BuildModel(cfg StackConfig, seed int64) (*Model, error) {
	m := &Model{Cfg: cfg}
	b := &builder{m: m, rng: rand.New(rand.NewSource(seed))}
	H, inter := cfg.Hidden, cfg.Inter

	for li, kind := range cfg.Types {
		w := hostmath.HybridLayerWeights{IsLinear: kind == LinearAttention}
		p := func(s string) string { return name(li, s) }
		// common: norms (vec) + SwiGLU MLP (mat).
		w.InputNorm = b.vec(p("input_norm"), H)
		w.PostNorm = b.vec(p("post_norm"), H)
		w.MLP.Gate = b.mat(p("mlp.gate"), inter, H)
		w.MLP.Up = b.mat(p("mlp.up"), inter, H)
		w.MLP.Down = b.mat(p("mlp.down"), H, inter)

		switch kind {
		case FullAttention:
			qDim, kvDim := cfg.Heads*cfg.HeadDim, cfg.KVHeads*cfg.HeadDim
			w.Attn.Wq = b.mat(p("attn.q"), qDim, H)
			w.Attn.Wk = b.mat(p("attn.k"), kvDim, H)
			w.Attn.Wv = b.mat(p("attn.v"), kvDim, H)
			w.Attn.Wo = b.mat(p("attn.o"), H, qDim)
			w.Attn.QNorm = b.vec(p("attn.qnorm"), cfg.HeadDim)
			w.Attn.KNorm = b.vec(p("attn.knorm"), cfg.HeadDim)
		case LinearAttention:
			keyDim := cfg.GDNKeyHeads * cfg.GDNHeadDim
			valDim := cfg.GDNValueHeads * cfg.GDNHeadDim
			hv, K := cfg.GDNValueHeads, cfg.GDNConvK
			w.GDN.Wq = b.mat(p("gdn.q"), keyDim, H)
			w.GDN.Wk = b.mat(p("gdn.k"), keyDim, H)
			w.GDN.Wv = b.mat(p("gdn.v"), valDim, H)
			w.GDN.Wbeta = b.mat(p("gdn.beta"), hv, H)
			w.GDN.Walpha = b.mat(p("gdn.alpha"), hv, H)
			w.GDN.Wz = b.mat(p("gdn.z"), valDim, H)
			w.GDN.Wout = b.mat(p("gdn.out"), H, valDim)
			// small conv kernels + biases + per-head scalars stay host-Sign.
			w.GDN.ConvQ = b.vec(p("gdn.convq"), keyDim*K)
			w.GDN.ConvK = b.vec(p("gdn.convk"), keyDim*K)
			w.GDN.ConvV = b.vec(p("gdn.convv"), valDim*K)
			w.GDN.ConvBiasQ = b.vec(p("gdn.cbq"), keyDim)
			w.GDN.ConvBiasK = b.vec(p("gdn.cbk"), keyDim)
			w.GDN.ConvBiasV = b.vec(p("gdn.cbv"), valDim)
			w.GDN.TimeStep = b.vec(p("gdn.dt"), hv)
			w.GDN.A = b.vec(p("gdn.a"), hv)
			w.GDN.Norm = b.vec(p("gdn.norm"), valDim)
		}
		m.Weights = append(m.Weights, w)
		m.Dims = append(m.Dims, cfg.layerDims(kind))
		if kind == LinearAttention {
			m.States = append(m.States, make([]float32, cfg.GDNValueHeads*cfg.GDNHeadDim*cfg.GDNHeadDim))
		} else {
			m.States = append(m.States, nil)
		}
	}

	m.X = make([]float32, cfg.Tokens*H)
	for i := range m.X {
		m.X[i] = float32(b.rng.NormFloat64() * 0.3)
	}
	m.Target = make([]float32, cfg.Tokens*H)
	for i := range m.Target {
		m.Target[i] = float32(b.rng.NormFloat64() * 0.3)
	}

	var err error
	if m.matPlan, err = optimizer.CompilePlan(len(m.matW), b.matSpecs); err != nil {
		return nil, err
	}
	if m.vecPlan, err = optimizer.CompilePlan(len(m.vecW), b.vecSpecs); err != nil {
		return nil, err
	}
	return m, nil
}

func name(layer int, tensor string) string {
	return "layers." + itoa(layer) + "." + tensor
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}

// loss computes the squared-error loss of out against the fixed target and the
// cotangent dOut = out - target for the top layer.
func (m *Model) loss(out []float32) (loss float64, dTop []float32) {
	dTop = make([]float32, len(out))
	for i := range out {
		e := float64(out[i]) - float64(m.Target[i])
		loss += 0.5 * e * e
		dTop[i] = float32(e)
	}
	return loss, dTop
}

// gradMats returns one layer's resident-Muon matrix GRADIENT slices in the exact
// order BuildModel appended the matrices (mlp gate/up/down, then attn q/k/v/o or
// gdn q/k/v/beta/alpha/z/out).
func gradMats(g hostmath.HybridDecoderLayerGrads) [][]float32 {
	out := [][]float32{g.DMLP.Gate, g.DMLP.Up, g.DMLP.Down}
	if g.IsLinear {
		out = append(out, g.DGDN.DWq, g.DGDN.DWk, g.DGDN.DWv, g.DGDN.DWbeta, g.DGDN.DWalpha, g.DGDN.DWz, g.DGDN.DWout)
	} else {
		out = append(out, g.DAttn.Wq, g.DAttn.Wk, g.DAttn.Wv, g.DAttn.Wo)
	}
	return out
}

// gradVecs returns one layer's host-Sign GRADIENT slices in BuildModel's vec order.
func gradVecs(g hostmath.HybridDecoderLayerGrads) [][]float32 {
	out := [][]float32{g.DInputNorm, g.DPostNorm}
	if g.IsLinear {
		out = append(out,
			g.DGDN.DConvQ, g.DGDN.DConvK, g.DGDN.DConvV,
			g.DGDN.DConvBiasQ, g.DGDN.DConvBiasK, g.DGDN.DConvBiasV,
			g.DGDN.DTimeStep, g.DGDN.DA, g.DGDN.DNorm)
	} else {
		out = append(out, g.DAttn.QNorm, g.DAttn.KNorm)
	}
	return out
}

// packGrads flattens per-layer grads into the flat matG/vecG buffers matching the
// matW/vecW layout (concatenation in stack order).
func (m *Model) packGrads(grads []hostmath.HybridDecoderLayerGrads) (matG, vecG []float32) {
	matG = make([]float32, len(m.matW))
	vecG = make([]float32, len(m.vecW))
	mi, vi := 0, 0
	for _, g := range grads {
		for _, s := range gradMats(g) {
			copy(matG[mi:mi+len(s)], s)
			mi += len(s)
		}
		for _, s := range gradVecs(g) {
			copy(vecG[vi:vi+len(s)], s)
			vi += len(s)
		}
	}
	return matG, vecG
}

// TrainHost runs K steps of the reference host trainer (Muon on the resident-set
// matrices via the host optimizer, Sign on the vector params) and returns the loss
// trajectory. This is the parity oracle for the device resident loop.
func (m *Model) TrainHost(steps int, cfg optimizer.Config) ([]float64, error) {
	matGrad := make([]float32, len(m.matW))
	vecGrad := make([]float32, len(m.vecW))
	matOpt, err := optimizer.New(m.matW, matGrad, m.matPlan, cfg)
	if err != nil {
		return nil, err
	}
	vecOpt, err := optimizer.New(m.vecW, vecGrad, m.vecPlan, cfg)
	if err != nil {
		return nil, err
	}

	traj := make([]float64, 0, steps)
	for step := 0; step < steps; step++ {
		out, inputs, caches := hostmath.HybridStackForward(m.X, m.Weights, m.Dims, m.States)
		loss, dTop := m.loss(out)
		traj = append(traj, loss)

		grads, _ := hostmath.HybridStackBackward(inputs, m.Weights, m.Dims, m.States, dTop, caches)
		matG, vecG := m.packGrads(grads)
		copy(matGrad, matG)
		copy(vecGrad, vecG)
		matOpt.Step()
		vecOpt.Step()
	}
	return traj, nil
}
