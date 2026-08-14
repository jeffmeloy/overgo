// Package hybridtrain assembles a multi-layer qwen3.5-style HYBRID decoder stack
// (a mix of full_attention and linear_attention/GDN layers) into a training loop
// and provides a host reference trajectory (this file) plus a device resident
// trajectory (hybrid_train_device_cuda_windows.go). The host reference is the
// parity oracle: it runs the same N-layer stack the device loop runs, K steps of
// the same optimizer, entirely on host with hostmath.HybridDecoderLayerForward /
// HybridDecoderLayerBackward.
package hybridtrain

import (
	"fmt"
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

// matSlot is one matrix's element offset and size within the flat matW buffer.
type matSlot struct{ Off, Size int }

// layerMatrixPlan is one hybrid layer's MATRIX-weight offsets within matW: the
// SwiGLU MLP plus the active mix's projection matrices. It is the device-free
// source the resident device loop turns into resident weight POINTERS
// (ResidentPtr(dW, Off)) and the layout test validates without a GPU -- the crux
// invariant of the no-weight-motion loop (every resident matrix addressed at the
// right offset, the slots tiling matW exactly).
type layerMatrixPlan struct {
	IsLinear                                   bool
	Gate, Up, Down                             matSlot
	Wq, Wk, Wv, Wo                             matSlot // full_attention
	GWq, GWk, GWv, GWbeta, GWalpha, GWz, GWout matSlot // GDN
}

// matrixPlans resolves every layer's matrix slots from the built mats (by the
// exact BuildModel tensor names), so the resident pointers and the layout test
// share one derivation. Errors if any expected matrix is missing.
func (m *Model) matrixPlans() ([]layerMatrixPlan, error) {
	byName := make(map[string]matSlot, len(m.mats))
	for _, md := range m.mats {
		byName[md.name] = matSlot{Off: md.off, Size: md.size}
	}
	slot := func(li int, tensor string) (matSlot, error) {
		s, ok := byName[name(li, tensor)]
		if !ok {
			return matSlot{}, fmt.Errorf("matrixPlans: missing matrix %q", name(li, tensor))
		}
		return s, nil
	}
	plans := make([]layerMatrixPlan, len(m.Cfg.Types))
	for li, kind := range m.Cfg.Types {
		p := layerMatrixPlan{IsLinear: kind == LinearAttention}
		var err error
		get := func(dst *matSlot, tensor string) {
			if err == nil {
				*dst, err = slot(li, tensor)
			}
		}
		get(&p.Gate, "mlp.gate")
		get(&p.Up, "mlp.up")
		get(&p.Down, "mlp.down")
		if p.IsLinear {
			get(&p.GWq, "gdn.q")
			get(&p.GWk, "gdn.k")
			get(&p.GWv, "gdn.v")
			get(&p.GWbeta, "gdn.beta")
			get(&p.GWalpha, "gdn.alpha")
			get(&p.GWz, "gdn.z")
			get(&p.GWout, "gdn.out")
		} else {
			get(&p.Wq, "attn.q")
			get(&p.Wk, "attn.k")
			get(&p.Wv, "attn.v")
			get(&p.Wo, "attn.o")
		}
		if err != nil {
			return nil, err
		}
		plans[li] = p
	}
	return plans, nil
}

// slots returns every matrix slot in this plan (mix-appropriate), for tiling
// validation and pointer construction.
func (p layerMatrixPlan) slots() []matSlot {
	s := []matSlot{p.Gate, p.Up, p.Down}
	if p.IsLinear {
		return append(s, p.GWq, p.GWk, p.GWv, p.GWbeta, p.GWalpha, p.GWz, p.GWout)
	}
	return append(s, p.Wq, p.Wk, p.Wv, p.Wo)
}

// validateMatrixTiling asserts the plans' matrix slots exactly tile [0,total):
// no gaps, no overlaps, sizes positive. This is the device-free guard that the
// resident weight pointers address every matrix correctly -- meaningful in CI
// without a GPU.
func validateMatrixTiling(plans []layerMatrixPlan, total int) error {
	covered := make([]bool, total)
	sum := 0
	for _, p := range plans {
		for _, s := range p.slots() {
			if s.Size <= 0 || s.Off < 0 || s.Off+s.Size > total {
				return fmt.Errorf("validateMatrixTiling: slot off=%d size=%d out of [0,%d)", s.Off, s.Size, total)
			}
			for i := s.Off; i < s.Off+s.Size; i++ {
				if covered[i] {
					return fmt.Errorf("validateMatrixTiling: element %d covered twice", i)
				}
				covered[i] = true
			}
			sum += s.Size
		}
	}
	if sum != total {
		return fmt.Errorf("validateMatrixTiling: covered %d of %d matrix elements (gap or stray)", sum, total)
	}
	return nil
}

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

// aliasFixup records a weight-slice field to rebind onto the FINAL flat buffer
// after the build completes: mat/vec append into matW/vecW, and append reallocates
// the backing array mid-build, so a slice captured when a tensor is first written
// detaches from the final buffer. rebindAliases re-points every field at the final
// backing array (by offset) so an optimizer step on the flat matW/vecW IS seen by
// the forward -- the aliasing the Model doc promises.
type aliasFixup struct {
	dst       *[]float32 // the layer weight field to repoint
	mat       bool       // true: sub-slice of matW; false: sub-slice of vecW
	off, size int
}

// builder accumulates the flat weight buffers while recording matrix descriptors,
// optimizer group specs, and the alias fixups rebound after the buffers are final.
type builder struct {
	m        *Model
	rng      *rand.Rand
	matSpecs []optimizer.GroupSpec
	vecSpecs []optimizer.GroupSpec
	fixups   []aliasFixup
}

// randn appends n N(0,0.3) weights (well-conditioned small init, matching the layer
// tests) into the growing flat buffer dst.
func (b *builder) randn(dst *[]float32, n int) {
	for i := 0; i < n; i++ {
		*dst = append(*dst, float32(b.rng.NormFloat64()*0.3))
	}
}

// mat appends a [rows,cols] matrix into matW, records it as a resident-Muon group,
// and registers dst for post-build alias rebinding.
func (b *builder) mat(dst *[]float32, name string, rows, cols int) {
	off := len(b.m.matW)
	b.randn(&b.m.matW, rows*cols)
	b.m.mats = append(b.m.mats, matDesc{name: name, off: off, size: rows * cols, rows: rows, cols: cols})
	b.matSpecs = append(b.matSpecs, optimizer.GroupSpec{Name: name, Start: off, End: off + rows*cols, Rows: rows, Cols: cols})
	b.fixups = append(b.fixups, aliasFixup{dst: dst, mat: true, off: off, size: rows * cols})
}

// vec appends an n-element host-Sign param into vecW (recorded as a 1xN Sign group)
// and registers dst for post-build alias rebinding.
func (b *builder) vec(dst *[]float32, name string, n int) {
	off := len(b.m.vecW)
	b.randn(&b.m.vecW, n)
	b.vecSpecs = append(b.vecSpecs, optimizer.GroupSpec{Name: name, Start: off, End: off + n, Rows: 1, Cols: n})
	b.fixups = append(b.fixups, aliasFixup{dst: dst, mat: false, off: off, size: n})
}

// rebindAliases repoints every recorded weight-slice field onto the final matW/vecW
// backing array, restoring the aliasing an optimizer step relies on. Run once, after
// all appends.
func (b *builder) rebindAliases() {
	for _, f := range b.fixups {
		if f.mat {
			*f.dst = b.m.matW[f.off : f.off+f.size]
		} else {
			*f.dst = b.m.vecW[f.off : f.off+f.size]
		}
	}
}

// BuildModel constructs the hybrid stack deterministically from seed. Two calls with
// the same seed and config produce bit-identical initial weights -- the host and
// device trajectories start from the same point.
func BuildModel(cfg StackConfig, seed int64) (*Model, error) {
	m := &Model{Cfg: cfg}
	b := &builder{m: m, rng: rand.New(rand.NewSource(seed))}
	H, inter := cfg.Hidden, cfg.Inter

	// Pre-size m.Weights so each field's address is stable across the build: the
	// alias fixups store &field pointers, which must survive into the final model.
	m.Weights = make([]hostmath.HybridLayerWeights, len(cfg.Types))
	m.Dims = make([]hostmath.HybridLayerDims, len(cfg.Types))
	m.States = make([][]float32, len(cfg.Types))

	for li, kind := range cfg.Types {
		w := &m.Weights[li]
		w.IsLinear = kind == LinearAttention
		p := func(s string) string { return name(li, s) }
		// common: norms (vec) + SwiGLU MLP (mat).
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
			// small conv kernels + biases + per-head scalars stay host-Sign.
			b.vec(&w.GDN.ConvQ, p("gdn.convq"), keyDim*K)
			b.vec(&w.GDN.ConvK, p("gdn.convk"), keyDim*K)
			b.vec(&w.GDN.ConvV, p("gdn.convv"), valDim*K)
			b.vec(&w.GDN.ConvBiasQ, p("gdn.cbq"), keyDim)
			b.vec(&w.GDN.ConvBiasK, p("gdn.cbk"), keyDim)
			b.vec(&w.GDN.ConvBiasV, p("gdn.cbv"), valDim)
			b.vec(&w.GDN.TimeStep, p("gdn.dt"), hv)
			b.vec(&w.GDN.A, p("gdn.a"), hv)
			b.vec(&w.GDN.Norm, p("gdn.norm"), valDim)
		}
		m.Dims[li] = cfg.layerDims(kind)
		if kind == LinearAttention {
			m.States[li] = make([]float32, cfg.GDNValueHeads*cfg.GDNHeadDim*cfg.GDNHeadDim)
		}
	}
	// Rebind every weight slice onto the now-final matW/vecW backing arrays.
	b.rebindAliases()

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
