// Package hybridtrain trains mixed attention stacks.
package hybridtrain

import (
	"fmt"
	"math/rand"

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

// LayerKind selects a layer's mix variant.
type LayerKind int

const (
	FullAttention LayerKind = iota
	LinearAttention
)

// StackConfig declares stack and mix geometry.
type StackConfig struct {
	Types                 []LayerKind
	Tokens, Hidden, Inter int
	Eps                   float64
	// Attention geometry.
	Heads, KVHeads, HeadDim int
	RopeDim                 int
	RopeTheta               float64
	// GDN geometry.
	GDNKeyHeads, GDNValueHeads, GDNHeadDim, GDNConvK int
}

// matDesc binds matrix geometry to a resident range.
type matDesc struct {
	name       string
	off, size  int
	rows, cols int
}

// Model owns aliased matrix/vector optimizer slabs.
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
	program trainingprogram.TrainingProgram
}

// MatrixParamCount / VectorParamCount report optimizer extents.
func (m *Model) MatrixParamCount() int                    { return len(m.matW) }
func (m *Model) VectorParamCount() int                    { return len(m.vecW) }
func (m *Model) Program() trainingprogram.TrainingProgram { return m.program }

// matSlot is one matrix's element offset and size within the flat matW buffer.
type matSlot struct{ Off, Size int }

// layerMatrixPlan binds one layer to matrix-slab ranges.
type layerMatrixPlan struct {
	IsLinear                                   bool
	Gate, Up, Down                             matSlot
	Wq, Wk, Wv, Wo                             matSlot // full_attention
	GWq, GWk, GWv, GWbeta, GWalpha, GWz, GWout matSlot // GDN
}

// matrixPlans resolves resident ranges from compiled tensor names.
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

// slots returns active matrix ranges.
func (p layerMatrixPlan) slots() []matSlot {
	s := []matSlot{p.Gate, p.Up, p.Down}
	if p.IsLinear {
		return append(s, p.GWq, p.GWk, p.GWv, p.GWbeta, p.GWalpha, p.GWz, p.GWout)
	}
	return append(s, p.Wq, p.Wk, p.Wv, p.Wo)
}

// validateMatrixTiling requires an exact positive partition.
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

// layerDims resolves mix geometry.
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

// aliasFixup defers field binding until slab growth ends.
type aliasFixup struct {
	dst       *[]float32 // the layer weight field to repoint
	mat       bool       // true: sub-slice of matW; false: sub-slice of vecW
	off, size int
}

// builder compiles slabs, groups, and final aliases.
type builder struct {
	m        *Model
	rng      *rand.Rand
	matSpecs []optimizer.GroupSpec
	vecSpecs []optimizer.GroupSpec
	fixups   []aliasFixup
}

// randn appends configured bootstrap draws.
func (b *builder) randn(dst *[]float32, n int) {
	for i := 0; i < n; i++ {
		*dst = append(*dst, float32(b.rng.NormFloat64()*0.3))
	}
}

// mat appends one resident matrix group.
func (b *builder) mat(dst *[]float32, name string, rows, cols int) {
	off := len(b.m.matW)
	b.randn(&b.m.matW, rows*cols)
	b.m.mats = append(b.m.mats, matDesc{name: name, off: off, size: rows * cols, rows: rows, cols: cols})
	b.matSpecs = append(b.matSpecs, optimizer.GroupSpec{Name: name, Start: off, End: off + rows*cols, Rows: rows, Cols: cols})
	b.fixups = append(b.fixups, aliasFixup{dst: dst, mat: true, off: off, size: rows * cols})
}

func (b *builder) matValues(dst *[]float32, name string, rows, cols int, values []float32) error {
	if len(values) != rows*cols {
		return fmt.Errorf("hybrid model: matrix %q has %d values, need %d", name, len(values), rows*cols)
	}
	off := len(b.m.matW)
	b.m.matW = append(b.m.matW, values...)
	b.m.mats = append(b.m.mats, matDesc{name: name, off: off, size: len(values), rows: rows, cols: cols})
	b.matSpecs = append(b.matSpecs, optimizer.GroupSpec{Name: name, Start: off, End: off + len(values), Rows: rows, Cols: cols})
	b.fixups = append(b.fixups, aliasFixup{dst: dst, mat: true, off: off, size: len(values)})
	return nil
}

// vec appends one host vector group.
func (b *builder) vec(dst *[]float32, name string, n int) {
	off := len(b.m.vecW)
	b.randn(&b.m.vecW, n)
	b.vecSpecs = append(b.vecSpecs, optimizer.GroupSpec{Name: name, Start: off, End: off + n, Rows: 1, Cols: n})
	b.fixups = append(b.fixups, aliasFixup{dst: dst, mat: false, off: off, size: n})
}

func (b *builder) vecValues(dst *[]float32, name string, values []float32) error {
	if len(values) == 0 {
		return fmt.Errorf("hybrid model: vector %q is empty", name)
	}
	off := len(b.m.vecW)
	b.m.vecW = append(b.m.vecW, values...)
	b.vecSpecs = append(b.vecSpecs, optimizer.GroupSpec{Name: name, Start: off, End: off + len(values), Rows: 1, Cols: len(values)})
	b.fixups = append(b.fixups, aliasFixup{dst: dst, mat: false, off: off, size: len(values)})
	return nil
}

// rebindAliases publishes final slab views.
func (b *builder) rebindAliases() {
	for _, f := range b.fixups {
		if f.mat {
			*f.dst = b.m.matW[f.off : f.off+f.size]
		} else {
			*f.dst = b.m.vecW[f.off : f.off+f.size]
		}
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

func (b *builder) finish() error {
	var err error
	if b.m.matPlan, err = optimizer.CompilePlan(len(b.m.matW), b.matSpecs); err != nil {
		return err
	}
	if b.m.vecPlan, err = optimizer.CompilePlan(len(b.m.vecW), b.vecSpecs); err != nil {
		return err
	}
	all := append([]optimizer.GroupSpec(nil), b.matSpecs...)
	for _, spec := range b.vecSpecs {
		spec.Start += len(b.m.matW)
		spec.End += len(b.m.matW)
		all = append(all, spec)
	}
	plan, err := optimizer.CompilePlan(len(b.m.matW)+len(b.m.vecW), all)
	if err != nil {
		return err
	}
	parameters := make([]trainingprogram.ParameterSpec, len(all))
	for index, spec := range all {
		parameters[index] = trainingprogram.ParameterSpec{Name: spec.Name, Rows: spec.Rows, Cols: spec.Cols, Trainable: !spec.Frozen}
	}
	b.m.program, err = trainingprogram.CompileTrainingProgram(trainingprogram.ProgramSpec{
		Objective: trainingprogram.ObjectiveTokenPrediction,
		Operators: []trainingprogram.OperatorSpec{
			{ID: "hybrid-forward", Phase: trainingprogram.PhaseForward},
			{ID: "squared-error", Phase: trainingprogram.PhaseLoss},
			{ID: "hybrid-backward", Phase: trainingprogram.PhaseBackward},
			{ID: "optimizer-step", Phase: trainingprogram.PhaseOptimize},
		},
		Parameters: parameters,
		Optimizer:  plan,
	})
	return err
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

func lossGradient(out, target, dTop []float32) (float64, []float32) {
	if cap(dTop) < len(out) {
		dTop = make([]float32, len(out))
	} else {
		dTop = dTop[:len(out)]
	}
	var loss float64
	for i := range out {
		e := float64(out[i]) - float64(target[i])
		loss += 0.5 * e * e
		dTop[i] = float32(e)
	}
	return loss, dTop
}

func pack(dst []float32, offset int, groups ...[]float32) int {
	for _, group := range groups {
		offset += copy(dst[offset:], group)
	}
	return offset
}

func (m *Model) packGradients(grads []hostmath.HybridDecoderLayerGrads, mat, vec []float32) {
	mi, vi := 0, 0
	for _, g := range grads {
		mi = pack(mat, mi, g.DMLP.Gate, g.DMLP.Up, g.DMLP.Down)
		vi = pack(vec, vi, g.DInputNorm, g.DPostNorm)
		if g.IsLinear {
			mi = pack(mat, mi, g.DGDN.DWq, g.DGDN.DWk, g.DGDN.DWv, g.DGDN.DWbeta, g.DGDN.DWalpha, g.DGDN.DWz, g.DGDN.DWout)
			vi = pack(vec, vi, g.DGDN.DConvQ, g.DGDN.DConvK, g.DGDN.DConvV, g.DGDN.DConvBiasQ, g.DGDN.DConvBiasK, g.DGDN.DConvBiasV, g.DGDN.DTimeStep, g.DGDN.DA, g.DGDN.DNorm)
		} else {
			mi = pack(mat, mi, g.DAttn.Wq, g.DAttn.Wk, g.DAttn.Wv, g.DAttn.Wo)
			vi = pack(vec, vi, g.DAttn.QNorm, g.DAttn.KNorm)
		}
	}
}

type trainingBackend interface {
	Forward() ([]float32, error)
	Backward([]float32) error
	Step(int) error
}

type trainingState struct {
	backend trainingBackend
	target  []float32
	output  []float32
	dTop    []float32
	loss    float64
	step    int
}

func (state *trainingState) forward() (err error) {
	state.output, err = state.backend.Forward()
	return err
}

func (state *trainingState) lossGradient() error {
	state.loss, state.dTop = lossGradient(state.output, state.target, state.dTop)
	return nil
}

func (state *trainingState) backward() error { return state.backend.Backward(state.dTop) }
func (state *trainingState) optimize() error { return state.backend.Step(state.step + 1) }
func runTraining(program trainingprogram.TrainingProgram, backend trainingBackend, target []float32, steps int) ([]float64, error) {
	return runTrainingObserved(program, backend, target, steps, nil)
}

// runTrainingObserved runs the compiled loop, reporting each committed step's
// loss to observe; an observer error stops training with the trajectory so far.
func runTrainingObserved(program trainingprogram.TrainingProgram, backend trainingBackend, target []float32, steps int, observe func(step int, loss float64) error) ([]float64, error) {
	execution, err := trainingprogram.Bind(program, []trainingprogram.Binding[trainingState]{
		{Operator: "hybrid-forward", Execute: (*trainingState).forward},
		{Operator: "squared-error", Execute: (*trainingState).lossGradient},
		{Operator: "hybrid-backward", Execute: (*trainingState).backward},
		{Operator: "optimizer-step", Execute: (*trainingState).optimize},
	})
	if err != nil {
		return nil, err
	}
	trajectory := make([]float64, steps)
	state := trainingState{backend: backend, target: target}
	for state.step = range steps {
		state.loss = 0
		if err := execution.Run(&state); err != nil {
			return nil, err
		}
		trajectory[state.step] = state.loss
		if observe != nil {
			if err := observe(state.step, state.loss); err != nil {
				return trajectory[:state.step+1], err
			}
		}
	}
	return trajectory, nil
}

type hostTraining struct {
	model            *Model
	trace            hostmath.HybridStackTrace
	grads            []hostmath.HybridDecoderLayerGrads
	matGrad, vecGrad []float32
	matOpt, vecOpt   *optimizer.Optimizer
}

func (training *hostTraining) Forward() ([]float32, error) {
	output, trace := hostmath.HybridStackForward(training.model.X, training.model.Weights, training.model.Dims, training.model.States)
	training.trace = trace
	return output, nil
}

func (training *hostTraining) Backward(dTop []float32) error {
	model := training.model
	training.grads, _ = hostmath.HybridStackBackward(training.trace, model.Weights, model.Dims, model.States, dTop, training.grads)
	model.packGradients(training.grads, training.matGrad, training.vecGrad)
	return nil
}

func (training *hostTraining) Step(_ int) error {
	training.matOpt.Step()
	training.vecOpt.Step()
	return nil
}

// TrainHost runs the host parity lane.
func (m *Model) TrainHost(steps int, cfg optimizer.Config) ([]float64, error) {
	training := &hostTraining{model: m, matGrad: make([]float32, len(m.matW)), vecGrad: make([]float32, len(m.vecW))}
	var err error
	training.matOpt, err = optimizer.New(m.matW, training.matGrad, m.matPlan, cfg)
	if err != nil {
		return nil, err
	}
	training.vecOpt, err = optimizer.New(m.vecW, training.vecGrad, m.vecPlan, cfg)
	if err != nil {
		return nil, err
	}
	return runTraining(m.program, training, m.Target, steps)
}
