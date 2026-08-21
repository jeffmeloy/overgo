package hybridtrain

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/gguf"
	"overgo/internal/hostmath"
	"overgo/internal/model"
	"overgo/internal/tensor/reference"
)

// stackGeometry: mix geometry read once from the artifact's own declarations.
// No model facts live here beyond what the spec metadata declares.
type stackGeometry struct {
	hidden, inter int
	// Full-attention mix.
	heads, kvHeads, headDim, ropeDim int
	ropeTheta                        float64
	// Gated-delta mix.
	gdnKeyHeads, gdnValueHeads, gdnHeadDim, gdnConvK int
}

func readStackGeometry(spec model.Spec) (stackGeometry, error) {
	g := stackGeometry{
		hidden:        int(spec.EmbeddingLength),
		inter:         int(spec.FeedForwardLength),
		heads:         int(spec.HeadCount),
		kvHeads:       int(spec.HeadCountKV),
		headDim:       int(spec.KeyLength),
		ropeDim:       int(spec.RopeDimensionCount),
		ropeTheta:     float64(spec.RopeFrequencyBase),
		gdnKeyHeads:   int(spec.SSMGroupCount),
		gdnValueHeads: int(spec.SSMTimeStepRank),
		gdnHeadDim:    int(spec.SSMStateSize),
		gdnConvK:      int(spec.SSMConvKernel),
	}
	if g.hidden <= 0 || g.inter <= 0 {
		return g, errors.New("hybrid stack: hidden and feed-forward lengths are invalid")
	}
	if g.heads <= 0 || g.kvHeads <= 0 || g.headDim <= 0 || g.ropeTheta <= 0 {
		return g, errors.New("hybrid stack: attention geometry is invalid")
	}
	if spec.ValueLength != spec.KeyLength {
		return g, fmt.Errorf("hybrid stack: value length %d differs from key length %d", spec.ValueLength, spec.KeyLength)
	}
	if g.gdnKeyHeads <= 0 || g.gdnValueHeads <= 0 || g.gdnHeadDim <= 0 || g.gdnConvK <= 0 {
		return g, errors.New("hybrid stack: gated-delta geometry is invalid")
	}
	return g, nil
}

// matrixElements / vectorElements: exact slab extents per layer kind, used to
// pre-size the flat optimizer slabs so a multi-gigabyte append never regrows.
func (g stackGeometry) matrixElements(recurrent, convBias bool) (mat, vec int) {
	mat = 3 * g.inter * g.hidden // mlp gate/up/down
	vec = 2 * g.hidden           // input/post norms
	if recurrent {
		keyDim := g.gdnKeyHeads * g.gdnHeadDim
		valDim := g.gdnValueHeads * g.gdnHeadDim
		mat += (2*keyDim + 3*valDim + 2*g.gdnValueHeads) * g.hidden
		vec += (2*keyDim+valDim)*g.gdnConvK + 2*g.gdnValueHeads + g.gdnHeadDim
		if convBias {
			vec += 2*keyDim + valDim
		}
		return mat, vec
	}
	qDim, kvDim := g.heads*g.headDim, g.kvHeads*g.headDim
	mat += (2*qDim + 2*kvDim) * g.hidden
	vec += 2 * g.headDim
	return mat, vec
}

// requireValues rejects absent tensors by name before any binding starts.
func requireValues(layer int, fields map[string][]float32) error {
	for name, data := range fields {
		if len(data) == 0 {
			return fmt.Errorf("hybrid stack: layer %d tensor %s is absent or empty", layer, name)
		}
	}
	return nil
}

// hostData returns a host tensor's values, or nil when the slot is absent.
func hostData(value *reference.Value) []float32 {
	if value == nil {
		return nil
	}
	return value.Data
}

// vecBinding: one host-vector group in canonical pack order.
type vecBinding struct {
	dst  *[]float32
	name string
	data []float32
}

// bindAttentionLayer binds one full-attention layer in canonical pack order:
// matrices mlp.gate/up/down then attn q/k/v/o; vectors input/post norm then
// per-head q/k RMSNorm scales.
func bindAttentionLayer(b *builder, li int, g stackGeometry, host *model.HostLayer) error {
	w := &b.m.Weights[li]
	w.IsLinear = false
	p := func(s string) string { return name(li, s) }
	qDim, kvDim := g.heads*g.headDim, g.kvHeads*g.headDim

	gate := hostData(host.FeedForwardGate)
	up := hostData(host.FeedForwardUp)
	down := hostData(host.FeedForwardDown)
	q := hostData(host.AttentionQ)
	k := hostData(host.AttentionK)
	v := hostData(host.AttentionV)
	o := hostData(host.AttentionOutput)
	if err := requireValues(li, map[string][]float32{
		"attn_norm": hostData(host.AttentionNorm), "ffn_norm": hostData(host.FeedForwardNorm),
		"ffn_gate": gate, "ffn_up": up, "ffn_down": down,
		"attn_q": q, "attn_k": k, "attn_v": v, "attn_output": o,
		"attn_q_norm": hostData(host.AttentionQNorm), "attn_k_norm": hostData(host.AttentionKNorm),
	}); err != nil {
		return err
	}
	// The projection may fuse a per-head output gate after each head's query
	// rows ([q(headDim) | gate(headDim)] per head, the serving GroupSlice
	// split). The training mix models ungated attention, so only the query
	// rows enter the trainable slab.
	switch len(q) {
	case qDim * g.hidden:
	case 2 * qDim * g.hidden:
		query := make([]float32, qDim*g.hidden)
		for h := 0; h < g.heads; h++ {
			src := q[(h*2*g.headDim)*g.hidden : (h*2*g.headDim+g.headDim)*g.hidden]
			copy(query[h*g.headDim*g.hidden:], src)
		}
		q = query
	default:
		return fmt.Errorf("hybrid stack: layer %d query projection has %d values, need %d or %d",
			li, len(q), qDim*g.hidden, 2*qDim*g.hidden)
	}
	if err := b.vecValues(&w.InputNorm, p("input_norm"), hostData(host.AttentionNorm)); err != nil {
		return err
	}
	if err := b.vecValues(&w.PostNorm, p("post_norm"), hostData(host.FeedForwardNorm)); err != nil {
		return err
	}
	for _, binding := range []struct {
		dst        *[]float32
		name       string
		rows, cols int
		data       []float32
	}{
		{&w.MLP.Gate, p("mlp.gate"), g.inter, g.hidden, gate},
		{&w.MLP.Up, p("mlp.up"), g.inter, g.hidden, up},
		{&w.MLP.Down, p("mlp.down"), g.hidden, g.inter, down},
		{&w.Attn.Wq, p("attn.q"), qDim, g.hidden, q},
		{&w.Attn.Wk, p("attn.k"), kvDim, g.hidden, k},
		{&w.Attn.Wv, p("attn.v"), kvDim, g.hidden, v},
		{&w.Attn.Wo, p("attn.o"), g.hidden, qDim, o},
	} {
		if err := b.matValues(binding.dst, binding.name, binding.rows, binding.cols, binding.data); err != nil {
			return err
		}
	}
	for _, binding := range []vecBinding{
		{&w.Attn.QNorm, p("attn.qnorm"), hostData(host.AttentionQNorm)},
		{&w.Attn.KNorm, p("attn.knorm"), hostData(host.AttentionKNorm)},
	} {
		if len(binding.data) != g.headDim {
			return fmt.Errorf("hybrid stack: layer %d %s has %d values, need %d", li, binding.name, len(binding.data), g.headDim)
		}
		if err := b.vecValues(binding.dst, binding.name, binding.data); err != nil {
			return err
		}
	}
	return nil
}

// bindRecurrentLayer binds one gated-delta layer in canonical pack order:
// matrices mlp.gate/up/down then gdn q/k/v/beta/alpha/z/out; vectors input/post
// norm then conv kernels, conv biases (when declared), dt, a, norm.
func bindRecurrentLayer(b *builder, li int, g stackGeometry, host *model.HostLayer) error {
	w := &b.m.Weights[li]
	w.IsLinear = true
	p := func(s string) string { return name(li, s) }
	keyDim := g.gdnKeyHeads * g.gdnHeadDim
	valDim := g.gdnValueHeads * g.gdnHeadDim

	qkv := hostData(host.AttentionQKV)
	conv := hostData(host.SSMConv1D)
	if err := requireValues(li, map[string][]float32{
		"attn_norm": hostData(host.AttentionNorm), "ffn_norm": hostData(host.FeedForwardNorm),
		"ffn_gate": hostData(host.FeedForwardGate), "ffn_up": hostData(host.FeedForwardUp), "ffn_down": hostData(host.FeedForwardDown),
		"attn_qkv": qkv, "attn_gate": hostData(host.AttentionGate), "ssm_conv1d": conv,
		"ssm_dt": hostData(host.SSMTimeStep), "ssm_a": hostData(host.SSMA),
		"ssm_beta": hostData(host.SSMBeta), "ssm_alpha": hostData(host.SSMAlpha),
		"ssm_norm": hostData(host.SSMNorm), "ssm_out": hostData(host.SSMOutput),
	}); err != nil {
		return err
	}
	if host.AttentionQKVBias != nil {
		return fmt.Errorf("hybrid stack: layer %d gated-delta projection bias is unsupported", li)
	}
	if len(qkv) != (2*keyDim+valDim)*g.hidden {
		return fmt.Errorf("hybrid stack: layer %d fused projection has %d values, need %d", li, len(qkv), (2*keyDim+valDim)*g.hidden)
	}
	convQEnd := keyDim * g.gdnConvK
	convKEnd := 2 * convQEnd
	if len(conv) != convKEnd+valDim*g.gdnConvK {
		return fmt.Errorf("hybrid stack: layer %d convolution has %d values, need %d", li, len(conv), convKEnd+valDim*g.gdnConvK)
	}
	if err := b.vecValues(&w.InputNorm, p("input_norm"), hostData(host.AttentionNorm)); err != nil {
		return err
	}
	if err := b.vecValues(&w.PostNorm, p("post_norm"), hostData(host.FeedForwardNorm)); err != nil {
		return err
	}
	for _, binding := range []struct {
		dst        *[]float32
		name       string
		rows, cols int
		data       []float32
	}{
		{&w.MLP.Gate, p("mlp.gate"), g.inter, g.hidden, hostData(host.FeedForwardGate)},
		{&w.MLP.Up, p("mlp.up"), g.inter, g.hidden, hostData(host.FeedForwardUp)},
		{&w.MLP.Down, p("mlp.down"), g.hidden, g.inter, hostData(host.FeedForwardDown)},
		{&w.GDN.Wq, p("gdn.q"), keyDim, g.hidden, qkv[:keyDim*g.hidden]},
		{&w.GDN.Wk, p("gdn.k"), keyDim, g.hidden, qkv[keyDim*g.hidden : 2*keyDim*g.hidden]},
		{&w.GDN.Wv, p("gdn.v"), valDim, g.hidden, qkv[2*keyDim*g.hidden:]},
		{&w.GDN.Wbeta, p("gdn.beta"), g.gdnValueHeads, g.hidden, hostData(host.SSMBeta)},
		{&w.GDN.Walpha, p("gdn.alpha"), g.gdnValueHeads, g.hidden, hostData(host.SSMAlpha)},
		{&w.GDN.Wz, p("gdn.z"), valDim, g.hidden, hostData(host.AttentionGate)},
		{&w.GDN.Wout, p("gdn.out"), g.hidden, valDim, hostData(host.SSMOutput)},
	} {
		if err := b.matValues(binding.dst, binding.name, binding.rows, binding.cols, binding.data); err != nil {
			return err
		}
	}
	vectors := []vecBinding{
		{&w.GDN.ConvQ, p("gdn.convq"), conv[:convQEnd]},
		{&w.GDN.ConvK, p("gdn.convk"), conv[convQEnd:convKEnd]},
		{&w.GDN.ConvV, p("gdn.convv"), conv[convKEnd:]},
	}
	if host.SSMConv1DBias != nil {
		// Bias vectors bind between the conv kernels and dt so the slab layout
		// matches the gradient pack order exactly.
		bias := hostData(host.SSMConv1DBias)
		if len(bias) != 2*keyDim+valDim {
			return fmt.Errorf("hybrid stack: layer %d convolution bias shape differs", li)
		}
		vectors = append(vectors,
			vecBinding{&w.GDN.ConvBiasQ, p("gdn.cbq"), bias[:keyDim]},
			vecBinding{&w.GDN.ConvBiasK, p("gdn.cbk"), bias[keyDim : 2*keyDim]},
			vecBinding{&w.GDN.ConvBiasV, p("gdn.cbv"), bias[2*keyDim:]},
		)
	}
	vectors = append(vectors,
		vecBinding{&w.GDN.TimeStep, p("gdn.dt"), hostData(host.SSMTimeStep)},
		vecBinding{&w.GDN.A, p("gdn.a"), hostData(host.SSMA)},
		vecBinding{&w.GDN.Norm, p("gdn.norm"), hostData(host.SSMNorm)},
	)
	for _, binding := range vectors {
		if err := b.vecValues(binding.dst, binding.name, binding.data); err != nil {
			return err
		}
	}
	return nil
}

// LoadStackArtifact binds every layer of the artifact — full-attention and
// recurrent gated-delta alike, per the spec's own layer cadence — into one
// trainable mixed stack with f32 masters decoded from the serving storage,
// plus the real token rows the Model's hidden-state objective consumes.
func LoadStackArtifact(ctx context.Context, path string, tokenIDs []uint32) (*Model, error) {
	if len(tokenIDs) < 2 {
		return nil, errors.New("hybrid stack: at least two token IDs are required")
	}
	file, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("hybrid stack: open artifact: %w", err)
	}
	defer file.Close()

	spec, err := model.ReadSpec(file)
	if err != nil {
		return nil, fmt.Errorf("hybrid stack: read specification: %w", err)
	}
	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		return nil, fmt.Errorf("hybrid stack: read weight catalog: %w", err)
	}
	if len(weights.Layers) == 0 {
		return nil, errors.New("hybrid stack: artifact declares no layers")
	}
	geometry, err := readStackGeometry(spec)
	if err != nil {
		return nil, err
	}

	// Layer census and exact slab extents before any tensor is decoded, so
	// the multi-gigabyte master slabs allocate exactly once.
	types := make([]LayerKind, len(weights.Layers))
	matTotal, vecTotal := 0, 0
	for li := range weights.Layers {
		recurrent := spec.IsRecurrentLayer(uint32(li))
		if recurrent != weights.Layers[li].Recurrent {
			return nil, fmt.Errorf("hybrid stack: layer %d cadence disagrees with its tensor inventory", li)
		}
		if inter := int(spec.LayerFeedForwardLength(uint32(li))); inter != geometry.inter {
			return nil, fmt.Errorf("hybrid stack: layer %d feed-forward length %d differs from %d", li, inter, geometry.inter)
		}
		types[li] = FullAttention
		if recurrent {
			types[li] = LinearAttention
		}
		mat, vec := geometry.matrixElements(recurrent, weights.Layers[li].SSMConv1DBias != nil)
		matTotal += mat
		vecTotal += vec
	}

	cfg := StackConfig{
		Types:         types,
		Tokens:        len(tokenIDs) - 1,
		Hidden:        geometry.hidden,
		Inter:         geometry.inter,
		Eps:           float64(spec.RMSNormEpsilon),
		Heads:         geometry.heads,
		KVHeads:       geometry.kvHeads,
		HeadDim:       geometry.headDim,
		RopeDim:       geometry.ropeDim,
		RopeTheta:     geometry.ropeTheta,
		GDNKeyHeads:   geometry.gdnKeyHeads,
		GDNValueHeads: geometry.gdnValueHeads,
		GDNHeadDim:    geometry.gdnHeadDim,
		GDNConvK:      geometry.gdnConvK,
	}
	m := &Model{
		Cfg:     cfg,
		Weights: make([]hostmath.HybridLayerWeights, len(types)),
		Dims:    make([]hostmath.HybridLayerDims, len(types)),
		States:  make([][]float32, len(types)),
	}
	m.matW = make([]float32, 0, matTotal)
	m.vecW = make([]float32, 0, vecTotal)
	b := &builder{m: m}

	for li, kind := range types {
		host, err := model.LoadHostLayer(ctx, file, weights.Layers[li])
		if err != nil {
			return nil, fmt.Errorf("hybrid stack: load layer %d: %w", li, err)
		}
		if kind == LinearAttention {
			err = bindRecurrentLayer(b, li, geometry, &host)
			m.States[li] = make([]float32, geometry.gdnValueHeads*geometry.gdnHeadDim*geometry.gdnHeadDim)
		} else {
			err = bindAttentionLayer(b, li, geometry, &host)
		}
		if err != nil {
			return nil, err
		}
		m.Dims[li] = cfg.layerDims(kind)
	}
	if len(m.matW) != matTotal || len(m.vecW) != vecTotal {
		return nil, fmt.Errorf("hybrid stack: bound %d+%d parameters, expected %d+%d",
			len(m.matW), len(m.vecW), matTotal, vecTotal)
	}
	b.rebindAliases()
	if err := b.finish(); err != nil {
		return nil, err
	}

	input, err := model.LoadHostRows(ctx, file, weights.TokenEmbedding, tokenIDs[:len(tokenIDs)-1])
	if err != nil {
		return nil, fmt.Errorf("hybrid stack: load input token rows: %w", err)
	}
	target, err := model.LoadHostRows(ctx, file, weights.TokenEmbedding, tokenIDs[1:])
	if err != nil {
		return nil, fmt.Errorf("hybrid stack: load target token rows: %w", err)
	}
	m.X = input.Data
	m.Target = target.Data
	return m, nil
}
