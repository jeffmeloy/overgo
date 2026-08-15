package hybridtrain

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/gguf"
	"overgo/internal/hostmath"
	"overgo/internal/model"
)

// LoadRecurrentLayerArtifact binds one gated-delta layer and real token rows.
func LoadRecurrentLayerArtifact(ctx context.Context, path string, layer uint32, tokenIDs []uint32) (*Model, error) {
	if len(tokenIDs) < 2 {
		return nil, errors.New("hybrid model: at least two token IDs are required")
	}
	file, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("hybrid model: open artifact: %w", err)
	}
	defer file.Close()

	spec, err := model.ReadSpec(file)
	if err != nil {
		return nil, fmt.Errorf("hybrid model: read specification: %w", err)
	}
	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		return nil, fmt.Errorf("hybrid model: read weight catalog: %w", err)
	}
	if layer >= uint32(len(weights.Layers)) || !spec.IsRecurrentLayer(layer) || !weights.Layers[layer].Recurrent {
		return nil, fmt.Errorf("hybrid model: layer %d is not a recurrent gated-delta layer", layer)
	}
	host, err := model.LoadHostLayer(ctx, file, weights.Layers[layer])
	if err != nil {
		return nil, fmt.Errorf("hybrid model: load layer %d: %w", layer, err)
	}
	if host.AttentionQKV == nil || host.AttentionGate == nil || host.SSMConv1D == nil ||
		host.SSMTimeStep == nil || host.SSMA == nil || host.SSMBeta == nil || host.SSMAlpha == nil ||
		host.SSMNorm == nil || host.SSMOutput == nil {
		return nil, errors.New("hybrid model: gated-delta tensor set is incomplete")
	}
	if host.AttentionQKVBias != nil {
		return nil, errors.New("hybrid model: gated-delta projection bias is unsupported")
	}

	hidden := int(spec.EmbeddingLength)
	headDim := int(spec.SSMStateSize)
	keyHeads := int(spec.SSMGroupCount)
	valueHeads := int(spec.SSMTimeStepRank)
	keyDim := keyHeads * headDim
	valueDim := valueHeads * headDim
	convK := int(spec.SSMConvKernel)
	inter := int(spec.LayerFeedForwardLength(layer))
	if hidden <= 0 || headDim <= 0 || keyHeads <= 0 || valueHeads <= 0 || convK <= 0 || inter <= 0 {
		return nil, errors.New("hybrid model: gated-delta dimensions are invalid")
	}
	qkv := host.AttentionQKV.Data
	if len(qkv) != (2*keyDim+valueDim)*hidden {
		return nil, fmt.Errorf("hybrid model: fused projection has %d values, need %d", len(qkv), (2*keyDim+valueDim)*hidden)
	}

	cfg := StackConfig{
		Types:         []LayerKind{LinearAttention},
		Tokens:        len(tokenIDs) - 1,
		Hidden:        hidden,
		Inter:         inter,
		Eps:           float64(spec.RMSNormEpsilon),
		GDNKeyHeads:   keyHeads,
		GDNValueHeads: valueHeads,
		GDNHeadDim:    headDim,
		GDNConvK:      convK,
	}
	m := &Model{
		Cfg:     cfg,
		Weights: make([]hostmath.HybridLayerWeights, 1),
		Dims:    []hostmath.HybridLayerDims{cfg.layerDims(LinearAttention)},
		States:  [][]float32{make([]float32, valueHeads*headDim*headDim)},
	}
	b := &builder{m: m}
	w := &m.Weights[0]
	w.IsLinear = true
	p := func(s string) string { return name(int(layer), s) }

	for _, binding := range []struct {
		dst        *[]float32
		name       string
		rows, cols int
		data       []float32
	}{
		{&w.MLP.Gate, p("mlp.gate"), inter, hidden, host.FeedForwardGate.Data},
		{&w.MLP.Up, p("mlp.up"), inter, hidden, host.FeedForwardUp.Data},
		{&w.MLP.Down, p("mlp.down"), hidden, inter, host.FeedForwardDown.Data},
		{&w.GDN.Wq, p("gdn.q"), keyDim, hidden, qkv[:keyDim*hidden]},
		{&w.GDN.Wk, p("gdn.k"), keyDim, hidden, qkv[keyDim*hidden : 2*keyDim*hidden]},
		{&w.GDN.Wv, p("gdn.v"), valueDim, hidden, qkv[2*keyDim*hidden:]},
		{&w.GDN.Wbeta, p("gdn.beta"), valueHeads, hidden, host.SSMBeta.Data},
		{&w.GDN.Walpha, p("gdn.alpha"), valueHeads, hidden, host.SSMAlpha.Data},
		{&w.GDN.Wz, p("gdn.z"), valueDim, hidden, host.AttentionGate.Data},
		{&w.GDN.Wout, p("gdn.out"), hidden, valueDim, host.SSMOutput.Data},
	} {
		if err := b.matValues(binding.dst, binding.name, binding.rows, binding.cols, binding.data); err != nil {
			return nil, err
		}
	}
	conv := host.SSMConv1D.Data
	convQEnd := keyDim * convK
	convKEnd := 2 * convQEnd
	if len(conv) != convKEnd+valueDim*convK {
		return nil, fmt.Errorf("hybrid model: convolution has %d values, need %d", len(conv), convKEnd+valueDim*convK)
	}
	for _, binding := range []struct {
		dst  *[]float32
		name string
		data []float32
	}{
		{&w.InputNorm, p("input_norm"), host.AttentionNorm.Data},
		{&w.PostNorm, p("post_norm"), host.FeedForwardNorm.Data},
		{&w.GDN.ConvQ, p("gdn.convq"), conv[:convQEnd]},
		{&w.GDN.ConvK, p("gdn.convk"), conv[convQEnd:convKEnd]},
		{&w.GDN.ConvV, p("gdn.convv"), conv[convKEnd:]},
		{&w.GDN.TimeStep, p("gdn.dt"), host.SSMTimeStep.Data},
		{&w.GDN.A, p("gdn.a"), host.SSMA.Data},
		{&w.GDN.Norm, p("gdn.norm"), host.SSMNorm.Data},
	} {
		if err := b.vecValues(binding.dst, binding.name, binding.data); err != nil {
			return nil, err
		}
	}
	if host.SSMConv1DBias != nil {
		bias := host.SSMConv1DBias.Data
		if len(bias) != 2*keyDim+valueDim {
			return nil, errors.New("hybrid model: convolution bias shape differs")
		}
		for _, binding := range []struct {
			dst  *[]float32
			name string
			data []float32
		}{
			{&w.GDN.ConvBiasQ, p("gdn.cbq"), bias[:keyDim]},
			{&w.GDN.ConvBiasK, p("gdn.cbk"), bias[keyDim : 2*keyDim]},
			{&w.GDN.ConvBiasV, p("gdn.cbv"), bias[2*keyDim:]},
		} {
			if err := b.vecValues(binding.dst, binding.name, binding.data); err != nil {
				return nil, err
			}
		}
	}
	b.rebindAliases()
	if err := b.finish(); err != nil {
		return nil, err
	}

	input, err := model.LoadHostRows(ctx, file, weights.TokenEmbedding, tokenIDs[:len(tokenIDs)-1])
	if err != nil {
		return nil, fmt.Errorf("hybrid model: load input token rows: %w", err)
	}
	target, err := model.LoadHostRows(ctx, file, weights.TokenEmbedding, tokenIDs[1:])
	if err != nil {
		return nil, fmt.Errorf("hybrid model: load target token rows: %w", err)
	}
	m.X = input.Data
	m.Target = target.Data
	return m, nil
}
