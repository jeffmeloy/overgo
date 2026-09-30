package e4btrain

import (
	"context"
	"errors"
	"fmt"
	"reflect"

	"overgo/internal/gguf"
	"overgo/internal/hostmath"
	"overgo/internal/model"
	"overgo/internal/tensor/reference"
)

// modeledLayerTensors are the layer tensors the trainer reads. An artifact
// whose layer declares any other is refused by that tensor's name, so the
// trainer never trains a different function than the artifact serves.
var modeledLayerTensors = map[string]bool{
	"AttentionNorm": true, "AttentionQ": true, "AttentionQNorm": true, "AttentionK": true, "AttentionKNorm": true,
	"AttentionV": true, "AttentionOutput": true, "AttentionPostNorm": true, "RopeFactors": true,
	"FeedForwardNorm": true, "FeedForwardGate": true, "FeedForwardUp": true, "FeedForwardDown": true,
	"FeedForwardPostNorm": true, "LayerOutputScale": true,
	"PerLayerInputGate": true, "PerLayerProjection": true, "PerLayerPostNorm": true,
}

// LoadArtifact binds a GGUF artifact that declares a Gemma-4 E4B decoder
// for full-parameter training on the token window ids: every trainable
// tensor decodes from its serving storage into the f32 master slab, and the
// geometry comes from the artifact's own declarations. The tied output
// table and the window's per-layer embedding rows load as frozen inputs.
func LoadArtifact(ctx context.Context, path string, ids []uint32) (*Model, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("e4b model: open artifact: %w", err)
	}
	defer file.Close()
	spec, err := model.ReadSpec(file)
	if err != nil {
		return nil, err
	}
	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		return nil, err
	}
	if err := requireModeled(spec, weights); err != nil {
		return nil, err
	}
	shape, err := readGeometry(ctx, file, spec, weights)
	if err != nil {
		return nil, err
	}
	table := weights.TokenEmbedding
	if weights.Output != nil {
		table = *weights.Output
	}
	head, err := model.LoadHostTensor(ctx, file, table)
	if err != nil {
		return nil, err
	}
	if len(head.Data)%shape.hidden != 0 {
		return nil, fmt.Errorf("e4b model: output table %s is not rows of the %d-wide hidden state", table.Name, shape.hidden)
	}
	shape.vocab = len(head.Data) / shape.hidden
	window := make([]int, len(ids))
	for index, id := range ids {
		window[index] = int(id)
	}
	var perLayerRows []float32
	if shape.perLayer > 0 && len(ids) >= shortestWindow {
		rows, err := model.LoadHostRows(ctx, file, *weights.PerLayerTokenEmbedding, ids[:len(ids)-1])
		if err != nil {
			return nil, err
		}
		perLayerRows = rows.Data
	}
	m, err := newModel(shape, head.Data, perLayerRows, window)
	if err != nil {
		return nil, err
	}
	for index := range weights.Layers {
		host, err := model.LoadHostLayer(ctx, file, weights.Layers[index])
		if err != nil {
			return nil, fmt.Errorf("e4b model: load layer %d: %w", index, err)
		}
		if err := m.bindLayer(index, host); err != nil {
			return nil, err
		}
	}
	globals := []struct {
		name string
		dst  []float32
		info *gguf.TensorInfo
	}{
		{rolePerLayerProjection, m.global.perLayerProjection, weights.PerLayerModelProjection},
		{rolePerLayerNorm, m.global.perLayerNorm, weights.PerLayerProjectionNorm},
		{roleOutputNorm, m.global.outputNorm, &weights.OutputNorm},
	}
	for _, global := range globals {
		if global.dst == nil {
			continue
		}
		value, err := model.LoadHostTensor(ctx, file, *global.info)
		if err != nil {
			return nil, err
		}
		if err := bind(global.name, global.dst, &value); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// requireModeled refuses an artifact that declares what the trainer does
// not model: a layer tensor beyond modeledLayerTensors, input or output
// tensors beyond the tied table and final norm, rotary scaling, attention
// soft caps or bidirectional attention.
func requireModeled(spec model.Spec, weights model.Weights) error {
	if spec.RopeScalingType != "" || spec.AttentionSoftcap != 0 || spec.NonCausalAttention {
		return errors.New("e4b model: the artifact declares rotary scaling, an attention soft cap or bidirectional attention")
	}
	if weights.OutputBias != nil || weights.OutputNormBias != nil || weights.TokenEmbeddingNorm != nil ||
		weights.PositionEmbedding != nil || weights.TokenTypeEmbedding != nil || weights.AltUpProjection != nil {
		return errors.New("e4b model: the artifact declares input or output tensors beyond a tied table and a final norm")
	}
	if spec.EmbeddingPerLayer > 0 && (weights.PerLayerTokenEmbedding == nil || weights.PerLayerModelProjection == nil || weights.PerLayerProjectionNorm == nil) {
		return errors.New("e4b model: the artifact declares per-layer inputs without their tables")
	}
	for index, layer := range weights.Layers {
		value := reflect.ValueOf(layer)
		for field := range value.NumField() {
			slot := value.Field(field)
			if name := value.Type().Field(field).Name; slot.Kind() == reflect.Pointer && !slot.IsNil() && !modeledLayerTensors[name] {
				return fmt.Errorf("e4b model: layer %d declares %s, which the trainer does not model", index, name)
			}
		}
	}
	return nil
}

// readGeometry reads the decoder's shape from the artifact's declarations:
// per layer, its heads and head width, feed-forward width, sliding window,
// the layer whose keys and values it attends, and its rotary width, base
// and frequency factors.
func readGeometry(ctx context.Context, file *gguf.File, spec model.Spec, weights model.Weights) (geometry, error) {
	shape := geometry{
		hidden: int(spec.EmbeddingLength), perLayer: int(spec.EmbeddingPerLayer), eps: float64(spec.RMSNormEpsilon),
		attentionScale: float64(spec.AttentionScale), embeddingScale: float64(spec.InputEmbeddingScale()),
		logitScale: float64(spec.OutputLogitMultiplier()), softcap: float64(spec.FinalLogitSoftcap),
		perLayerScale: float64(spec.Profile().Runtime.ActivationResidualScale),
		layers:        make([]layerGeometry, len(weights.Layers)),
	}
	if shape.hidden <= 0 || !(shape.attentionScale > 0) {
		return geometry{}, errors.New("e4b model: the artifact declares no hidden width or attention scale")
	}
	for index := range weights.Layers {
		layer := uint32(index)
		if spec.LayerKeyLength(layer) != spec.LayerValueLength(layer) || !spec.UsesRoPE(layer) {
			return geometry{}, fmt.Errorf("e4b model: layer %d keys and values differ in width, or it is not rotary", index)
		}
		width := spec.LayerRopeDimensionCount(layer)
		base := spec.RopeFrequencyBase
		window := 0
		if spec.IsSlidingLayer(layer) {
			base, window = spec.RopeFrequencySWA, int(spec.SlidingWindow)
		}
		if width == 0 || width%pairWidth != 0 || !(base > 0) || spec.IsSlidingLayer(layer) && window <= 0 {
			return geometry{}, fmt.Errorf("e4b model: layer %d rotary width, base or window is invalid", index)
		}
		invFreq := hostmath.RopeInvFreq(float64(base), int(width))
		if factors := weights.Layers[index].RopeFactors; factors != nil {
			value, err := model.LoadHostTensor(ctx, file, *factors)
			if err != nil {
				return geometry{}, err
			}
			if len(value.Data) != len(invFreq) {
				return geometry{}, fmt.Errorf("e4b model: layer %d has %d rotary factors for %d pairs", index, len(value.Data), len(invFreq))
			}
			for pair, factor := range value.Data {
				invFreq[pair] /= float64(factor)
			}
		}
		source := index
		if !spec.LayerHasKV(layer) {
			source = int(spec.LayerSharedKVSource(layer))
		}
		shape.layers[index] = layerGeometry{
			heads: int(spec.LayerHeadCount(layer)), kvHeads: int(spec.LayerKVHeadCount(layer)),
			headDim: int(spec.LayerKeyLength(layer)), inter: int(spec.LayerFeedForwardLength(layer)),
			window: window, source: source, invFreq: invFreq,
			scaled: weights.Layers[index].LayerOutputScale != nil,
		}
	}
	return shape, nil
}

// bindLayer copies one loaded layer's tensors into the layer's views.
func (m *Model) bindLayer(index int, host model.HostLayer) error {
	w := m.layers[index]
	for _, binding := range []struct {
		role string
		dst  []float32
		src  *reference.Value
	}{
		{roleAttentionNorm, w.attentionNorm, host.AttentionNorm},
		{roleQuery, w.query, host.AttentionQ},
		{roleQueryNorm, w.queryNorm, host.AttentionQNorm},
		{roleKey, w.key, host.AttentionK},
		{roleKeyNorm, w.keyNorm, host.AttentionKNorm},
		{roleValue, w.value, host.AttentionV},
		{roleAttentionOutput, w.attentionOutput, host.AttentionOutput},
		{rolePostAttentionNorm, w.postAttentionNorm, host.AttentionPostNorm},
		{roleFeedForwardNorm, w.feedForwardNorm, host.FeedForwardNorm},
		{roleGate, w.gate, host.FeedForwardGate},
		{roleUp, w.up, host.FeedForwardUp},
		{roleDown, w.down, host.FeedForwardDown},
		{rolePostFeedForward, w.postFeedForward, host.FeedForwardPostNorm},
		{roleAdapterGate, w.adapterGate, host.PerLayerInputGate},
		{roleAdapterProjection, w.adapterProjection, host.PerLayerProjection},
		{roleAdapterNorm, w.adapterNorm, host.PerLayerPostNorm},
		{roleOutputScale, w.outputScale, host.LayerOutputScale},
	} {
		if err := bind(fmt.Sprintf("layer %d %s", index, binding.role), binding.dst, binding.src); err != nil {
			return err
		}
	}
	return nil
}

// bind copies a loaded tensor into the view the geometry laid out for it.
// A tensor the geometry has no view for must be absent, and one it has a
// view for present, with exactly the view's values.
func bind(name string, dst []float32, src *reference.Value) error {
	switch {
	case dst == nil && src == nil:
		return nil
	case dst == nil || src == nil:
		return fmt.Errorf("e4b model: %s is declared where the geometry has no tensor, or missing where it has one", name)
	case len(src.Data) != len(dst):
		return fmt.Errorf("e4b model: %s has %d values, the geometry lays out %d", name, len(src.Data), len(dst))
	}
	copy(dst, src.Data)
	return nil
}
