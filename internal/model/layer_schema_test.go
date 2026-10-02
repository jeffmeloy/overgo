package model

import (
	"bytes"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// TestHostLayerGraphInputsAliasLoadedTensors drives the unified layer schema
// through LoadHostLayer and GraphInputs: every bound slot loads, becomes a
// named graph input whose feed aliases the host tensor, unset slots stay
// unbound, and runtime-only fields never enter the weight catalog.
func TestHostLayerGraphInputsAliasLoadedTensors(t *testing.T) {
	for _, slot := range layerBindingSchema {
		if layerRuntimeOnlySlots[slot.name] {
			t.Errorf("runtime-only field %q entered weight catalog", slot.name)
		}
	}

	data := hostTensorFixture(t, "weight", 4)
	file, err := gguf.Parse(bytes.NewReader(data), uint64(len(data)), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	info := file.Tensors[0]
	layer := LayerWeights{
		AttentionQ:     &info,
		AttentionQNorm: &info, FeedForwardRouterBias: &info,
		SSMQueryConv: &info, TimeMixW1: &info, ShortConvKernel: &info,
	}
	host, err := LoadHostLayer(t.Context(), file, layer)
	if err != nil {
		t.Fatal(err)
	}
	if host.AttentionQ == nil || len(host.AttentionQ.Data) != 4 || host.AttentionK != nil {
		t.Fatalf("host layer loaded %+v", host.AttentionQ)
	}
	builder := tensor.NewBuilder()
	graph, feeds, err := host.GraphInputs(builder, "blk.0.")
	if err != nil {
		t.Fatal(err)
	}
	if len(feeds) != 6 || graph.AttentionK != nil {
		t.Fatalf("bound %d graph inputs, want 6", len(feeds))
	}
	for name, pair := range map[string]struct {
		host []float32
		node *tensor.Tensor
	}{
		"blk.0.attention_q":              {host.AttentionQ.Data, graph.AttentionQ},
		"blk.0.attention_q_norm":         {host.AttentionQNorm.Data, graph.AttentionQNorm},
		"blk.0.feed_forward_router_bias": {host.FeedForwardRouterBias.Data, graph.FeedForwardRouterBias},
		"blk.0.ssm_query_conv":           {host.SSMQueryConv.Data, graph.SSMQueryConv},
		"blk.0.time_mix_w1":              {host.TimeMixW1.Data, graph.TimeMixW1},
		"blk.0.short_conv_kernel":        {host.ShortConvKernel.Data, graph.ShortConvKernel},
	} {
		if pair.node == nil || pair.node.Name != name {
			t.Errorf("%s graph node = %+v", name, pair.node)
			continue
		}
		var feed reference.Value
		ok := false
		for _, binding := range feeds {
			if binding.Node == pair.node {
				feed, ok = binding.Value, true
				break
			}
		}
		if !ok {
			t.Errorf("%s graph node has no feed", name)
			continue
		}
		if len(feed.Data) != len(pair.host) || &feed.Data[0] != &pair.host[0] {
			t.Errorf("%s graph feed does not alias the host tensor", name)
		}
	}
}
