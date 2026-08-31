package model

import (
	"bytes"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// legacyLayerCatalogSlots is the exact slot set the deleted reflection
// synchronizer (layer_graph_field_catalog.go) compiled from the three
// mirrored structs. The unified schema must keep covering every one.
var legacyLayerCatalogSlots = []string{
	"AttentionNorm", "AttentionNormBias", "AttentionNorm2", "AttentionNorm2Bias",
	"AttentionQ", "AttentionQB", "AttentionK", "AttentionV", "AttentionOutput",
	"AttentionQScale", "AttentionKScale", "AttentionVScale", "AttentionOutputScale",
	"AttentionSubNorm", "AttentionQBias", "AttentionKBias", "AttentionVBias",
	"AttentionOutputBias", "AttentionQNorm", "AttentionKNorm", "AttentionQNormBias",
	"AttentionKNormBias", "AttentionPostNorm", "AttentionPostNormBias",
	"AttentionRelativeBias", "CrossAttentionNorm", "CrossAttentionQ",
	"CrossAttentionK", "CrossAttentionV", "CrossAttentionOutput",
	"AttentionOutputGate", "AttentionSinks", "RopeFactors", "FeedForwardNorm",
	"FeedForwardNormBias", "FeedForwardExpertNorm", "FeedForwardGate",
	"FeedForwardUp", "FeedForwardDown", "FeedForwardGateScale",
	"FeedForwardUpScale", "FeedForwardDownScale", "FeedForwardActivationScale",
	"FeedForwardSubNorm", "FeedForwardGateBias", "FeedForwardUpBias",
	"FeedForwardDownBias", "FeedForwardPostNorm", "FeedForwardPostNormBias",
	"FeedForwardPreNorm2", "FeedForwardPostNorm1", "FeedForwardPostNorm2",
	"FeedForwardRouter", "FeedForwardRouterBias", "FeedForwardRouterScale",
	"FeedForwardGateUpExperts", "FeedForwardGateExperts", "FeedForwardUpExperts",
	"FeedForwardDownExperts", "FeedForwardDownExpertsScale",
	"FeedForwardGateChunkExperts", "FeedForwardUpChunkExperts",
	"FeedForwardDownChunkExperts", "FeedForwardExpertBias",
	"FeedForwardLatentDown", "FeedForwardLatentUp", "FeedForwardSharedGate",
	"FeedForwardSharedUp", "FeedForwardSharedDown", "FeedForwardSharedRouter",
	"LayerOutputScale", "PerLayerInputGate", "PerLayerProjection",
	"PerLayerPostNorm", "ShortConvKernel", "ShortConvInput", "ShortConvOutput",
	"AttentionKVAMQA", "AttentionKVANorm", "AttentionKVB", "AttentionKB",
	"AttentionVB", "IndexerKNorm", "IndexerKNormBias", "IndexerProjection",
	"IndexerAttentionK", "IndexerAttentionQB", "AttentionOutputA",
	"AttentionCompressorKV", "AttentionCompressorGate", "AttentionCompressorAPE",
	"AttentionCompressorNorm", "IndexerCompressorKV", "IndexerCompressorGate",
	"IndexerCompressorAPE", "IndexerCompressorNorm", "HyperAttentionFN",
	"HyperAttentionBase", "HyperAttentionScale", "HyperFeedForwardFN",
	"HyperFeedForwardBase", "HyperFeedForwardScale", "HyperHeadFN",
	"HyperHeadBase", "HyperHeadScale", "FeedForwardHashExperts",
	"AltUpCorrectCoefficient", "AltUpCorrectScale", "AltUpPredictCoefficient",
	"AltUpRouter", "AltUpRouterNorm", "LaurelLeft", "LaurelRight",
	"LaurelPostNorm", "AttentionQKV", "AttentionQKVBias", "AttentionGate",
	"SSMConv1D", "SSMConv1DBias", "SSMInput", "SSMX", "SSMTimeStepWeight",
	"SSMTimeStep", "SSMTimeStepNorm", "SSMA", "SSMD", "SSMBNorm", "SSMCNorm",
	"SSMBeta", "SSMAlpha", "SSMBetaAlpha", "SSMNorm", "SSMOutput",
	"SSMQueryConv", "SSMKeyConv", "SSMValueConv", "SSMForgetA", "SSMForgetB",
	"SSMOutputGateA", "SSMOutputGateB", "TimeMixW1", "TimeMixW2", "TimeMixW0",
	"TimeMixA0", "TimeMixA1", "TimeMixA2", "TimeMixV0", "TimeMixV1", "TimeMixV2",
	"TimeMixG1", "TimeMixG2", "TimeMixKK", "TimeMixKA", "TimeMixRK",
	"TimeMixLerpX", "TimeMixLerpFused", "TimeMixLerpW", "TimeMixLerpK",
	"TimeMixLerpV", "TimeMixLerpR", "TimeMixLerpG", "TimeMixFirst",
	"TimeMixDecay", "TimeMixDecayW1", "TimeMixDecayW2", "TimeMixKey",
	"TimeMixValue", "TimeMixReceptance", "TimeMixGate", "TimeMixLN",
	"TimeMixLNBias", "TimeMixOutput", "ChannelMixLerpK", "ChannelMixLerpR",
	"ChannelMixKey", "ChannelMixValue", "ChannelMixReceptance",
}

func TestUnifiedLayerBindingSchema(t *testing.T) {
	slots := make(map[string]layerBindingSlot, len(layerBindingSchema))
	for _, slot := range layerBindingSchema {
		if _, exists := slots[slot.name]; exists {
			t.Fatalf("duplicate slot %q", slot.name)
		}
		if slot.inputName == "" {
			t.Fatalf("slot %q has no graph input name", slot.name)
		}
		slots[slot.name] = slot
	}
	if len(layerBindingSchema) != len(legacyLayerCatalogSlots) {
		t.Errorf("compiled schema has %d slots, legacy catalog had %d",
			len(layerBindingSchema), len(legacyLayerCatalogSlots))
	}
	for _, name := range legacyLayerCatalogSlots {
		if _, ok := slots[name]; !ok {
			t.Errorf("legacy catalog slot %q was lost", name)
		}
	}
	for name := range layerRuntimeOnlySlots {
		if _, ok := slots[name]; ok {
			t.Errorf("runtime-only field %q entered weight catalog", name)
		}
	}
	for input, want := range map[string]string{
		"AttentionQNorm": "attention_q_norm",
		"SSMQueryConv":   "ssm_query_conv",
		"TimeMixW1":      "time_mix_w1",
	} {
		if got := slots[input].inputName; got != want {
			t.Errorf("slot %q input name = %q, want %q", input, got, want)
		}
	}

	// Host and graph views of one loaded layer must expose the same
	// underlying data: the graph feed for every bound slot aliases the
	// host slice, so weights flow without copies or divergence.
	data := hostTensorFixture(t)
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
	builder := tensor.NewBuilder()
	graph, feeds, err := host.GraphInputs(builder, "blk.0.")
	if err != nil {
		t.Fatal(err)
	}
	if len(feeds) != 6 {
		t.Fatalf("bound %d graph inputs, want 6", len(feeds))
	}
	for name, pair := range map[string]struct {
		host []float32
		node *tensor.Tensor
	}{
		"attention Q":       {host.AttentionQ.Data, graph.AttentionQ},
		"attention Q norm":  {host.AttentionQNorm.Data, graph.AttentionQNorm},
		"router bias":       {host.FeedForwardRouterBias.Data, graph.FeedForwardRouterBias},
		"Kimi query conv":   {host.SSMQueryConv.Data, graph.SSMQueryConv},
		"RWKV mix":          {host.TimeMixW1.Data, graph.TimeMixW1},
		"short convolution": {host.ShortConvKernel.Data, graph.ShortConvKernel},
	} {
		if pair.node == nil {
			t.Errorf("%s graph node was not bound", name)
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

func TestGraphInputName(t *testing.T) {
	for input, want := range map[string]string{
		"AttentionQNorm": "attention_q_norm",
		"SSMQueryConv":   "ssm_query_conv",
		"TimeMixW1":      "time_mix_w1",
	} {
		if got := graphInputName(input); got != want {
			t.Errorf("graphInputName(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestLoadHostLayerGraphFields(t *testing.T) {
	data := hostTensorFixture(t)
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
	host := HostLayer{}
	if err := loadHostLayerGraphFields(t.Context(), file, &layer, &host); err != nil {
		t.Fatal(err)
	}
	if host.AttentionQ == nil || host.AttentionQ.Shape.Rank == 0 || len(host.AttentionQ.Data) != 4 {
		t.Fatalf("required attention Q was not loaded: %+v", host.AttentionQ)
	}
	for name, value := range map[string]any{
		"attention Q norm": host.AttentionQNorm, "router bias": host.FeedForwardRouterBias,
		"Kimi query convolution": host.SSMQueryConv, "RWKV mix": host.TimeMixW1,
		"short convolution": host.ShortConvKernel,
	} {
		if value == nil {
			t.Errorf("%s was not loaded", name)
		}
	}
}

func TestBindDeviceLayerGraphInputsPreservesStorageTypes(t *testing.T) {
	const (
		hiddenWidth   = 32
		feedForwardUp = 64
		normPointer   = driver.DevicePtr(11)
		gatePointer   = driver.DevicePtr(22)
	)
	info := func(name string, storage dtype.Type, dimensions ...uint64) *gguf.TensorInfo {
		result := gguf.TensorInfo{Name: name, Type: storage, Dimensions: uint32(len(dimensions))}
		copy(result.Shape[:], dimensions)
		return &result
	}
	layer := LayerWeights{
		AttentionNorm:   info("blk.0.attn_norm.weight", dtype.F32, hiddenWidth),
		FeedForwardGate: info("blk.0.ffn_gate.weight", dtype.Q8_0, hiddenWidth, feedForwardUp),
	}
	pointers := map[string]driver.DevicePtr{
		layer.AttentionNorm.Name:   normPointer,
		layer.FeedForwardGate.Name: gatePointer,
	}
	builder := tensor.NewBuilder()
	graph, feeds, err := BindDeviceLayerGraphInputs(builder, layer, func(
		builder *tensor.Builder,
		info gguf.TensorInfo,
	) (*tensor.Tensor, driver.DevicePtr, error) {
		shape, shapeErr := tensor.NewShape(info.Shape[:info.Dimensions]...)
		if shapeErr != nil {
			return nil, 0, shapeErr
		}
		return builder.Input(info.Name, info.Type, shape), pointers[info.Name], builder.Err()
	})
	if err != nil {
		t.Fatal(err)
	}
	if graph.AttentionNorm.Type != dtype.F32 || graph.FeedForwardGate.Type != dtype.Q8_0 {
		t.Fatalf("storage types = %s/%s, want F32/Q8_0", graph.AttentionNorm.Type, graph.FeedForwardGate.Type)
	}
	if len(feeds) != 2 ||
		feeds[0].Node != graph.AttentionNorm || feeds[0].Value != normPointer ||
		feeds[1].Node != graph.FeedForwardGate || feeds[1].Value != gatePointer {
		t.Fatalf("device feeds = %v", feeds)
	}
}
